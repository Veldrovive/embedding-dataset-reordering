"""
The main code for reordering embeddings
"""

import tempfile
import fsspec
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from embedding_dataset_reordering.helper import RuntimeAnalyzer

ra = RuntimeAnalyzer()


def get_shard(file_path):
    basename = os.path.basename(file_path)
    return basename.split("_")[-1].split(".")[0]


def reorder_shard(
    embedding_file,
    metadata_file,
    output_folder,
    index_width,
    output_shard_width,
    verbose=False,
    max_tries=3,
    return_queue=None,
):
    """
    Reorders a single shard
    """
    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    print(f"Reordering embedding shard {embedding_file}")
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(max_tries):
            try:
                # Load the parquet into pandas
                with fsspec.open(metadata_file, "rb") as f:
                    metadata_df = pd.read_parquet(f)
                log("Read metadata")
                metadata_df["img_shard"] = metadata_df.image_path.str[:-index_width].astype(int)
                metadata_df["img_index"] = metadata_df.image_path.str[-index_width:].astype(int)
                metadata_df["original_index"] = metadata_df.index
                metadata_df.drop(
                    metadata_df.columns.difference(["original_index", "img_shard", "img_index"]), 1, inplace=True
                )
                metadata_df = metadata_df.sort_values(by=["img_shard", "img_index"])
                log("Sorted metadata")

                # Read the embeddings into numpy
                with fsspec.open(embedding_file, "rb") as f:
                    embeddings = np.load(f)

                assert len(embeddings) == len(metadata_df)
                log("Read embeddings")

                embedding_shape = embeddings.shape[1:]

                # Now we iterate over the metadata and build new embedding shards named fro the img_shard and in the order of img_index
                # We can find the embedding by indexing the embeddings array at original_index
                shard_embeddings = []
                current_shard = None
                last_index = -1
                for original_index, img_shard, img_index in zip(
                    metadata_df["original_index"], metadata_df["img_shard"], metadata_df["img_index"]
                ):
                    embedding = embeddings[original_index]
                    if current_shard is None or current_shard != img_shard:
                        if current_shard is not None:
                            np.save(
                                os.path.join(tmpdir, f"img_emb_{str(current_shard).zfill(output_shard_width)}.npy"),
                                shard_embeddings,
                            )
                            log(f"Saved shard {current_shard}")
                        shard_embeddings = []
                        current_shard = img_shard
                        last_index = -1
                    if img_index > last_index + 1:
                        # Then we need to fill this gap of embeddings with zeros
                        fill_number = img_index - last_index - 1
                        fill_embeddings = np.zeros((fill_number, *embedding_shape))
                        shard_embeddings.extend(fill_embeddings)
                        if fill_number > 5:
                            log(f"Filling {fill_number} embeddings with zeros")
                    shard_embeddings.append(embedding)
                    last_index = img_index
                np.save(
                    os.path.join(tmpdir, f"img_emb_{str(current_shard).zfill(output_shard_width)}.npy"),
                    shard_embeddings,
                )

                log("Shards", os.listdir(tmpdir))

                # Then we upload all files in s3 to the output folder file system
                output_fs, output_path = fsspec.core.url_to_fs(output_folder)
                for file in os.listdir(tmpdir):
                    output_fs.put(os.path.join(tmpdir, file), os.path.join(output_path, file))
                return_data = {"shard": get_shard(embedding_file), "success": True}
                if return_queue is not None:
                    return_queue.put(return_data)
                return [return_data["shard"], True]
            except Exception as e:  # pylint: disable=broad-except
                # Sometimes we get a broken pipe or similar
                print(f"Error reordering shard {embedding_file} on try {i}: {e}")
        print(f"Failed to reorder shard {embedding_file} after {max_tries} tries")
        return_data = {"shard": get_shard(embedding_file), "success": False}
        if return_queue is not None:
            return_queue.put(return_data)
        return [return_data["shard"], False]


def reorder_embeddings(
    output_folder,
    embeddings_folder,
    metadata_folder,
    index_width,
    output_shard_width,
    limit=None,
    run_concurrent=1,
    concurrent_backend="multiprocessing",
    verbose=False,
):
    """
    Initializes multiprocessing of shard reordering
    """
    assert concurrent_backend in ["multiprocessing", "spark"], "Only multiprocessing and spark are supported"
    assert run_concurrent > 0, "Concurrent workers must be greater than 0"
    ra.start()
    embeddings_fs, embeddings_fs_path = fsspec.core.url_to_fs(embeddings_folder)
    embedding_files = embeddings_fs.ls(embeddings_fs_path)

    def get_protocol(fs):
        return fs.protocol if isinstance(fs.protocol, str) else fs.protocol[-1]

    embedding_paths = [get_protocol(embeddings_fs) + "://" + embedding_file for embedding_file in embedding_files]

    metadata_fs, metadata_fs_path = fsspec.core.url_to_fs(metadata_folder)
    metadata_files = metadata_fs.ls(metadata_fs_path)
    metadata_paths = [get_protocol(metadata_fs) + "://" + metadata_file for metadata_file in metadata_files]

    # Now we sort these paths by shard
    embedding_paths.sort(key=lambda filename: int(get_shard(filename)))
    metadata_paths.sort(key=lambda filename: int(get_shard(filename)))

    # Then we make sure embeddings and metadata shards line up
    assert len(embedding_paths) == len(metadata_paths)
    embedding_shards = [int(get_shard(embedding_path)) for embedding_path in embedding_paths]
    metadata_shards = [int(get_shard(metadata_path)) for metadata_path in metadata_paths]
    assert embedding_shards == metadata_shards

    # Use multiprocessing/spark to run this in parallel
    if run_concurrent == 1:
        count = 0
        for embedding_path, metadata_path in tqdm(zip(embedding_paths, metadata_paths), total=len(embedding_paths)):
            reorder_shard(embedding_path, metadata_path, output_folder, 4, 6, verbose)
            count += 1
            if limit is not None and count >= limit:
                break
    elif run_concurrent > 1:
        data = list(
            zip(
                embedding_paths,
                metadata_paths,
                [output_folder] * len(embedding_paths),
                [index_width] * len(embedding_paths),
                [output_shard_width] * len(embedding_paths),
                [verbose] * len(embedding_paths),
            )
        )
        data = data[:limit] if limit is not None else data
        if concurrent_backend == "multiprocessing":
            from multiprocessing import Pool, Queue  # pylint: disable=import-outside-toplevel

            return_queue = Queue()
            with Pool(run_concurrent) as pool:
                pool.starmap(
                    lambda d: reorder_shard(
                        d[0], d[1], output_folder, index_width, output_shard_width, verbose, 3, return_queue
                    ),
                    data,
                )
            # Convert queue to list
            successes = []
            while not return_queue.empty():
                successes.append(return_queue.get())
        elif concurrent_backend == "spark":
            import findspark  # pylint: disable=import-outside-toplevel

            findspark.init()
            from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel

            spark = (
                SparkSession.builder.master(f"local[{run_concurrent}]")
                .appName("Embedding Reorder")
                .config("spark.ui.showConsoleProgress", "true")
                .getOrCreate()
            )
            # Turn data into an rdd
            data_rdd = spark.sparkContext.parallelize(data, numSlices=run_concurrent)
            # print(data_rdd.take(10))
            # Then we can foreach the rdd to run the reorder
            res_data = data_rdd.map(lambda row: reorder_shard(*row))  # reorder_shard(*row))
            res_data = res_data.collect()
            successes = [{"shard": d[0], "success": d[1]} for d in res_data]
        failed_shards = [row["shard"] for row in successes if not row["success"]]
        if len(failed_shards) > 0:
            print("Failed to reorder shards:", failed_shards)
        else:
            print(f"Successfully reordered {len(successes)} shards")

    ra.end()
    print("Full Runtime:", ra.full_run_time)
