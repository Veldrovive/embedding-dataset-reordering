import tempfile
import fsspec
import os
import pandas as pd
import numpy as np
from s3fs.core import S3FileSystem
from tqdm import tqdm

from helper import RuntimeAnalyzer
ra = RuntimeAnalyzer()

def get_shard(file_path):
    basename = os.path.basename(file_path)
    return  basename.split("_")[-1].split(".")[0]

def reorder_shard(
    embedding_file,
    metadata_file,
    output_folder,
    index_width,
    output_shard_width,
    verbose=False
):
    s3 = S3FileSystem()
    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    print("Reordering embedding shard {}".format(embedding_file))
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load the parquet into pandas
        end_read_metadata_timer = ra.start_timer("Read Metadata")
        with fsspec.open(metadata_file, 'rb') as f:
            metadata_df = pd.read_parquet(f)
        log("Read metadata")
        metadata_df["img_shard"] = metadata_df.image_path.str[:-index_width].astype(int)
        metadata_df["img_index"] = metadata_df.image_path.str[-index_width:].astype(int)
        metadata_df["original_index"] = metadata_df.index
        metadata_df.drop(metadata_df.columns.difference(['original_index', 'img_shard','img_index']), 1, inplace=True)
        end_read_metadata_timer()
        end_sort_metadata_timer = ra.start_timer("Sort Metadata")
        metadata_df = metadata_df.sort_values(by=['img_shard', 'img_index'])
        end_sort_metadata_timer()
        log("Sorted metadata")

        # Show a few rows
        log(metadata_df.head())

        # Read the embeddings into numpy
        end_read_embeddings_timer = ra.start_timer("Read Embeddings")
        with fsspec.open(embedding_file, 'rb') as f:
            embeddings = np.load(f)
        end_read_embeddings_timer()

        assert len(embeddings) == len(metadata_df)
        log("Read embeddings")

        # Reading directly is faster, but we need to try other steps before we can tell if it will be a bottleneck
        # end_read_embeddings_directly_timer = ra.start_timer("Read Embeddings Directly")
        # embeddings = np.load(s3.open(embedding_file, 'rb'))
        # end_read_embeddings_directly_timer()

        embedding_shape = embeddings.shape[1:]

        # Now we iterate over the metadata and build new embedding shards named fro the img_shard and in the order of img_index
        # We can find the embedding by indexing the embeddings array at original_index
        end_reorder_embeddings_timer = ra.start_timer("Reorder Embeddings")
        shard_embeddings = []
        current_shard = None
        last_index = -1
        for i, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), disable=not verbose):
            img_shard = row.img_shard
            img_index = row.img_index
            embedding = embeddings[row.original_index]
            if current_shard is None or current_shard != img_shard:
                if current_shard is not None:
                    np.save(os.path.join(tmpdir, f"img_emb_{str(current_shard).zfill(output_shard_width)}.npy"), shard_embeddings)
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
                    log("Filling {} embeddings with zeros".format(fill_number))
            shard_embeddings.append(embedding)
            last_index = img_index
        end_reorder_embeddings_timer()
        np.save(os.path.join(tmpdir, f"img_emb_{str(current_shard).zfill(output_shard_width)}.npy"), shard_embeddings)

        log("Shards", os.listdir(tmpdir))

        # Then we upload all files in s3 to the output folder file system
        end_upload_embeddings_timer = ra.start_timer("Upload Embeddings")
        output_fs, output_path = fsspec.core.url_to_fs(output_folder)
        for file in os.listdir(tmpdir):
            output_fs.put(os.path.join(tmpdir, file), os.path.join(output_path, file))
        end_upload_embeddings_timer()
        


def reorder_embeddings(
    output_folder,
    embeddings_folder,
    metadata_folder,
    run_concurrent=1,
    concurrent_backend="multiprocessing",
    verbose=False
):
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

    # Use multiprocessing to run this in parallel
    if run_concurrent == 1:
        for embedding_path, metadata_path in tqdm(zip(embedding_paths, metadata_paths), total=len(embedding_paths)):
            reorder_shard(embedding_path, metadata_path, output_folder, 4, 6, verbose)
    elif run_concurrent > 1:
        data = list(zip(embedding_paths, metadata_paths, [output_folder] * len(embedding_paths), [4] * len(embedding_paths), [6] * len(embedding_paths), [verbose] * len(embedding_paths)))
        data = data[:64]
        if concurrent_backend == "multiprocessing":
            from multiprocessing import Pool

            with Pool(run_concurrent) as pool:
                pool.starmap(reorder_shard, data)
        elif concurrent_backend == "spark":
            import findspark
            findspark.init()
            from pyspark.sql import SparkSession

            spark = (
                SparkSession.builder.master(f"local[{run_concurrent}]")
                .appName("Embedding Reorder")
                .config("spark.ui.showConsoleProgress", "true")
                .getOrCreate()
            ) 
            # Turn data into an rdd
            data_rdd = spark.sparkContext.parallelize(data)
            # Then we can foreach the rdd to run the reorder
            data_rdd.foreach(lambda row: reorder_shard(row[0], row[1], row[2], row[3], row[4], row[5]))
    
    ra.end()
    print("Full Runtime:", ra.full_run_time)
    # ra.graph()
    # ra.graph(average=True)

if __name__ == "__main__":
    reorder_embeddings(
        output_folder='s3a://dalle2-training-dataset/reorderv2',
        embeddings_folder= 's3a://laion-us-east-1/embeddings/vit-l-14/laion2B-en/img_emb',
        metadata_folder='s3a://laion-us-east-1/embeddings/vit-l-14/laion2B-en/metadata',
        run_concurrent=32,
        concurrent_backend="spark",
        verbose=False
    )