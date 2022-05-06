"""Main functionality"""

import os
import shutil
from embedding_reader import EmbeddingReader
import numpy as np
import pandas as pd
import math
import fsspec
from tqdm import tqdm
import time
import plotext as plt
from embedding_dataset_reordering.helper import RuntimeAnalyzer

import findspark

findspark.init()

from pyspark.sql import SparkSession  # pylint: disable=wrong-import-position
import pyspark.sql.functions as F  # pylint: disable=wrong-import-position
from pyspark.sql.window import Window  # pylint: disable=wrong-import-position

import os
os.environ['PYSPARK_SUBMIT_ARGS'] = "--packages=com.amazonaws:aws-java-sdk-bundle:1.11.271,org.apache.hadoop:hadoop-aws:3.1.2 pyspark-shell"

ra = RuntimeAnalyzer()

def download_embeddings_parallel(spark, embeddings_folder, metadata_folder, output_folder, cores, shard_width, start_at=None, stop_after=None):
    """
    Uses pyspark to parallelize downloading and formatting metadata parquet files to also include embeddings.
    """
    embeddings_fs, embeddings_path = fsspec.core.url_to_fs(embeddings_folder)
    metadata_fs, metadata_path = fsspec.core.url_to_fs(metadata_folder)
    output_fs, output_path = fsspec.core.url_to_fs(output_folder)
    embedding_files = embeddings_fs.ls(embeddings_path)
    metadata_files = metadata_fs.ls(metadata_path)
    get_shard = lambda filename: filename.split("_")[-1].split(".")[0]
    embedding_files.sort(key=lambda filename: int(get_shard(filename)))
    metadata_files.sort(key=lambda filename: int(get_shard(filename)))

    files = []
    output_fs.makedirs(output_path, exist_ok=True)
    for emb_file, meta_file in zip(embedding_files, metadata_files):
        assert get_shard(emb_file) == get_shard(meta_file)
        out_file = os.path.join(output_path, f"emb_meta_{get_shard(emb_file)}.parquet")
        files.append((
            (emb_file, embeddings_fs.to_json()),
            (meta_file, metadata_fs.to_json()),
            (out_file, output_fs.to_json())
        ))
    if stop_after is not None:
        files = files[:stop_after]
    if start_at is not None:
        files = files[start_at:]
    files_rdd = spark.sparkContext.parallelize(files)

    def combine(row):
        """
        Downloads and formats a single parquet metadata file
        """
        try:
            emb_file_data, meta_file_data, out_file_data = row
            print(f"Downloading embedding {int(get_shard(emb_file_data[0]))}: {emb_file_data[0]} {meta_file_data[0]}")
            emb_fs = fsspec.AbstractFileSystem.from_json(emb_file_data[1])
            meta_fs = fsspec.AbstractFileSystem.from_json(meta_file_data[1])
            out_fs = fsspec.AbstractFileSystem.from_json(out_file_data[1])
            with emb_fs.open(emb_file_data[0], "rb") as emb_f, meta_fs.open(meta_file_data[0], "rb") as meta_f, out_fs.open(out_file_data[0], "wb") as out_f:
                emb = np.load(emb_f)
                meta = pd.read_parquet(meta_f, columns=["image_path"])
                meta["img_shard"] = meta.image_path.str[:shard_width].astype(int)
                meta["img_index"] = meta.image_path.str[shard_width:].astype(int)
                meta.drop("image_path", axis=1, inplace=True)
                meta["embeddings"] = emb.tolist()
                meta.to_parquet(out_f, index=False)
        except Exception as e:
            print("Failed to download file", emb_file_data[0])

    files_rdd.foreach(combine)
        

def save_row(row, fs_path: str):
    """
    Saves a single row of the ordered and grouped dataset.
    At this stage, the embeddings are already combined in the correct order.
    We just need to convert them to a numpy array and save them under the correct shard id.

    Files will be saved to the fs_path filesystem.
    """
    fs, fs_base_path = fsspec.core.url_to_fs(fs_path)
    shard_index = row.img_shard
    embeddings = row.embeddings
    if "partition_group" in row:
        partition_group = row.partition_group
        filename = f"img_emb_{shard_index}-{partition_group}.npy"
    else:
        filename = f"img_emb_{shard_index}.npy"
    np_embeddings = np.array(embeddings)
    fs.makedirs(fs_base_path, exist_ok=True)
    save_path = os.path.join(fs_base_path, filename)
    # print(f"Saving: {save_path}")
    with fs.open(save_path, "wb") as f:
        np.save(f, np_embeddings)

def create_empty_embeddings(row, output_folder, shape, skip_fill_range_over):
    """
    Sometimes there will be gaps in the embeddings where an image was skipped due to an error.
    Since our system relies on the index of the image matching the index in the array, we need to add in empty data where such a skip occurred.
    """
    output_fs, output_path = fsspec.core.url_to_fs(output_folder)
    shard = row.img_shard
    start_indices, end_indices = row.first_missing_index, row.next_full_index
    added_data = []
    for first_missing_index, next_full_index in zip(start_indices, end_indices):
        if first_missing_index is None:
            first_missing_index = 0
        if skip_fill_range_over is not None:
            if next_full_index - first_missing_index > skip_fill_range_over:
                print(f"Found long fill range for {shard} starting at {first_missing_index} and going for {next_full_index - first_missing_index}")
                continue
        if next_full_index - first_missing_index > 10:
            print(f"Filling large range from {first_missing_index} to {next_full_index} in shard {shard}")
        for missing_index in range(first_missing_index, next_full_index):
            added_data.append((shard, missing_index, np.zeros(shape).tolist()))
    df = pd.DataFrame(data=added_data, columns=["img_shard", "img_index", "embeddings"])
    with output_fs.open(os.path.join(output_path, f'empty_{shard}.parquet'), "wb") as f:
        df.to_parquet(f)


def reorder_embeddings(
    output_base_path,
    embeddings_folder,
    metadata_folder,
    output_folder="reordered_embeddings",
    intermediate_folder="tmp",
    start_at=0,
    num_shards=None,
    shard_width=5,
    cores=1,
    memory=16,
    overwrite=False,
    skip_fill_range_over=None,
    skip_format_embed=False,
    skip_fill=False,
    skip_sort=False,
    test_recall_single=False
):
    """
    Reorders embeddings such that they are in the same order as the webdataset that was used to generate them.
    If using the index of the image in the webdataset, it is not necessary to fill missing values.
    If using the filepath itself, failing to fill missing values will result in a mismatch between images and embeddings
    """
    working_fs, working_fs_path = fsspec.core.url_to_fs(output_base_path)
    output_folder_path = os.path.join(working_fs_path, output_folder)
    intermediate_folder_path = os.path.join(working_fs_path, intermediate_folder)
    meta_embed_folder_path = os.path.join(intermediate_folder_path, "meta_embed")
    empty_embed_folder_path = os.path.join(intermediate_folder_path, "empty")

    def rm_folder(fs, folder_path):
        try:
            fs.rm(folder_path, recursive=True)
        except FileNotFoundError:
            pass
    
    # Some unexpected behavior can occur if we do not remove existing folders so we do that at the top
    if working_fs.exists(output_folder_path) and len(working_fs.ls(output_folder_path)) > 0 and not skip_sort:
        # Then we should delete it if we are overwriting or error if we aren't
        if overwrite:
            rm_folder(working_fs, output_folder_path)
        else:
            raise RuntimeError("Output folder already has embeddings in it")
    # Create output
    if not skip_sort:
        working_fs.makedirs(output_folder_path, exist_ok=True)


    if working_fs.exists(meta_embed_folder_path) and len(working_fs.ls(meta_embed_folder_path)) > 0 and not skip_format_embed:
        # Then if we are not skipping embed, we need to check if we enabled overwrites
        if overwrite:
            rm_folder(working_fs, meta_embed_folder_path)
            # otherwise if we are not overwriting and not skipping this should error
        else:
            raise RuntimeError("Itermediate embedding folder for metadata+embeddings already has data in it")
    if not skip_format_embed:
        working_fs.makedirs(meta_embed_folder_path, exist_ok=True)

    # And we need the same for the empty path
    if working_fs.exists(empty_embed_folder_path) and len(working_fs.ls(empty_embed_folder_path)) > 0 and not skip_fill:
        # Then if we are not skipping empty, we need to check if we enabled overwrites
        if overwrite:
            rm_folder(working_fs, empty_embed_folder_path)
            # otherwise if we are not overwriting and not skipping this should error
        else:
            raise RuntimeError("Itermediate embedding folder for empty embeddings already has data in it")
    if not skip_fill:
        working_fs.makedirs(empty_embed_folder_path, exist_ok=True)

    print("========= Starting Reorder =========")
    print(f"  Embedding Source: {embeddings_folder}")
    print(f"  Metadata Source: {metadata_folder}")
    print(f"  Output Folder: {output_folder_path}")
    print(f"  Overwriting Old Embeddings: {overwrite}")

    """
    One shard that includes both embeddings and metadata should be about 1483.7mb
    I'm not sure, but I think spark may want to load the entire thing into memory. That means each core should have around 4g of memory.
    """

    spark = (
        SparkSession.builder.master(f"local[{cores}]")
        .appName("Embedding Reorder")
        .config("spark.ui.showConsoleProgress", "true")
        .config("spark.driver.memory", f"{memory}g")
        .config("spark.executor.memory", f"{memory}g")
        .config("spark.sql.shuffle.partitions", "1000000")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer", "1800m")
        .getOrCreate()
    )  # TODO: Add in the ability to have nodes
    sc = spark.sparkContext

    if "s3a" in working_fs.protocol:
        # Swap to using s3a
        print(f"Using s3 for intermediate and output with new basepath {output_base_path}")
        hadoop_conf = sc._jsc.hadoopConfiguration()
        # We also need to get hadoop to be able to use s3a file systems and tell it how to find the credentials
        hadoop_conf.set("com.amazonaws.services.s3.enableV4", "true")
        hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        hadoop_conf.set("fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        hadoop_conf.set("fs.AbstractFileSystem.s3a.impl", "org.apache.hadoop.fs.s3a.S3A")
    
    output_folder_remote_path = os.path.join(output_base_path, output_folder)  # Used in original format since save_row makes it's own fs
    meta_embed_remote_path = os.path.join(output_base_path, intermediate_folder,  "meta_embed")
    empty_embed_remote_path = os.path.join(output_base_path, intermediate_folder,  "empty")

    print("Created spark instance.\nLoading embeddings.")

    ra.start()

    if not skip_format_embed:
        print("========= Formatting Intermediate Embeddings =========")
        end_read_embeddings_timer = ra.start_timer("Read embeddings")
        end = None if num_shards is None else (num_shards + (0 if start_at is None else start_at))  # Idk just go with it. Defaults are just all for num_shards and 0 for start.
        download_embeddings_parallel(spark, embeddings_folder, metadata_folder, meta_embed_remote_path, cores=cores, shard_width=shard_width, start_at=start_at, stop_after=end)
        end_read_embeddings_timer()

    print("========= Recalling and Reordering Embeddings =========")
    end_recall_timer = ra.start_timer("Recall Embeddings")
    # Recall the data that was saved by each worker into a single dataframe so that we can do a full sort
    remote_path = os.path.join(meta_embed_remote_path, "*.parquet" if not test_recall_single else "emb_meta_"+"".zfill(shard_width)+".parquet")
    print("Recalling data from worker paths:", remote_path)
    data = spark.read.parquet(remote_path)  # TODO: Verify this work with remote files. It doesn't. It needs to be able to connect to the s3 file system.
    print(f"Data has {data.rdd.getNumPartitions()} partitions")

    end_recall_timer()

    end_fill_timer = ra.start_timer("Fill Missing Embeddings")
    if not skip_fill:
        print("========= Inserting Missing Data =========")
        # If an image returned a error during webdataset creation, it will still take up an index, but will not be included in the embeddings
        # This means if we do not account for these missing indices, we will be off by one for all subsequent embeddings in the shard
        # In order to fix these, we insert an empty embedding into every location where one is missing
        example_embedding = np.array(data.first().embeddings)
        data.createOrReplaceTempView("df")
        missing_values = spark.sql(
            """
      SELECT img_shard, COALESCE(last_img_index + 1, 0) AS first_missing_index, img_index AS next_full_index FROM (
        SELECT
          img_shard, img_index,
          LAG(img_index, 1) OVER (PARTITION BY img_shard ORDER BY img_index) AS last_img_index
        FROM df ) list
      WHERE img_index - last_img_index > 1 OR (last_img_index IS NULL AND img_index > 0)
    """
        )  # The where clause catches both the case where any index besides the first is skipped and the case where the first index is skipped
        # missing_values.write.csv("./missing_values.csv")
        print(f"Found {missing_values.count()} missing ranges. Skipping filling ranges over {skip_fill_range_over} in length")

        missing_values_by_shard = missing_values.groupBy("img_shard").agg(
            F.collect_list("first_missing_index").alias("first_missing_index"),
            F.collect_list("next_full_index").alias("next_full_index")
        )
        missing_values_by_shard.foreach(lambda row: create_empty_embeddings(row, empty_embed_remote_path, example_embedding.shape, skip_fill_range_over))
    empty_path = os.path.join(empty_embed_remote_path, "empty_*.parquet")
    added_df = spark.read.parquet(empty_path)
    print(f"Adding {added_df.count()} missing embeddings to data")
    data = data.union(added_df)

    end_fill_timer()
    end_export_timer = ra.start_timer("Sort & Export")
    if not skip_sort:
        print("========= Grouping and Saving =========")
        print("Number of partitions", data.rdd.getNumPartitions())
        data = data.withColumn("partition_group", F.floor(F.col("img_index") / 1000))
        data = data.sort("img_index")
        data = data.groupBy("img_shard", "partition_group")
        data = data.agg(F.collect_list("embeddings").alias("embeddings"))

        # Parallelize saving the grouped embeddings as this also takes a while
        data.foreach(lambda row: save_row(row, output_folder_remote_path))
    end_export_timer()
    ra.end()

    runtimes = ra.get_runtimes()
    print("Total Time", ra.full_run_time)
    ra.graph()
