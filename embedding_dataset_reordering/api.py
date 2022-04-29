"""Main functionality"""

import os
from embedding_reader import EmbeddingReader
import numpy as np
import pandas as pd
import math

import findspark

findspark.init()

from pyspark.sql import SparkSession  # pylint: disable=wrong-import-position
import pyspark.sql.functions as F  # pylint: disable=wrong-import-position


def download_embedding_range(
    r, embeddings_folder, metadata_folder, folder="numpy_parquet", shard_width=5, batch_size=10e3
):
    """
    Uses embedding-reader to create parquet files that have the image dataset
    shard id and index as well as the corresponding embedding.
    Since this is done in parallel with other downloaders, each needs to have it's own embedding reader object.
    """
    os.makedirs(folder, exist_ok=True)
    partition_number, start, end = r
    embedding_generator = EmbeddingReader(
        embeddings_folder=embeddings_folder,
        metadata_folder=metadata_folder,
        meta_columns=["image_path"],
        file_format="parquet_npy",
    )(batch_size=batch_size, start=start, end=end)
    frames = []
    for data, meta in embedding_generator:
        # In order to split the image_path into the shard and index, we use a constant called shard width that defines how many digits make up the shard name
        meta["img_shard"] = meta.image_path.str[:shard_width].astype(int)
        meta["img_index"] = meta.image_path.str[shard_width:].astype(int)
        meta.drop("image_path", axis=1, inplace=True)
        meta.drop("i", axis=1, inplace=True)
        meta["embeddings"] = data.tolist()
        frames.append(meta)
    df = pd.concat(frames)
    df.to_parquet(os.path.join(folder, f"meta_embeddings_{partition_number}.parquet"), index=False)


def download_embeddings(
    embedding_reader, output_folder, shards=None, start=0, end=None, shard_width=5, batch_size=10e3
):
    """
    Uses an existing embedding reader to download the embeddings and metadata.
    Optionally split the output into smaller shards each containing a fraction of the data
    """
    os.makedirs(output_folder, exist_ok=True)
    if end is None:
        end = embedding_reader.count
    embedding_generator = embedding_reader(batch_size=batch_size, start=start, end=end)
    frames = []
    shard_number = 0
    current_shard_size = 0
    shard_size_limit = math.ceil(end - start if shards is None else (end - start) / shards)

    def save_frames():
        # Used to save a single shard of the data so that they can potentially be distributed
        nonlocal frames, shard_number, current_shard_size
        df = pd.concat(frames)
        df.to_parquet(os.path.join(output_folder, f"meta_embeddings_{shard_number}.parquet"), index=False)
        frames.clear()
        shard_number += 1
        current_shard_size = 0

    for data, meta in embedding_generator:
        meta["img_shard"] = meta.image_path.str[:shard_width].astype(int)
        meta["img_index"] = meta.image_path.str[shard_width:].astype(int)
        meta.drop("image_path", axis=1, inplace=True)
        meta.drop("i", axis=1, inplace=True)
        meta["embeddings"] = data.tolist()
        frames.append(meta)
        current_shard_size += data.shape[0]
        if current_shard_size >= shard_size_limit:
            save_frames()  # This won't ensure shards are always under shard_size_limit, but will make sure they are less than one batch size larger
    if len(frames) > 0:
        save_frames()


def save_row(row, output_folder):
    """
    Saves a single row of the ordered and grouped dataset.
    At this stage, the embeddings are already combined in the correct order.
    We just need to convert them to a numpy array and save them under the correct shard id.
    """
    shard_index = row.img_shard
    embeddings = row.embeddings
    np_embeddings = np.array(embeddings)
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, f"img_emb_{shard_index}.npy")
    with open(save_path, "wb") as f:
        np.save(f, np_embeddings)


def reorder_embeddings(
    embeddings_folder,
    metadata_folder,
    output_folder,
    intermediate_folder="./tmp",
    intermediate_partitions=5,
    start=0,
    end=None,
    shard_width=5,
    overwrite=False,
    parallelize_reading=False,
    fill_missing=True,
):
    """
    Reorders embeddings such that they are in the same order as the webdataset that was used to generate them.
    If using the index of the image in the webdataset, it is not necessary to fill missing values.
    If using the filepath itself, failing to fill missing values will result in a mismatch between images and embeddings
    """
    if not overwrite and os.path.exists(output_folder) and len(os.listdir(output_folder)) > 0:
        raise RuntimeError("Output folder already has embeddings in it")

    print("========= Starting Reorder =========")
    print(f"  Embedding Source: {embeddings_folder}")
    print(f"  Metadata Source: {metadata_folder}")
    print(f"  Output Folder: {output_folder}")
    print(f"  Intermediate Partitions: {intermediate_partitions}")
    print(f"  Overwriting Old Embeddings: {overwrite}")

    spark = (
        SparkSession.builder.master("local")
        .appName("Embedding Reorder")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )  # TODO: Add in the ability to have nodes
    sc = spark.sparkContext

    print("Created spark instance.\nLoading embeddings.")

    embedding_reader = EmbeddingReader(  # TODO: Figure out if some kind of authorization will be necessary
        embeddings_folder=embeddings_folder,
        metadata_folder=metadata_folder,
        meta_columns=["image_path"],
        file_format="parquet_npy",
    )

    start_index = max(0, start)
    end_index = embedding_reader.count if end is None else min(end, embedding_reader.count)

    print(f"Embedding reader found {embedding_reader.count} embeddings")

    print("========= Formatting Intermediate Embeddings =========")
    if parallelize_reading:
        # Parallelize the downloading of the embeddings
        partition_width = (end_index - start_index) / intermediate_partitions
        ends = [int(round(partition_width * i)) + start_index for i in range(intermediate_partitions + 1)]
        ranges = list(zip(range(len(ends) - 1), ends, ends[1:]))
        ranges_rdd = sc.parallelize(ranges)
        ranges_rdd.foreach(
            lambda r: download_embedding_range(
                r, embeddings_folder, metadata_folder, shard_width=shard_width, folder=intermediate_folder
            )
        )
    else:
        batch_size = min(math.ceil((end_index - start_index) / intermediate_partitions), 10e3)
        download_embeddings(
            embedding_reader,
            intermediate_folder,
            shards=intermediate_partitions,
            start=start_index,
            end=end_index,
            shard_width=shard_width,
            batch_size=batch_size,
        )

    print("========= Recalling and Reordering Embeddings =========")
    # Recall the data that was saved by each worker into a single dataframe so that we can do a full sort
    data = spark.read.parquet(intermediate_folder)
    example_embedding = np.array(data.first().embeddings)

    if fill_missing:
        print("========= Inserting Missing Data =========")
        # If an image returned a error during webdataset creation, it will still take up an index, but will not be included in the embeddings
        # This means if we do not account for these missing indices, we will be off by one for all subsequent embeddings in the shard
        # In order to fix these, we insert an empty embedding into every location where one is missing
        data.createOrReplaceTempView("df")
        missing_values = spark.sql(
            """
      SELECT img_shard, last_img_index + 1 AS first_missing_index, img_index AS next_full_index FROM (
        SELECT
          img_shard, img_index,
          LAG(img_index, 1) OVER (ORDER BY img_shard, img_index) AS last_img_index
        FROM df ) list
      WHERE img_index - last_img_index > 1
    """
        )
        added_data = []
        for row in missing_values.collect():
            shard = row.img_shard
            first_missing_index, next_full_index = row.first_missing_index, row.next_full_index
            for missing_index in range(first_missing_index, next_full_index):
                added_data.append((shard, missing_index, np.zeros_like(example_embedding).tolist()))
        added_df = spark.createDataFrame(added_data, ["img_shard", "img_index", "embeddedings"])
        data = data.union(added_df)

    print("========= Grouping and Saving =========")
    grouped = (
        data.orderBy("img_shard", "img_index")
        .groupBy("img_shard")
        .agg(F.collect_list("embeddings").alias("embeddings"))
    )
    # Parallelize saving the grouped embeddings as this also takes a while
    grouped.foreach(lambda row: save_row(row, output_folder))
    shards = [row.img_shard for row in grouped.select("img_shard").collect()]
    shutil.rmtree(intermediate_folder)
    return [os.path.join(output_folder, f"img_emb_{shard_index}.npy") for shard_index in shards]
