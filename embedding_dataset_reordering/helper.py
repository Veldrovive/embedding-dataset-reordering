"""Most relevant for testing. Main script is in api.py"""
import os
import pandas as pd
import numpy as np
from img2dataset import download
from clip_retrieval import clip_inference


def get_example_key(metadata_folder="./"):
    """
    Prints example keys for the metadata
    """
    from_each = 2
    example_parquets = os.listdir(metadata_folder)
    example_keys = {}
    for example_parquet in example_parquets:
        shard = int(example_parquet.split("_")[-1].split(".")[0])
        example_keys[shard] = []
        df = pd.read_parquet(os.path.join(metadata_folder, example_parquet))
        example_rows = df.sample(n=from_each)
        for _, row in example_rows.iterrows():
            example_keys[shard].append(row.image_path)
    print("Example Keys:")
    for shard, keys in example_keys.items():
        print(f"Shard {shard} has keys {keys}")


def download_test_data(
    csv_path="./example.csv", output_folder="./test_data", samples_per_shard=10e4, thread_count=64, image_size=256
):
    download(
        url_list=csv_path,
        input_format="csv",
        caption_col="caption",
        output_folder=output_folder,
        output_format="webdataset",
        thread_count=thread_count,
        image_size=image_size,
        number_sample_per_shard=samples_per_shard,
    )


def test_inference(
    input_folder="./test_data", output_folder="./test_data_inference", samples_per_shard=10e4, batch_size=256
):
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)
    num_tars = len([name for name in os.listdir(input_folder) if ".tar" in name])
    print(f"Found {num_tars} shards of webdataset in {input_folder}.")
    input_path = '"' + input_folder + "/0000{0.." + str(num_tars - 1) + '}.tar"'
    clip_inference(
        input_dataset=input_path,
        output_folder=output_folder,
        batch_size=batch_size,
        input_format="webdataset",
        wds_number_file_per_input_file=samples_per_shard,
        output_partition_count=num_tars,
        enable_metadata=True,
    )


def verify_reorder(embeddings_folder, metadata_folder, reordered_embeddings_folder, shard_width=5):
    """
    Verifies that embeddings are in the shard and index specified by image_path
    This method reads all data into memory so should only be performed on toy datasets
    """
    # Read embeddings
    shard_original_embeddings = {}
    for embedding_filename in [name for name in os.listdir(embeddings_folder) if ".npy" in name]:
        embedding = np.load(os.path.join(embeddings_folder, embedding_filename))
        shard = int(embedding_filename.split("_")[-1].split(".")[0])
        shard_original_embeddings[shard] = embedding

    shard_reordered_embeddings = {}
    for embedding_filename in [name for name in os.listdir(reordered_embeddings_folder) if ".npy" in name]:
        embedding = np.load(os.path.join(reordered_embeddings_folder, embedding_filename))
        shard = int(embedding_filename.split("_")[-1].split(".")[0])
        shard_reordered_embeddings[shard] = embedding

    # Read metadata
    shard_metadata = {}
    for metadata_filename in [name for name in os.listdir(metadata_folder) if ".parquet" in name]:
        metadatum = pd.read_parquet(os.path.join(metadata_folder, metadata_filename))
        shard = int(metadata_filename.split("_")[-1].split(".")[0])
        shard_metadata[shard] = metadatum

    errors = []
    for shard, metadata in shard_metadata.items():
        # original_embeddings = shard_original_embeddings[shard]
        for original_index, metadatum in metadata.iterrows():
            # print(shard, original_index, metadatum.image_path)
            embedding = shard_original_embeddings[shard][original_index]
            img_shard = int(metadatum.image_path[:shard_width])
            img_index = int(metadatum.image_path[shard_width:])
            reordered_embedding = shard_reordered_embeddings[img_shard][img_index]
            correct = np.array_equal(embedding, reordered_embedding)
            if not correct:
                errors.append([shard, original_index, img_shard, img_index])
    return errors
