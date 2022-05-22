"""Most relevant for testing. Main script is in api.py"""
import os
import pandas as pd
import numpy as np
from img2dataset import download
from clip_retrieval import clip_inference
import time
import plotext as plt
from collections import OrderedDict


def get_example_key(metadata_folder="./"):
    """
    Prints example keys for the metadata
    """
    from_each = 2
    example_parquets = os.listdir(metadata_folder)
    # Filter for parquets
    example_parquets = [name for name in example_parquets if ".parquet" in name]
    example_keys = {}
    for example_parquet in example_parquets:
        shard = int(example_parquet.split("_")[-1].split(".")[0])
        example_keys[shard] = []
        df = pd.read_parquet(os.path.join(metadata_folder, example_parquet))
        example_rows = df.sample(n=from_each)
        for _, row in example_rows.iterrows():
            if "image_path" in row:
                example_keys[shard].append(row.image_path)
            elif "key" in row:
                example_keys[shard].append(row.key)
            else:
                raise Exception("No key or image_path in row. Maybe img2dataset changed its output? Raise an issue on the github repo.")
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


class RuntimeAnalyzer:
    """
    Used for tracking runtimes where multiple iterations are performed possibly in parallel
    Can graph the runtimes
    """
    def __init__(self):
        self.runtimes = OrderedDict()  # Dictionary of lists of runtimes
        self.running = False
        self.run_start = None
        self.full_run_time = None

    def start(self):
        """
        Initializes the runtime analyzer. Must be called before any other method
        """
        self.runtimes.clear()
        self.running = True
        self.run_start = time.perf_counter()

    def end(self):
        """
        Ends the analysis. Should be called before summaries are generated
        """
        self.running = False
        self.full_run_time = time.perf_counter() - self.run_start

    def add_runtime(self, key, start, end):
        """
        Adds a runtime to the list of runtimes
        Should rarely be used instaed of start_timer
        """
        if not self.running:
            raise RuntimeError("Tried to add {key} while analyzer is not running")
        if key not in self.runtimes:
            self.runtimes[key] = []
        self.runtimes[key].append(end - start)

    def start_timer(self, key):
        """
        Starts a timer for a given key
        Returns a method to stop the timer
        """
        if not self.running:
            raise RuntimeError("Tried to time {key} while analyzer is not running")
        start_time = time.perf_counter()
        return lambda: self.add_runtime(key, start_time, time.perf_counter())

    def get_runtimes(self):
        """
        Returns a dict of all runtimes
        """
        total_runtimes = OrderedDict()
        for key, runtimes in self.runtimes.items():
            total_runtimes[key] = sum(runtimes)
        return total_runtimes

    def get_average_runtimes(self):
        """
        Returns a dict of average runtimes
        """
        total_runtimes = self.get_runtimes()
        for key, total_runtime in total_runtimes.items():
            total_runtimes[key] = total_runtime / len(self.runtimes[key])
        return total_runtimes

    def get_runtime_fractions(self):
        """
        Returns a dict of the proportion of the total runtime each key took up
        """
        return {key: runtime / self.full_run_time for key, runtime in self.get_runtimes().items()}

    def graph(self, average=False):
        """
        Graphs the runtimes in terminal
        """
        if average:
            total_runtimes = self.get_average_runtimes()
        else:
            total_runtimes = self.get_runtimes()
        tasks = list(total_runtimes.keys())
        times = list(total_runtimes.values())
        plt.bar(tasks, times, orientation="horizontal", width=0.3)
        plt.clc()
        if average:
            plt.title("Average runtimes")
        else:
            plt.title("Runtimes")
        plt.xlabel("Execution Time (s)")
        plt.xlim(0, max(times))
        plt.show()
