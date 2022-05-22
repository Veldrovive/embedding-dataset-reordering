"""Main entrypoint for the cli"""

import os
from embedding_dataset_reordering.api import reorder_embeddings
from embedding_dataset_reordering.helper import download_test_data, test_inference, get_example_key
import click


@click.group()
def cli():
    pass


@cli.command()
@click.option("--output-folder", required=True, help="The folder to write the output shards to", type=click.STRING)
@click.option(
    "--embeddings-folder", required=True, help="The folder containing the embeddings to reorder", type=click.STRING
)
@click.option(
    "--metadata-folder", required=True, help="The folder containing the metadata to reorder", type=click.STRING
)
@click.option("--index-width", default=4, help="The assumed number of digits in the index", type=click.INT)
@click.option("--output-shard-width", default=6, help="The number of digits in the output shard", type=click.INT)
@click.option("--limit", default=None, help="The number of shards to reorder", type=click.INT)
@click.option("--run-concurrent", default=1, help="The number of concurrent workers to use", type=click.INT)
@click.option(
    "--concurrent-backend",
    default="multiprocessing",
    help="The backend to use for concurrent workers. Only `multiprocessing` or `spark` are supported",
    type=click.Choice(["multiprocessing", "spark"]),
)
@click.option("--verbose", default=False, help="Whether to print out progress", type=click.BOOL)
@click.option("--tmp-folder", default=None, help="The folder to use for temporary files", type=click.STRING)
def reorder(
    output_folder,
    embeddings_folder,
    metadata_folder,
    index_width,
    output_shard_width,
    limit,
    run_concurrent,
    concurrent_backend,
    verbose,
    tmp_folder,
):
    """Reorder embeddings"""
    if tmp_folder is not None:
        os.environ["TMPDIR"] = os.path.abspath(tmp_folder)
        print("TMPDIR:", os.environ["TMPDIR"])
        os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    reorder_embeddings(
        output_folder,
        embeddings_folder,
        metadata_folder,
        index_width,
        output_shard_width,
        limit,
        run_concurrent,
        concurrent_backend,
        verbose,
    )


@cli.command()
@click.option("--metadata-folder", required=True, help="The folder containing the metadata to be reordered")
def example_key(metadata_folder):
    """Get an example key from a local metadata folder"""
    get_example_key(metadata_folder)


@cli.command()
@click.option("--csv-path", default="./example.csv", help="Path to csv file", type=click.STRING)
@click.option("--output-folder", default="./test_data", help="Path to output folder", type=click.STRING)
@click.option("--samples-per_shard", default=10e4, help="Number of samples per shard", type=click.INT)
@click.option("--thread-count", default=64, help="Number of threads to use", type=click.INT)
@click.option("--image-size", default=256, help="Size of images", type=click.INT)
def download_data(csv_path, output_folder, samples_per_shard, thread_count, image_size):
    """
    Runs img2dataset.download to download the images from the csv file
    """
    download_test_data(csv_path, output_folder, samples_per_shard, thread_count, image_size)


@cli.command()
@click.option(
    "--input-folder", default="./test_data", help="Folder containing the webdataset shards", type=click.STRING
)
@click.option(
    "--output-folder", default="./test_data_inference", help="Folder to save the inference results", type=click.STRING
)
@click.option("--samples-per-shard", default=10e4, help="Number of samples per shard", type=click.INT)
@click.option("--batch-size", default=256, help="Batch size", type=click.INT)
def clip_inference(input_folder, output_folder, samples_per_shard, batch_size):
    """
    Runs clip_retrieval.clip_inference
    """
    test_inference(input_folder, output_folder, samples_per_shard, batch_size)
