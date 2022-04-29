"""Main entrypoint for the cli"""

from embedding_dataset_reordering.api import reorder_embeddings
from embedding_dataset_reordering.helper import download_test_data, test_inference, get_example_key
import fire


def test(text: int, second):
    print(type(text), second)


def main():
    """Main entry point"""
    fire.Fire(
        {
            "reorder": reorder_embeddings,
            "get-example-key": get_example_key,
            "download-test-data": download_test_data,
            "test-inference": test_inference,
        }
    )


if __name__ == "__main__":
    main()
