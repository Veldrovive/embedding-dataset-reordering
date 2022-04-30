import pytest
import os
from embedding_dataset_reordering.helper import verify_reorder


def test_reorder():
    # TODO: Make this not half-assed. Or don't.
    embeddings_folder = "./examples/test_data_inference/img_emb"
    metadata_folder = "./examples/test_data_inference/metadata"
    reordered_embeddings_folder = "./examples/test_reordered/reordered_embeddings"
    assert os.path.exists(embeddings_folder), f"Embeddings folder not found"
    assert os.path.exists(metadata_folder), f"Metadata folder not found"
    assert os.path.exists(reordered_embeddings_folder), f"Reordered embeddings folder not found"
    shard_width = 5
    errors = verify_reorder(embeddings_folder, metadata_folder, reordered_embeddings_folder, shard_width)
    assert len(errors) == 0, f"{len(errors)} errors found: {errors}"
