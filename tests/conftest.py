import os

import pytest


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def _data_available() -> bool:
    root = os.environ.get("HNL_DATA", "/path/to/data")
    return os.path.isdir(os.path.join(root, "unpacked", "WmuHNL10GeV100mm_nopileup_10K"))


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="needs a CUDA device (run through Slurm on a GPU node)")
    skip_data = pytest.mark.skip(reason="needs the unpacked npz files ($HNL_DATA/unpacked)")
    has_gpu, has_data = _cuda_available(), _data_available()
    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)
        if "data" in item.keywords and not has_data:
            item.add_marker(skip_data)
