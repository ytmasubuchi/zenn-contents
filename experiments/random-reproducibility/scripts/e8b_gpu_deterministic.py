"""E8b: the same measurements as E8a, but with PyTorch's documented
"make CUDA reproducible" recipe turned on:

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    CUBLAS_WORKSPACE_CONFIG=:4096:8   (must be set in the environment
                                       *before* the process starts, so
                                       run_gpu.sh passes it via `docker run
                                       -e`; we also os.environ.setdefault
                                       it here as a belt-and-braces fallback)

The article (chapter 7, written before a GPU was available) claims, based
on the PyTorch docs, that `index_add_`'s CUDA implementation raises a
RuntimeError under `use_deterministic_algorithms(True)` rather than
silently returning a (possibly non-reproducible) result. This script is
what actually checks that claim on torch==2.5.1 for every op we probe, not
just index_add_: `_common.run_with_fallback` labels each measurement's
outcome as
  - "ok"                         -> ran under the deterministic path
  - "error_no_deterministic_impl" -> raised, message mentions determinism
  - "error_cuda_oom"              -> raised, message is a CUDA OOM
  - "error_other"                 -> raised, anything else
so the per-op determinism behavior is directly visible in the result JSON
without re-deriving it by hand.

Run this identical script 3x in fresh containers and diff with compare.py,
exactly like E8a.
"""
import os

# Must be set before any CUDA/cuBLAS call in this process. run_gpu.sh sets
# it via `docker run -e CUBLAS_WORKSPACE_CONFIG=:4096:8`; this is a
# fallback for anyone running the script directly inside a container shell.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from _common import emit, run_with_fallback
from e8a_gpu_repeat import (
    SEED,
    atomic_add_repeat,
    conv_forward_backward,
    cumsum_demo,
    embedding_bag_demo,
    matmul,
    mlp_train,
    philox_streams,
)


def main():
    import sys

    import numpy as np
    import torch

    outdir = sys.argv[1] if len(sys.argv) > 1 else None
    npz_store: dict = {}

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    result = {
        "experiment": "E8b_gpu_deterministic",
        "seed": SEED,
        "note": (
            "torch.use_deterministic_algorithms(True), "
            "torch.backends.cudnn.benchmark=False, "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 all set before any measurement."
        ),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config_env": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "philox_streams": run_with_fallback(
            lambda n: philox_streams(n, npz_store), [1000, 100, 10, 1]
        ),
        "matmul_512": run_with_fallback(lambda s: matmul(s, npz_store), [512, 128, 32, 8]),
        "conv_forward_backward": run_with_fallback(
            lambda b: conv_forward_backward(b, npz_store), [16, 4, 1]
        ),
        "mlp_train": run_with_fallback(lambda n: mlp_train(n, npz_store), [200, 50, 10]),
        "atomic_add_repeat": run_with_fallback(
            lambda n: atomic_add_repeat(n, npz_store=npz_store),
            [1_000_000, 100_000, 10_000, 1_000],
        ),
        "embedding_bag": run_with_fallback(
            lambda n: embedding_bag_demo(n, npz_store=npz_store), [1_000_000, 10_000, 100]
        ),
        "cumsum": run_with_fallback(
            lambda n: cumsum_demo(n, npz_store=npz_store), [1_000_000, 10_000, 100]
        ),
    }
    if outdir:
        np.savez(f"{outdir}/raw_arrays.npz", **npz_store)
    emit(result)


if __name__ == "__main__":
    main()
