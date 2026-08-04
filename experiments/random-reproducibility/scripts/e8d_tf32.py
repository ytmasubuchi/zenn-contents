"""E8d: TF32 (TensorFloat-32) effect on float32 matmul.

TF32 is a reduced-mantissa (10-bit, vs float32's 23-bit) execution mode
Ampere+ GPUs (this RTX 4090 is Ada Lovelace, sm_89, also TF32-capable) can
use internally for float32 matmul/conv while keeping float32 storage. It
changes results at the bit level without changing dtype, so a hash compare
of "the same float32 matmul" with TF32 on vs off is exactly the kind of
thing this project's SHA256-of-raw-bytes methodology is built to catch.

torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
default to *different* values as of torch 2.5.1 (recorded empirically
below, not assumed) -- this script records both defaults, then explicitly
forces both False ("tf32_off") and both True ("tf32_on") for the same
input and computes:
  - SHA256 hash of the float32 output (bit-exact compare)
  - max abs diff / max rel diff between tf32_off and tf32_on (direct
    measurement of the error TF32 introduces, not just "match/no match")

The input matrices are generated once on CPU with a fixed seed and moved
to CUDA, so the only thing that can differ between the two GPU runs is the
TF32 setting itself. Raw arrays are dumped to <outdir>/raw_arrays.npz
(argv[1]) for e8e_ulp_diff.py's ULP-level breakdown.

Usage (inside container): python e8d_tf32.py /work/out > result.json
"""
import sys

import numpy as np

from _common import emit, run_with_fallback, sha256_of_array

SEED = 42
SIZE = 512


def record_tf32_defaults():
    import torch

    return {
        "tf32_matmul_default": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn_default": torch.backends.cudnn.allow_tf32,
    }


def matmul_with_tf32(allow_tf32: bool, a_np, b_np, size, npz_store, npz_key):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    a = torch.from_numpy(a_np[:size, :size]).cuda()
    b = torch.from_numpy(b_np[:size, :size]).cuda()
    out = (a @ b).cpu().numpy().astype(np.float32)
    if size == SIZE:
        npz_store[npz_key] = out
    return {
        "allow_tf32": allow_tf32,
        "shape": [size, size],
        "matmul_sha256": sha256_of_array(out, dtype=np.float32),
    }


def diff_stats_f32(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.ravel()
    b = b.ravel()
    absdiff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    denom = np.maximum(np.abs(a.astype(np.float64)), np.abs(b.astype(np.float64)))
    with np.errstate(divide="ignore", invalid="ignore"):
        reldiff = np.where(denom > 0, absdiff / denom, 0.0)
    return {
        "n": int(a.size),
        "max_abs_diff": float(absdiff.max()),
        "max_rel_diff": float(reldiff.max()),
        "mean_abs_diff": float(absdiff.mean()),
    }


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else None
    npz_store: dict = {}

    defaults = record_tf32_defaults()

    rng = np.random.default_rng(SEED)
    a_np = rng.standard_normal((SIZE, SIZE)).astype(np.float32)
    b_np = rng.standard_normal((SIZE, SIZE)).astype(np.float32)

    off_attempt = run_with_fallback(
        lambda size: matmul_with_tf32(False, a_np, b_np, size, npz_store, "e8d_matmul_tf32_off"),
        [SIZE, 128, 32, 8],
    )
    on_attempt = run_with_fallback(
        lambda size: matmul_with_tf32(True, a_np, b_np, size, npz_store, "e8d_matmul_tf32_on"),
        [SIZE, 128, 32, 8],
    )

    both_ok = off_attempt.get("status") == "ok" and on_attempt.get("status") == "ok"
    hashes_match = both_ok and off_attempt["matmul_sha256"] == on_attempt["matmul_sha256"]
    diff = None
    if both_ok and "e8d_matmul_tf32_off" in npz_store and "e8d_matmul_tf32_on" in npz_store:
        diff = diff_stats_f32(npz_store["e8d_matmul_tf32_off"], npz_store["e8d_matmul_tf32_on"])

    result = {
        "experiment": "E8d_tf32",
        "seed": SEED,
        "tf32_defaults": defaults,
        "tf32_off": off_attempt,
        "tf32_on": on_attempt,
        "hashes_match": hashes_match,
        "diff_tf32_off_vs_on": diff,
    }
    if outdir:
        np.savez(f"{outdir}/raw_arrays.npz", **npz_store)
    emit(result)


if __name__ == "__main__":
    main()
