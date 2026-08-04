"""E8e: max-abs-diff / max-rel-diff / max-ULP-diff for whichever E8a-E8d
items turn out to mismatch, mirroring e7_compare.py's methodology exactly
(same ULP ordering scheme) but for the GPU experiments.

Unlike E7 (which only had to explain two known mismatch sets from E4/E5),
E8a and E8b's raw arrays are captured for *every* run (see the npz_store
plumbing added to e8a_gpu_repeat.py / e8b_gpu_deterministic.py), so this
script can diff any pair of runs -- run1 vs run2, run1 vs run3 -- for every
measured quantity, plus the CPU-vs-GPU (E8c) and TF32-on-vs-off (E8d) raw
captures.

If a GPU measurement never actually produced output in a given run (e.g.
every attempt hit a CUDA OOM because the GPU's VRAM was almost entirely
occupied by another process), the corresponding npz key is simply absent;
this script records that as "insufficient_data" for that item rather than
failing, since "we could not measure this at all" is itself a fact worth
recording, not an error in this script.

Usage: python e8e_ulp_diff.py /work/results
Reads:
  <results>/e8_raw/e8a_run{1,2,3}/raw_arrays.npz
  <results>/e8_raw/e8b_run{1,2,3}/raw_arrays.npz
  <results>/e8_raw/e8c/raw_arrays.npz
  <results>/e8_raw/e8d/raw_arrays.npz
Writes diff-statistics JSON to stdout.
"""
import sys

import numpy as np


def _ordered_int(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.float64:
        i = a.view(np.int64)
        return np.where(i >= 0, i, np.int64(-(2**63)) - i)
    if a.dtype == np.float32:
        i = a.view(np.int32)
        return np.where(i >= 0, i, np.int32(-(2**31)) - i)
    raise TypeError(a.dtype)


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.shape != b.shape:
        return {"comparable": False, "reason": f"shape mismatch {a.shape} vs {b.shape}"}
    if a.dtype != b.dtype:
        return {"comparable": False, "reason": f"dtype mismatch {a.dtype} vs {b.dtype}"}

    bits_a = a.view(np.int64 if a.dtype == np.float64 else np.int32)
    bits_b = b.view(np.int64 if a.dtype == np.float64 else np.int32)
    mismatch_mask = bits_a != bits_b
    n_mismatch = int(mismatch_mask.sum())

    absdiff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    denom = np.maximum(np.abs(a.astype(np.float64)), np.abs(b.astype(np.float64)))
    with np.errstate(divide="ignore", invalid="ignore"):
        reldiff = np.where(denom > 0, absdiff / denom, 0.0)

    oa = _ordered_int(a).astype(object)
    ob = _ordered_int(b).astype(object)
    ulp = np.abs(oa - ob)
    max_ulp = int(ulp.max()) if len(ulp) else 0

    stats = {
        "comparable": True,
        "dtype": str(a.dtype),
        "n_total": int(a.size),
        "n_mismatch": n_mismatch,
        "mismatch_fraction": n_mismatch / a.size if a.size else 0.0,
        "max_abs_diff": float(absdiff.max()) if a.size else None,
        "max_rel_diff": float(reldiff.max()) if a.size else None,
        "max_ulp_diff": max_ulp,
    }
    if n_mismatch:
        idx = int(np.argmax(ulp))
        stats["worst_element"] = {
            "index": idx,
            "a_hex": float(a[idx]).hex(),
            "b_hex": float(b[idx]).hex(),
            "abs_diff": float(absdiff[idx]),
        }
    return stats


def load_npz(path: str):
    try:
        return np.load(path)
    except FileNotFoundError:
        return None


def compare_key(npz_a, npz_b, key: str) -> dict:
    if npz_a is None or npz_b is None:
        return {"comparable": False, "reason": "npz file(s) not found"}
    if key not in npz_a.files or key not in npz_b.files:
        return {
            "comparable": False,
            "reason": "insufficient_data",
            "present_in_a": key in (npz_a.files if npz_a else []),
            "present_in_b": key in (npz_b.files if npz_b else []),
        }
    return diff_stats(npz_a[key], npz_b[key])


# Keys captured by e8a_gpu_repeat.py / e8b_gpu_deterministic.py, shared
# between both since they use the identical measurement functions.
# atomic_index_add_last / atomic_scatter_add_last come from the
# index_add_repeat / scatter_add_repeat measurements (split from a single
# atomic_add_repeat measurement so index_add_'s and scatter_add_'s outcomes
# under deterministic mode are observable independently); embedding_bag_sum
# is the embedding_bag_forward output, embedding_bag_grad_weight is the
# weight.grad from embedding_bag_backward (added because PyTorch's
# nondeterminism docs for embedding_bag concern backward, not forward).
E8AB_KEYS = [
    "philox_rand",
    "philox_randn",
    "matmul_512",
    "conv_output",
    "conv_grad_input",
    "conv_grad_weight",
    "mlp_loss_curve",
    "mlp_params",
    "atomic_index_add_last",
    "atomic_scatter_add_last",
    "embedding_bag_sum",
    "embedding_bag_grad_weight",
    "cumsum",
]


def main():
    results = sys.argv[1].rstrip("/")

    out = {
        "label": "E8e_ulp_diff",
        "note": (
            "diff statistics (max abs/rel/ULP) for E8a/E8b repeated-run "
            "pairs, E8c CPU-vs-GPU pairs, and E8d TF32-off-vs-on pairs. "
            "'insufficient_data' means the corresponding GPU measurement "
            "never produced output in one or both runs being compared -- "
            "the original experiment (e8a/e8b/e8c/e8d) failed to capture "
            "that raw array for some reason (e.g. every fallback size hit "
            "a CUDA OOM, or the op raised RuntimeError: no deterministic "
            "implementation under use_deterministic_algorithms(True), as "
            "happens for cumsum in E8b). Check that run's own JSON "
            "(e.g. e8b_run*.json's 'status'/'error_message' fields) for "
            "the specific reason in any given case."
        ),
    }

    # --- E8a: run1 vs run2, run1 vs run3 (repeated identical containers) --
    e8a = {}
    npz_a1 = load_npz(f"{results}/e8_raw/e8a_run1/raw_arrays.npz")
    npz_a2 = load_npz(f"{results}/e8_raw/e8a_run2/raw_arrays.npz")
    npz_a3 = load_npz(f"{results}/e8_raw/e8a_run3/raw_arrays.npz")
    e8a["run1_vs_run2"] = {k: compare_key(npz_a1, npz_a2, k) for k in E8AB_KEYS}
    e8a["run1_vs_run3"] = {k: compare_key(npz_a1, npz_a3, k) for k in E8AB_KEYS}
    out["e8a_gpu_repeat_default"] = e8a

    # --- E8b: run1 vs run2, run1 vs run3 (deterministic-mode containers) --
    e8b = {}
    npz_b1 = load_npz(f"{results}/e8_raw/e8b_run1/raw_arrays.npz")
    npz_b2 = load_npz(f"{results}/e8_raw/e8b_run2/raw_arrays.npz")
    npz_b3 = load_npz(f"{results}/e8_raw/e8b_run3/raw_arrays.npz")
    e8b["run1_vs_run2"] = {k: compare_key(npz_b1, npz_b2, k) for k in E8AB_KEYS}
    e8b["run1_vs_run3"] = {k: compare_key(npz_b1, npz_b3, k) for k in E8AB_KEYS}
    out["e8b_gpu_deterministic"] = e8b

    # --- E8c: CPU vs GPU -------------------------------------------------
    npz_c = load_npz(f"{results}/e8_raw/e8c/raw_arrays.npz")
    e8c_pairs = {
        "rand_stream": ("e8c_rand_cpu", "e8c_rand_gpu"),
        "matmul": ("e8c_matmul_cpu", "e8c_matmul_gpu"),
        "mlp_params": ("e8c_mlp_params_cpu", "e8c_mlp_params_gpu"),
    }
    e8c = {}
    for name, (ka, kb) in e8c_pairs.items():
        if npz_c is not None and ka in npz_c.files and kb in npz_c.files:
            e8c[name] = diff_stats(npz_c[ka], npz_c[kb])
        else:
            e8c[name] = {"comparable": False, "reason": "insufficient_data"}
    out["e8c_cpu_vs_gpu"] = e8c

    # --- E8d: TF32 off vs on ----------------------------------------------
    npz_d = load_npz(f"{results}/e8_raw/e8d/raw_arrays.npz")
    if npz_d is not None and "e8d_matmul_tf32_off" in npz_d.files and "e8d_matmul_tf32_on" in npz_d.files:
        e8d = diff_stats(npz_d["e8d_matmul_tf32_off"], npz_d["e8d_matmul_tf32_on"])
    else:
        e8d = {"comparable": False, "reason": "insufficient_data"}
    out["e8d_tf32_off_vs_on"] = e8d

    import json

    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
