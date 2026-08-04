"""E7 comparison: max abs diff / max rel diff / max ULP diff for the
items that mismatched in E4 and E5.

Reads the raw captures produced by e7_capture_raw.py from
  <results>/e7_raw/{old-amd64,base-amd64,base-arm64}/raw_arrays.npz
plus the per-env JSON (<results>/e7_<env>.json), cross-checks the
recomputed hashes against the original e4_*.json / e5_*.json, and writes
the diff statistics JSON to stdout.

ULP distance (per experiment spec): reinterpret the float bits as a
signed integer, order-normalize negatives (bits b < 0 -> INT_MIN - b,
i.e. the standard monotone mapping such that adjacent floats differ by
1), then take |a - b| exactly (python ints, no overflow).

Usage: python e7_compare.py /work/results
"""
import json
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
    assert a.shape == b.shape and a.dtype == b.dtype

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
        "dtype": str(a.dtype),
        "n_total": int(a.size),
        "n_mismatch": n_mismatch,
        "mismatch_fraction": n_mismatch / a.size if a.size else 0.0,
        "max_abs_diff": float(absdiff.max()),
        "max_rel_diff": float(reldiff.max()),
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


def get_in(d: dict, dotted: str):
    for k in dotted.split("."):
        d = d[k]
    return d


def main():
    results = sys.argv[1].rstrip("/")

    envs = ["old-amd64", "base-amd64", "base-arm64"]
    npz = {e: np.load(f"{results}/e7_raw/{e}/raw_arrays.npz") for e in envs}
    e7j = {e: json.load(open(f"{results}/e7_{e}.json")) for e in envs}

    # --- cross-check: E7 must reproduce the exact values E4/E5 hashed ----
    originals = {
        "old-amd64": json.load(open(f"{results}/e4_old.json")),
        "base-amd64": json.load(open(f"{results}/e5_amd64.json")),
        "base-arm64": json.load(open(f"{results}/e5_arm64.json")),
    }
    crosscheck = {}
    check_keys = {
        "old-amd64": [
            ("sklearn.logistic_regression.coef_sha256",) * 2,
            ("sklearn.random_forest.feature_importances_sha256",) * 2,
        ],
        "base-amd64": [
            ("float_math_ops.exp_sha256",) * 2,
            ("float_math_ops.matmul_sha256",) * 2,
            ("sklearn.logistic_regression.coef_sha256",) * 2,
            ("sklearn.random_forest.feature_importances_sha256",) * 2,
            ("torch_mlp.loss_curve_sha256",) * 2,
            ("torch_mlp.final_params_sha256",) * 2,
        ],
    }
    check_keys["base-arm64"] = check_keys["base-amd64"]
    for env, pairs in check_keys.items():
        env_checks = {}
        for e7_key, orig_key in pairs:
            v_new = get_in(e7j[env], e7_key)
            v_old = get_in(originals[env], orig_key)
            env_checks[e7_key] = {"match": v_new == v_old}
            if v_new != v_old:
                env_checks[e7_key].update({"e7": v_new, "original": v_old})
        crosscheck[env] = env_checks
    all_crosschecks_ok = all(
        c["match"] for env in crosscheck.values() for c in env.values()
    )

    # --- E4 pair: old-amd64 vs base-amd64 (mismatch was LR coef_ only) --
    e4_pair = {
        "lr_coef": diff_stats(npz["old-amd64"]["lr_coef"], npz["base-amd64"]["lr_coef"]),
        "lr_intercept": diff_stats(
            npz["old-amd64"]["lr_intercept"], npz["base-amd64"]["lr_intercept"]
        ),
        "lr_n_iter": {
            "old": e7j["old-amd64"]["sklearn"]["lr_n_iter"],
            "new": e7j["base-amd64"]["sklearn"]["lr_n_iter"],
        },
        "rf_feature_importances": diff_stats(
            npz["old-amd64"]["rf_feature_importances"],
            npz["base-amd64"]["rf_feature_importances"],
        ),
    }

    # --- E5 pair: base-amd64 vs base-arm64 (6 mismatching items) --------
    A, B = npz["base-amd64"], npz["base-arm64"]
    torch_losses_a64 = A["torch_losses_f64"]
    torch_losses_b64 = B["torch_losses_f64"]
    e5_pair = {
        "np_exp_100k": diff_stats(A["exp_vals"], B["exp_vals"]),
        "matmul_512": diff_stats(A["matmul"], B["matmul"]),
        "rf_feature_importances": diff_stats(
            A["rf_feature_importances"], B["rf_feature_importances"]
        ),
        "lr_coef": diff_stats(A["lr_coef"], B["lr_coef"]),
        "torch_loss_curve_as_float64": diff_stats(torch_losses_a64, torch_losses_b64),
        # loss.item() values originate from float32 tensors, so the
        # float32 ULP distance is the physically meaningful one.
        "torch_loss_curve_as_float32": diff_stats(
            torch_losses_a64.astype(np.float32), torch_losses_b64.astype(np.float32)
        ),
        "torch_final_params_float32": diff_stats(
            np.concatenate([A[f"torch_param_{i}"].ravel() for i in range(4)]),
            np.concatenate([B[f"torch_param_{i}"].ravel() for i in range(4)]),
        ),
        # sanity: items that matched in E5 must have 0 mismatches here too
        "sanity_sin_100k": diff_stats(A["sin_vals"], B["sin_vals"]),
    }

    out = {
        "label": "E7_ulp_diff",
        "note": (
            "diff statistics for the items that mismatched in E4 "
            "(old-amd64 vs base-amd64) and E5 (base-amd64 vs base-arm64); "
            "raw values captured by e7_capture_raw.py with identical "
            "procedure/seed as e4_version_stream.py / e5_arch_compare.py"
        ),
        "hash_crosscheck_vs_original_e4_e5": {
            "all_ok": all_crosschecks_ok,
            "detail": crosscheck,
        },
        "e4_pair_old_amd64_vs_base_amd64": e4_pair,
        "e5_pair_base_amd64_vs_base_arm64": e5_pair,
        "metadata": {env: e7j[env]["metadata"] for env in envs},
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
