"""E6 repeat: summarize whether 5 independent container runs per
OMP_NUM_THREADS setting produced identical hashes.

The original E6 ran e6_parallel_nondeterminism.py only once per setting;
this reruns it 5x per setting (fresh container each time, driven by
run_e6_repeat.sh) and checks that every compared hash is identical
across all runs of the same setting, and also across the two settings.

Usage: python e6_repeat_summary.py /work/results
Reads  <results>/e6_repeat/e6_omp{1,8}_run{1..5}.json
Writes summary JSON to stdout.
"""
import json
import sys

# The hashes/values that constitute the reproducibility claim.
KEYS = [
    "non_associativity.large_array.forward_sum_hex",
    "non_associativity.large_array.reversed_sum_hex",
    "non_associativity.large_array.shuffled_sum_hex",
    "non_associativity.large_array.naive_sequential_sum_hex",
    "blas_threads.matmul_sha256",
    "blas_threads.sum_hex",
    "torch_threads.threads_1_sha256",
    "torch_threads.threads_8_sha256",
    "sklearn_njobs.n_jobs_1.predictions_sha256",
    "sklearn_njobs.n_jobs_1.feature_importances_sha256",
    "sklearn_njobs.n_jobs_minus1.predictions_sha256",
    "sklearn_njobs.n_jobs_minus1.feature_importances_sha256",
]


def get_in(d, dotted):
    for k in dotted.split("."):
        d = d[k]
    return d


def main():
    results = sys.argv[1].rstrip("/")
    n_runs = 5
    settings = {}
    values = {}  # (setting, key) -> value from run1
    for omp in (1, 8):
        runs = [
            json.load(open(f"{results}/e6_repeat/e6_omp{omp}_run{i}.json"))
            for i in range(1, n_runs + 1)
        ]
        per_key = {}
        for key in KEYS:
            vals = [get_in(r, key) for r in runs]
            per_key[key] = {
                "all_runs_identical": len(set(vals)) == 1,
                "value": vals[0] if len(set(vals)) == 1 else vals,
            }
            values[(omp, key)] = vals[0]
        settings[f"omp{omp}"] = {
            "n_runs": n_runs,
            "n_keys_compared": len(KEYS),
            "all_keys_identical_across_runs": all(
                v["all_runs_identical"] for v in per_key.values()
            ),
            "per_key": per_key,
        }

    cross = {
        key: values[(1, key)] == values[(8, key)]
        for key in KEYS
    }
    out = {
        "label": "E6_repeat_summary",
        "note": (
            "e6_parallel_nondeterminism.py run 5x per OMP_NUM_THREADS "
            "setting, each in a fresh container of random-repro:base-amd64"
        ),
        "settings": settings,
        "omp1_vs_omp8_same_value": {
            "all_match": all(cross.values()),
            "per_key": cross,
        },
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
