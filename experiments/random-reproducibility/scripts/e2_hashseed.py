"""E2: PYTHONHASHSEED effects on hash() and set ordering.

This script does NOT set PYTHONHASHSEED itself -- it reports whatever is in
effect for the current process so that run_all.sh can invoke it multiple
times with different env (unset vs PYTHONHASHSEED=0) and diff the results.

Includes the classic DS footgun: building a feature-name list via
`list(set(columns))` and getting a different column order every run,
which silently reorders model inputs / feature_importances_ between runs.
"""
import os

from _common import emit, sha256_of_str_list

FEATURE_COLUMNS = [
    "age", "income", "region", "tenure_months", "num_purchases",
    "avg_basket_size", "is_churned", "signup_channel", "device_type",
    "last_login_days",
]


def hash_of_strings():
    words = ["hello", "world", "reproducibility", "seed", "pandas", "numpy"]
    return {w: hash(w) for w in words}


def set_ordering_demo():
    """The classic bug: feature columns reordered by set() every run."""
    s = set(FEATURE_COLUMNS)
    ordered_via_set = list(s)
    return {
        "original_order": FEATURE_COLUMNS,
        "order_after_set_roundtrip": ordered_via_set,
        "order_matches_original": ordered_via_set == FEATURE_COLUMNS,
        "order_sha256": sha256_of_str_list(ordered_via_set),
    }


def main():
    result = {
        "experiment": "E2_pythonhashseed",
        "pythonhashseed_env": os.environ.get("PYTHONHASHSEED"),
        "hash_of_strings": hash_of_strings(),
        "set_ordering_demo": set_ordering_demo(),
    }
    emit(result)


if __name__ == "__main__":
    main()
