"""E1: same-process / repeated-process-run reproducibility baseline.

Generates fixed-seed random streams from the four most common DS/ML entry
points (stdlib random, numpy legacy RandomState/MT19937, numpy new
Generator/PCG64, torch CPU) plus a small sklearn train/test/RandomForest
pipeline, and hashes every output exactly (bit-for-bit, not printed values).

Run this script multiple times (separate `python` invocations, or separate
`docker run` invocations for E3) and diff the JSON: if E1 holds, every hash
below must be byte-identical across runs on the same machine/image.
"""
import random

import numpy as np

from _common import emit, sha256_of_array, sha256_of_str_list

N = 1000
SEED = 42


def stdlib_random_stream():
    random.seed(SEED)
    vals = [random.random() for _ in range(N)]
    return sha256_of_array(vals, dtype=np.float64)


def numpy_randomstate_stream():
    rs = np.random.RandomState(SEED)  # MT19937 legacy API
    vals = rs.random_sample(N)
    ints = rs.randint(0, 2**63 - 1, size=N, dtype=np.int64)
    return {
        "float_sha256": sha256_of_array(vals, dtype=np.float64),
        "int_sha256": sha256_of_array(ints, dtype=np.int64),
    }


def numpy_global_seed_stream():
    """np.random.seed(42) legacy *global* state (distinct from RandomState instance)."""
    np.random.seed(SEED)
    vals = np.random.random(N)
    return sha256_of_array(vals, dtype=np.float64)


def numpy_generator_stream():
    rng = np.random.default_rng(SEED)  # PCG64
    vals = rng.random(N)
    ints = rng.integers(0, 2**63 - 1, size=N, dtype=np.int64)
    return {
        "float_sha256": sha256_of_array(vals, dtype=np.float64),
        "int_sha256": sha256_of_array(ints, dtype=np.int64),
    }


def torch_cpu_stream():
    try:
        import torch
    except ImportError:
        return {"skipped": "torch not installed"}
    torch.manual_seed(SEED)
    vals = torch.rand(N, dtype=torch.float64).numpy()
    ints = torch.randint(0, 2**62, (N,), dtype=torch.int64).numpy()
    return {
        "float_sha256": sha256_of_array(vals, dtype=np.float64),
        "int_sha256": sha256_of_array(ints, dtype=np.int64),
    }


def sklearn_pipeline():
    try:
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
    except ImportError:
        return {"skipped": "sklearn not installed"}

    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=SEED,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED
    )
    split_sha = sha256_of_array(np.concatenate([X_train.ravel(), X_test.ravel()]), dtype=np.float64)

    clf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)

    return {
        "train_test_split_sha256": split_sha,
        "predictions_sha256": sha256_of_array(preds, dtype=np.int64),
        "predict_proba_sha256": sha256_of_array(proba, dtype=np.float64),
        "feature_importances_sha256": sha256_of_array(clf.feature_importances_, dtype=np.float64),
    }


def main():
    result = {
        "experiment": "E1_baseline_reproducibility",
        "seed": SEED,
        "n": N,
        "streams": {
            "stdlib_random": stdlib_random_stream(),
            "numpy_randomstate_mt19937": numpy_randomstate_stream(),
            "numpy_global_seed_legacy": numpy_global_seed_stream(),
            "numpy_generator_pcg64": numpy_generator_stream(),
            "torch_cpu": torch_cpu_stream(),
        },
        "sklearn": sklearn_pipeline(),
    }
    emit(result)


if __name__ == "__main__":
    main()
