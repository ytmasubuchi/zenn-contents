"""E4: does the raw PRNG stream / sklearn model output survive a library
version change on the *same* machine/architecture?

Run this identical script inside two different Docker images (old:
python3.11 + numpy 1.26.x + sklearn 1.3.x vs new: python3.12 + numpy 2.x +
sklearn 1.5.x+) and diff the resulting JSON.

Expectation being tested:
  - RandomState (MT19937) stream: numpy guarantees stream stability across
    versions for the legacy API -> should match.
  - Generator (PCG64) stream: no formal guarantee, but PCG64 itself hasn't
    changed -> usually matches in practice.
  - sklearn RandomForest with fixed random_state: algorithm/defaults can
    change between minor versions (e.g. criterion defaults, tie-breaking,
    n_estimators default) -> may NOT match even with same random_state.
"""
import numpy as np

from _common import emit, sha256_of_array

N = 2000
SEED = 42


def randomstate_raw_stream():
    rs = np.random.RandomState(SEED)
    raw_uint32 = rs.randint(0, 2**32, size=N, dtype=np.uint64).astype(np.uint32)
    uniform = rs.random_sample(N)
    normal = rs.standard_normal(N)
    return {
        "raw_uint32_sha256": sha256_of_array(raw_uint32, dtype=np.uint32),
        "uniform_float64_sha256": sha256_of_array(uniform, dtype=np.float64),
        "normal_float64_sha256": sha256_of_array(normal, dtype=np.float64),
    }


def generator_pcg64_stream():
    rng = np.random.default_rng(SEED)
    raw_uint64 = rng.integers(0, 2**63 - 1, size=N, dtype=np.int64)
    uniform = rng.random(N)
    normal = rng.standard_normal(N)
    return {
        "raw_int64_sha256": sha256_of_array(raw_uint64, dtype=np.int64),
        "uniform_float64_sha256": sha256_of_array(uniform, dtype=np.float64),
        "normal_float64_sha256": sha256_of_array(normal, dtype=np.float64),
    }


def sklearn_models():
    try:
        import sklearn
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except ImportError:
        return {"skipped": "sklearn not installed"}

    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=10, n_redundant=5, random_state=SEED
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    lr = LogisticRegression(random_state=SEED, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    return {
        "sklearn_version": sklearn.__version__,
        "random_forest": {
            "predictions_sha256": sha256_of_array(rf_preds, dtype=np.int64),
            "feature_importances_sha256": sha256_of_array(rf.feature_importances_, dtype=np.float64),
        },
        "logistic_regression": {
            "predictions_sha256": sha256_of_array(lr_preds, dtype=np.int64),
            "coef_sha256": sha256_of_array(lr.coef_, dtype=np.float64),
        },
    }


def main():
    result = {
        "experiment": "E4_version_stream",
        "seed": SEED,
        "n": N,
        "numpy_randomstate_mt19937_raw": randomstate_raw_stream(),
        "numpy_generator_pcg64_raw": generator_pcg64_stream(),
        "sklearn": sklearn_models(),
    }
    emit(result)


if __name__ == "__main__":
    main()
