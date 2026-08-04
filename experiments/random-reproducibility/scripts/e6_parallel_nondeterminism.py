"""E6: parallelism / thread-count / summation-order non-determinism on a
single machine (no cross-process or cross-arch variable at all).

Four sub-experiments:
  1. Floating point is not associative: summing the same values in a
     different order can change the bit pattern of the result. Includes a
     minimal 3-float example.
  2. BLAS thread count (OMP_NUM_THREADS): this script reports the matmul /
     np.sum hash under whatever OMP_NUM_THREADS is set in its environment.
     run_all.sh invokes it twice (once with OMP_NUM_THREADS=1, once with
     OMP_NUM_THREADS=8) as *separate processes*, since most BLAS
     implementations read the thread count once at library init time and
     ignore later in-process changes.
  3. torch CPU thread count: unlike BLAS, torch.set_num_threads() can be
     changed at runtime, so this is done in-process for both settings.
  4. sklearn RandomForest with n_jobs=1 vs n_jobs=-1: expected to match,
     since each tree gets an independent seed derived from random_state
     regardless of how the trees are scheduled across workers.
"""
import os

import numpy as np

from _common import emit, sha256_of_array

SEED = 42


def non_associativity_demo():
    # Minimal 3-value example where (a+b)+c != a+(b+c) in float64.
    a, b, c = 1.0, 1e16, -1e16
    left_assoc = (a + b) + c
    right_assoc = a + (b + c)
    minimal_example = {
        "a": a, "b": b, "c": c,
        "(a+b)+c": left_assoc,
        "a+(b+c)": right_assoc,
        "differ": bool(left_assoc != right_assoc),
    }

    rng = np.random.default_rng(SEED)
    values = rng.standard_normal(200_000)

    forward_sum = np.sum(values)
    reversed_sum = np.sum(values[::-1])
    shuffled = values.copy()
    rng.shuffle(shuffled)
    shuffled_sum = np.sum(shuffled)

    # Pairwise (numpy default, tree/pairwise summation) vs naive sequential
    # Python-level sum forces strict left-to-right accumulation.
    naive_sum = 0.0
    for v in values:
        naive_sum += v

    return {
        "minimal_3_value_example": minimal_example,
        "large_array": {
            "forward_sum_hex": float(forward_sum).hex(),
            "reversed_sum_hex": float(reversed_sum).hex(),
            "shuffled_sum_hex": float(shuffled_sum).hex(),
            "naive_sequential_sum_hex": float(naive_sum).hex(),
            "forward_vs_reversed_differ": bool(forward_sum != reversed_sum),
            "forward_vs_naive_differ": bool(forward_sum != naive_sum),
        },
    }


def blas_thread_demo():
    rng = np.random.default_rng(SEED)
    a = rng.standard_normal((1024, 1024))
    b = rng.standard_normal((1024, 1024))
    matmul = a @ b
    big = rng.standard_normal(20_000_000)
    total = np.sum(big)

    thread_info = None
    try:
        import threadpoolctl

        thread_info = threadpoolctl.threadpool_info()
    except ImportError:
        pass

    return {
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads_env": os.environ.get("OPENBLAS_NUM_THREADS"),
        "mkl_num_threads_env": os.environ.get("MKL_NUM_THREADS"),
        "threadpool_info": thread_info,
        "matmul_sha256": sha256_of_array(matmul, dtype=np.float64),
        "sum_hex": float(total).hex(),
    }


def torch_thread_demo():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return {"skipped": "torch not installed"}

    def train_with_threads(n_threads: int):
        torch.set_num_threads(n_threads)
        torch.manual_seed(SEED)

        rng = np.random.default_rng(SEED)
        X = rng.standard_normal((200, 10)).astype(np.float32)
        true_w = rng.standard_normal(10).astype(np.float32)
        y = (X @ true_w + 0.1 * rng.standard_normal(200).astype(np.float32)).reshape(-1, 1)
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)

        torch.manual_seed(SEED)
        model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        losses = []
        for _ in range(50):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return sha256_of_array(np.array(losses, dtype=np.float64), dtype=np.float64)

    return {
        "threads_1_sha256": train_with_threads(1),
        "threads_8_sha256": train_with_threads(min(8, os.cpu_count() or 8)),
    }


def sklearn_njobs_demo():
    try:
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
    except ImportError:
        return {"skipped": "sklearn not installed"}

    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=10, n_redundant=5, random_state=SEED
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

    def fit(n_jobs: int):
        clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=n_jobs)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        return {
            "predictions_sha256": sha256_of_array(preds, dtype=np.int64),
            "feature_importances_sha256": sha256_of_array(clf.feature_importances_, dtype=np.float64),
        }

    r1 = fit(1)
    rminus1 = fit(-1)
    return {
        "n_jobs_1": r1,
        "n_jobs_minus1": rminus1,
        "predictions_match": r1["predictions_sha256"] == rminus1["predictions_sha256"],
        "feature_importances_match": r1["feature_importances_sha256"] == rminus1["feature_importances_sha256"],
    }


def main():
    result = {
        "experiment": "E6_parallel_nondeterminism",
        "seed": SEED,
        "cpu_count": os.cpu_count(),
        "non_associativity": non_associativity_demo(),
        "blas_threads": blas_thread_demo(),
        "torch_threads": torch_thread_demo(),
        "sklearn_njobs": sklearn_njobs_demo(),
    }
    emit(result)


if __name__ == "__main__":
    main()
