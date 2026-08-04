"""E5: x86_64 vs arm64 (QEMU) reproducibility.

Run this identical script (same Docker image, same source) once with
`--platform linux/amd64` and once with `--platform linux/arm64` and diff
the JSON. Sizes are kept small since ARM64 runs under QEMU emulation.

What's being probed:
  a. raw PRNG integer stream                -> expected to match (pure integer arithmetic)
  b. uniform/normal float64 stream           -> expected to match (same bit-exact algorithm)
  c. transcendental funcs / BLAS matmul / sum reduction -> NOT guaranteed
     (different libm/BLAS implementation and SIMD widths per architecture
     can change rounding in the last bit(s), and summation order for
     reductions is implementation-defined)
  d. sklearn RandomForest / LogisticRegression training -> depends on (c)
  e. torch CPU small-MLP training (1 thread, fixed seed) -> depends on (c)
"""
import numpy as np

from _common import emit, sha256_of_array

SEED = 42
N = 2000


def raw_prng_streams():
    rs = np.random.RandomState(SEED)
    rs_ints = rs.randint(0, 2**32, size=N, dtype=np.int64)
    rs_uniform = rs.random_sample(N)
    rs_normal = rs.standard_normal(N)

    rng = np.random.default_rng(SEED)
    gen_ints = rng.integers(0, 2**63 - 1, size=N, dtype=np.int64)
    gen_uniform = rng.random(N)
    gen_normal = rng.standard_normal(N)

    return {
        "randomstate_mt19937": {
            "int_sha256": sha256_of_array(rs_ints, dtype=np.int64),
            "uniform_sha256": sha256_of_array(rs_uniform, dtype=np.float64),
            "normal_sha256": sha256_of_array(rs_normal, dtype=np.float64),
        },
        "generator_pcg64": {
            "int_sha256": sha256_of_array(gen_ints, dtype=np.int64),
            "uniform_sha256": sha256_of_array(gen_uniform, dtype=np.float64),
            "normal_sha256": sha256_of_array(gen_normal, dtype=np.float64),
        },
    }


def float_math_ops():
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-10, 10, size=100_000)

    sin_vals = np.sin(x)
    exp_vals = np.exp(x / 10.0)

    a = rng.standard_normal((512, 512))
    b = rng.standard_normal((512, 512))
    matmul = a @ b

    big = rng.standard_normal(20_000_000)
    total = np.sum(big)

    return {
        "sin_sha256": sha256_of_array(sin_vals, dtype=np.float64),
        "exp_sha256": sha256_of_array(exp_vals, dtype=np.float64),
        "matmul_sha256": sha256_of_array(matmul, dtype=np.float64),
        "big_array_sum_hex": float(total).hex(),
        "big_array_sum_sha256": sha256_of_array(np.array([total]), dtype=np.float64),
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

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


def torch_mlp():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return {"skipped": "torch not installed"}

    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    # Fixed synthetic data (generated from numpy with a fixed seed so the
    # *data* is identical across architectures; only torch's own compute
    # path differs).
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((200, 10)).astype(np.float32)
    true_w = rng.standard_normal(10).astype(np.float32)
    y = (X @ true_w + 0.1 * rng.standard_normal(200).astype(np.float32)).reshape(-1, 1)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    torch.manual_seed(SEED)
    model = nn.Sequential(
        nn.Linear(10, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
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

    param_bytes = b"".join(p.detach().numpy().tobytes() for p in model.parameters())
    import hashlib

    return {
        "torch_version": torch.__version__,
        "num_threads": torch.get_num_threads(),
        "final_loss_hex": float(losses[-1]).hex(),
        "loss_curve_sha256": sha256_of_array(np.array(losses, dtype=np.float64), dtype=np.float64),
        "final_params_sha256": hashlib.sha256(param_bytes).hexdigest(),
    }


def main():
    result = {
        "experiment": "E5_arch_compare",
        "seed": SEED,
        "raw_prng_streams": raw_prng_streams(),
        "float_math_ops": float_math_ops(),
        "sklearn": sklearn_models(),
        "torch_mlp": torch_mlp(),
    }
    emit(result)


if __name__ == "__main__":
    main()
