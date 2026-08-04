"""E7: capture the RAW float values behind the E4/E5 mismatching hashes.

E4 (old-amd64 vs base-amd64) mismatched only on:
  - sklearn.logistic_regression.coef_
E5 (base-amd64 vs base-arm64) mismatched on:
  - float_math_ops.exp (np.exp on 100k float64)
  - float_math_ops.matmul (512x512 float64)
  - sklearn.random_forest.feature_importances_
  - sklearn.logistic_regression.coef_
  - torch loss curve (50 steps) and final params

Because the original scripts only stored SHA256 hashes, we re-run the
EXACT same generation procedure (same seed, same RNG consumption order as
e4_version_stream.py / e5_arch_compare.py) and dump the raw values:
  - small arrays: float.hex() string lists in the JSON on stdout
  - large arrays (exp 100k, matmul 512x512): full raw arrays in an .npz
    written to the output dir given as argv[1]
  - recomputed SHA256 hashes are included so results can be cross-checked
    against the original e4_*.json / e5_*.json files.

Usage (inside container):  python e7_capture_raw.py /work/out > result.json
"""
import hashlib
import sys

import numpy as np

from _common import emit, sha256_of_array

SEED = 42


def float_math_ops(npz_store: dict):
    """Identical procedure to e5_arch_compare.float_math_ops()."""
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-10, 10, size=100_000)

    sin_vals = np.sin(x)
    exp_vals = np.exp(x / 10.0)

    a = rng.standard_normal((512, 512))
    b = rng.standard_normal((512, 512))
    matmul = a @ b

    big = rng.standard_normal(20_000_000)
    total = np.sum(big)

    npz_store["exp_vals"] = exp_vals.astype(np.float64)
    npz_store["matmul"] = matmul.astype(np.float64)
    npz_store["sin_vals"] = sin_vals.astype(np.float64)

    return {
        "sin_sha256": sha256_of_array(sin_vals, dtype=np.float64),
        "exp_sha256": sha256_of_array(exp_vals, dtype=np.float64),
        "matmul_sha256": sha256_of_array(matmul, dtype=np.float64),
        "big_array_sum_hex": float(total).hex(),
        "big_array_sum_sha256": sha256_of_array(np.array([total]), dtype=np.float64),
    }


def sklearn_models(npz_store: dict):
    """Identical procedure to e4_version_stream/e5_arch_compare sklearn_models()."""
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

    fi = np.asarray(rf.feature_importances_, dtype=np.float64)
    coef = np.asarray(lr.coef_, dtype=np.float64).ravel()
    intercept = np.asarray(lr.intercept_, dtype=np.float64).ravel()

    npz_store["rf_feature_importances"] = fi
    npz_store["lr_coef"] = coef
    npz_store["lr_intercept"] = intercept

    return {
        "sklearn_version": sklearn.__version__,
        "lr_n_iter": [int(v) for v in np.atleast_1d(lr.n_iter_)],
        "random_forest": {
            "predictions_sha256": sha256_of_array(rf_preds, dtype=np.int64),
            "feature_importances_sha256": sha256_of_array(fi, dtype=np.float64),
            "feature_importances_hex": [float(v).hex() for v in fi],
        },
        "logistic_regression": {
            "predictions_sha256": sha256_of_array(lr_preds, dtype=np.int64),
            "coef_sha256": sha256_of_array(lr.coef_, dtype=np.float64),
            "coef_hex": [float(v).hex() for v in coef],
            "intercept_hex": [float(v).hex() for v in intercept],
        },
    }


def torch_mlp(npz_store: dict):
    """Identical procedure to e5_arch_compare.torch_mlp()."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return {"skipped": "torch not installed"}

    torch.set_num_threads(1)
    torch.manual_seed(SEED)

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

    param_arrays = [p.detach().numpy().astype(np.float32) for p in model.parameters()]
    param_bytes = b"".join(p.detach().numpy().tobytes() for p in model.parameters())

    losses64 = np.array(losses, dtype=np.float64)
    npz_store["torch_losses_f64"] = losses64
    for i, p in enumerate(param_arrays):
        npz_store[f"torch_param_{i}"] = p

    return {
        "torch_version": torch.__version__,
        "num_threads": torch.get_num_threads(),
        "final_loss_hex": float(losses[-1]).hex(),
        "loss_curve_hex": [float(v).hex() for v in losses],
        "loss_curve_sha256": sha256_of_array(losses64, dtype=np.float64),
        "final_params_sha256": hashlib.sha256(param_bytes).hexdigest(),
        "param_shapes": [list(p.shape) for p in param_arrays],
    }


def main():
    outdir = sys.argv[1]
    npz_store: dict = {}
    result = {
        "experiment": "E7_capture_raw",
        "seed": SEED,
        "float_math_ops": float_math_ops(npz_store),
        "sklearn": sklearn_models(npz_store),
        "torch_mlp": torch_mlp(npz_store),
    }
    np.savez(f"{outdir}/raw_arrays.npz", **npz_store)
    emit(result)


if __name__ == "__main__":
    main()
