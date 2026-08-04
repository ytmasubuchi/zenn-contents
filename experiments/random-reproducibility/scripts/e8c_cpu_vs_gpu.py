"""E8c: CPU vs GPU, same seed / same input, on the SAME machine and process.

Three probes, each structured as "compute the CPU-side reference value
unconditionally, then attempt the GPU side wrapped in run_with_fallback" so
that even if the GPU half fails (OOM, etc.) the CPU reference value is
still recorded rather than lost:

  a. torch.manual_seed(42) then torch.rand() on CPU vs on CUDA. These use
     different RNG implementations/streams by design (PyTorch docs: the
     CPU and CUDA default generators are independent), so a mismatch here
     is *expected*, not a bug -- this probe exists to confirm that
     empirically rather than assume it.
  b. Same float32 input (generated once on CPU with a fixed seed, then
     `.to('cuda')` for the GPU side) run through matmul on CPU (native
     BLAS) vs GPU (cuBLAS). Isolates the compute backend from RNG.
  c. Same initial parameters (constructed once on CPU, then copied to a
     CUDA copy of the model) trained for 50 steps on the same data, on CPU
     vs GPU. Compares loss curve + final params.

Raw float32 arrays for the matmul and MLP-params comparisons are dumped to
`<outdir>/raw_arrays.npz` (outdir given as argv[1]) for e8e_ulp_diff.py to
compute max-abs-diff / max-rel-diff / max-ULP-diff on, mirroring the
e7_capture_raw.py convention.

Usage (inside container): python e8c_cpu_vs_gpu.py /work/out > result.json
"""
import copy
import hashlib
import sys

import numpy as np

from _common import emit, run_with_fallback, sha256_of_array

SEED = 42


def rand_stream_probe(npz_store: dict):
    import torch

    torch.manual_seed(SEED)
    cpu_r = torch.rand(1000, dtype=torch.float32)
    cpu_hash = sha256_of_array(cpu_r.numpy(), dtype=np.float32)
    npz_store["e8c_rand_cpu"] = cpu_r.numpy()

    def gpu_side(n):
        torch.manual_seed(SEED)
        gpu_r = torch.rand(n, device="cuda", dtype=torch.float32)
        gpu_hash = sha256_of_array(gpu_r.cpu().numpy(), dtype=np.float32)
        if n == 1000:
            npz_store["e8c_rand_gpu"] = gpu_r.cpu().numpy()
        return {"gpu_sha256": gpu_hash}

    gpu_attempt = run_with_fallback(gpu_side, [1000, 100, 10, 1])
    return {
        "n": 1000,
        "cpu_sha256": cpu_hash,
        "gpu_attempt": gpu_attempt,
        "match": (
            gpu_attempt.get("status") == "ok" and gpu_attempt.get("gpu_sha256") == cpu_hash
        ),
    }


def matmul_probe(npz_store: dict):
    import torch

    rng = np.random.default_rng(SEED)
    size = 512
    a_np = rng.standard_normal((size, size)).astype(np.float32)
    b_np = rng.standard_normal((size, size)).astype(np.float32)
    a_cpu = torch.from_numpy(a_np)
    b_cpu = torch.from_numpy(b_np)

    cpu_out = (a_cpu @ b_cpu).numpy()
    cpu_hash = sha256_of_array(cpu_out, dtype=np.float32)
    npz_store["e8c_matmul_cpu"] = cpu_out.astype(np.float32)

    def gpu_side(sz):
        a_g = torch.from_numpy(a_np[:sz, :sz]).cuda()
        b_g = torch.from_numpy(b_np[:sz, :sz]).cuda()
        out = (a_g @ b_g).cpu().numpy()
        gpu_hash = sha256_of_array(out, dtype=np.float32)
        if sz == size:
            npz_store["e8c_matmul_gpu"] = out.astype(np.float32)
        return {"gpu_sha256": gpu_hash}

    gpu_attempt = run_with_fallback(gpu_side, [size, 128, 32, 8])
    return {
        "shape": [size, size],
        "cpu_sha256": cpu_hash,
        "gpu_attempt": gpu_attempt,
        "match": (
            gpu_attempt.get("status") == "ok" and gpu_attempt.get("gpu_sha256") == cpu_hash
        ),
    }


def _build_model():
    import torch.nn as nn

    return nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1))


def _train(model, X_t, y_t, n_steps=50):
    import torch

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    losses = []
    for _ in range(n_steps):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def mlp_probe(npz_store: dict):
    import torch

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((200, 10)).astype(np.float32)
    true_w = rng.standard_normal(10).astype(np.float32)
    y = (X @ true_w + 0.1 * rng.standard_normal(200).astype(np.float32)).reshape(-1, 1)

    torch.manual_seed(SEED)
    model_cpu = _build_model()
    init_state = copy.deepcopy(model_cpu.state_dict())

    X_cpu = torch.from_numpy(X)
    y_cpu = torch.from_numpy(y)
    losses_cpu = _train(model_cpu, X_cpu, y_cpu)
    param_bytes_cpu = b"".join(p.detach().numpy().tobytes() for p in model_cpu.parameters())
    cpu_params = np.concatenate(
        [p.detach().numpy().ravel() for p in model_cpu.parameters()]
    ).astype(np.float32)
    npz_store["e8c_mlp_params_cpu"] = cpu_params

    cpu_result = {
        "final_loss_hex": float(losses_cpu[-1]).hex(),
        "loss_curve_sha256": sha256_of_array(np.array(losses_cpu, dtype=np.float64), dtype=np.float64),
        "final_params_sha256": hashlib.sha256(param_bytes_cpu).hexdigest(),
    }

    def gpu_side(n_samples):
        model_gpu = _build_model()
        model_gpu.load_state_dict(init_state)
        model_gpu = model_gpu.cuda()
        X_t = torch.from_numpy(X[:n_samples]).cuda()
        y_t = torch.from_numpy(y[:n_samples]).cuda()
        losses_gpu = _train(model_gpu, X_t, y_t)
        param_bytes_gpu = b"".join(
            p.detach().cpu().numpy().tobytes() for p in model_gpu.parameters()
        )
        out = {
            "n_samples": n_samples,
            "final_loss_hex": float(losses_gpu[-1]).hex(),
            "loss_curve_sha256": sha256_of_array(
                np.array(losses_gpu, dtype=np.float64), dtype=np.float64
            ),
            "final_params_sha256": hashlib.sha256(param_bytes_gpu).hexdigest(),
        }
        if n_samples == 200:
            gpu_params = np.concatenate(
                [p.detach().cpu().numpy().ravel() for p in model_gpu.parameters()]
            ).astype(np.float32)
            npz_store["e8c_mlp_params_gpu"] = gpu_params
        return out

    gpu_attempt = run_with_fallback(gpu_side, [200, 50, 10])
    return {
        "cpu": cpu_result,
        "gpu_attempt": gpu_attempt,
        "loss_curve_match": (
            gpu_attempt.get("status") == "ok"
            and gpu_attempt.get("loss_curve_sha256") == cpu_result["loss_curve_sha256"]
        ),
        "final_params_match": (
            gpu_attempt.get("status") == "ok"
            and gpu_attempt.get("final_params_sha256") == cpu_result["final_params_sha256"]
        ),
    }


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else None
    npz_store: dict = {}
    result = {
        "experiment": "E8c_cpu_vs_gpu",
        "seed": SEED,
        "rand_stream": rand_stream_probe(npz_store),
        "matmul": matmul_probe(npz_store),
        "mlp_train": mlp_probe(npz_store),
    }
    if outdir:
        np.savez(f"{outdir}/raw_arrays.npz", **npz_store)
    emit(result)


if __name__ == "__main__":
    main()
