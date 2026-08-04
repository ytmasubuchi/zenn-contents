"""E8a: GPU reproducibility under repeated runs, default (non-deterministic)
PyTorch settings -- i.e. NOT calling torch.use_deterministic_algorithms(),
NOT setting CUBLAS_WORKSPACE_CONFIG. cudnn.benchmark is left at whatever
torch's own default is (recorded in
metadata.gpu.cudnn_benchmark_at_metadata_collection, not overridden here).

Run this identical script 3x, each in a *fresh container*, and diff the
JSON with compare.py -- mirrors E1/E3 but on CUDA instead of CPU.

What's being probed (see README "GPU実験" section for the article framing):
  a. torch.manual_seed(42) -> CUDA Philox rand/randn stream
  b. float32 512x512 matmul via cuBLAS
  c. small Conv2d forward+backward via cuDNN
  d. small MLP training, 50 steps, fixed seed for data+init
  e. atomicAdd-backed scatter ops (index_add_ / scatter_add_) with heavy
     index collisions (1e6 random indices into 16 output slots), repeated
     20x *within the same process* so we can directly see whether the
     result hash varies run-to-run without even restarting the container
  f. one documented-nondeterministic op each: embedding_bag(mode='sum')
     and cumsum, both with CUDA float32 tensors

Every measurement is wrapped by `_common.run_with_fallback`, which retries
at smaller tensor sizes if a CUDA RuntimeError (most likely OOM, since this
GPU is shared with a resident vLLM server that holds ~23/24.5GB) is raised,
and records the raw error message if even the smallest size fails.

Every measurement function also takes an optional `npz_store` dict; when
given, it stashes its raw float32 output array(s) into that dict (only for
the *first* scale actually attempted, i.e. the intended full-size run) so
that e8e_ulp_diff.py can compute max-abs-diff / max-rel-diff / max-ULP-diff
between repeated container runs for whichever items compare.py flags as
mismatching, without having to re-run anything.
"""
import hashlib

import numpy as np

from _common import emit, run_with_fallback, sha256_of_array

SEED = 42


def philox_streams(n, npz_store=None, full_n=1000):
    import torch

    torch.manual_seed(SEED)
    r = torch.rand(n, device="cuda", dtype=torch.float32)
    rn = torch.randn(n, device="cuda", dtype=torch.float32)
    r_np = r.cpu().numpy()
    rn_np = rn.cpu().numpy()
    if npz_store is not None and n == full_n:
        npz_store["philox_rand"] = r_np
        npz_store["philox_randn"] = rn_np
    return {
        "n": n,
        "rand_sha256": sha256_of_array(r_np, dtype=np.float32),
        "randn_sha256": sha256_of_array(rn_np, dtype=np.float32),
    }


def matmul(size, npz_store=None, full_size=512):
    import torch

    torch.manual_seed(SEED)
    a = torch.randn(size, size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, size, device="cuda", dtype=torch.float32)
    c = (a @ b).cpu().numpy()
    if npz_store is not None and size == full_size:
        npz_store["matmul_512"] = c.astype(np.float32)
    return {
        "shape": [size, size],
        "matmul_sha256": sha256_of_array(c, dtype=np.float32),
    }


def conv_forward_backward(batch, npz_store=None, full_batch=16):
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1).cuda()
    x = torch.randn(batch, 3, 32, 32, device="cuda", requires_grad=True)
    y = conv(x)
    loss = y.sum()
    loss.backward()
    y_np = y.detach().cpu().numpy()
    gi_np = x.grad.detach().cpu().numpy()
    gw_np = conv.weight.grad.detach().cpu().numpy()
    if npz_store is not None and batch == full_batch:
        npz_store["conv_output"] = y_np.astype(np.float32)
        npz_store["conv_grad_input"] = gi_np.astype(np.float32)
        npz_store["conv_grad_weight"] = gw_np.astype(np.float32)
    return {
        "batch": batch,
        "cudnn_benchmark_effective": torch.backends.cudnn.benchmark,
        "output_sha256": sha256_of_array(y_np, dtype=np.float32),
        "grad_input_sha256": sha256_of_array(gi_np, dtype=np.float32),
        "grad_weight_sha256": sha256_of_array(gw_np, dtype=np.float32),
    }


def mlp_train(n_samples, npz_store=None, full_n=200):
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n_samples, 10)).astype(np.float32)
    true_w = rng.standard_normal(10).astype(np.float32)
    y = (X @ true_w + 0.1 * rng.standard_normal(n_samples).astype(np.float32)).reshape(-1, 1)
    X_t = torch.from_numpy(X).cuda()
    y_t = torch.from_numpy(y).cuda()

    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1)).cuda()
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

    param_arrays = [p.detach().cpu().numpy().astype(np.float32) for p in model.parameters()]
    param_bytes = b"".join(p.tobytes() for p in param_arrays)
    if npz_store is not None and n_samples == full_n:
        npz_store["mlp_loss_curve"] = np.array(losses, dtype=np.float64)
        npz_store["mlp_params"] = np.concatenate([p.ravel() for p in param_arrays])
    return {
        "n_samples": n_samples,
        "final_loss_hex": float(losses[-1]).hex(),
        "loss_curve_sha256": sha256_of_array(np.array(losses, dtype=np.float64), dtype=np.float64),
        "final_params_sha256": hashlib.sha256(param_bytes).hexdigest(),
    }


def atomic_add_repeat(n_indices, out_size=16, repeats=20, npz_store=None, full_n=1_000_000):
    """The core atomicAdd probe: 1e6 (or fewer, on fallback) random indices
    all colliding into a 16-slot output, via index_add_ and scatter_add_.
    Both are on the docs' nondeterministic-op list because CUDA's
    implementation resolves the collisions with atomicAdd, whose add order
    depends on scheduling. We repeat the *identical* op 20x in the same
    process (same idx/vals tensors, freshly zeroed output each time) so we
    can directly count how many distinct result hashes show up -- this is
    the most direct in-process observation of the non-determinism the
    article's chapter 7 describes only in theory.
    """
    import torch

    torch.manual_seed(SEED)
    idx = torch.randint(0, out_size, (n_indices,), device="cuda")
    vals = torch.randn(n_indices, device="cuda", dtype=torch.float32)

    index_add_hashes = []
    index_add_last_raw = None
    for _ in range(repeats):
        out = torch.zeros(out_size, device="cuda", dtype=torch.float32)
        out.index_add_(0, idx, vals)
        out_np = out.cpu().numpy()
        index_add_last_raw = out_np
        index_add_hashes.append(sha256_of_array(out_np, dtype=np.float32))

    scatter_add_hashes = []
    scatter_add_last_raw = None
    for _ in range(repeats):
        out2 = torch.zeros(out_size, device="cuda", dtype=torch.float32)
        out2.scatter_add_(0, idx, vals)
        out2_np = out2.cpu().numpy()
        scatter_add_last_raw = out2_np
        scatter_add_hashes.append(sha256_of_array(out2_np, dtype=np.float32))

    if npz_store is not None and n_indices == full_n:
        npz_store["atomic_index_add_last"] = index_add_last_raw.astype(np.float32)
        npz_store["atomic_scatter_add_last"] = scatter_add_last_raw.astype(np.float32)

    return {
        "n_indices": n_indices,
        "out_size": out_size,
        "repeats": repeats,
        "index_add_hashes": index_add_hashes,
        "index_add_unique_count": len(set(index_add_hashes)),
        "scatter_add_hashes": scatter_add_hashes,
        "scatter_add_unique_count": len(set(scatter_add_hashes)),
    }


def embedding_bag_demo(n_indices, num_embeddings=16, embedding_dim=8, npz_store=None, full_n=1_000_000):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(SEED)
    weight = torch.randn(num_embeddings, embedding_dim, device="cuda", dtype=torch.float32)
    idx = torch.randint(0, num_embeddings, (n_indices,), device="cuda")
    offsets = torch.tensor([0], device="cuda", dtype=torch.long)
    out = F.embedding_bag(idx, weight, offsets, mode="sum")
    out_np = out.cpu().numpy()
    if npz_store is not None and n_indices == full_n:
        npz_store["embedding_bag_sum"] = out_np.astype(np.float32)
    return {
        "n_indices": n_indices,
        "embedding_bag_sum_sha256": sha256_of_array(out_np, dtype=np.float32),
    }


def cumsum_demo(n, npz_store=None, full_n=1_000_000):
    import torch

    torch.manual_seed(SEED)
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.cumsum(x, dim=0)
    out_np = out.cpu().numpy()
    if npz_store is not None and n == full_n:
        npz_store["cumsum"] = out_np.astype(np.float32)
    return {
        "n": n,
        "cumsum_sha256": sha256_of_array(out_np, dtype=np.float32),
    }


def main():
    import sys

    outdir = sys.argv[1] if len(sys.argv) > 1 else None
    npz_store: dict = {}

    result = {
        "experiment": "E8a_gpu_repeat_default",
        "seed": SEED,
        "note": (
            "Default (non-deterministic) PyTorch CUDA settings: "
            "use_deterministic_algorithms not called, CUBLAS_WORKSPACE_CONFIG "
            "not set, cudnn.benchmark left at its library default."
        ),
        "philox_streams": run_with_fallback(
            lambda n: philox_streams(n, npz_store), [1000, 100, 10, 1]
        ),
        "matmul_512": run_with_fallback(lambda s: matmul(s, npz_store), [512, 128, 32, 8]),
        "conv_forward_backward": run_with_fallback(
            lambda b: conv_forward_backward(b, npz_store), [16, 4, 1]
        ),
        "mlp_train": run_with_fallback(lambda n: mlp_train(n, npz_store), [200, 50, 10]),
        "atomic_add_repeat": run_with_fallback(
            lambda n: atomic_add_repeat(n, npz_store=npz_store),
            [1_000_000, 100_000, 10_000, 1_000],
        ),
        "embedding_bag": run_with_fallback(
            lambda n: embedding_bag_demo(n, npz_store=npz_store), [1_000_000, 10_000, 100]
        ),
        "cumsum": run_with_fallback(
            lambda n: cumsum_demo(n, npz_store=npz_store), [1_000_000, 10_000, 100]
        ),
    }
    if outdir:
        np.savez(f"{outdir}/raw_arrays.npz", **npz_store)
    emit(result)


if __name__ == "__main__":
    main()
