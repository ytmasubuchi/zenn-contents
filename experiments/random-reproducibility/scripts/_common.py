"""Shared helpers for the random-reproducibility experiments.

Design rule (per experiment spec): never compare printed/rounded floats.
Always dump the exact float64 (or int) bit pattern to bytes, hex-encode,
and SHA256 the result. This is the only way to detect a single differing
ULP between two runs/platforms/library versions.
"""
import hashlib
import json
import os
import platform
import socket
import sys
from typing import Any, Iterable


def sha256_of_array(arr, dtype=None) -> str:
    """Exact-bit hash of a numpy array (or array-like) via its raw bytes."""
    import numpy as np

    a = np.asarray(arr)
    if dtype is not None:
        a = a.astype(dtype)
    return hashlib.sha256(a.tobytes()).hexdigest()


def hexdump_of_array(arr, dtype=None, limit: int = 8) -> str:
    """First `limit` elements as raw hex bytes, for human-readable spot checks."""
    import numpy as np

    a = np.asarray(arr)
    if dtype is not None:
        a = a.astype(dtype)
    return a[:limit].tobytes().hex()


def sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_of_str_list(items: Iterable[str]) -> str:
    joined = "\x1f".join(items).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def try_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not-installed"


def get_gpu_metadata() -> dict:
    """Best-effort GPU/CUDA metadata for the E8 (GPU) experiments.

    Deliberately defensive: must never raise, even when torch isn't
    installed, when there is no GPU, or when the GPU exists but a CUDA
    context can't actually be created (e.g. another process such as vLLM
    has claimed almost all VRAM). In that last case ``cuda_available`` can
    still be True (it only reflects driver/device visibility) while every
    real allocation later fails with a CUDA OOM RuntimeError -- that's a
    real, reproducible finding of E8, not a bug in this helper.
    """
    info: dict = {"cuda_available": False}
    try:
        import torch
    except Exception:
        return info

    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as e:
        info["cuda_available_error"] = str(e)
        return info

    info["cuda_runtime_version_torch"] = getattr(torch.version, "cuda", None)
    try:
        info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception as e:
        info["cudnn_version_error"] = str(e)
    try:
        info["tf32_matmul_default"] = torch.backends.cuda.matmul.allow_tf32
        info["tf32_cudnn_default"] = torch.backends.cudnn.allow_tf32
        info["cudnn_benchmark_default"] = torch.backends.cudnn.benchmark
        info["cudnn_deterministic_default"] = torch.backends.cudnn.deterministic
    except Exception as e:
        info["backend_flags_error"] = str(e)

    if not info["cuda_available"]:
        return info

    try:
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as e:
        info["device_query_error"] = str(e)

    # torch.cuda.mem_get_info() needs a live CUDA context, which itself can
    # fail with OOM when VRAM is nearly exhausted by another process. Shell
    # out to nvidia-smi instead, which reports driver-level memory/version
    # info without creating a CUDA context in this process.
    try:
        import subprocess

        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.used,memory.free,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        info["nvidia_smi_query"] = out.stdout.strip()
    except Exception as e:
        info["nvidia_smi_error"] = str(e)

    return info


def get_metadata() -> dict:
    return {
        "hostname": socket.gethostname(),
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": try_version("numpy"),
        "scipy_version": try_version("scipy"),
        "sklearn_version": try_version("sklearn"),
        "torch_version": try_version("torch"),
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "in_container": os.path.exists("/.dockerenv"),
        "gpu": get_gpu_metadata(),
    }


def run_with_fallback(fn, scales: Iterable) -> dict:
    """Call ``fn(scale)`` for each scale in `scales` (preferred/largest
    first), catching CUDA RuntimeErrors (OOM, or "no deterministic
    implementation") and falling back to the next (usually smaller) scale.

    `fn(scale)` must return a dict describing the successful measurement.
    On success, returns that dict merged with
    ``{"status": "ok", "scale_used": scale, "scales_tried": [...]}``.
    If every scale fails, returns a dict describing the *last* failure with
    ``status`` set to one of ``"error_cuda_oom"``,
    ``"error_no_deterministic_impl"``, or ``"error_other"`` plus
    ``error_type``/``error_message``/``scales_tried``.
    """
    tried = []
    last_exc: Exception | None = None
    for scale in scales:
        tried.append(scale)
        try:
            result = fn(scale)
            out = {"status": "ok", "scale_used": scale, "scales_tried": tried}
            out.update(result)
            return out
        except RuntimeError as e:
            last_exc = e
            continue
    msg = str(last_exc) if last_exc else ""
    low = msg.lower()
    if "out of memory" in low:
        status = "error_cuda_oom"
    elif "deterministic" in low:
        status = "error_no_deterministic_impl"
    else:
        status = "error_other"
    return {
        "status": status,
        "scales_tried": tried,
        "error_type": type(last_exc).__name__ if last_exc else None,
        "error_message": msg,
    }


def emit(result: dict) -> None:
    """Write the final JSON blob (metadata + payload) to stdout only."""
    out = {"metadata": get_metadata(), **result}
    sys.stdout.write(json.dumps(out, indent=2, sort_keys=False))
    sys.stdout.write("\n")


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)
