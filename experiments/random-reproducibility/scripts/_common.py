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
        },
        "in_container": os.path.exists("/.dockerenv"),
    }


def emit(result: dict) -> None:
    """Write the final JSON blob (metadata + payload) to stdout only."""
    out = {"metadata": get_metadata(), **result}
    sys.stdout.write(json.dumps(out, indent=2, sort_keys=False))
    sys.stdout.write("\n")


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)
