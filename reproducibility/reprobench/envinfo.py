"""実行環境メタデータの採取。

`collect_env()` は、プローブ(別タスク)が計算結果を出したときの「環境が
何であったか」を1つの辞書に記録する。比較器(別タスク)が「この2環境は
結果が食い違った」と報告したとき、その理由を説明できる唯一の手がかりが
この出力であるため、取得できる値は可能な限り取り、取得できない項目でも
キー自体は必ず存在させ値を `None` にする。

このモジュールの責務は「実行環境が何であるかを集めて返す」ことのみ。
"""
import glob
import importlib.metadata
import os
import platform
from typing import Any


def _safe(fn):
    """呼び出して失敗したら None を返す。環境採取はどの項目も失敗して
    他の項目を道連れにしてはならないため、個々のフィールドをこれで包む。
    """
    try:
        return fn()
    except Exception:
        return None


def _host() -> str | None:
    return platform.node() or None


def _arch() -> str | None:
    return platform.machine() or None


def _kernel() -> str | None:
    return platform.release() or None


def _python() -> str | None:
    return platform.python_version()


def _libc() -> dict | None:
    """glibc/musl を区別する。

    `platform.libc_ver()` は musl 環境では `('', '')` を返すため、その
    場合は `/lib/ld-musl-*.so.1` の有無で musl ローダーを検出する
    フォールバックを使う。どちらの検出方法で判定できたかを `method` に
    記録する。
    """
    name, version = platform.libc_ver()
    if name:
        return {"name": name, "version": version, "method": "platform.libc_ver"}
    if glob.glob("/lib/ld-musl-*.so.1") or glob.glob("/lib64/ld-musl-*.so.1"):
        return {"name": "musl", "version": None, "method": "musl_loader_glob"}
    return None


def _numpy_cpu_features() -> dict | None:
    """NumPyが検出したCPU機能・ベースライン・ディスパッチ候補を取得する。

    NumPy 2系では `numpy.core._multiarray_umath` が
    `numpy._core._multiarray_umath` に移動したため、両方を試す。新しい方を
    先に試す(古い方は非推奨警告(DeprecationWarning)を発し、`-W error`
    下では例外になるため、`ImportError` に限らず広く捕捉して次を試す)。
    """
    for modname in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        return {
            "module": modname,
            "features": dict(getattr(mod, "__cpu_features__", {})),
            "baseline": list(getattr(mod, "__cpu_baseline__", [])),
            "dispatch": list(getattr(mod, "__cpu_dispatch__", [])),
        }
    return None


def _blas() -> Any:
    """BLAS/LAPACK構成情報。`numpy.__config__.show(mode="dicts")` を優先し、
    無ければ `np.show_config()` の文字列表現にフォールバックする。
    """
    import numpy as np

    show = getattr(np.__config__, "show", None)
    if show is not None:
        try:
            return show(mode="dicts")
        except TypeError:
            pass
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        np.show_config()
    return buf.getvalue()


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_CONTROL_ENV_VARS = (
    "NPY_DISABLE_CPU_FEATURES",
    "NPY_ENABLE_CPU_FEATURES",
    "MKL_CBWR",
    "CUBLAS_WORKSPACE_CONFIG",
)


def _env_subset(names: tuple) -> dict:
    return {name: os.environ.get(name) for name in names}


def _import_torch():
    """torchをimportして返す。未インストール(ModuleNotFoundError)に加え、
    テストが `sys.modules["torch"] = None` で再現する「import済みだが壊れて
    いる」状態(素のImportError)も区別せずNoneとして扱う。
    """
    try:
        import torch
    except ImportError:
        return None
    return torch


def _torch_info() -> dict | None:
    torch = _import_torch()
    if torch is None:
        return None

    return {
        "version": _safe(lambda: torch.__version__),
        "cuda_version": _safe(lambda: torch.version.cuda),
        "cudnn_version": _safe(lambda: torch.backends.cudnn.version()),
        "matmul_allow_tf32": _safe(lambda: torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": _safe(lambda: torch.backends.cudnn.allow_tf32),
    }


def _cuda_info() -> dict | None:
    torch = _import_torch()
    if torch is None:
        return None

    return {
        "available": _safe(lambda: bool(torch.cuda.is_available())),
        "device_count": _safe(lambda: torch.cuda.device_count()),
    }


def _gpus() -> list:
    torch = _import_torch()
    if torch is None or not _safe(lambda: torch.cuda.is_available()):
        return []

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append(
            {
                "name": _safe(lambda: props.name),
                "compute_capability": _safe(lambda: f"{props.major}.{props.minor}"),
                "multi_processor_count": _safe(lambda: props.multi_processor_count),
                "total_memory": _safe(lambda: props.total_memory),
            }
        )
    return gpus


_PACKAGE_NAMES = ("numpy", "scipy", "scikit-learn", "lightgbm", "torch", "transformers")


def _packages() -> dict:
    """distribution名→バージョン文字列(無ければNone)の辞書。
    未インストール(`PackageNotFoundError`)はもちろん、1パッケージぶんの
    メタデータ破損など想定外の失敗も、他のパッケージの結果を道連れに
    しないよう1件ずつ捕捉する。
    """
    out = {}
    for name in _PACKAGE_NAMES:
        try:
            out[name] = importlib.metadata.version(name)
        except Exception:
            out[name] = None
    return out


def collect_env() -> dict:
    """現在の実行環境のメタデータを1つのJSONシリアライズ可能な辞書で返す。

    取得できなかった項目もキー自体は必ず存在し、値は `None` になる。
    どのフィールドで失敗しても例外は送出しない。
    """
    return {
        "host": _safe(_host),
        "arch": _safe(_arch),
        "libc": _safe(_libc),
        "kernel": _safe(_kernel),
        "python": _safe(_python),
        "packages": _safe(_packages) or {},
        "numpy_cpu_features": _safe(_numpy_cpu_features),
        "blas": _safe(_blas),
        "thread_env": _safe(lambda: _env_subset(_THREAD_ENV_VARS)) or {},
        "torch": _safe(_torch_info),
        "cuda": _safe(_cuda_info),
        "gpus": _safe(_gpus) or [],
        "image": os.environ.get("REPRO_IMAGE"),
        "config_label": os.environ.get("REPRO_CONFIG"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "control_env": _safe(lambda: _env_subset(_CONTROL_ENV_VARS)) or {},
    }
