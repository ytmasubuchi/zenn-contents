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
import sys
from typing import Any


def _safe(name: str, fn):
    """呼び出して失敗したら None を返す。環境採取はどの項目も失敗して
    他の項目を道連れにしてはならないため、個々のフィールドをこれで包む。

    ただし例外を無言で握りつぶすと、「この環境には本当にその項目が無い」
    場合と「コードのバグで例外を飲んだ」場合が、どちらも同じ `None` に
    なり区別できなくなる。そこで例外を再送出はしない(返り値の契約は
    `None` のまま変えない)が、フィールド名と例外(repr)を stderr に
    出力し、後から原因を追跡できるようにする。
    """
    try:
        return fn()
    except Exception as exc:
        print(f"envinfo: field {name!r} failed: {exc!r}", file=sys.stderr)
        return None


def _host() -> str | None:
    """計測ホストの識別子。`REPRO_HOST` が設定されていれば優先する
    (`image`/`config_label` が `REPRO_IMAGE`/`REPRO_CONFIG` を読むのと
    同じ優先順位)。コンテナ内では `platform.node()` がコンテナ自身の
    生成ホスト名を返し、4台の実ホスト名(MacBook/anakin/blade/
    dgxspark)を特定できないため、`REPRO_HOST` 未設定時のみフォール
    バックとして使う。どちらを使ったかは `_host_source` が返す。
    """
    repro_host = os.environ.get("REPRO_HOST")
    if repro_host:
        return repro_host
    return platform.node() or None


def _host_source() -> str | None:
    """`_host` がどちらの手段で値を得たかを記録する("REPRO_HOST" /
    "platform.node")。`_libc` が検出方法を `method` に記録するのと同じ
    理由で、host は `image`/`config_label` と同じくプレーンな文字列の
    ままにし、出所は隣接する `host_source` キーに記録する。
    """
    if os.environ.get("REPRO_HOST"):
        return "REPRO_HOST"
    if platform.node():
        return "platform.node"
    return None


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
        "version": _safe("torch.version", lambda: torch.__version__),
        "cuda_version": _safe("torch.cuda_version", lambda: torch.version.cuda),
        "cudnn_version": _safe("torch.cudnn_version", lambda: torch.backends.cudnn.version()),
        "matmul_allow_tf32": _safe(
            "torch.matmul_allow_tf32", lambda: torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": _safe(
            "torch.cudnn_allow_tf32", lambda: torch.backends.cudnn.allow_tf32
        ),
    }


def _cuda_info() -> dict | None:
    torch = _import_torch()
    if torch is None:
        return None

    return {
        "available": _safe("cuda.available", lambda: bool(torch.cuda.is_available())),
        "device_count": _safe("cuda.device_count", lambda: torch.cuda.device_count()),
    }


def _gpus() -> list:
    torch = _import_torch()
    if torch is None:
        return []
    if not _safe("gpus.cuda_available_check", lambda: torch.cuda.is_available()):
        return []

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append(
            {
                "name": _safe(f"gpus[{i}].name", lambda: props.name),
                "compute_capability": _safe(
                    f"gpus[{i}].compute_capability", lambda: f"{props.major}.{props.minor}"
                ),
                "multi_processor_count": _safe(
                    f"gpus[{i}].multi_processor_count", lambda: props.multi_processor_count
                ),
                "total_memory": _safe(f"gpus[{i}].total_memory", lambda: props.total_memory),
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
        "host": _safe("host", _host),
        "host_source": _safe("host_source", _host_source),
        "arch": _safe("arch", _arch),
        "libc": _safe("libc", _libc),
        "kernel": _safe("kernel", _kernel),
        "python": _safe("python", _python),
        "packages": _safe("packages", _packages) or {},
        "numpy_cpu_features": _safe("numpy_cpu_features", _numpy_cpu_features),
        "blas": _safe("blas", _blas),
        "thread_env": _safe("thread_env", lambda: _env_subset(_THREAD_ENV_VARS)) or {},
        "torch": _safe("torch", _torch_info),
        "cuda": _safe("cuda", _cuda_info),
        "gpus": _safe("gpus", _gpus) or [],
        "image": os.environ.get("REPRO_IMAGE"),
        "config_label": os.environ.get("REPRO_CONFIG"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "control_env": _safe("control_env", lambda: _env_subset(_CONTROL_ENV_VARS)) or {},
    }
