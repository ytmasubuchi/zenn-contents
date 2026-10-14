"""reprobench.envinfo のテスト: 実行環境メタデータの採取を検証する。"""
import json
import platform
import sys
import types

from reprobench.envinfo import collect_env


def _fake_torch(n_gpus: int = 1, cuda_available: bool = True):
    """`_torch_info`/`_cuda_info`/`_gpus` が参照する属性パスだけを備えた
    偽torchモジュール。CUDAを模擬するのが目的ではなく、実装が読みに
    行く属性パス(キー名のタイプミスを含む)をピン止めするのが目的。
    """
    device_props = types.SimpleNamespace(
        name="Fake GPU",
        major=8,
        minor=9,
        multi_processor_count=42,
        total_memory=8_000_000_000,
    )
    backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),
        cudnn=types.SimpleNamespace(allow_tf32=False, version=lambda: 90100),
    )
    cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        device_count=lambda: n_gpus,
        get_device_properties=lambda i: device_props,
    )
    return types.SimpleNamespace(
        __version__="2.3.1+cpu",
        version=types.SimpleNamespace(cuda="12.1"),
        backends=backends,
        cuda=cuda,
    )


# --- タスクブリーフ記載のテスト(そのまま) ---

REQUIRED = {"host","arch","libc","kernel","python","packages","numpy_cpu_features",
            "blas","thread_env","torch","cuda","gpus","image","config_label","pythonhashseed"}


def test_collect_env_has_all_required_keys():
    assert REQUIRED <= set(collect_env())


def test_collect_env_is_json_serializable():
    json.dumps(collect_env())


def test_collect_env_records_pythonhashseed_from_environment(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    assert collect_env()["pythonhashseed"] == "0"


def test_collect_env_survives_missing_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert collect_env()["torch"] is None


# --- 曖昧さの解決事項に対する追加テスト ---


def test_collect_env_control_env_key_present():
    # control_env はブリーフの必須キーに含まれないが、解決事項5で追加される。
    # REQUIRED <= set(...) は余分なキーがあっても壊れないことの確認でもある。
    env = collect_env()
    assert "control_env" in env
    assert REQUIRED <= set(env)


def test_collect_env_control_env_tracks_declared_variables(monkeypatch):
    monkeypatch.setenv("NPY_DISABLE_CPU_FEATURES", "AVX512F")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env = collect_env()
    assert env["control_env"]["NPY_DISABLE_CPU_FEATURES"] == "AVX512F"
    assert env["control_env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert env["control_env"]["NPY_ENABLE_CPU_FEATURES"] is None
    assert env["control_env"]["MKL_CBWR"] is None


def test_collect_env_numpy_cpu_features_has_baseline_and_dispatch():
    features = collect_env()["numpy_cpu_features"]
    assert features is not None
    assert {"features", "baseline", "dispatch"} <= set(features)


def test_collect_env_thread_env_has_all_declared_vars():
    thread_env = collect_env()["thread_env"]
    assert {
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    } <= set(thread_env)


def test_collect_env_image_and_config_label_from_environment(monkeypatch):
    monkeypatch.setenv("REPRO_IMAGE", "debian-py311")
    monkeypatch.setenv("REPRO_CONFIG", "threads1")
    env = collect_env()
    assert env["image"] == "debian-py311"
    assert env["config_label"] == "threads1"


def test_collect_env_image_and_config_label_default_to_none(monkeypatch):
    monkeypatch.delenv("REPRO_IMAGE", raising=False)
    monkeypatch.delenv("REPRO_CONFIG", raising=False)
    env = collect_env()
    assert env["image"] is None
    assert env["config_label"] is None


def test_collect_env_gpus_is_a_list():
    # CUDA非搭載環境(CI含む)では空リストになる。
    assert isinstance(collect_env()["gpus"], list)


def test_collect_env_packages_reports_known_distribution_versions():
    packages = collect_env()["packages"]
    assert packages["numpy"] is not None
    for name in ("scipy", "scikit-learn", "lightgbm", "torch", "transformers"):
        assert name in packages


def test_collect_env_survives_torch_module_not_found_error():
    # このテストコンテナにはtorchをインストールしていないため、import torch は
    # 実際に(monkeypatchなしで)ModuleNotFoundErrorになる。brief記載の
    # monkeypatchケース(sys.modules["torch"]=None、素のImportError)とは
    # 異なる自然な欠如ケースをカバーする。
    env = collect_env()
    assert env["torch"] is None


# --- レビュー fix round 1: torch/cuda/gpus の辞書形状をピン止めするテスト ---


def test_collect_env_torch_dict_has_expected_keys(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    torch_info = collect_env()["torch"]
    assert set(torch_info) == {
        "version",
        "cuda_version",
        "cudnn_version",
        "matmul_allow_tf32",
        "cudnn_allow_tf32",
    }
    assert torch_info["version"] == "2.3.1+cpu"
    assert torch_info["cuda_version"] == "12.1"
    assert torch_info["cudnn_version"] == 90100
    assert torch_info["matmul_allow_tf32"] is True
    assert torch_info["cudnn_allow_tf32"] is False


def test_collect_env_cuda_dict_has_expected_keys(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(n_gpus=2))
    cuda_info = collect_env()["cuda"]
    assert set(cuda_info) == {"available", "device_count"}
    assert cuda_info["available"] is True
    assert cuda_info["device_count"] == 2


def test_collect_env_gpus_dicts_have_expected_keys(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(n_gpus=2))
    gpus = collect_env()["gpus"]
    assert len(gpus) == 2
    for gpu in gpus:
        assert set(gpu) == {
            "name",
            "compute_capability",
            "multi_processor_count",
            "total_memory",
        }
    assert gpus[0]["name"] == "Fake GPU"
    assert gpus[0]["compute_capability"] == "8.9"
    assert gpus[0]["multi_processor_count"] == 42
    assert gpus[0]["total_memory"] == 8_000_000_000


def test_collect_env_gpus_empty_when_cuda_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False))
    assert collect_env()["gpus"] == []


# --- レビュー fix round 1: host の REPRO_HOST 優先順位テスト ---


def test_collect_env_host_prefers_repro_host_env_var(monkeypatch):
    monkeypatch.setenv("REPRO_HOST", "anakin")
    env = collect_env()
    assert env["host"] == "anakin"
    assert env["host_source"] == "REPRO_HOST"


def test_collect_env_host_falls_back_to_platform_node_when_unset(monkeypatch):
    monkeypatch.delenv("REPRO_HOST", raising=False)
    monkeypatch.setattr(platform, "node", lambda: "container-abc123")
    env = collect_env()
    assert env["host"] == "container-abc123"
    assert env["host_source"] == "platform.node"
