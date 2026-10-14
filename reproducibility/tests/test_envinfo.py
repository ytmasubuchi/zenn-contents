"""reprobench.envinfo のテスト: 実行環境メタデータの採取を検証する。"""
import json
import sys

from reprobench.envinfo import collect_env


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
