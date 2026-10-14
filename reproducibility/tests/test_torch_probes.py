"""reprobench.probes.{s7_torch_cpu,s8_torch_gpu} のテスト:
PyTorch CPU/GPUの非決定性を検証する。

torchはこのテストコンテナ(python:3.11-slim + numpy + pytest)には
インストールされていないため、`run_probe`は常に`SkippedProbe`を返す。
digestへ直接アクセスするテストはそのままでは`AttributeError`になって
しまうため、該当テストは`pytest.importorskip`でライブラリ不在時に
スキップへ丸める(標準スイートを壊さないため。検証コンテナ
(torch CPU版をpip installしたもの)ではそのまま実行される)。

`test_gpu_probes_skip_without_cuda`は`check_requirement`そのものを
monkeypatchで差し替えるため、実機にtorchが無くても(requiresの"torch"判定
より前に"cuda"判定でスキップするため)問題なく動く。
"""
import pytest

import reprobench.probes  # モジュールimportの副作用でS7/S8プローブ群を登録する
from reprobench import registry
from reprobench.registry import ProbeResult, SkippedProbe, run_probe


# --- タスクブリーフ記載のテスト(そのまま、ただしtorch不在環境用にimportorskipを追加) ---


def test_torch_cpu_probe_deterministic_within_process():
    pytest.importorskip("torch")
    assert run_probe("S7_mlp_train10").digest == run_probe("S7_mlp_train10").digest


def test_gpu_probes_skip_without_cuda(monkeypatch):
    monkeypatch.setattr(registry, "check_requirement", lambda r: r != "cuda")
    for name in ("S8_mlp_train10_cuda", "S8_matmul_tf32_on"):
        r = run_probe(name)
        assert isinstance(r, SkippedProbe) and "cuda" in r.reason


def test_curand_sizes_probe_records_per_size_digests():
    r = run_probe("S8_curand_randn_sizes")
    if isinstance(r, ProbeResult):
        assert set(r.meta["digests"]) == {1024, 65536, 4194304, 67108864}


def test_tf32_probes_record_their_flag_state():
    # CUDA が無い環境では skipped になるので、そのときはこのテストを skip
    r = run_probe("S8_matmul_tf32_on")
    if isinstance(r, ProbeResult):
        assert r.meta["tf32"] is True


# --- 曖昧さの解決事項に対する追加テスト ---


def test_s7_probes_record_thread_count_and_version():
    pytest.importorskip("torch")
    r = run_probe("S7_mlp_forward")
    assert "threads" in r.meta
    assert "torch_version" in r.meta


def test_s8_probes_record_both_tf32_flags_and_sm_count():
    # SM数解釈に要るため(brief記載)。CUDA無し環境ではSkippedProbeになり
    # 検証不能なので、その場合はスキップ扱いにする。
    for name in ("S8_mlp_train10_cuda", "S8_matmul_tf32_on", "S8_matmul_tf32_off",
                 "S8_mlp_train10_cuda_bf16", "S8_curand_randn_sizes"):
        r = run_probe(name)
        if isinstance(r, ProbeResult):
            assert "matmul_allow_tf32" in r.meta
            assert "cudnn_allow_tf32" in r.meta
            assert "multi_processor_count" in r.meta


def test_s8_deterministic_probe_skips_without_cublas_workspace_config(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(registry, "check_requirement", lambda r: True)
    r = run_probe("S8_mlp_train10_cuda_deterministic")
    assert isinstance(r, SkippedProbe)
    assert "CUBLAS_WORKSPACE_CONFIG" in r.reason


def test_s8_tf32_probes_restore_ambient_flags_after_running():
    pytest.importorskip("torch")
    import torch
    before_matmul = torch.backends.cuda.matmul.allow_tf32
    before_cudnn = torch.backends.cudnn.allow_tf32
    run_probe("S8_matmul_tf32_on")
    run_probe("S8_matmul_tf32_off")
    assert torch.backends.cuda.matmul.allow_tf32 == before_matmul
    assert torch.backends.cudnn.allow_tf32 == before_cudnn
