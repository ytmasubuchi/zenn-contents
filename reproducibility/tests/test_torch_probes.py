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
    # fix round 1 finding 9: meta["digests"] is string-keyed (not int-keyed)
    # so it round-trips through JSON (results.json) the same way it reads
    # in-process -- the brief's literal snippet asserted an int-keyed set,
    # which only held before this fix.
    r = run_probe("S8_curand_randn_sizes")
    if isinstance(r, ProbeResult):
        assert set(r.meta["digests"]) == {"1024", "65536", "4194304", "67108864"}


def test_tf32_probes_record_their_flag_state():
    # CUDA が無い環境では skipped になるので、そのときはこのテストを skip
    r = run_probe("S8_matmul_tf32_on")
    if isinstance(r, ProbeResult):
        assert r.meta["tf32"] is True


# --- 曖昧さの解決事項に対する追加テスト ---


def test_s7_probes_record_the_actual_compute_dtype_not_the_digest_dtype():
    # fix round 1 finding 7: meta["dtype"] used to report the numpy array's
    # dtype *after* the float64 upcast done for digesting, which was always
    # "float64" regardless of what precision the MLP actually computed in
    # (float32, torch's and nn.Linear's default). It must report the real
    # compute dtype; the upcast dtype used for digest/encode_sample is kept
    # separately as "digest_dtype" so neither fact is lost.
    pytest.importorskip("torch")
    for name in ("S7_mlp_forward", "S7_mlp_train10"):
        r = run_probe(name)
        assert r.meta["dtype"] == "float32"
        assert r.meta["digest_dtype"] == "float64"


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


# --- fix round 1 (coordinator review) ---


def test_s8_deterministic_probe_restores_prior_flag_value_not_hardcoded_false(monkeypatch):
    # fix round 1 finding 6: the old version unconditionally set
    # use_deterministic_algorithms(False) in `finally`, silently turning it
    # off for every later probe in the process even if it had been True
    # before this probe ran. We force check_requirement to always report
    # satisfied (bypassing the registry-level "missing requirement: cuda"
    # short-circuit, which would otherwise skip the probe body entirely and
    # make this test vacuous) and set CUBLAS_WORKSPACE_CONFIG so the probe's
    # own env-var gate passes too. The probe body then reaches
    # `_train10_cuda`, which fails inside `.to("cuda")` since this container
    # has no GPU -- but that failure happens *inside* the
    # `_deterministic_algorithms` context manager, so its `finally` restore
    # still runs before the exception propagates out to run_probe (which
    # converts it to a SkippedProbe("raised: ...")). What we're checking is
    # that the global flag comes back to what it was before, not what the
    # probe returned.
    pytest.importorskip("torch")
    import torch

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(registry, "check_requirement", lambda r: True)
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        run_probe("S8_mlp_train10_cuda_deterministic")
        assert torch.are_deterministic_algorithms_enabled() is True
    finally:
        torch.use_deterministic_algorithms(before)


def test_s8_matmul_and_training_probes_generate_inputs_on_cpu_and_record_their_digest():
    # fix round 1 finding 3: inputs must come from a CPU generator (not
    # torch.randn(..., device="cuda")), so the curand GPU-model-dependence
    # that S8_curand_randn_sizes exists to measure cannot leak into the
    # matmul/training probes and confound their results. The input digests
    # must be present in meta so this separation is visible in the record,
    # not only in the source.
    for name in ("S8_mlp_train10_cuda", "S8_mlp_train10_cuda_bf16"):
        r = run_probe(name)
        if isinstance(r, ProbeResult):
            assert set(r.meta["input_digests"]) == {"x_float32_source", "target_float32_source"}
    r = run_probe("S8_matmul_tf32_on")
    if isinstance(r, ProbeResult):
        assert "input_a_digest" in r.meta
        assert "input_b_digest" in r.meta


def test_index_add_atomics_probes_are_registered_with_correct_requirements(monkeypatch):
    monkeypatch.setattr(registry, "check_requirement", lambda r: r != "cuda")
    for name in ("S8_index_add_atomics", "S8_index_add_atomics_deterministic"):
        r = run_probe(name)
        assert isinstance(r, SkippedProbe) and "cuda" in r.reason


def test_index_add_atomics_probe_self_reports_stability():
    # finding 0: this probe is expected to be unstable in-process by design
    # (atomicAdd accumulation order on CUDA is scheduling-dependent), so it
    # must self-report rather than assert raw digest equality -- it is
    # registered in test_all_probes_smoke.py's _SELF_REPORTED_STABILITY.
    r = run_probe("S8_index_add_atomics")
    if isinstance(r, ProbeResult):
        assert "stable_within_process" in r.meta
        assert r.meta["deterministic_algorithms"] is False


def test_index_add_atomics_deterministic_probe_records_the_flag():
    r = run_probe("S8_index_add_atomics_deterministic")
    if isinstance(r, ProbeResult):
        assert r.meta["deterministic_algorithms"] is True
        assert "stable_within_process" in r.meta
