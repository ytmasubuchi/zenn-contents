"""reprobench.probes.s12_cudnn のテスト。

`result_key`/`stable_within_process`はtorch非依存の純粋関数なので、
モデルや外部依存なしでそのままテストできる(brief Step1)。`run_probe`は
CUDAの無い環境では常に`SkippedProbe`を返すため、プローブ本体のテストは
その分岐を検証する(brief Step5: 「CUDAの無いbaseイメージでは
SkippedProbeを返し、skip経路の分岐が通る」)。

**`benchmark=True`側が実際にプロセス内で不安定になるかどうかはテストで
固定しない。** それは実測で確かめる対象であり、事前登録された予測の
1つに過ぎない。テストが検証してよいのは「各設定についてdigestと
stable_within_processが記録されていること」までである(brief Step1)。
"""
import reprobench.probes  # モジュールimportの副作用でS12プローブ群を登録する
from reprobench import registry
from reprobench.registry import ProbeResult, SkippedProbe, all_probes, run_probe
from reprobench.probes.s12_cudnn import (
    CUDNN_SETTINGS,
    result_key,
    stable_within_process,
)


# --- 定数が仕様書の型と一致していること ------------------------------------


def test_cudnn_settings_is_a_2x2_grid_of_benchmark_and_deterministic():
    assert len(CUDNN_SETTINGS) == 4
    assert set(CUDNN_SETTINGS) == {
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    }


def test_probe_is_registered_with_torch_and_cuda_requirement():
    assert all_probes()["S12_cudnn_benchmark_deterministic"].requires == ("torch", "cuda")


# --- result_key()/stable_within_process()の純粋関数としての性質 -------------


def test_result_key_formats_benchmark_and_deterministic():
    assert result_key(True, False) == "benchmark=True,deterministic=False"
    assert result_key(False, True) == "benchmark=False,deterministic=True"


def test_stable_within_process_true_when_all_digests_equal():
    assert stable_within_process(["a", "a", "a"]) is True


def test_stable_within_process_false_when_a_digest_differs():
    assert stable_within_process(["a", "b", "a"]) is False


def test_stable_within_process_true_for_single_digest():
    assert stable_within_process(["a"]) is True


# --- プローブ本体(brief Step1相当) ------------------------------------------


def test_s12_skips_without_cuda_and_records_within_process_stability():
    r = run_probe("S12_cudnn_benchmark_deterministic")
    if isinstance(r, SkippedProbe):
        assert "missing requirement" in r.reason
        return
    assert isinstance(r, ProbeResult)
    for bench, det in CUDNN_SETTINGS:
        cell = r.meta["results"][result_key(bench, det)]
        assert "digest" in cell
        assert cell["stable_within_process"] in (True, False)


def test_s12_skips_without_cuda_via_forced_requirement(monkeypatch):
    # S8のtest_gpu_probes_skip_without_cudaと同じ流儀: torchの有無に
    # 関わらず"cuda"判定だけを先に落とす。
    monkeypatch.setattr(registry, "check_requirement", lambda r: r != "cuda")
    r = run_probe("S12_cudnn_benchmark_deterministic")
    assert isinstance(r, SkippedProbe) and "cuda" in r.reason
