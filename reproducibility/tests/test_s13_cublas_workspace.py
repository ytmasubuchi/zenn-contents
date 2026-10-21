"""reprobench.probes.s13_cublas_workspace のテスト。

`record_outcome`はtorch非依存の純粋関数なので、組み込みの例外クラスを
使ってモデルや外部依存なしでそのままテストできる(brief Step1)。
`run_probe`はCUDAの無い環境では常に`SkippedProbe`を返すため、プローブ
本体のテストはその分岐を検証する(brief Step5)。

**「どの演算が実際に例外を送出するか」はテストで固定しない。** それは
実測で確かめる対象であり、事前登録された予測の1つに過ぎない(brief
「例外が出ることが期待挙動」節)。テストが検証してよいのは「演算ごとに
digestまたはraisedのいずれかが記録されていること」までである。
"""
import os

import reprobench.probes  # モジュールimportの副作用でS13プローブ群を登録する
from reprobench import registry
from reprobench.registry import ProbeResult, SkippedProbe, all_probes, run_probe
from reprobench.probes.s13_cublas_workspace import CUBLAS_OPS, record_outcome


# --- 定数が仕様書の型と一致していること ------------------------------------


def test_cublas_ops_covers_multiple_matmul_like_operations():
    # brief: 「行列積を含む複数」の演算を測る。
    assert len(CUBLAS_OPS) >= 2
    assert len(set(CUBLAS_OPS)) == len(CUBLAS_OPS)


def test_probe_is_registered_with_torch_and_cuda_requirement():
    assert all_probes()["S13_cublas_workspace_config"].requires == ("torch", "cuda")


# --- record_outcome()の純粋関数としての性質 ---------------------------------


def test_record_outcome_records_digest_when_no_exception():
    assert record_outcome("abc123", None) == {"digest": "abc123"}


def test_record_outcome_records_exception_type_and_message_when_raised():
    cell = record_outcome(None, ValueError("boom"))
    assert cell == {"raised": {"type": "ValueError", "message": "boom"}}


def test_record_outcome_does_not_swallow_the_exception_into_a_fake_success():
    # 例外が出た場合、digestキーは一切含まれない(握りつぶして正常終了
    # したように見せない。brief Review Focus 3)。
    cell = record_outcome(None, RuntimeError("cublas workspace required"))
    assert "digest" not in cell
    assert cell["raised"]["type"] == "RuntimeError"
    assert cell["raised"]["message"] == "cublas workspace required"


# --- プローブ本体(brief Step1相当) ------------------------------------------


def test_s13_records_exception_or_result_without_swallowing():
    r = run_probe("S13_cublas_workspace_config")
    if isinstance(r, SkippedProbe):
        assert "missing requirement" in r.reason
        return
    assert isinstance(r, ProbeResult)
    assert r.meta["cublas_workspace_config"] in (None, "") or isinstance(
        r.meta["cublas_workspace_config"], str
    )
    for op in r.meta["ops"]:
        cell = r.meta["ops"][op]
        assert ("digest" in cell) or ("raised" in cell)


def test_s13_does_not_set_cublas_workspace_config_itself(monkeypatch):
    # brief: このプローブ自身はCUBLAS_WORKSPACE_CONFIGを設定しない
    # (config側が決める)。環境変数を明示的に未設定にしてから実行しても、
    # プローブが自分でそれを設定して回避しようとしていないことを確認する。
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    r = run_probe("S13_cublas_workspace_config")
    if isinstance(r, ProbeResult):
        assert r.meta["cublas_workspace_config"] is None
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None


def test_s13_skips_without_cuda_via_forced_requirement(monkeypatch):
    monkeypatch.setattr(registry, "check_requirement", lambda r: r != "cuda")
    r = run_probe("S13_cublas_workspace_config")
    assert isinstance(r, SkippedProbe) and "cuda" in r.reason
