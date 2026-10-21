"""reprobench.probes.s14_dataloader_seed のテスト。

`duplicate_sample_count`は外部依存なしの純粋関数なので、そのままテスト
できる(brief Step1相当)。`run_probe`はtorchが無い環境では常に
`SkippedProbe`を返すため、該当テストは`pytest.importorskip`でライブラリ
不在時にスキップへ丸める(test_torch_probes.pyと同じ流儀)。

**`numpy`版が全worker数で重複するかどうかはテストで固定しない。** それは
実測で確かめる対象であり、事前登録された予測の1つに過ぎない。テストが
検証してよいのは「各組み合わせについて記録(digestと、
duplicate_sample_countまたはworker_start_failedのいずれか)が存在する
こと」までである(brief Step1の例の通り)。
"""
import pytest

import reprobench.probes  # モジュールimportの副作用でS14プローブ群を登録する
from reprobench.registry import ProbeResult, SkippedProbe, run_probe
from reprobench.probes.s14_dataloader_seed import (
    RNG_SOURCES,
    WORKER_COUNTS,
    duplicate_sample_count,
)


# --- 定数が仕様書の値と一致していること ------------------------------------


def test_rng_sources_matches_brief():
    assert RNG_SOURCES == ("torch", "numpy")


def test_worker_counts_matches_brief():
    assert WORKER_COUNTS == (0, 2, 4)


# --- duplicate_sample_count()の純粋関数としての性質 -------------------------


def test_duplicate_sample_count_is_zero_for_all_unique_values():
    assert duplicate_sample_count([0.1, 0.2, 0.3, 0.4]) == 0


def test_duplicate_sample_count_counts_excess_occurrences():
    # 0.5が3回、0.1が2回出現する: ユニーク値3種に対し総数6個、差分は3。
    assert duplicate_sample_count([0.5, 0.5, 0.5, 0.1, 0.1, 0.9]) == 3


def test_duplicate_sample_count_handles_empty_list():
    assert duplicate_sample_count([]) == 0


# --- プローブ本体(brief Step1相当) ------------------------------------------


def test_probe_records_each_rng_source_and_worker_count():
    r = run_probe("S14_dataloader_worker_seed")
    if isinstance(r, SkippedProbe):
        assert "missing requirement" in r.reason
        return
    assert isinstance(r, ProbeResult)
    for src in RNG_SOURCES:
        for n in WORKER_COUNTS:
            cell = r.meta["results"][src][str(n)]
            assert "digest" in cell
            assert "duplicate_sample_count" in cell or "worker_start_failed" in cell


def test_probe_meta_keys_are_json_safe_strings():
    # worker数はJSONの整数キー非対応のため文字列化されている必要がある
    # (brief明記)。トップレベルのrng_sourceキーも含め、すべて`str`である
    # ことを確認する。
    pytest.importorskip("torch")
    r = run_probe("S14_dataloader_worker_seed")
    assert isinstance(r, ProbeResult)
    for src, by_workers in r.meta["results"].items():
        assert isinstance(src, str)
        for n_key in by_workers:
            assert isinstance(n_key, str)


def test_probe_is_stable_across_two_in_process_calls():
    pytest.importorskip("torch")
    first = run_probe("S14_dataloader_worker_seed")
    second = run_probe("S14_dataloader_worker_seed")
    assert isinstance(first, ProbeResult) and isinstance(second, ProbeResult)
    assert first.digest == second.digest
