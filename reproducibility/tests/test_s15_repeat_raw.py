"""reprobench.probes.s15_repeat_raw のテスト。

`round_digests_and_stability`/`max_abs_diffs_from_first_round`はtorch/
sklearnに依存しない純粋関数なので、GPU・sklearn無しでもそのままテスト
できる(brief Step1)。`run_probe`は、CUDAの無い環境では
`S15_index_add_repeats`について常に`SkippedProbe`を返すため、プローブ
本体のテストはその分岐も検証する(`test_s12_cudnn.py`と同じ流儀)。

**`S15_index_add_repeats`の各回のdigestが一致するか否かはテストで固定
しない。** `index_add_`はatomicAdd由来で意図的に非決定であり(brief
参照)、固定するとテストが実測の偶然に依存してflakyになる。テストが
検証してよいのは「各回のdigestが列として記録されていること」までである
(brief Step1のテストをそのまま使う)。
"""
import numpy as np
import pytest

import reprobench.probes  # モジュールimportの副作用でS15プローブ群を登録する
from reprobench import registry
from reprobench.probes.s15_repeat_raw import (
    _N_ROUNDS,
    max_abs_diffs_from_first_round,
    round_digests_and_stability,
)
from reprobench.registry import ProbeResult, SkippedProbe, all_probes, run_probe


# --- round_digests_and_stability()/max_abs_diffs_from_first_round()の純粋関数としての性質 ---


def test_round_digests_and_stability_true_when_all_arrays_equal():
    arrays = [np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0])]
    digests, stable = round_digests_and_stability(arrays)
    assert len(digests) == 3
    assert len(set(digests)) == 1
    assert stable is True


def test_round_digests_and_stability_false_when_an_array_differs():
    arrays = [np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([9.0, 9.0])]
    digests, stable = round_digests_and_stability(arrays)
    assert len(digests) == 3
    assert len(set(digests)) == 2
    assert stable is False


def test_round_digests_and_stability_true_for_single_array():
    digests, stable = round_digests_and_stability([np.array([1.0])])
    assert len(digests) == 1
    assert stable is True


def test_max_abs_diffs_from_first_round_first_entry_is_zero():
    arrays = [np.array([1.0, 2.0]), np.array([1.0, 5.0]), np.array([1.0, 2.0])]
    diffs = max_abs_diffs_from_first_round(arrays)
    assert diffs[0] == 0.0
    assert diffs == [0.0, 3.0, 0.0]


# --- 登録内容(requires) ---


def test_index_add_repeats_is_registered_with_torch_and_cuda_requirement():
    assert all_probes()["S15_index_add_repeats"].requires == ("torch", "cuda")


def test_kmeans_repeats_is_registered_with_sklearn_requirement():
    assert all_probes()["S15_kmeans_repeats"].requires == ("sklearn",)


# --- skip経路 ---


def test_index_add_repeats_skips_without_cuda_via_forced_requirement(monkeypatch):
    # S8のtest_gpu_probes_skip_without_cuda/S12のtest_s12_skips_without_cuda_via_forced_requirement
    # と同じ流儀: torchの有無に関わらず"cuda"判定だけを先に落とす。
    monkeypatch.setattr(registry, "check_requirement", lambda r: r != "cuda")
    r = run_probe("S15_index_add_repeats")
    assert isinstance(r, SkippedProbe) and "cuda" in r.reason


def test_kmeans_repeats_skips_cleanly_without_library(monkeypatch):
    monkeypatch.setattr(registry, "check_requirement", lambda r: r != "sklearn")
    r = run_probe("S15_kmeans_repeats")
    assert isinstance(r, SkippedProbe) and "sklearn" in r.reason


# --- brief Step1のテストそのもの ---


def test_index_add_repeats_records_per_round_digests():
    # brief Step1の原文は`run_probe(registry.get("S15_index_add_repeats"))`
    # だが、`reprobench.registry`に`get`関数は存在しない(`run_probe`は
    # 文字列名をそのまま受け取る契約 -- 他の全テストファイルと同じ呼び方)。
    # ここは実際のAPIに合わせて`run_probe("S15_index_add_repeats")`と書く。
    r = run_probe("S15_index_add_repeats")
    if isinstance(r, SkippedProbe):
        assert "missing requirement" in r.reason
        return
    assert len(r.meta["round_digests"]) == r.meta["n_rounds"]
    assert r.meta["n_rounds"] >= 5


def test_kmeans_repeats_records_per_round_digests():
    r = run_probe("S15_kmeans_repeats")
    if isinstance(r, SkippedProbe):
        assert "missing requirement" in r.reason
        return
    assert len(r.meta["round_digests"]) == r.meta["n_rounds"]


# --- 自己申告安定性(tests/test_all_probes_smoke.pyの_SELF_REPORTED_STABILITY参照) ---


def test_index_add_repeats_self_reports_stability():
    # index_add_はatomicAdd由来で意図的に非決定なので、ここでdigestの
    # 一致/不一致そのものをassertしない。自己申告の仕組み
    # (meta["stable_within_process"])が存在することだけを確認する。
    r = run_probe("S15_index_add_repeats")
    if isinstance(r, ProbeResult):
        assert "stable_within_process" in r.meta
        assert len(r.meta["round_digests"]) == _N_ROUNDS


def test_kmeans_repeats_self_reports_stability():
    pytest.importorskip("sklearn")
    r = run_probe("S15_kmeans_repeats")
    assert isinstance(r, ProbeResult)
    assert "stable_within_process" in r.meta
    assert len(r.meta["round_digests"]) == _N_ROUNDS
    assert r.meta["effective_thread_count"] is not None
    assert r.meta["effective_thread_count"] > 0


# --- その他のmeta ---


def test_index_add_repeats_records_max_abs_diff_scalar():
    r = run_probe("S15_index_add_repeats")
    if isinstance(r, ProbeResult):
        assert "max_abs_diff_from_first_round" in r.scalars
        assert r.scalars["max_abs_diff_from_first_round"] >= 0.0
        per_round = r.meta["max_abs_diff_from_first_round_per_round"]
        assert len(per_round) == _N_ROUNDS
        assert per_round[0] == 0.0


def test_kmeans_repeats_uses_same_dataset_and_params_as_s5_kmeans_centers():
    # brief: S5_kmeans_centersと同じデータ・同じパラメータを使うこと。
    pytest.importorskip("sklearn")
    from reprobench.probes.s5_sklearn import _SEED as _S5_SEED

    r = run_probe("S15_kmeans_repeats")
    assert isinstance(r, ProbeResult)
    assert r.meta["seed"] == _S5_SEED
    assert r.meta["n_clusters"] == 4
    assert r.meta["n_init"] == 10
