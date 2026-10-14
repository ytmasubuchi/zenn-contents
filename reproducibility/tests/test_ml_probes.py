"""reprobench.probes.{s5_sklearn,s6_lightgbm} のテスト:
scikit-learnとLightGBMの非決定性を検証する。

sklearn/lightgbmはこのテストコンテナ(python:3.11-slim + numpy + pytest)には
インストールされていないため、`run_probe`は常に`SkippedProbe`を返す。
digest/metaへ直接アクセスするテストはそのままでは`AttributeError`になって
しまうため、該当テストは`pytest.importorskip`でライブラリ不在時に
スキップへ丸める(標準スイートを壊さないため。検証コンテナ
(sklearn/lightgbmをpip installしたもの)ではそのまま実行される)。
"""
import numpy as np
import pytest

import reprobench.probes  # モジュールimportの副作用でS5/S6プローブ群を登録する
from reprobench import registry
from reprobench.registry import ProbeResult, SkippedProbe, run_probe


# --- タスクブリーフ記載のテスト(そのまま、ただしsklearn不在環境用にimportorskipを追加) ---


def test_sklearn_probes_deterministic_within_process():
    # fix round 1/2/3: 測定した相関のみを書く。S5_kmeans_centersは有効
    # スレッド数が1・2・4のときは安定、8(測定環境のnproc)のときは不安定
    # という相関を30回×6設定の実測で確認した(s5_sklearn.pyのモジュール
    # docstring参照)。このテストを走らせる検証コンテナは8 CPUで、既定
    # (未設定)の環境ではその8 CPU上で不安定側に入ることを確認済みだが、
    # これは「このコンテナでの既定」についての記述であり、有効スレッド数が
    # 1・2・4のホスト(そこでは同じ表が安定を予測する)についての主張では
    # ない。したがってここでdigestの一致を前提にはできない。このプローブ
    # 自身が`meta["stable_within_process"]`として自己申告する対象であり
    # (test_all_probes_smoke.pyの`_SELF_REPORTED_STABILITY`参照)、ここでは
    # それが存在することだけを確認する。
    pytest.importorskip("sklearn")
    for name in ("S5_logreg_coef", "S5_train_test_split_index"):
        assert run_probe(name).digest == run_probe(name).digest
    r = run_probe("S5_kmeans_centers")
    assert "stable_within_process" in r.meta


def test_train_test_split_probe_is_pure_l0_on_indices():
    pytest.importorskip("sklearn")
    r = run_probe("S5_train_test_split_index")
    assert r.meta["n_train"] + r.meta["n_test"] == 20_000


def test_ml_probes_skip_without_library(monkeypatch):
    monkeypatch.setattr(registry, "check_requirement", lambda r: r != "lightgbm")
    assert isinstance(run_probe("S6_lgbm_pred"), SkippedProbe)


# --- 曖昧さの解決事項に対する追加テスト ---


def test_random_forest_n_jobs_follows_repro_threads(monkeypatch):
    # fix round 1 small finding: `isinstance(r, SkippedProbe) or ...` passes
    # vacuously for ANY skip reason (including a bug-indicating "raised: ..."),
    # which defeats the `importorskip` guard above it. Since sklearn is
    # confirmed present at this point, a SkippedProbe here would itself be
    # the bug worth catching, not something to wave through.
    pytest.importorskip("sklearn")
    monkeypatch.setenv("REPRO_THREADS", "3")
    r = run_probe("S5_random_forest_pred")
    assert isinstance(r, ProbeResult)
    assert r.meta["n_jobs"] == 3


def test_logreg_probe_records_solver_and_max_iter():
    pytest.importorskip("sklearn")
    r = run_probe("S5_logreg_coef")
    assert r.meta["solver"] == "lbfgs"
    assert r.meta["max_iter"] == 200


def test_lgbm_num_threads_follows_repro_threads(monkeypatch):
    # fix round 1 small finding: see test_random_forest_n_jobs_follows_repro_threads.
    pytest.importorskip("lightgbm")
    monkeypatch.setenv("REPRO_THREADS", "2")
    r = run_probe("S6_lgbm_pred")
    assert isinstance(r, ProbeResult)
    assert r.meta["num_threads"] == 2


def test_lgbm_tree_structure_differs_from_prediction_digest():
    # 木構造ダイジェストと予測値ダイジェストは別の問いに答えるため、
    # 同じ値に潰れていないかを確認する(finding: 予測だけでなく木構造
    # 自体が変わるかを見る、というbrief記載の意図を裏切らないため)。
    pytest.importorskip("lightgbm")
    pred = run_probe("S6_lgbm_pred")
    tree = run_probe("S6_lgbm_tree_structure")
    assert pred.digest != tree.digest


def test_sklearn_probes_skip_cleanly_without_library(monkeypatch):
    monkeypatch.setattr(registry, "check_requirement", lambda r: r != "sklearn")
    for name in ("S5_logreg_coef", "S5_random_forest_pred", "S5_kmeans_centers", "S5_train_test_split_index"):
        r = run_probe(name)
        assert isinstance(r, SkippedProbe) and "sklearn" in r.reason


# --- fix round 1 (coordinator review) ---


def test_kmeans_dataset_has_genuine_cluster_structure_and_a_unique_optimum():
    # finding 2: the classification dataset was not built for clustering, so
    # k=4 on it had a huge family of near-tied local optima. With the new
    # well-separated-blobs dataset, repeated fits converge via the identical
    # path (n_iter_ never varies, measured over 180 fits across 6
    # thread-count configurations -- see s5_sklearn.py's module docstring)
    # -- that is the signature of "one dominant optimum", not "a different
    # local optimum chosen each time". inertia_ itself can still differ by a
    # tiny relative amount (measured: up to ~3.9e-16, across two independent
    # 30-fit runs). rel=1e-12 below is a safety bound with a ~2500x margin
    # over that measurement, not itself a measured tolerance.
    pytest.importorskip("sklearn")
    from reprobench.probes.s5_sklearn import _kmeans_dataset
    from sklearn.cluster import KMeans

    X = _kmeans_dataset()
    a = KMeans(n_clusters=4, n_init=10, random_state=20261014).fit(X)
    b = KMeans(n_clusters=4, n_init=10, random_state=20261014).fit(X)
    assert a.n_iter_ == b.n_iter_
    assert a.inertia_ == pytest.approx(b.inertia_, rel=1e-12)


def test_kmeans_centers_digest_is_invariant_to_row_order():
    # finding 2 (canonicalization): KMeans assigns no stable row order to
    # cluster_centers_, so two hosts that find the identical clustering with
    # rows permuted must not be reported as disagreeing.
    from reprobench.probes.s5_sklearn import _canonical_centers

    centers = np.array([[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    permuted = centers[[2, 0, 1]]
    assert np.array_equal(_canonical_centers(centers), _canonical_centers(permuted))


def test_kmeans_centers_records_effective_thread_count():
    pytest.importorskip("sklearn")
    r = run_probe("S5_kmeans_centers")
    assert isinstance(r, ProbeResult)
    assert r.meta["effective_thread_count"] is not None
    assert r.meta["effective_thread_count"] > 0


def test_split_digests_are_recorded_and_s5_s6_use_independent_splits():
    # finding 10: S5 and S6 use the same dataset-generation formula but
    # independent split algorithms (S6 avoids sklearn on purpose -- see
    # s6_lightgbm.py's module docstring). Record both so a reader can see
    # they differ, rather than silently assuming they match.
    pytest.importorskip("sklearn")
    pytest.importorskip("lightgbm")
    s5 = run_probe("S5_logreg_coef")
    s6 = run_probe("S6_lgbm_pred")
    assert isinstance(s5, ProbeResult) and isinstance(s6, ProbeResult)
    assert "train_idx_digest" in s5.meta
    assert "train_idx_digest" in s6.meta
    assert s5.meta["train_idx_digest"] != s6.meta["train_idx_digest"]
