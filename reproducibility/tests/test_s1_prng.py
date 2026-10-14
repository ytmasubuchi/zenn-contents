"""reprobench.probes.s1_prng のテスト: PRNGの生ビット列を検証する(L0の実証)。"""
import random

import reprobench.probes  # モジュールimportの副作用でS1プローブ群を登録する
from reprobench.registry import run_probe


# --- タスクブリーフ記載のテスト(そのまま) ---


def test_stdlib_mt19937_matches_published_value():
    # MT19937 は公開仕様。CPython の random.seed(0) 直後の最初の値は
    # 実装を跨いで一致する既知値であり、これが L0(規格レベル)の実証になる。
    random.seed(0)
    assert random.random() == 0.8444218515250481


def test_prng_probe_records_seed_and_count():
    r = run_probe("S1_stdlib_mt19937")
    assert r.meta["seed"] == 20261014 and r.meta["n"] == 10_000


def test_prng_probes_are_deterministic_within_process():
    for name in ("S1_stdlib_mt19937", "S1_numpy_randomstate", "S1_numpy_pcg64", "S1_numpy_philox"):
        assert run_probe(name).digest == run_probe(name).digest


def test_randomstate_and_generator_differ():
    assert run_probe("S1_numpy_randomstate").digest != run_probe("S1_numpy_pcg64").digest


def test_three_mt19937_paths_give_three_different_streams():
    # 同じ MT19937、同じ seed、3つの異なる seed→state 写像。
    # 仕様が固定しているのは state→出力であって seed→state ではない。
    ds = {run_probe(n).digest for n in
          ("S1_stdlib_mt19937", "S1_numpy_randomstate", "S1_numpy_mt19937_generator")}
    assert len(ds) == 3
