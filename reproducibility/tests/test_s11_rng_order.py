"""reprobench.probes.s11_rng_order のテスト。

`generate`は numpy のみに依存する純粋関数なので、モデルや外部依存
なしでそのままテストできる(brief Step1)。

**頑健版(`spawn`)が4パターン全てを生き延びるかどうかはテストで固定
しない。** それは実測で確かめる対象であり、事前登録された予測の1つに
過ぎない。テストが検証してよいのは「各組み合わせについて比較結果が
記録されていること」までであり、`matches_baseline`が`True`であることを
assertしてはならない(brief Step6)。
"""
import reprobench.probes  # モジュールimportの副作用でS11プローブ群を登録する
from reprobench.registry import ProbeResult, run_probe
from reprobench.probes.s11_rng_order import (
    IMPL_NAMES,
    REFACTOR_PATTERNS,
    _make_generators,
    generate,
)


# --- 定数が仕様書の値と一致していること ------------------------------------


def test_refactor_patterns_matches_brief():
    assert REFACTOR_PATTERNS == (
        "baseline",
        "loop_split",
        "swap_order",
        "drop_dead_call",
        "early_return",
    )


def test_impl_names_matches_brief():
    assert IMPL_NAMES == ("naive", "spawn")


# --- generate()の純粋関数としての性質 --------------------------------------


def test_generate_returns_values_and_consumption_count():
    values, n_consumed = generate("naive", "baseline", seed=20261014)
    assert len(values) > 0
    assert n_consumed > 0


def test_naive_baseline_is_reproducible_for_same_seed():
    assert generate("naive", "baseline", 20261014) == generate("naive", "baseline", 20261014)


def test_spawn_baseline_is_reproducible_for_same_seed():
    assert generate("spawn", "baseline", 20261014) == generate("spawn", "baseline", 20261014)


def test_all_impl_pattern_combinations_are_generatable():
    for impl in IMPL_NAMES:
        for pattern in REFACTOR_PATTERNS:
            values, n = generate(impl, pattern, 20261014)
            assert len(values) > 0
            assert n > 0


def test_spawn_impl_uses_independent_streams_per_component():
    """spawnは成分ごとに独立したGenerator、naiveは全成分が同一の共有
    Generatorを参照する(`_make_generators`のdocstring参照)。この性質を
    「ある成分から余計に何回描画しても、他の成分の次の描画値は変わらない
    (spawn)/変わる(naive)」という形で直接検証する。

    (fix round 1, レビュー指摘対応: 当初の実装は関数名・docstringが
    「独立したストリームを使うことを見る」と謳いながら
    `n_naive > 0 and n_spawn > 0` しか確認しておらず、独立性そのものは
    何も検証していなかった。`matches_baseline`の実測値を数値で固定する
    のではなく、`SeedSequence.spawn`が与える「他成分の消費から無関係」
    という設計上の契約そのものを直接確かめる形に書き直した。これは
    `REFACTOR_PATTERNS`のどれが生き残るかという実験結果の先取りではない
    — 対象のGeneratorオブジェクトが物理的に同一か別かという、
    `_make_generators`自身の実装から自明に決まる性質である。)
    """
    seed = 20261014

    def second_component_first_draw(impl: str, extra_draws_on_first_component: int) -> float:
        generators = _make_generators(impl, seed)
        for _ in range(extra_draws_on_first_component):
            generators[0].standard_normal()
        return float(generators[1].standard_normal())

    # spawn: 成分0から余計に5回描画しても、成分1の最初の描画値は変わらない
    # (別々のSeedSequenceから作った、物理的に別のGeneratorだから)。
    spawn_untouched = second_component_first_draw("spawn", extra_draws_on_first_component=0)
    spawn_perturbed = second_component_first_draw("spawn", extra_draws_on_first_component=5)
    assert spawn_untouched == spawn_perturbed

    # naive: 成分0・成分1は同一のGeneratorインスタンスを参照するため、
    # 成分0から余計に描画した分だけ共有ストリームの位置が進み、成分1の
    # 「最初の描画値」そのものが変わる。
    naive_untouched = second_component_first_draw("naive", extra_draws_on_first_component=0)
    naive_perturbed = second_component_first_draw("naive", extra_draws_on_first_component=5)
    assert naive_untouched != naive_perturbed


def test_generate_rejects_unknown_impl():
    import pytest

    with pytest.raises(ValueError):
        generate("bogus", "baseline", 20261014)


def test_generate_rejects_unknown_pattern():
    import pytest

    with pytest.raises(ValueError):
        generate("naive", "bogus", 20261014)


def test_all_patterns_produce_the_same_number_of_observed_values():
    # どのパターンも「値の意味」は変えない設計: 観測値列は常に
    # 成分数×2(a, b)の長さで揃っているはず。
    for impl in IMPL_NAMES:
        lengths = {len(generate(impl, pattern, 20261014)[0]) for pattern in REFACTOR_PATTERNS}
        assert lengths == {6}


# --- プローブ本体 -----------------------------------------------------------


def test_probe_records_all_combinations_against_own_baseline():
    r = run_probe("S11_rng_consumption_order")
    assert isinstance(r, ProbeResult)
    for impl in IMPL_NAMES:
        for pattern in REFACTOR_PATTERNS:
            assert r.meta["results"][impl][pattern]["matches_baseline"] in (True, False)


def test_probe_never_compares_naive_values_against_spawn_values():
    # 比較は常に同じ実装のbaselineとの間でのみ行う。meta構造そのものが
    # impl毎に独立したサブ辞書であり、implをまたいだ比較キーが存在しない
    # ことを確認する。
    r = run_probe("S11_rng_consumption_order")
    assert set(r.meta["results"].keys()) == set(IMPL_NAMES)
    for impl in IMPL_NAMES:
        assert set(r.meta["results"][impl].keys()) == set(REFACTOR_PATTERNS)


def test_probe_is_stable_across_two_in_process_calls():
    first = run_probe("S11_rng_consumption_order")
    second = run_probe("S11_rng_consumption_order")
    assert isinstance(first, ProbeResult) and isinstance(second, ProbeResult)
    assert first.digest == second.digest
