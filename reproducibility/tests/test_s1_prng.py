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


# --- 曖昧さの解決事項に対する追加テスト(fix round 1) ---


def test_normal_probes_record_bit_generator_method_and_seeding():
    # L0(一様分布)とL2(正規分布変換)の確認は別物なので、どちらの
    # 組み合わせを計算したか(ビットジェネレータ・変換・seed→state写像)が
    # JSON上からも分かるようにする(findings 2/3、fix round 2 finding A)。
    expectations = {
        "S1_numpy_normal_randomstate": ("MT19937", "polar", "int_seed"),
        "S1_numpy_normal_randomstate_seedseq": ("MT19937", "polar", "bitgen_seedsequence"),
        "S1_numpy_normal_generator": ("PCG64", "ziggurat", "bitgen_seedsequence"),
        "S1_numpy_normal_generator_mt19937": ("MT19937", "ziggurat", "bitgen_seedsequence"),
    }
    for name, (bit_generator, method, seeding) in expectations.items():
        r = run_probe(name)
        assert r.meta["bit_generator"] == bit_generator
        assert r.meta["method"] == method
        assert r.meta["seeding"] == seeding


def test_normal_transform_bit_generator_and_seeding_are_independently_isolated():
    # fix round 2 finding A: S1_numpy_normal_randomstate(RandomState(int)、
    # mt19937_seed経由)とS1_numpy_normal_generator_mt19937
    # (Generator(MT19937(int))、SeedSequence経由)は見た目には「legacyの
    # MT19937+polar」対「GeneratorのMT19937+ziggurat」に見えるが、
    # seed→state写像も同時に変わってしまうため、その差は「変換だけの差」
    # ではない。変換だけを単離するには、RandomStateの側もSeedSequence
    # 経由でシードさせた S1_numpy_normal_randomstate_seedseq
    # (RandomState(MT19937(int)))が要る。
    randomstate_seedseq = run_probe("S1_numpy_normal_randomstate_seedseq")
    generator_mt19937 = run_probe("S1_numpy_normal_generator_mt19937")
    generator_pcg64 = run_probe("S1_numpy_normal_generator")

    # 変換だけを単離(ビットジェネレータもseed→state写像もMT19937+
    # SeedSequenceで共通)。
    assert randomstate_seedseq.digest != generator_mt19937.digest
    # ビットジェネレータだけを単離(変換(ziggurat)もseed→state写像
    # (SeedSequence)も共通)。
    assert generator_mt19937.digest != generator_pcg64.digest


def test_randomstate_int_seed_and_bitgen_seedseq_differ():
    # RandomState(int)はNumPy独自のmt19937_seed、RandomState(MT19937(int))
    # はSeedSequence — 同じMT19937コア・同じseedでも異なるseed→state写像
    # なので、正規分布に変換する前の時点ですでにストリームが違う
    # (fix round 2 finding Aの前提そのもの)。
    randomstate_int = run_probe("S1_numpy_normal_randomstate")
    randomstate_seedseq = run_probe("S1_numpy_normal_randomstate_seedseq")
    assert randomstate_int.digest != randomstate_seedseq.digest
