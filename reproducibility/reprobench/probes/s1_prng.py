"""S1: PRNGの生ビット列(L0の証明)と、正規分布変換(L2)の内訳。

公開されたPRNGアルゴリズムの生ビット列は、実装を跨いでも一致するはず
(=仕様で固定されている)というのが本記事のL0主張である。**この主張が
直接当てはまるのは一様分布の5本**(`S1_stdlib_mt19937` /
`S1_numpy_randomstate` / `S1_numpy_mt19937_generator` / `S1_numpy_pcg64` /
`S1_numpy_philox`)**と、NumPy 1.22のビット修正を検出する
`S1_generator_random_float32` のみ**である。ここではCPython `random`・
NumPy legacy `RandomState`・NumPy `Generator`(PCG64/Philox/MT19937の3
ビットジェネレータ)から同一seedで10,000個を生成し、ダイジェストで比較
できるようにする。numpyと標準ライブラリのみを使い、重い依存は無いため、
モジュール先頭で通常通りimportする(重い依存を関数内部に置く規律は
alpine等でimport自体が失敗しうる場合のためのものであり、本モジュールには
該当しない)。

`S1_numpy_mt19937_generator` は同じMT19937コアを使うが、`Generator` の
seed→state写像(`SeedSequence`)はCPython `random` の参照実装
`init_by_array` ともNumPy `RandomState` 独自の `mt19937_seed` とも異なる。
MT19937の仕様が固定しているのはstate→出力のtwist+temperingであって
seed→stateではないため、この3本は**同一seed・同一コアでも異なるストリーム**
になる。

`S1_generator_random_float32` はNumPy 1.22.0でfloat32一様分布の最下位ビット
のバグが修正されたため、1.21.x以前と2.x以降でストリームが変わる
(NEP 19が明言する「Generatorはストリーム互換を保証しない」の実例)。

正規分布の4本(`S1_numpy_normal_*`)は**L0ではなくL2相当**である。一様乱数を
Box-Muller系の変換(polar法/ziggurat法)に通した値であり、その変換自体が
`log`/`exp`を呼ぶ(S4がまさに対象とする、IEEE 754が正しい丸めを要求しない
「recommended」演算)。したがってこれらがホスト間で食い違っても、それは
PRNG層ではなくlibm層由来でありうる。この confound をJSON上でも追えるよう、
各正規分布プローブの`meta`に`bit_generator`(`"MT19937"`/`"PCG64"`)と
`method`(`"polar"`/`"ziggurat"`)、さらに`seeding`(`"int_seed"`/
`"bitgen_seedsequence"`)を記録する。

正規分布プローブは「変換アルゴリズム」「ビットジェネレータ」「seed→state
写像」という3つの独立変数を区別するために4本ある。**重要な注意**:
`RandomState(int)`(`S1_numpy_normal_randomstate`が使う)はNumPy独自の
`mt19937_seed`でシードするが、`Generator(MT19937(int))`
(`S1_numpy_normal_generator_mt19937`が使う)は`SeedSequence`でシードする
——上のパラグラフで一様分布プローブについて述べた通り、この2つは
**同じMT19937コアでも異なるシード→state写像**である
(`S1_stdlib_mt19937`/`S1_numpy_randomstate`/`S1_numpy_mt19937_generator`が
3つの異なるストリームになることの再掲)。したがって
`S1_numpy_normal_randomstate`と`S1_numpy_normal_generator_mt19937`を直接
比べても「変換だけの差」にはならず、変換とseed→state写像の両方が同時に
変わる。真に変換だけを単離するには、`RandomState`にも`MT19937`インスタンス
を渡してSeedSequence経由でシードさせた`S1_numpy_normal_randomstate_seedseq`
が要る(`RandomState`はintだけでなくBitGeneratorインスタンスも受け付ける)。

3本の比較はそれぞれ次だけを単離する:
- `S1_numpy_normal_randomstate_seedseq` vs `S1_numpy_normal_generator_mt19937`
  — **変換だけ**(polar/ziggurat)。ビットジェネレータもseed→state写像も
  同一(どちらもMT19937インスタンスをSeedSequence経由でシード)。
- `S1_numpy_normal_generator_mt19937` vs `S1_numpy_normal_generator`
  — **ビットジェネレータだけ**(MT19937/PCG64)。変換(ziggurat)も
  seed→state写像(どちらもSeedSequence経由)も同一。
- `S1_numpy_normal_randomstate` vs `S1_numpy_normal_generator`
  — legacyの既定とGeneratorの既定(`default_rng`)という**現実的な
  比較**だが、変換・ビットジェネレータ・seed→state写像の3軸が同時に
  変わる。これは「何かの単離」ではなく、素朴に2つのAPIを比べると何が
  起きるかを示すプローブとして位置づける。
"""
import random

import numpy as np

from reprobench.digest import digest, encode_sample
from reprobench.registry import ProbeResult, register

_SEED = 20261014
_N = 10_000


def _result(name: str, arr, **extra_meta) -> ProbeResult:
    """生成済み配列1本からProbeResultを組み立てる共通処理。

    `digest`/`sample_b64` は渡された配列そのもののdtypeで計算する(float32
    ストリームのビット差を検出するため、float64への昇格より前の値を使う
    必要がある)。`scalars` はJSON化しやすいfloat64のPython floatに統一する。
    JSONのfloat表現はfloat64の全ビットを往復させる保証が無いため、各
    scalarに`float.hex()`を`meta["{key}_hex"]`として残す
    (s2_reduction.pyのtotal_hexと同じ理由)。`extra_meta` は正規分布
    プローブが`bit_generator`/`method`/`seeding`を足すために使う
    (一様分布プローブは渡さない)。
    """
    arr = np.asarray(arr)
    arr64 = arr.astype(np.float64)
    scalars = {
        "first": float(arr64[0]),
        "last": float(arr64[-1]),
        "mean": float(arr64.mean()),
    }
    meta = {"seed": _SEED, "n": _N, "dtype": str(arr.dtype)}
    for key, value in scalars.items():
        meta[f"{key}_hex"] = value.hex()
    meta.update(extra_meta)
    return ProbeResult(
        probe=name,
        digest=digest(arr),
        scalars=scalars,
        sample_b64=encode_sample(arr),
        meta=meta,
    )


@register("S1_stdlib_mt19937")
def _stdlib_mt19937() -> ProbeResult:
    rng = random.Random(_SEED)
    return _result("S1_stdlib_mt19937", [rng.random() for _ in range(_N)])


@register("S1_numpy_randomstate")
def _numpy_randomstate() -> ProbeResult:
    rs = np.random.RandomState(_SEED)
    return _result("S1_numpy_randomstate", rs.random_sample(_N))


@register("S1_numpy_mt19937_generator")
def _numpy_mt19937_generator() -> ProbeResult:
    gen = np.random.Generator(np.random.MT19937(_SEED))
    return _result("S1_numpy_mt19937_generator", gen.random(_N))


@register("S1_numpy_pcg64")
def _numpy_pcg64() -> ProbeResult:
    gen = np.random.Generator(np.random.PCG64(_SEED))
    return _result("S1_numpy_pcg64", gen.random(_N))


@register("S1_numpy_philox")
def _numpy_philox() -> ProbeResult:
    gen = np.random.Generator(np.random.Philox(_SEED))
    return _result("S1_numpy_philox", gen.random(_N))


@register("S1_numpy_normal_randomstate")
def _numpy_normal_randomstate() -> ProbeResult:
    # legacyの既定の使い方(intを直接渡す)。seed→stateはNumPy独自の
    # mt19937_seedを通る。
    rs = np.random.RandomState(_SEED)
    return _result(
        "S1_numpy_normal_randomstate",
        rs.standard_normal(_N),
        bit_generator="MT19937",
        method="polar",
        seeding="int_seed",
    )


@register("S1_numpy_normal_randomstate_seedseq")
def _numpy_normal_randomstate_seedseq() -> ProbeResult:
    # RandomStateはintだけでなくBitGeneratorインスタンスも受け付ける。
    # MT19937インスタンスを渡すと、Generator(MT19937(...))と同じ
    # SeedSequence経由でシードされる(ビットジェネレータのraw stateが
    # 両者で一致することを確認済み)。これにより
    # S1_numpy_normal_generator_mt19937との差分が「変換だけ」になる。
    rs = np.random.RandomState(np.random.MT19937(_SEED))
    return _result(
        "S1_numpy_normal_randomstate_seedseq",
        rs.standard_normal(_N),
        bit_generator="MT19937",
        method="polar",
        seeding="bitgen_seedsequence",
    )


@register("S1_numpy_normal_generator")
def _numpy_normal_generator() -> ProbeResult:
    gen = np.random.default_rng(_SEED)  # 既定のビットジェネレータはPCG64
    return _result(
        "S1_numpy_normal_generator",
        gen.standard_normal(_N),
        bit_generator="PCG64",
        method="ziggurat",
        seeding="bitgen_seedsequence",
    )


@register("S1_numpy_normal_generator_mt19937")
def _numpy_normal_generator_mt19937() -> ProbeResult:
    # S1_numpy_normal_randomstate_seedseq(MT19937+polar、SeedSequence)
    # とはビットジェネレータもseed→state写像も同じでziggurat変換だけが
    # 違い、S1_numpy_normal_generator(PCG64+ziggurat、SeedSequence)とは
    # 変換もseed→state写像も同じでビットジェネレータだけが違う。
    gen = np.random.Generator(np.random.MT19937(_SEED))
    return _result(
        "S1_numpy_normal_generator_mt19937",
        gen.standard_normal(_N),
        bit_generator="MT19937",
        method="ziggurat",
        seeding="bitgen_seedsequence",
    )


@register("S1_generator_random_float32")
def _generator_random_float32() -> ProbeResult:
    gen = np.random.default_rng(_SEED)
    return _result("S1_generator_random_float32", gen.random(_N, dtype=np.float32))
