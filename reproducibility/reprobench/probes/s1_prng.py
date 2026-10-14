"""S1: PRNGの生ビット列(L0の証明)。

公開されたPRNGアルゴリズムの生ビット列は、実装を跨いでも一致するはず
(=仕様で固定されている)というのが本記事のL0主張である。ここではCPython
`random`・NumPy legacy `RandomState`・NumPy `Generator`(PCG64/Philox/
MT19937の3ビットジェネレータ)から同一seedで10,000個を生成し、ダイジェスト
で比較できるようにする。numpyと標準ライブラリのみを使い、重い依存は無い
ため、モジュール先頭で通常通りimportする(重い依存を関数内部に置く規律は
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
"""
import random

import numpy as np

from reprobench.digest import digest, encode_sample
from reprobench.registry import ProbeResult, register

_SEED = 20261014
_N = 10_000


def _result(name: str, arr) -> ProbeResult:
    """生成済み配列1本からProbeResultを組み立てる共通処理。

    `digest`/`sample_b64` は渡された配列そのもののdtypeで計算する(float32
    ストリームのビット差を検出するため、float64への昇格より前の値を使う
    必要がある)。`scalars` はJSON化しやすいfloat64のPython floatに統一する。
    """
    arr = np.asarray(arr)
    arr64 = arr.astype(np.float64)
    return ProbeResult(
        probe=name,
        digest=digest(arr),
        scalars={
            "first": float(arr64[0]),
            "last": float(arr64[-1]),
            "mean": float(arr64.mean()),
        },
        sample_b64=encode_sample(arr),
        meta={"seed": _SEED, "n": _N, "dtype": str(arr.dtype)},
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
    rs = np.random.RandomState(_SEED)
    return _result("S1_numpy_normal_randomstate", rs.standard_normal(_N))


@register("S1_numpy_normal_generator")
def _numpy_normal_generator() -> ProbeResult:
    gen = np.random.default_rng(_SEED)
    return _result("S1_numpy_normal_generator", gen.standard_normal(_N))


@register("S1_generator_random_float32")
def _generator_random_float32() -> ProbeResult:
    gen = np.random.default_rng(_SEED)
    return _result("S1_generator_random_float32", gen.random(_N, dtype=np.float32))
