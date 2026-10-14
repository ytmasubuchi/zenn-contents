"""S2: 縮約(reduction)の非結合性。

浮動小数点の加算は結合的ではないため、同じ値の集合でも足す順序が変われば
結果のビットパターンは変わりうる。SIMD幅が違えばNumPyの`sum`/`mean`が
内部で使うペアワイズ和のブロック分割も変わるため、ここでの差はホスト間の
SIMD幅の違い(AVX2/AVX-512/SVE2)を検出する材料になる。

`S2_sum_shuffled_float32` は `S2_sum_float32` と**同じ値**を固定の置換
(`default_rng(20261014).permutation(n)`)で並べ替えてから和を取る。置換を
固定するのは、非決定な置換だとプローブ自体が環境と無関係にばらつき、
非結合性の実証という目的を果たさなくなるため(置換の生成に使うPCG64の
ビット列自体はS1で独立にL0として検証されるため、ここでの依存で循環は
生じない)。
"""
import numpy as np

from reprobench.digest import digest
from reprobench.registry import ProbeResult, register

_SEED = 20261014
_PERM_SEED = 20261014
_N = 10_000_000


def _base_values() -> np.ndarray:
    """縮約対象のfloat64配列(S2の全プローブが共有する「同じ値」)。"""
    rng = np.random.default_rng(_SEED)
    return rng.standard_normal(_N)


def _scalar_result(name: str, key: str, value: float, dtype: str, **extra_meta) -> ProbeResult:
    """スカラー1個(合計または平均)を返すプローブの共通組み立て処理。

    JSONの浮動小数点表現はfloat64の全ビットを往復させる保証が無いため、
    `meta["{key}_hex"]` に `float.hex()` を残し、比較器が正確なビットで
    突き合わせられるようにする。`extra_meta` は
    `S2_sum_shuffled_float32` が置換の情報(`permuted`/`perm_seed`)を
    足すために使う。
    """
    meta = {
        "seed": _SEED,
        "n": _N,
        "dtype": dtype,
        f"{key}_hex": value.hex(),
    }
    meta.update(extra_meta)
    return ProbeResult(
        probe=name,
        digest=digest(value),
        scalars={key: value},
        sample_b64=None,
        meta=meta,
    )


@register("S2_sum_float32")
def _sum_float32() -> ProbeResult:
    arr = _base_values().astype(np.float32)
    total = float(np.sum(arr))
    return _scalar_result("S2_sum_float32", "total", total, "float32")


@register("S2_sum_float64")
def _sum_float64() -> ProbeResult:
    arr = _base_values()
    total = float(np.sum(arr))
    return _scalar_result("S2_sum_float64", "total", total, "float64")


@register("S2_mean_float32")
def _mean_float32() -> ProbeResult:
    arr = _base_values().astype(np.float32)
    mean = float(np.mean(arr))
    return _scalar_result("S2_mean_float32", "mean", mean, "float32")


@register("S2_sum_shuffled_float32")
def _sum_shuffled_float32() -> ProbeResult:
    arr = _base_values().astype(np.float32)
    perm = np.random.default_rng(_PERM_SEED).permutation(_N)
    total = float(np.sum(arr[perm]))
    return _scalar_result(
        "S2_sum_shuffled_float32",
        "total",
        total,
        "float32",
        permuted=True,
        perm_seed=_PERM_SEED,
    )
