"""S4: libm(超越関数)の実装差。

`exp`/`log`/`sin` はIEEE 754が正しい丸めを要求しない「recommended」演算
(IEEE 754-2019 §9.2)であるため、glibc/musl/Appleのlibmやコンパイラの
ベクトル化経路によって最終ビットが異なりうる。NumPyのAVX-512経路と
非AVX-512経路だけでも最大約4ULPの差が出ると報告されている。

各関数の入力範囲は、overflow/divide-by-zero/invalidのいずれのFPU例外も
発生させない(`pytest -W error`下でも安全な)値域を選んでいる: `exp`は
`[-20, 20]`(`exp(20)`でもfloat64の範囲に十分収まる)、`log`は厳密な正の値
`[1e-9, 1e9]`(ゼロ割を避ける)、`sin`は `[-1e6, 1e6]`(有限値である限り
NumPyのsin ufuncはFPU例外を立てない)。
"""
import numpy as np

from reprobench.digest import digest, encode_sample
from reprobench.registry import ProbeResult, register

_SEED = 20261014
_N = 1_000_000


def _libm_result(name: str, low: float, high: float, fn) -> ProbeResult:
    rng = np.random.default_rng(_SEED)
    x = rng.uniform(low, high, _N)
    y = fn(x)
    return ProbeResult(
        probe=name,
        digest=digest(y),
        scalars={
            "first": float(y[0]),
            "last": float(y[-1]),
            "mean": float(y.mean()),
        },
        sample_b64=encode_sample(y),
        meta={
            "seed": _SEED,
            "n": _N,
            "dtype": str(y.dtype),
            "domain": [low, high],
        },
    )


@register("S4_exp_float64")
def _exp_float64() -> ProbeResult:
    return _libm_result("S4_exp_float64", -20.0, 20.0, np.exp)


@register("S4_log_float64")
def _log_float64() -> ProbeResult:
    return _libm_result("S4_log_float64", 1e-9, 1e9, np.log)


@register("S4_sin_float64")
def _sin_float64() -> ProbeResult:
    return _libm_result("S4_sin_float64", -1e6, 1e6, np.sin)
