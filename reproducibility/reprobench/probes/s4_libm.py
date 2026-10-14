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

`S4_sin_float64`/`S4_log_float64`の既存の値域だけでは、本プローブが検出
したい効果そのものを取り逃がす恐れがある: NumPyのベクトル化された
`sin`/`cos`経路は歴史的に振幅がおよそ1.17e5未満の入力にしか適用されず、
`[-1e6, 1e6]`の一様分布ではその閾値を超える入力がほとんどになり
AVX-512経路を素通りしてしまう。また`[1e-9, 1e9]`の一様分布は指数的に
広い範囲を線形に敷くため、サンプルの大半(約99.9%)が`1e6`超に偏り、
下限の`1e-9`付近はほぼ実現されない(logの指数スイープにならない)。
そこで既存の2本は変えず(両者とも別の問いに答えるプローブとして残す)、
`S4_sin_small_float64`(振幅をAVX-512経路が実際に使われる範囲に収めた
sin)と`S4_log_loguniform_float64`(指数を一様に敷いてlog一様サンプリング
したlog)を追加する。
"""
import numpy as np

from reprobench.digest import digest, encode_sample
from reprobench.registry import ProbeResult, register

_SEED = 20261014
_N = 1_000_000


def _libm_result(name: str, x: np.ndarray, fn, meta_extra: dict) -> ProbeResult:
    y = fn(x)
    scalars = {
        "first": float(y[0]),
        "last": float(y[-1]),
        "mean": float(y.mean()),
    }
    meta = {"seed": _SEED, "n": _N, "dtype": str(y.dtype), **meta_extra}
    # JSONのfloat表現はfloat64の全ビットを往復させる保証が無いため、
    # 比較器が正確なビットで突き合わせられるようscalarsの各値にhex表現を
    # 残す(s2_reduction.pyのtotal_hexと同じ理由)。
    for key, value in scalars.items():
        meta[f"{key}_hex"] = value.hex()
    return ProbeResult(
        probe=name,
        digest=digest(y),
        scalars=scalars,
        sample_b64=encode_sample(y),
        meta=meta,
    )


def _uniform(low: float, high: float) -> np.ndarray:
    return np.random.default_rng(_SEED).uniform(low, high, _N)


def _log_uniform(low: float, high: float) -> np.ndarray:
    """`[low, high]` の指数を一様に敷いてサンプリングする(対数一様分布)。

    通常の線形一様分布だと、`low`と`high`が何桁も離れている場合に標本の
    ほとんどが`high`付近に偏り、`low`に近い桁はほぼ実現されない。指数
    自体を一様乱数で決めることで、全桁にわたって均等にサンプルが乗る。
    """
    rng = np.random.default_rng(_SEED)
    exponents = rng.uniform(np.log10(low), np.log10(high), _N)
    return np.power(10.0, exponents)


@register("S4_exp_float64")
def _exp_float64() -> ProbeResult:
    low, high = -20.0, 20.0
    return _libm_result(
        "S4_exp_float64", _uniform(low, high), np.exp,
        {"domain": [low, high], "sampling": "uniform"},
    )


@register("S4_log_float64")
def _log_float64() -> ProbeResult:
    low, high = 1e-9, 1e9
    return _libm_result(
        "S4_log_float64", _uniform(low, high), np.log,
        {"domain": [low, high], "sampling": "uniform"},
    )


@register("S4_sin_float64")
def _sin_float64() -> ProbeResult:
    low, high = -1e6, 1e6
    return _libm_result(
        "S4_sin_float64", _uniform(low, high), np.sin,
        {"domain": [low, high], "sampling": "uniform"},
    )


@register("S4_sin_small_float64")
def _sin_small_float64() -> ProbeResult:
    # NumPyのベクトル化sin/cos経路が実際に使われる振幅(閾値はおよそ
    # 1.17e5)に収めることで、S4_sin_float64が取り逃がしうる
    # AVX-512経路そのものを確実に踏ませる。
    low, high = -1e4, 1e4
    return _libm_result(
        "S4_sin_small_float64", _uniform(low, high), np.sin,
        {"domain": [low, high], "sampling": "uniform"},
    )


@register("S4_log_loguniform_float64")
def _log_loguniform_float64() -> ProbeResult:
    # S4_log_float64と同じ値域だが、線形一様ではなく指数を一様に敷く
    # ことで、全桁にわたって均等にlogの引数をサンプリングする。
    low, high = 1e-9, 1e9
    return _libm_result(
        "S4_log_loguniform_float64", _log_uniform(low, high), np.log,
        {"domain": [low, high], "sampling": "log_uniform"},
    )
