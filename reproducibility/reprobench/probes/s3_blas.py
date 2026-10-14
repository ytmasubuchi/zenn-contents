"""S3: BLAS行列積のスレッド数依存性。

`A @ B` の結果はBLAS実装がどうタイル分割し、どの順序で部分積を足し合わせる
かに依存するため、スレッド数やBLAS実装(OpenBLAS/MKL)が変われば浮動小数点
のビットパターンが変わりうる。2048×2048は共有マシンでの10分予算に収まる
最大級のサイズとして選んだ。

積の全要素をダイジェストに掛けるとメモリと時間を食うため、`scalars` には
`trace`(対角和)と`fro_norm`(フロベニウスノルム)という2つの要約統計量を
入れ、`encode_sample`で4096点だけを保存する。`digest`自体は全要素から計算
する(SHA-256は十分高速なため、全要素を見ること自体はコストにならない)。
"""
import numpy as np

from reprobench.digest import digest, encode_sample
from reprobench.registry import ProbeResult, register

_SEED = 20261014
_SIZE = 2048


def _matmul_result(name: str, dtype) -> ProbeResult:
    rng = np.random.default_rng(_SEED)
    a = rng.standard_normal((_SIZE, _SIZE)).astype(dtype)
    b = rng.standard_normal((_SIZE, _SIZE)).astype(dtype)
    c = a @ b
    trace = float(np.trace(c))
    fro_norm = float(np.linalg.norm(c))
    return ProbeResult(
        probe=name,
        digest=digest(c),
        scalars={"trace": trace, "fro_norm": fro_norm},
        sample_b64=encode_sample(c, n=4096),
        meta={
            "seed": _SEED,
            "shape": list(c.shape),
            "dtype": str(c.dtype),
            # JSONのfloat表現はfloat64の全ビットを往復させる保証が無いため、
            # 比較器が正確なビットで突き合わせられるようhex表現を残す
            # (s2_reduction.pyのtotal_hexと同じ理由)。
            "trace_hex": trace.hex(),
            "fro_norm_hex": fro_norm.hex(),
        },
    )


@register("S3_matmul_float32")
def _matmul_float32() -> ProbeResult:
    return _matmul_result("S3_matmul_float32", np.float32)


@register("S3_matmul_float64")
def _matmul_float64() -> ProbeResult:
    return _matmul_result("S3_matmul_float64", np.float64)
