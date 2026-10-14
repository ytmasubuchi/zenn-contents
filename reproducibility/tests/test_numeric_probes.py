"""reprobench.probes.{s2_reduction,s3_blas,s4_libm} のテスト:
縮約・BLAS・libmの非決定性を検証する。
"""
import reprobench.probes  # モジュールimportの副作用でS2/S3/S4プローブ群を登録する
from reprobench.registry import run_probe


# --- タスクブリーフ記載のテスト(そのまま) ---


def test_shuffled_sum_differs_from_ordered_sum_float32():
    a = run_probe("S2_sum_float32")
    b = run_probe("S2_sum_shuffled_float32")
    assert a.scalars["total"] != b.scalars["total"]  # 非結合性の直接証明


def test_numeric_probes_deterministic_within_process():
    for name in ("S2_sum_float64", "S3_matmul_float64", "S4_exp_float64"):
        assert run_probe(name).digest == run_probe(name).digest


def test_matmul_probe_records_shape_and_dtype():
    r = run_probe("S3_matmul_float32")
    assert r.meta["shape"] == [2048, 2048] and r.meta["dtype"] == "float32"
