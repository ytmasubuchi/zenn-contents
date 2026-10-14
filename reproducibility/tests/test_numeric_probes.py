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


# --- 曖昧さの解決事項に対する追加テスト(fix round 1) ---


def test_s3_scalars_have_lossless_hex_in_meta():
    # trace/fro_normはJSONのfloat表現だけでは全ビットを往復できないため、
    # hex表現もmetaに残す(finding 5)。
    r = run_probe("S3_matmul_float32")
    assert r.meta["trace_hex"] == r.scalars["trace"].hex()
    assert r.meta["fro_norm_hex"] == r.scalars["fro_norm"].hex()


def test_s4_scalars_have_lossless_hex_in_meta():
    r = run_probe("S4_exp_float64")
    for key in ("first", "last", "mean"):
        assert r.meta[f"{key}_hex"] == r.scalars[key].hex()


def test_s2_shuffled_sum_records_permutation_metadata():
    # permの存在とseedがmetaに残っていないと、「置換した」という事実が
    # 結果JSONから読み取れなくなる(finding 8)。
    r = run_probe("S2_sum_shuffled_float32")
    assert r.meta["permuted"] is True
    assert r.meta["perm_seed"] == 20261014


def test_s4_additional_domain_probes_cover_the_effect_s4_targets():
    # S4_sin_float64/S4_log_float64の既存の値域だけでは、AVX-512 sin経路の
    # 閾値未満に収まらなかったり、logの指数スイープが実現されなかったり
    # する恐れがあるため追加した2本(finding 6)。
    sin_small = run_probe("S4_sin_small_float64")
    assert sin_small.meta["domain"] == [-1e4, 1e4]

    log_loguniform = run_probe("S4_log_loguniform_float64")
    assert log_loguniform.meta["sampling"] == "log_uniform"
    assert log_loguniform.meta["n"] == 1_000_000
