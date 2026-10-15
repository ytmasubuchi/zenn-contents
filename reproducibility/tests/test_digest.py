"""reprobench.digest のテスト: ダイジェストの正準化と数値比較プリミティブを検証する。"""
import numpy as np

from reprobench.digest import (
    canonical_bytes,
    compare_arrays,
    decode_sample,
    digest,
    encode_sample,
    ulp_diff,
)


# --- タスクブリーフ記載のテスト(そのまま) ---


def test_digest_distinguishes_dtype():
    a = np.arange(8, dtype=np.float32)
    b = np.arange(8, dtype=np.float64)
    assert digest(a) != digest(b)


def test_digest_distinguishes_shape():
    a = np.arange(8, dtype=np.float64)
    assert digest(a) != digest(a.reshape(2, 4))


def test_digest_stable_for_non_contiguous_view():
    base = np.arange(16, dtype=np.float64)
    assert digest(base[::2]) == digest(np.ascontiguousarray(base[::2]))


def test_encode_decode_sample_roundtrip():
    arr = np.linspace(0, 1, 10_000, dtype=np.float32)
    out = decode_sample(encode_sample(arr, n=256))
    assert out.shape == (256,) and out.dtype == np.float64


def test_encode_sample_is_deterministic():
    arr = np.linspace(0, 1, 10_000, dtype=np.float64)
    assert encode_sample(arr, n=256) == encode_sample(arr, n=256)


def test_ulp_diff_of_adjacent_floats_is_one():
    a = np.array([1.0])
    b = np.nextafter(a, 2.0)
    assert ulp_diff(a, b)[0] == 1


def test_compare_arrays_reports_bit_equal():
    a = np.linspace(0, 1, 100)
    r = compare_arrays(a, a.copy())
    assert r["bit_equal"] is True and r["max_ulp"] == 0


def test_compare_arrays_reports_difference():
    a = np.linspace(1, 2, 100)
    r = compare_arrays(a, np.nextafter(a, 3.0))
    assert r["bit_equal"] is False and r["max_ulp"] == 1 and r["max_rel"] > 0


# --- 曖昧さの解決事項に対する追加テスト ---


def test_digest_distinguishes_python_types():
    # 型タグが先頭に付くため、int/float/strは値が"1"相当でも衝突しない。
    assert digest(1) != digest(1.0)
    assert digest(1.0) != digest("1")
    assert digest(1) != digest("1")


def test_canonical_bytes_dict_ignores_insertion_order():
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert canonical_bytes(a) == canonical_bytes(b)
    assert digest(a) == digest(b)


def test_ulp_diff_same_nan_payload_is_zero():
    nan_bits = np.array([0x7FF8000000000001], dtype=np.int64).view(np.float64)
    a = nan_bits.copy()
    b = nan_bits.copy()
    assert ulp_diff(a, b)[0] == 0


def test_ulp_diff_mismatched_nonfinite_is_int64_max():
    a = np.array([np.nan])
    b = np.array([np.inf])
    assert ulp_diff(a, b)[0] == np.iinfo(np.int64).max


def test_ulp_diff_matching_infinities_is_zero():
    a = np.array([np.inf, -np.inf])
    b = np.array([np.inf, -np.inf])
    assert np.all(ulp_diff(a, b) == 0)


def test_compare_arrays_shape_mismatch_reports_sentinel():
    a = np.arange(4, dtype=np.float64)
    b = np.arange(6, dtype=np.float64)
    r = compare_arrays(a, b)
    assert r == {
        "bit_equal": False,
        "max_abs": float("inf"),
        "max_rel": float("inf"),
        "max_ulp": np.iinfo(np.int64).max,
    }


def test_compare_arrays_dtype_mismatch_reports_sentinel():
    a = np.arange(4, dtype=np.float32)
    b = np.arange(4, dtype=np.float64)
    r = compare_arrays(a, b)
    assert r["bit_equal"] is False and r["max_ulp"] == np.iinfo(np.int64).max


def test_encode_sample_uses_every_element_when_smaller_than_n():
    arr = np.linspace(0, 1, 10, dtype=np.float64)
    out = decode_sample(encode_sample(arr, n=4096))
    assert out.shape == (10,)
    assert np.array_equal(out, arr)


def test_ulp_diff_negative_zero_vs_positive_zero_is_zero():
    assert ulp_diff(np.array([-0.0]), np.array([0.0]))[0] == 0


def test_ulp_diff_of_adjacent_negative_floats_is_one():
    a = np.array([-1.0])
    b = np.nextafter(a, -2.0)
    assert ulp_diff(a, b)[0] == 1


def test_ulp_diff_additive_along_negative_side():
    # -1.0 から -2.0 方向に1 ULPずつ2回進めた距離が、まとめて測った距離と
    # 一致することを確認する(符号ビット側での単調性・加法性のテスト)。
    a = np.array([-1.0])
    b = np.nextafter(a, -2.0)
    c = np.nextafter(b, -2.0)
    assert ulp_diff(a, b)[0] == 1
    assert ulp_diff(b, c)[0] == 1
    assert ulp_diff(a, c)[0] == 2


def test_ulp_diff_cross_zero_crossing_is_two():
    # 0 を挟んで隣り合う(-0 の1つ隣と +0 の1つ隣)ペアはULP距離2になる。
    neg = np.nextafter(np.array([-0.0]), np.array([-1.0]))
    pos = np.nextafter(np.array([0.0]), np.array([1.0]))
    assert ulp_diff(neg, pos)[0] == 2


def test_ulp_diff_cross_sign_symmetry_pins_the_remap():
    # 符号をまたぐ距離は「bits(1.0)を原点からの距離として2倍したもの」に
    # 一致するはず(符号ビット反転の写像が正しく対称であることのピン止め)。
    one_bits = int(np.array([1.0], dtype=np.float64).view(np.int64)[0])
    assert ulp_diff(np.array([-1.0]), np.array([1.0]))[0] == 2 * one_bits
    assert ulp_diff(np.array([-1.0]), np.array([1.0]))[0] == 9214364837600034816
