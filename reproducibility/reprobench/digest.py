"""ダイジェストと数値比較の基盤。

プローブ(別タスクで追加)が計算結果を記録するための正準バイト表現
(`canonical_bytes` / `digest`)と、2つの環境の出力をbit単位・ULP単位で
突き合わせるための比較プリミティブ(`encode_sample` / `decode_sample` /
`ulp_diff` / `compare_arrays`)を提供する。

このモジュールの責務は「正準バイト表現」と「数値比較」のみ。
"""
import base64
import hashlib
import struct
from typing import Any

import numpy as np

_INT64_MIN = np.iinfo(np.int64).min
_INT64_MAX = np.iinfo(np.int64).max


def _length_prefixed(b: bytes) -> bytes:
    """長さを8バイト(big-endian)で前置する。list/tuple/dict の要素区切りに
    任意のバイト列が混ざっても衝突しないようにするため。
    """
    return len(b).to_bytes(8, "big") + b


def canonical_bytes(value: Any) -> bytes:
    """値を型を区別できる正準バイト列に変換する。

    int/float/str/bytes/bool/list/tuple/dict/numpy.ndarray を受け付ける。
    先頭に型タグを付けるため、`digest(1)` と `digest(1.0)` と `digest("1")`
    は常に異なる値になる。ndarray は `dtype.str` + `shape` + C連続バイト列
    (`np.ascontiguousarray` 経由)を連結するため、非連続なビューでもその
    連続コピーと同じダイジェストになる。dict はキーでソートしてから連結
    するため、挿入順序の違い(本記事が計測対象とする非決定性の一種)は
    ダイジェストに影響しない。
    """
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return (
            b"ndarray:"
            + arr.dtype.str.encode("ascii")
            + b":"
            + str(arr.shape).encode("ascii")
            + b":"
            + arr.tobytes()
        )
    if isinstance(value, bool):
        # bool は int のサブクラスなので int より先に判定する。
        return b"bool:" + (b"1" if value else b"0")
    if isinstance(value, int):
        return b"int:" + str(value).encode("ascii")
    if isinstance(value, float):
        return b"float:" + struct.pack("<d", value)
    if isinstance(value, str):
        return b"str:" + value.encode("utf-8")
    if isinstance(value, bytes):
        return b"bytes:" + value
    if isinstance(value, (list, tuple)):
        tag = b"list:" if isinstance(value, list) else b"tuple:"
        body = b"".join(_length_prefixed(canonical_bytes(v)) for v in value)
        return tag + len(value).to_bytes(8, "big") + body
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: canonical_bytes(kv[0]))
        body = b"".join(
            _length_prefixed(canonical_bytes(k)) + _length_prefixed(canonical_bytes(v))
            for k, v in items
        )
        return b"dict:" + len(items).to_bytes(8, "big") + body
    raise TypeError(f"canonical_bytes: unsupported type {type(value)!r}")


def digest(value: Any) -> str:
    """`canonical_bytes(value)` の SHA-256 16進ダイジェストを返す。"""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def encode_sample(arr: np.ndarray, n: int = 4096) -> str:
    """ndarray を float64 に昇格し、固定インデックスで部分抽出してbase64化する。

    抽出前に1次元へ平坦化する。固定インデックスは
    `np.linspace(0, size-1, n)` を整数化したもので、常に同じ値を指すため
    結果は決定的になる。配列サイズが n 以下の場合は全要素を使う
    (結果の shape は `(arr.size,)`)。
    """
    flat = np.ascontiguousarray(arr).ravel().astype(np.float64)
    size = flat.size
    if size <= n:
        sample = flat
    else:
        idx = np.linspace(0, size - 1, n).astype(np.int64)
        sample = flat[idx]
    return base64.b64encode(sample.tobytes()).decode("ascii")


def decode_sample(s: str) -> np.ndarray:
    """`encode_sample` が返した文字列から float64 配列を復元する。"""
    raw = base64.b64decode(s)
    return np.frombuffer(raw, dtype=np.float64).copy()


def _ordered_bits(bits: np.ndarray) -> np.ndarray:
    """float64のビットパターン(int64として解釈したもの)を、対応する浮動小数
    点値の大小関係と一致する単調な符号付き整数列に正規化する。負数(符号
    ビットが立っている値)は `0x8000000000000000 - bits` に写す。
    """
    offset = np.int64(_INT64_MIN)  # 0x8000000000000000 相当(int64としてのラップ後の値)
    return np.where(bits < 0, offset - bits, bits)


def ulp_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """float64配列同士の要素ごとのULP距離を返す。

    ビットパターンが完全一致する要素(同じペイロードのNaN同士、符号も
    含めて一致するinf同士を含む)は距離0。非有限値(NaN/inf)でビット
    パターンが一致しない要素は `np.iinfo(np.int64).max` を返す。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    bits_a = a.view(np.int64)
    bits_b = b.view(np.int64)

    # オーバーフローを避けるためPythonの多倍長整数(object dtype)で差を取る。
    oa = _ordered_bits(bits_a).astype(object)
    ob = _ordered_bits(bits_b).astype(object)
    diff = np.abs(oa - ob)
    diff = np.minimum(diff, _INT64_MAX).astype(np.int64)

    nonfinite_mismatch = (~np.isfinite(a) | ~np.isfinite(b)) & (bits_a != bits_b)
    diff = np.where(nonfinite_mismatch, _INT64_MAX, diff)
    return diff.astype(np.int64)


def compare_arrays(a: np.ndarray, b: np.ndarray) -> dict:
    """2つの配列をbit完全一致・絶対誤差・相対誤差・ULP距離の観点で比較する。

    shape または dtype が異なる場合は比較不能とみなし、例外を送出せずに
    `{"bit_equal": False, "max_abs": inf, "max_rel": inf,
    "max_ulp": np.iinfo(np.int64).max}` を返す。これにより、shapeの時点で
    食い違った環境同士の結果も比較器(別タスク)が落ちずに扱える。
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return {
            "bit_equal": False,
            "max_abs": float("inf"),
            "max_rel": float("inf"),
            "max_ulp": int(_INT64_MAX),
        }

    bit_equal = bool(np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes())

    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    abs_diff = np.abs(af - bf)
    denom = np.abs(bf)
    with np.errstate(divide="ignore", invalid="ignore"):
        # denom==0の要素は np.where の不使用側でも演算されてしまうため、
        # 0除算由来の警告(RuntimeWarning)だけをここで無害化する。
        rel_diff = np.where(denom == 0, abs_diff, abs_diff / denom)
    ulp = ulp_diff(af, bf)

    return {
        "bit_equal": bit_equal,
        "max_abs": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "max_rel": float(np.max(rel_diff)) if rel_diff.size else 0.0,
        "max_ulp": int(np.max(ulp)) if ulp.size else 0,
    }
