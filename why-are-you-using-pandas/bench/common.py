"""共通ユーティリティ: ベンチマーク用の一意な固定長文字列生成など"""
import os

ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
BASE = len(ALPHABET)


def _base_n(i: int) -> str:
    if i == 0:
        return ALPHABET[0]
    digits = []
    while i:
        digits.append(ALPHABET[i % BASE])
        i //= BASE
    return "".join(reversed(digits))


def make_unique_strings(n: int, length: int):
    """0..n-1 を base62 でエンコードし、長さ length の一意な文字列を n 個生成する。
    長さ length は base62 で n 通りを表現するのに十分な桁数であること前提。

    パディングは ALPHABET[0](=数値の0に相当)によるゼロ埋め。_base_n は先頭に
    数値0の桁を出さないため、ゼロ埋めした固定長表現は i に対して単射になる。
    かつて 'x'(base62 の一桁として有効な文字)で埋めていた際は、埋め文字と
    エンコード結果の区別がつかず約1.9%の重複が生じていた(62^2 = 3,844通りの衝突)。
    """
    out = []
    pad = ALPHABET[0]
    for i in range(n):
        enc = _base_n(i)
        if len(enc) >= length:
            out.append(enc[-length:])
        else:
            out.append(pad * (length - len(enc)) + enc)
    return out


def env_info():
    import platform
    import psutil

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "total_mem_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }
