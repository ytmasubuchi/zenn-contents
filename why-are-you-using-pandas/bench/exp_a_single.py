"""実験A: 文字列格納のメモリ効率(単一条件を計測してJSON1行を標準出力に出す)。
プロセス分離してRSSを計測するため、CLIから1条件ずつ呼び出す設計。
"""
import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import make_unique_strings  # noqa: E402


def rss_bytes():
    import psutil

    gc.collect()
    return psutil.Process(os.getpid()).memory_info().rss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lib",
        required=True,
        choices=[
            "pandas_object",
            "pandas_default",
            "pandas_category",
            "pandas_pyarrow",
            "polars",
        ],
    )
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--length", type=int, required=True)
    args = ap.parse_args()

    # ライブラリのimportはベースラインRSSの取得前に済ませる。import自体が
    # 数十MB単位のメモリを使うため、ベースライン後にimportするとRSS増分が
    # 「Series作成コスト」ではなく「import+Series作成」の合算になってしまう。
    if args.lib.startswith("pandas"):
        import pandas as pd
    else:
        import polars as pl

    strings = make_unique_strings(args.n, args.length)
    baseline_rss = rss_bytes()

    # 注意: RSS増分は「元のPython文字列リストを保持したままSeriesを作った際の
    # 追加割り当て」を意味する。Arrow系(pyarrow/polars)は実データをバッファへ
    # コピーするため列の実サイズに近い値になるが、object/pythonストレージは
    # 元のPython文字列オブジェクトを共有するため文字列本体分は増分に現れない。
    # 条件間の直接比較にはapi_bytesを使うこと。
    api_bytes = None
    dtype_repr = None
    storage_repr = None
    if args.lib == "pandas_object":
        s = pd.Series(strings, dtype=object)
        api_bytes = int(s.memory_usage(deep=True))
        dtype_repr = str(s.dtype)
        storage_repr = getattr(s.dtype, "storage", None)
    elif args.lib == "pandas_default":
        # dtype引数を指定しない場合にpandasが選ぶデフォルトのdtypeを計測する
        # (pyarrow未インストール環境ではpythonストレージのstring dtypeにフォールバックする想定)
        s = pd.Series(strings)
        api_bytes = int(s.memory_usage(deep=True))
        dtype_repr = str(s.dtype)
        storage_repr = getattr(s.dtype, "storage", None)
    elif args.lib == "pandas_category":
        s = pd.Series(strings, dtype="category")
        api_bytes = int(s.memory_usage(deep=True))
        dtype_repr = str(s.dtype)
        storage_repr = getattr(s.dtype, "storage", None)
    elif args.lib == "pandas_pyarrow":
        # pandas_defaultと全く同じ呼び出し(dtype未指定)だが、pyarrowが
        # インストールされた環境で実行することで、pandas 3.0がデフォルトで
        # 選ぶstorage="pyarrow"のStringDtypeを計測する。
        s = pd.Series(strings)
        api_bytes = int(s.memory_usage(deep=True))
        dtype_repr = str(s.dtype)
        storage_repr = getattr(s.dtype, "storage", None)
    elif args.lib == "polars":
        s = pl.Series(strings)
        api_bytes = int(s.estimated_size())
        dtype_repr = str(s.dtype)

    after_rss = rss_bytes()

    result = {
        "lib": args.lib,
        "n": args.n,
        "length": args.length,
        "api_bytes": api_bytes,
        "api_bytes_per_elem": api_bytes / args.n,
        "rss_delta_bytes": after_rss - baseline_rss,
        "rss_delta_per_elem": (after_rss - baseline_rss) / args.n,
        "dtype_repr": dtype_repr,
        "storage_repr": storage_repr,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
