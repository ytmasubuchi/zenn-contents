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
        choices=["pandas_object", "pandas_arrow", "pandas_category", "polars"],
    )
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--length", type=int, required=True)
    args = ap.parse_args()

    strings = make_unique_strings(args.n, args.length)
    baseline_rss = rss_bytes()

    api_bytes = None
    if args.lib == "pandas_object":
        import pandas as pd

        s = pd.Series(strings, dtype=object)
        api_bytes = int(s.memory_usage(deep=True))
    elif args.lib == "pandas_arrow":
        import pandas as pd

        s = pd.Series(strings, dtype="string[pyarrow]")
        api_bytes = int(s.memory_usage(deep=True))
    elif args.lib == "pandas_category":
        import pandas as pd

        s = pd.Series(strings, dtype="category")
        api_bytes = int(s.memory_usage(deep=True))
    elif args.lib == "polars":
        import polars as pl

        s = pl.Series(strings)
        api_bytes = int(s.estimated_size())

    after_rss = rss_bytes()

    result = {
        "lib": args.lib,
        "n": args.n,
        "length": args.length,
        "api_bytes": api_bytes,
        "api_bytes_per_elem": api_bytes / args.n,
        "rss_delta_bytes": after_rss - baseline_rss,
        "rss_delta_per_elem": (after_rss - baseline_rss) / args.n,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
