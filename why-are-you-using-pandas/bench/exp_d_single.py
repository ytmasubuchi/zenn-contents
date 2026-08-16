"""実験D: polars DataFrame -> pandas DataFrame への変換コスト。単一条件、サブプロセスとして実行。

polars.DataFrame.to_pandas()は内部でpyarrow経由の変換を行うため、pyarrowが
インストールされた環境でのみ実行できる(他の実験とは別のDockerイメージを使う)。

use_pyarrow_extension_array=Falseは「pandas側でArrow配列をboxingしてPython
オブジェクトのnumpy配列に展開する」パス(文字列列で重くなることが予想される)、
Trueは「pandas側もArrowDtypeのまま受け取る」パス(ゼロコピーに近いことが予想される)。
"""
import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import make_unique_strings  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", required=True, choices=["numeric", "string"])
    ap.add_argument("--use_pyarrow_ext", action="store_true")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--length", type=int, default=20, help="--cols=string の場合の文字列長")
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()

    import numpy as np
    import polars as pl

    rng = np.random.default_rng(42)
    times = []

    for _ in range(args.reps):
        if args.cols == "numeric":
            df = pl.DataFrame({"v": rng.random(args.n)})
        else:
            df = pl.DataFrame({"v": make_unique_strings(args.n, args.length)})

        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        pdf = df.to_pandas(use_pyarrow_extension_array=args.use_pyarrow_ext)
        t1 = time.perf_counter()
        gc.enable()
        times.append(t1 - t0)
        del df, pdf

    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    print(
        json.dumps(
            {
                "cols": args.cols,
                "use_pyarrow_ext": args.use_pyarrow_ext,
                "n": args.n,
                "length": args.length,
                "reps": args.reps,
                "median_sec": median,
                "min_sec": times_sorted[0],
                "max_sec": times_sorted[-1],
            }
        )
    )


if __name__ == "__main__":
    main()
