"""実験B1: 統合済みブロックからの列削除コスト(単一条件、サブプロセスとして実行)"""
import argparse
import gc
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", choices=["pandas", "polars"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--threads", type=int, default=0, help="polars専用: POLARS_MAX_THREADSを固定する場合に指定")
    args = ap.parse_args()

    if args.threads:
        os.environ["POLARS_MAX_THREADS"] = str(args.threads)

    ncols = 101
    rng = np.random.default_rng(42)
    data = rng.random((args.n, ncols))
    colnames = [f"c{i}" for i in range(ncols)]

    times = []
    if args.lib == "pandas":
        import pandas as pd

        for _ in range(args.reps):
            df = pd.DataFrame(data, columns=colnames, copy=False)
            gc.collect()
            gc.disable()
            t0 = time.perf_counter()
            df2 = df.drop(columns=["c50"])
            t1 = time.perf_counter()
            gc.enable()
            times.append(t1 - t0)
            del df, df2
    else:
        import polars as pl

        for _ in range(args.reps):
            df = pl.DataFrame(data, schema=colnames)
            gc.collect()
            gc.disable()
            t0 = time.perf_counter()
            df2 = df.drop("c50")
            t1 = time.perf_counter()
            gc.enable()
            times.append(t1 - t0)
            del df, df2

    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    print(
        json.dumps(
            {
                "lib": args.lib,
                "n": args.n,
                "reps": args.reps,
                "threads": args.threads or None,
                "median_sec": median,
                "min_sec": times_sorted[0],
                "max_sec": times_sorted[-1],
            }
        )
    )


if __name__ == "__main__":
    main()
