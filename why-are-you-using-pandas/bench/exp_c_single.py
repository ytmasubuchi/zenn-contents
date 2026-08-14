"""実験C: numpy変換(.to_numpy())のコスト。単一条件、サブプロセスとして実行。

pandasは常にゼロコピーが期待できる一方、polarsは
「欠損なし・単一チャンク」というゼロコピー条件が崩れると
(concat等で複数チャンクになる/欠損値を含む)コピーが発生し得る、
という構造的な違いを見る。
"""
import argparse
import gc
import json
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lib",
        required=True,
        choices=["pandas", "polars_single_chunk", "polars_multi_chunk", "polars_nulls"],
    )
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    times = []
    n_chunks = None

    if args.lib == "pandas":
        import pandas as pd

        for _ in range(args.reps):
            s = pd.Series(rng.random(args.n))
            gc.collect()
            gc.disable()
            t0 = time.perf_counter()
            arr = s.to_numpy()
            t1 = time.perf_counter()
            gc.enable()
            times.append(t1 - t0)
            del s, arr

    elif args.lib == "polars_single_chunk":
        import polars as pl

        for _ in range(args.reps):
            s = pl.Series(rng.random(args.n))
            n_chunks = s.n_chunks()
            gc.collect()
            gc.disable()
            t0 = time.perf_counter()
            arr = s.to_numpy()
            t1 = time.perf_counter()
            gc.enable()
            times.append(t1 - t0)
            del s, arr

    elif args.lib == "polars_multi_chunk":
        import polars as pl

        n_parts = 10
        part_len = args.n // n_parts
        for _ in range(args.reps):
            parts = [pl.Series(rng.random(part_len)) for _ in range(n_parts)]
            s = pl.concat(parts, rechunk=False)
            n_chunks = s.n_chunks()
            gc.collect()
            gc.disable()
            t0 = time.perf_counter()
            arr = s.to_numpy()
            t1 = time.perf_counter()
            gc.enable()
            times.append(t1 - t0)
            del s, arr, parts

    elif args.lib == "polars_nulls":
        import polars as pl

        for _ in range(args.reps):
            values = rng.random(args.n).tolist()
            for i in range(0, args.n, 1000):  # 0.1%だけ欠損させる
                values[i] = None
            s = pl.Series(values)
            n_chunks = s.n_chunks()
            gc.collect()
            gc.disable()
            t0 = time.perf_counter()
            arr = s.to_numpy()
            t1 = time.perf_counter()
            gc.enable()
            times.append(t1 - t0)
            del s, arr

    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    print(
        json.dumps(
            {
                "lib": args.lib,
                "n": args.n,
                "reps": args.reps,
                "n_chunks": n_chunks,
                "median_sec": median,
                "min_sec": times_sorted[0],
                "max_sec": times_sorted[-1],
            }
        )
    )


if __name__ == "__main__":
    main()
