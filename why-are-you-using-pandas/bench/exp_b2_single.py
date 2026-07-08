"""実験B2: 逐次的な列追加(特徴量エンジニアリングの実務パターン)。1ライブラリ分を実行しJSONを出力。"""
import argparse
import gc
import json
import time
import warnings

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", choices=["pandas", "polars"], required=True)
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--start-cols", type=int, default=5)
    ap.add_argument("--add-cols", type=int, default=150)
    args = ap.parse_args()

    n = args.n
    rng = np.random.default_rng(42)
    result = {"lib": args.lib, "n": n, "start_cols": args.start_cols, "add_cols": args.add_cols}

    if args.lib == "pandas":
        import pandas as pd

        df = pd.DataFrame({f"c{i}": rng.random(n) for i in range(args.start_cols)})
        iter_times = []
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            for i in range(args.add_cols):
                t0 = time.perf_counter()
                df[f"f{i}"] = rng.random(n)
                t1 = time.perf_counter()
                iter_times.append(t1 - t0)
            frag_warnings = [str(w.message) for w in wlist if "fragmented" in str(w.message).lower()]
        result["iter_times"] = iter_times
        result["fragmentation_warning_seen"] = len(frag_warnings) > 0
        result["fragmentation_warning_text"] = frag_warnings[0] if frag_warnings else None
        result["n_blocks_after"] = len(df._mgr.blocks) if hasattr(df, "_mgr") else None

        gc.collect()
        t0 = time.perf_counter()
        _ = df.sum().sum()
        t1 = time.perf_counter()
        result["sum_time_fragmented_sec"] = t1 - t0

        df_c = df.copy()
        result["n_blocks_after_copy"] = len(df_c._mgr.blocks) if hasattr(df_c, "_mgr") else None
        gc.collect()
        t0 = time.perf_counter()
        _ = df_c.sum().sum()
        t1 = time.perf_counter()
        result["sum_time_consolidated_sec"] = t1 - t0

        base = pd.DataFrame({f"c{i}": rng.random(n) for i in range(args.start_cols)})
        new_cols = pd.DataFrame({f"f{i}": rng.random(n) for i in range(args.add_cols)})
        gc.collect()
        t0 = time.perf_counter()
        merged = pd.concat([base, new_cols], axis=1)
        t1 = time.perf_counter()
        result["batch_add_time_sec"] = t1 - t0

    else:
        import polars as pl

        df = pl.DataFrame({f"c{i}": rng.random(n) for i in range(args.start_cols)})
        iter_times = []
        for i in range(args.add_cols):
            t0 = time.perf_counter()
            df = df.with_columns(pl.Series(f"f{i}", rng.random(n)))
            t1 = time.perf_counter()
            iter_times.append(t1 - t0)
        result["iter_times"] = iter_times
        result["fragmentation_warning_seen"] = False
        result["fragmentation_warning_text"] = None
        result["n_blocks_after"] = None

        gc.collect()
        t0 = time.perf_counter()
        _ = df.select(pl.all().sum())
        t1 = time.perf_counter()
        result["sum_time_fragmented_sec"] = t1 - t0
        result["sum_time_consolidated_sec"] = None

        base = pl.DataFrame({f"c{i}": rng.random(n) for i in range(args.start_cols)})
        new_cols = pl.DataFrame({f"f{i}": rng.random(n) for i in range(args.add_cols)})
        gc.collect()
        t0 = time.perf_counter()
        merged = pl.concat([base, new_cols], how="horizontal")
        t1 = time.perf_counter()
        result["batch_add_time_sec"] = t1 - t0

    result["loop_add_time_total_sec"] = sum(result["iter_times"])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
