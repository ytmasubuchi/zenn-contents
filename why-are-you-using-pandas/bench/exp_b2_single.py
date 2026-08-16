"""実験B2: 逐次的な列追加(特徴量エンジニアリングの実務パターン)。1ライブラリ分を実行しJSONを出力。

計測設計:
- 追加する150列分のnumpy配列は計時窓の外で事前生成する。乱数生成のコストは
  直前のメモリ解放パターン(アロケータの状態)に依存して大きく揺れ、ライブラリ本体の
  差を飲み込んでしまうため。ループ・バッチとも「配列が揃った状態から」を計測起点とする。
- ループ: 1列ずつの挿入(pandas: df[col]=arr / polars: with_columns)のみを計時。
- バッチ: dict→DataFrame構築+横concatまで(=配列群から最終DataFrameを得るまで)を計時。
  pandasはこの構築時点で同dtype列のブロック統合(物理コピー)が発生するため、
  concat単体だけを計時するとCoWの遅延コピーにより実コストが見えなくなる。
- 上記シーケンス(ループ→断片化sum→copy→統合後sum→バッチ)を--reps回繰り返し、
  各計測値の中央値を報告する(B-1やC・Dと同じ流儀)。
"""
import argparse
import gc
import json
import statistics
import time
import warnings

import numpy as np


def timed(fn):
    gc.collect()
    gc.disable()
    t0 = time.perf_counter()
    ret = fn()
    t1 = time.perf_counter()
    gc.enable()
    return t1 - t0, ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", choices=["pandas", "polars", "pandas_pyarrow"], required=True)
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--start-cols", type=int, default=5)
    ap.add_argument("--add-cols", type=int, default=150)
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()

    n = args.n
    rng = np.random.default_rng(42)
    result = {
        "lib": args.lib,
        "n": n,
        "start_cols": args.start_cols,
        "add_cols": args.add_cols,
        "reps": args.reps,
    }

    loop_totals = []
    sum_frag_times = []
    copy_times = []
    sum_consol_times = []
    batch_times = []

    if args.lib == "pandas":
        import pandas as pd

        for rep in range(args.reps):
            new_arrays = {f"f{i}": rng.random(n) for i in range(args.add_cols)}
            df = pd.DataFrame({f"c{i}": rng.random(n) for i in range(args.start_cols)})
            iter_times = []
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                for name, arr in new_arrays.items():
                    t0 = time.perf_counter()
                    df[name] = arr
                    t1 = time.perf_counter()
                    iter_times.append(t1 - t0)
                frag_warnings = [str(w.message) for w in wlist if "fragmented" in str(w.message).lower()]
            loop_totals.append(sum(iter_times))
            if rep == 0:
                result["iter_times"] = iter_times
                result["fragmentation_warning_seen"] = len(frag_warnings) > 0
                result["fragmentation_warning_text"] = frag_warnings[0] if frag_warnings else None
                result["n_blocks_after"] = len(df._mgr.blocks) if hasattr(df, "_mgr") else None

            t, _ = timed(lambda: df.sum().sum())
            sum_frag_times.append(t)

            t, df_c = timed(lambda: df.copy())
            copy_times.append(t)
            if rep == 0:
                result["n_blocks_after_copy"] = len(df_c._mgr.blocks) if hasattr(df_c, "_mgr") else None
            t, _ = timed(lambda: df_c.sum().sum())
            sum_consol_times.append(t)
            del df, df_c

            base = pd.DataFrame({f"c{i}": rng.random(n) for i in range(args.start_cols)})
            t, _ = timed(lambda: pd.concat([base, pd.DataFrame(new_arrays)], axis=1))
            batch_times.append(t)
            del base, new_arrays

    elif args.lib == "pandas_pyarrow":
        import pandas as pd

        for rep in range(args.reps):
            new_arrays = {f"f{i}": rng.random(n) for i in range(args.add_cols)}
            df = pd.DataFrame(
                {f"c{i}": pd.array(rng.random(n), dtype="float64[pyarrow]") for i in range(args.start_cols)}
            )
            iter_times = []
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                for name, arr in new_arrays.items():
                    t0 = time.perf_counter()
                    df[name] = pd.array(arr, dtype="float64[pyarrow]")
                    t1 = time.perf_counter()
                    iter_times.append(t1 - t0)
                frag_warnings = [str(w.message) for w in wlist if "fragmented" in str(w.message).lower()]
            loop_totals.append(sum(iter_times))
            if rep == 0:
                result["iter_times"] = iter_times
                result["fragmentation_warning_seen"] = len(frag_warnings) > 0
                result["fragmentation_warning_text"] = frag_warnings[0] if frag_warnings else None
                result["n_blocks_after"] = len(df._mgr.blocks) if hasattr(df, "_mgr") else None
                result["dtype_repr"] = str(df.dtypes.iloc[0])

            t, _ = timed(lambda: df.sum().sum())
            sum_frag_times.append(t)

            t, df_c = timed(lambda: df.copy())
            copy_times.append(t)
            if rep == 0:
                result["n_blocks_after_copy"] = len(df_c._mgr.blocks) if hasattr(df_c, "_mgr") else None
            t, _ = timed(lambda: df_c.sum().sum())
            sum_consol_times.append(t)
            del df, df_c

            base = pd.DataFrame(
                {f"c{i}": pd.array(rng.random(n), dtype="float64[pyarrow]") for i in range(args.start_cols)}
            )
            t, _ = timed(
                lambda: pd.concat(
                    [base, pd.DataFrame({k: pd.array(v, dtype="float64[pyarrow]") for k, v in new_arrays.items()})],
                    axis=1,
                )
            )
            batch_times.append(t)
            del base, new_arrays

    else:
        import polars as pl

        for rep in range(args.reps):
            new_arrays = {f"f{i}": rng.random(n) for i in range(args.add_cols)}
            df = pl.DataFrame({f"c{i}": rng.random(n) for i in range(args.start_cols)})
            iter_times = []
            for name, arr in new_arrays.items():
                t0 = time.perf_counter()
                df = df.with_columns(pl.Series(name, arr))
                t1 = time.perf_counter()
                iter_times.append(t1 - t0)
            loop_totals.append(sum(iter_times))
            if rep == 0:
                result["iter_times"] = iter_times
                result["fragmentation_warning_seen"] = False
                result["fragmentation_warning_text"] = None
                result["n_blocks_after"] = None
                result["n_blocks_after_copy"] = None

            t, _ = timed(lambda: df.select(pl.all().sum()))
            sum_frag_times.append(t)
            copy_times.append(None)
            sum_consol_times.append(None)
            del df

            base = pl.DataFrame({f"c{i}": rng.random(n) for i in range(args.start_cols)})
            t, _ = timed(lambda: pl.concat([base, pl.DataFrame(new_arrays)], how="horizontal"))
            batch_times.append(t)
            del base, new_arrays

    def med(xs):
        vals = [x for x in xs if x is not None]
        return statistics.median(vals) if vals else None

    result["loop_add_time_total_sec"] = med(loop_totals)
    result["loop_add_time_total_all_sec"] = loop_totals
    result["sum_time_fragmented_sec"] = med(sum_frag_times)
    result["sum_time_fragmented_all_sec"] = sum_frag_times
    result["copy_time_sec"] = med(copy_times)
    result["copy_time_all_sec"] = copy_times
    result["sum_time_consolidated_sec"] = med(sum_consol_times)
    result["sum_time_consolidated_all_sec"] = sum_consol_times
    result["batch_add_time_sec"] = med(batch_times)
    result["batch_add_time_all_sec"] = batch_times
    print(json.dumps(result))


if __name__ == "__main__":
    main()
