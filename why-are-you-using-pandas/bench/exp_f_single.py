"""実験F: シングルスレッド vs マルチスレッド(スレッドスケーリング)。単一条件を計測してJSON1行を標準出力に出す。

測りたい本質: pandasの演算は基本シングルスレッド、polarsはマルチスレッド前提という
実行モデルの違いと、コア数に対するスケーリング。

計測設計:
- ワークロード: groupby集計。N行、カーディナリティ中程度(n_groups)のキー列と
  float64の値列2本を持つDataFrameを作り、キーごとにsum/meanを集計する
  (pandas: groupby(...).agg(...) / polars: group_by(...).agg(...))。
- POLARS_MAX_THREADSはpolarsのimport時に読み込まれる設定のため、スレッド数の
  条件ごとに別プロセス(subprocess)で計測する必要がある。本スクリプトはCLIから
  1条件(1lib × 1スレッド数)だけを実行する設計にし、ドライバ(run_exp_f.py)側で
  条件ごとにsubprocessを起動する。
- pandas(numpyバックエンド)はgroupby集計自体はCython実装でBLASを使わないが、
  環境によってはnumpyの他の演算がOpenBLAS/MKL/OpenMP経由で暗黙に並列化される
  ことがあるため、比較を公平にする目的でOMP_NUM_THREADS等のスレッド数を1に
  固定した状態で計測する(pandas・polars双方に同じ環境変数を設定するが、
  polars自身のスレッドプールはPOLARS_MAX_THREADSで別途制御する)。
- データ生成(乱数配列・DataFrame構築)は計時窓の外で行い、集計処理のみを計時する
  (他の実験と同じ流儀)。--reps回繰り返し中央値を報告する。
"""
import argparse
import gc
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", choices=["pandas", "polars"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--n-groups", type=int, required=True)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help="polars専用: POLARS_MAX_THREADSに設定するスレッド数(0の場合はpolarsのデフォルト挙動に任せる)",
    )
    args = ap.parse_args()

    # BLAS/OpenMP系のスレッド数はライブラリのimport時に読まれるため、import前に固定する。
    # pandasのgroupby集計自体はほぼ影響を受けないはずだが、公平のため両libで共通に固定する。
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[var] = "1"

    # POLARS_MAX_THREADSも同様にimport時に読まれる設定のため、polars import前に固定する。
    if args.lib == "polars" and args.threads:
        os.environ["POLARS_MAX_THREADS"] = str(args.threads)

    import numpy as np

    rng = np.random.default_rng(42)
    times = []
    threads_effective = None

    if args.lib == "pandas":
        import pandas as pd

        for _ in range(args.reps):
            keys = rng.integers(0, args.n_groups, args.n)
            v1 = rng.random(args.n)
            v2 = rng.random(args.n)
            df = pd.DataFrame({"key": keys, "v1": v1, "v2": v2})
            gc.collect()
            gc.disable()
            t0 = time.perf_counter()
            res = df.groupby("key", sort=False).agg(
                v1_sum=("v1", "sum"),
                v1_mean=("v1", "mean"),
                v2_sum=("v2", "sum"),
                v2_mean=("v2", "mean"),
            )
            t1 = time.perf_counter()
            gc.enable()
            times.append(t1 - t0)
            del df, res
    else:
        import polars as pl

        threads_effective = pl.thread_pool_size()
        for _ in range(args.reps):
            keys = rng.integers(0, args.n_groups, args.n)
            v1 = rng.random(args.n)
            v2 = rng.random(args.n)
            df = pl.DataFrame({"key": keys, "v1": v1, "v2": v2})
            gc.collect()
            gc.disable()
            t0 = time.perf_counter()
            res = df.group_by("key").agg(
                pl.col("v1").sum().alias("v1_sum"),
                pl.col("v1").mean().alias("v1_mean"),
                pl.col("v2").sum().alias("v2_sum"),
                pl.col("v2").mean().alias("v2_mean"),
            )
            t1 = time.perf_counter()
            gc.enable()
            times.append(t1 - t0)
            del df, res

    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    print(
        json.dumps(
            {
                "lib": args.lib,
                "n": args.n,
                "n_groups": args.n_groups,
                "reps": args.reps,
                "threads_requested": args.threads or None,
                "threads_effective": threads_effective,
                "median_sec": median,
                "min_sec": times_sorted[0],
                "max_sec": times_sorted[-1],
            }
        )
    )


if __name__ == "__main__":
    main()
