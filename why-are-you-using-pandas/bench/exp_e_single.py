"""実験E: 即時評価(pandas / polars eager) vs 遅延評価(polars lazy)。

「parquet読み込み→少数列選択→行フィルタ→集計」というパイプラインで、
predicate pushdown(行フィルタをparquetの読み込み段階まで押し下げる)/
projection pushdown(選択していない列を読み込み段階でスキップする)の
効果を計測する(単一条件、サブプロセスとして実行)。

4条件:
  - pandas_naive : read_parquet()で全列読み込み → フィルタ → 集計
  - pandas_manual: read_parquet(columns=...)で手動projection → フィルタ → 集計
                   (フィルタ自体はpandas側で行う。read_parquetのfilters引数は
                   使わない = あくまで「列選択だけ手で最適化した場合」を見る)
  - polars_eager : read_parquet()で全列読み込み → filter → select → 集計
  - polars_lazy  : scan_parquet() → filter → select → 集計 → collect()
                   (pushdownはpolarsのクエリオプティマイザが自動で適用する)

pandasのparquet読み込みはpyarrowを要求するため、この実験はpyarrow入り環境
(Dockerfile.pyarrow)で実行する。

各手法が同じ集計結果(group_keyごとのv0/v1平均とカウント)を返すことを
前提にしており、result_checksum()でその整合性を目視確認できるようにしている。
"""
import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.gen_exp_e_data import data_path  # noqa: E402

FILTER_THRESHOLD = 0.9
# pandas_manual が read_parquet(columns=...) で実際に読み込む列
# (フィルタに使うfilter_colと、集計に使うgroup_key/v0/v1のみ)
NEEDED_COLS = ["group_key", "filter_col", "v0", "v1"]


def run_pandas_naive(path):
    import pandas as pd

    df = pd.read_parquet(path)
    df = df[df["filter_col"] > FILTER_THRESHOLD]
    df = df[["group_key", "v0", "v1"]]
    agg = df.groupby("group_key").agg(v0_mean=("v0", "mean"), v1_mean=("v1", "mean"), cnt=("v0", "size"))
    return agg


def run_pandas_manual(path):
    import pandas as pd

    df = pd.read_parquet(path, columns=NEEDED_COLS)
    df = df[df["filter_col"] > FILTER_THRESHOLD]
    df = df[["group_key", "v0", "v1"]]
    agg = df.groupby("group_key").agg(v0_mean=("v0", "mean"), v1_mean=("v1", "mean"), cnt=("v0", "size"))
    return agg


def run_polars_eager(path):
    import polars as pl

    df = pl.read_parquet(path)
    df = df.filter(pl.col("filter_col") > FILTER_THRESHOLD).select(["group_key", "v0", "v1"])
    agg = df.group_by("group_key").agg(
        pl.col("v0").mean().alias("v0_mean"),
        pl.col("v1").mean().alias("v1_mean"),
        pl.len().alias("cnt"),
    )
    return agg


def build_lazy_query(path):
    """polars_lazy専用の遅延クエリ(run_exp_e.pyのexplain保存からも再利用する)"""
    import polars as pl

    lf = pl.scan_parquet(path)
    return (
        lf.filter(pl.col("filter_col") > FILTER_THRESHOLD)
        .select(["group_key", "v0", "v1"])
        .group_by("group_key")
        .agg(
            pl.col("v0").mean().alias("v0_mean"),
            pl.col("v1").mean().alias("v1_mean"),
            pl.len().alias("cnt"),
        )
    )


def run_polars_lazy(path):
    return build_lazy_query(path).collect()


RUNNERS = {
    "pandas_naive": run_pandas_naive,
    "pandas_manual": run_pandas_manual,
    "polars_eager": run_polars_eager,
    "polars_lazy": run_polars_lazy,
}


def result_checksum(method, result):
    """手法間で集計結果が一致しているかの目視確認用(group数と平均値の合計)。
    集計順序はgroup_byの実装により手法ごとに異なり得るが、合計値は順序に依存しないため
    突き合わせに使える(浮動小数点の加算順序differenceによる末尾桁のブレはあり得る)。
    """
    if method.startswith("pandas"):
        v0_sum = float(result["v0_mean"].sum())
        n_rows = int(len(result))
    else:
        v0_sum = float(result["v0_mean"].sum())
        n_rows = int(result.height)
    return round(v0_sum, 6), n_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(RUNNERS.keys()))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()

    path = data_path(args.n)
    if not os.path.exists(path):
        raise SystemExit(f"data file not found: {path} (run gen_exp_e_data.py first)")

    runner = RUNNERS[args.method]

    # OSページキャッシュを温めるためのウォームアップ読み込み(計時対象外)。
    # cold-disk読み込みのばらつきではなく、pushdownによる読み込み/演算量の差を
    # 見たいための措置。
    _ = runner(path)

    times = []
    checksum = None
    n_result_rows = None
    for _ in range(args.reps):
        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        result = runner(path)
        t1 = time.perf_counter()
        gc.enable()
        times.append(t1 - t0)
        if checksum is None:
            checksum, n_result_rows = result_checksum(args.method, result)
        del result

    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    print(
        json.dumps(
            {
                "method": args.method,
                "n": args.n,
                "reps": args.reps,
                "median_sec": median,
                "min_sec": times_sorted[0],
                "max_sec": times_sorted[-1],
                "checksum_v0_mean_sum": checksum,
                "n_result_rows": n_result_rows,
            }
        )
    )


if __name__ == "__main__":
    main()
