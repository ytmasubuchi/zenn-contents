"""実験G: ストリーミング/アウトオブコア実行の実処理本体。

条件:
- pandas_full:      pd.read_csv()で全行を読み込んでからgroupby集計する
                     (pandasは読み込み時点で全データをDataFrameとして実体化する)
- pandas_chunked:    pd.read_csv(chunksize=...)でチャンクごとに読み、
                     チャンク単位でgroupby集計した部分結果(sum/count)を
                     Python dictに蓄積し、最後に結合する手動アウトオブコア集計
                     (pandasでも書けるが、部分集計の結合ロジックを自前で書く必要があり
                     コードが複雑になる、という比較点)
- polars_eager:      pl.read_csv()で全行を読み込んでからgroup_by集計する
- polars_streaming:  pl.scan_csv()でLazyFrameを作り、
                     collect(engine="streaming")でバッチ単位のストリーミング実行に委ねる

このスクリプトは常にexp_g_single.pyの子プロセスとして起動される。
ピークRSSはこのスクリプト自身では計測しない(呼び出し元がresource.getrusage(RUSAGE_CHILDREN)
でこのプロセス全体のピークRSSを計測するため、ここでは所要時間と結果件数のみを標準出力に返す)。

group列はpandas 3.0のデフォルト文字列dtype(pyarrow有無で挙動が変わる、実験Aを参照)の
影響を受けないよう、明示的にdtype=objectで読み込み、この実験の主題(ストリーミング実行 vs
全実体化)以外の要因を極力排除している。
"""
import argparse
import gc
import json
import time


def run_pandas_full(path):
    import pandas as pd

    df = pd.read_csv(path, dtype={"id": "int64", "group": "object", "value": "float64"})
    agg = df.groupby("group", observed=True)["value"].agg(["sum", "mean", "count"])
    return {
        "n_result_rows": len(agg),
        "total_sum": float(agg["sum"].sum()),
        "total_count": int(agg["count"].sum()),
    }


def run_pandas_chunked(path, chunksize):
    import pandas as pd

    sums = {}
    counts = {}
    reader = pd.read_csv(
        path,
        dtype={"id": "int64", "group": "object", "value": "float64"},
        chunksize=chunksize,
    )
    for chunk in reader:
        partial = chunk.groupby("group", observed=True)["value"].agg(["sum", "count"])
        for name, row in partial.iterrows():
            sums[name] = sums.get(name, 0.0) + float(row["sum"])
            counts[name] = counts.get(name, 0) + int(row["count"])
    return {
        "n_result_rows": len(sums),
        "total_sum": float(sum(sums.values())),
        "total_count": int(sum(counts.values())),
    }


def run_polars_eager(path):
    import polars as pl

    df = pl.read_csv(path)
    agg = df.group_by("group").agg(
        pl.col("value").sum().alias("sum"),
        pl.col("value").mean().alias("mean"),
        pl.col("value").count().alias("count"),
    )
    return {
        "n_result_rows": agg.height,
        "total_sum": float(agg["sum"].sum()),
        "total_count": int(agg["count"].sum()),
    }


def run_polars_streaming(path):
    import polars as pl

    lf = pl.scan_csv(path)
    agg = (
        lf.group_by("group")
        .agg(
            pl.col("value").sum().alias("sum"),
            pl.col("value").mean().alias("mean"),
            pl.col("value").count().alias("count"),
        )
        .collect(engine="streaming")
    )
    return {
        "n_result_rows": agg.height,
        "total_sum": float(agg["sum"].sum()),
        "total_count": int(agg["count"].sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--condition",
        required=True,
        choices=["pandas_full", "pandas_chunked", "polars_eager", "polars_streaming"],
    )
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    args = ap.parse_args()

    # このプロセスはconditionにつき1回しか実行しない(replicationはexp_g_single.pyを
    # 都度新しいプロセスとして起動することで実現している)ため、ここでのgc.disable()は
    # 行わない。数秒〜数十分かかるI/Oバウンドなマクロベンチマークであり、GCの一時停止が
    # 所要時間計測のノイズになるほど短時間ではなく、むしろgc無効化はピークメモリ計測(この
    # 実験の主指標)を人為的に押し上げかねないため。
    gc.collect()
    t0 = time.perf_counter()
    if args.condition == "pandas_full":
        result = run_pandas_full(args.data_path)
    elif args.condition == "pandas_chunked":
        result = run_pandas_chunked(args.data_path, args.chunksize)
    elif args.condition == "polars_eager":
        result = run_polars_eager(args.data_path)
    else:
        result = run_polars_streaming(args.data_path)
    t1 = time.perf_counter()

    print(json.dumps({"elapsed_sec": t1 - t0, **result}))


if __name__ == "__main__":
    main()
