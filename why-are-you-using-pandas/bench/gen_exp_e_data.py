"""実験E用のparquetデータ生成: 列数を多め(40列)にした模擬テーブルを行数別に書き出す。

実験Eは「読み込み→少数列選択→行フィルタ→集計」パイプラインでの
predicate pushdown / projection pushdownの効果を見るため、以下の列構成にしている:
  - id: 行ID(int64)
  - group_key: 集計キー(int64, カーディナリティ N_GROUPS)
  - filter_col: フィルタ条件に使うfloat64(一様分布。>0.9で行の約10%が残る)
  - v0..v{N_NUMERIC_NOISE-1}: クエリで使わないfloat64ノイズ列
  - s0..s{N_STRING_NOISE-1}: クエリで使わない文字列列。projection pushdownで
    読み飛ばされることを期待する列(文字列のデコード/コピーは列読み込みの中でも
    特にコストが高いため、読み飛ばせるかどうかの差が出やすい)。
合計 3 + N_NUMERIC_NOISE + N_STRING_NOISE = 40列。

事前生成専用スクリプト。計測スクリプト(exp_e_single.py)は生成済みの
parquetファイルを読むだけで、ファイル生成そのものは計時対象に含めない。
row_group_size をあえて小さめに固定し、行数が増えるほど行グループ数も
増えるようにしている(parquetの行グループ単位でのpruningが効く条件を作るため)。
"""
import argparse
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import make_unique_strings  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data", "exp_e")

N_GROUPS = 20
N_NUMERIC_NOISE = 27
N_STRING_NOISE = 10
STRING_LENGTH = 16
ROW_GROUP_SIZE = 50_000

# 本計測用(数十万〜数百万行)。1回の計測(各手法の1回のパイプライン実行)が
# 数秒以内に収まることを想定したサイズ設計。
NS_FULL = [100_000, 300_000, 1_000_000, 3_000_000]
# スモークテスト用: パイプライン(生成→計測→CSV→グラフ)の動作確認だけが目的の極小サイズ。
NS_SMOKE = [2_000, 5_000]


def data_path(n: int) -> str:
    return os.path.join(DATA_DIR, f"n_{n}.parquet")


def make_df(n: int, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "id": np.arange(n, dtype=np.int64),
        "group_key": rng.integers(0, N_GROUPS, size=n, dtype=np.int64),
        "filter_col": rng.random(n),
    }
    for i in range(N_NUMERIC_NOISE):
        data[f"v{i}"] = rng.random(n)

    # 文字列ノイズ列は内容そのものはクエリで使わない(列を読み飛ばせるかどうかの
    # 差を見るためのダミー)。生成コストを抑えるため同じ文字列リストを使い回す。
    strings = make_unique_strings(n, STRING_LENGTH)
    for i in range(N_STRING_NOISE):
        data[f"s{i}"] = strings

    return pl.DataFrame(data)


def generate(ns, force=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    for n in ns:
        path = data_path(n)
        if os.path.exists(path) and not force:
            print(f"skip (exists): {path}")
            continue
        df = make_df(n)
        df.write_parquet(path, row_group_size=ROW_GROUP_SIZE)
        print(f"wrote {path} ({n} rows, {len(df.columns)} cols)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="スモークテスト用の小さい行数だけ生成する")
    ap.add_argument("--force", action="store_true", help="既存ファイルがあっても再生成する")
    args = ap.parse_args()
    ns = NS_SMOKE if args.smoke else NS_FULL
    generate(ns, force=args.force)


if __name__ == "__main__":
    main()
