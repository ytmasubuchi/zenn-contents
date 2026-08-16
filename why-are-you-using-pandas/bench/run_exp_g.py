"""実験Gのドライバ: データ生成(未生成時のみ) → 各条件(4条件 × reps回)を
サブプロセスとして実行 → CSVに集約する。

条件:
  - pandas_full:     pd.read_csv()で全行読み込み → groupby集計
  - pandas_chunked:  pd.read_csv(chunksize=...)によるチャンク単位の手動アウトオブコア集計
  - polars_eager:    pl.read_csv()で全行読み込み → group_by集計
  - polars_streaming: pl.scan_csv() → collect(engine="streaming")によるストリーミング実行

各条件・各repは exp_g_single.py を毎回新規プロセスとして起動することで実行する。
exp_g_single.py はさらにその子として exp_g_worker.py を1回だけ起動し、
resource.getrusage(RUSAGE_CHILDREN) でそのworkerのピークRSSを計測する
(詳細はexp_g_single.pyのdocstring参照)。プロセスを毎回新規に起動するため、
条件間でメモリ状態が引き継がれる心配もない。

パラメータはフルサイズ(本計測用)がデフォルト。動作確認だけしたい場合は
`--smoke` を付けるとごく小さいデータサイズ・reps=3で高速に流せる
(計測パイプライン=データ生成→ピークRSS取得→CSV集約→グラフ化の疎通確認が目的。
フルサイズの結果とは値が別物なので混同しないこと)。

使用例:
  本計測: python bench/run_exp_g.py
  スモーク: python bench/run_exp_g.py --smoke
"""
import argparse
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

sys.path.insert(0, ROOT)
from bench.gen_data_g import data_path, generate  # noqa: E402

CONDITIONS = ["pandas_full", "pandas_chunked", "polars_eager", "polars_streaming"]

# 本計測用パラメータ(フルサイズ): ピークメモリの差が明確に出るよう、数千万行・数GB規模にする。
# 1行あたり概算24〜25バイト("id,group,value\n")なので、1億行でおよそ2.3〜2.5GB程度のCSVになる。
ROWS_FULL = 100_000_000
N_GROUPS_FULL = 200
CHUNKSIZE_FULL = 2_000_000
GEN_CHUNK_ROWS_FULL = 2_000_000
# フルサイズは1回あたり数GBの読み込み・集計を伴い、条件×repsぶん繰り返すと時間がかかるため、
# 他実験(reps=7が標準)よりも少なめにする。
REPS_FULL = 3

# スモークテスト用パラメータ: パイプラインの動作確認だけが目的のごく小さいサイズ・回数。
ROWS_SMOKE = 200_000
N_GROUPS_SMOKE = 50
CHUNKSIZE_SMOKE = 20_000
GEN_CHUNK_ROWS_SMOKE = 50_000
REPS_SMOKE = 3

FIELDNAMES = [
    "condition",
    "rep",
    "rows",
    "n_groups",
    "chunksize",
    "elapsed_sec",
    "peak_rss_bytes",
    "n_result_rows",
    "total_sum",
    "total_count",
]


def run_condition(condition, path, chunksize):
    cmd = [
        sys.executable,
        os.path.join(HERE, "exp_g_single.py"),
        "--condition",
        condition,
        "--data-path",
        path,
        "--chunksize",
        str(chunksize),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="スモークテスト用の小さいデータサイズ・reps数で実行する")
    ap.add_argument("--force-regen", action="store_true", help="既存のCSVがあっても再生成する")
    args = ap.parse_args()

    rows = ROWS_SMOKE if args.smoke else ROWS_FULL
    n_groups = N_GROUPS_SMOKE if args.smoke else N_GROUPS_FULL
    chunksize = CHUNKSIZE_SMOKE if args.smoke else CHUNKSIZE_FULL
    gen_chunk_rows = GEN_CHUNK_ROWS_SMOKE if args.smoke else GEN_CHUNK_ROWS_FULL
    reps = REPS_SMOKE if args.smoke else REPS_FULL

    print(
        f"mode: {'SMOKE' if args.smoke else 'FULL'} rows={rows} n_groups={n_groups} "
        f"chunksize={chunksize} reps={reps}"
    )

    path = data_path(rows)
    generate(path, n_rows=rows, n_groups=n_groups, gen_chunk_rows=gen_chunk_rows, force=args.force_regen)
    print(f"data: {path} ({os.path.getsize(path) / 1024**2:.1f} MiB)")

    result_rows = []
    for condition in CONDITIONS:
        for rep in range(reps):
            data = run_condition(condition, path, chunksize)
            data["rep"] = rep
            data["rows"] = rows
            data["n_groups"] = n_groups
            result_rows.append(data)
            print(
                f"done: {condition} rep={rep} -> {data['elapsed_sec'] * 1000:.1f} ms, "
                f"peak_rss={data['peak_rss_bytes'] / 1024**2:.1f} MiB, "
                f"result_rows={data['n_result_rows']}"
            )

    out_path = os.path.join(ROOT, "results", "exp_g.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in result_rows:
            writer.writerow({k: r.get(k) for k in FIELDNAMES})
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
