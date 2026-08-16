"""実験Fのドライバ: pandasと、スレッド数を振ったpolarsの各条件をサブプロセスとして実行しCSVに集約する。

POLARS_MAX_THREADSはpolarsのimport時に読み込まれる設定のため、1プロセス内で使い回すと
2つ目以降の条件でスレッド数が反映されない。そのため条件(スレッド数)ごとにexp_f_single.pyを
別プロセスとして起動する設計にしている。

パラメータはフルサイズ(本計測用)がデフォルト。動作確認だけしたい場合は
`--smoke` を付けるとごく小さいデータサイズ・reps=3で高速に流せる
(CSV/グラフ生成パイプラインの検証用。フルサイズの結果とは値が別物なので混同しないこと)。
"""
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 本計測用パラメータ(フルサイズ): 数百万行、グループキーのカーディナリティ中程度。
N_FULL = 5_000_000
N_GROUPS_FULL = 1_000
REPS_FULL = 7

# スモークテスト用パラメータ: パイプライン検証だけが目的なのでごく小さいサイズ・回数にする。
N_SMOKE = 50_000
N_GROUPS_SMOKE = 100
REPS_SMOKE = 3

THREADS = [1, 2, 4, 8, 16, 32]

FIELDNAMES = [
    "lib",
    "n",
    "n_groups",
    "reps",
    "threads_requested",
    "threads_effective",
    "median_sec",
    "min_sec",
    "max_sec",
]


def run(lib, n, n_groups, reps, threads=0):
    cmd = [
        sys.executable,
        os.path.join(HERE, "exp_f_single.py"),
        "--lib",
        lib,
        "--n",
        str(n),
        "--n-groups",
        str(n_groups),
        "--reps",
        str(reps),
    ]
    if threads:
        cmd += ["--threads", str(threads)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    smoke = "--smoke" in sys.argv
    n = N_SMOKE if smoke else N_FULL
    n_groups = N_GROUPS_SMOKE if smoke else N_GROUPS_FULL
    reps = REPS_SMOKE if smoke else REPS_FULL

    rows = []

    # pandas(numpyバックエンド)はスレッド数を振らず1回だけ計測し、水平線の比較対象として使う。
    data = run("pandas", n, n_groups, reps)
    rows.append(data)
    print(f"done: pandas n={n} n_groups={n_groups} -> median {data['median_sec']*1000:.3f} ms")

    for threads in THREADS:
        data = run("polars", n, n_groups, reps, threads=threads)
        rows.append(data)
        print(
            f"done: polars threads={threads} (effective={data['threads_effective']}) "
            f"n={n} n_groups={n_groups} -> median {data['median_sec']*1000:.3f} ms"
        )

    out_path = os.path.join(ROOT, "results", "exp_f.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in FIELDNAMES})
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
