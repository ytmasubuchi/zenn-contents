"""実験Eのドライバ: データ生成 → 各条件(手法×行数)の計測 → CSV集約 → explainプラン保存。

即時評価(pandas naive/manual, polars eager)と遅延評価(polars lazy)の比較。
pandasのparquet読み込みはpyarrowを要求するため、pyarrow入り環境
(Dockerfile.pyarrow)で実行すること。他の実験と同様に、他ベンチマークと
CPUを競合させないよう単独で実行するのが望ましい。

環境変数 EXP_E_SMOKE=1 を設定すると、スモークテスト用の小さい行数・少ない
reps数だけで動作確認用に実行する(パイプライン/CSV/グラフ生成の疎通確認が目的で、
本計測の数値としては使わないこと)。

使用例:
  本計測: python bench/run_exp_e.py
  スモーク: EXP_E_SMOKE=1 python bench/run_exp_e.py
"""
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

sys.path.insert(0, ROOT)
from bench.gen_exp_e_data import NS_FULL, NS_SMOKE, data_path, generate  # noqa: E402
from bench.exp_e_single import FILTER_THRESHOLD, build_lazy_query  # noqa: E402

METHODS = ["pandas_naive", "pandas_manual", "polars_eager", "polars_lazy"]

SMOKE = os.environ.get("EXP_E_SMOKE") == "1"
NS = NS_SMOKE if SMOKE else NS_FULL
REPS = 3 if SMOKE else 7


def run(method, n):
    cmd = [
        sys.executable,
        os.path.join(HERE, "exp_e_single.py"),
        "--method",
        method,
        "--n",
        str(n),
        "--reps",
        str(REPS),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def save_explain(n):
    """polars lazyのクエリプランを最適化前後でテキスト保存する(記事掲載用)。"""
    path = data_path(n)
    q = build_lazy_query(path)

    out_path = os.path.join(ROOT, "results", "exp_e_explain.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# polars lazy .explain() for exp_e (n={n}, filter_col > {FILTER_THRESHOLD})\n\n")
        f.write("## optimized=False (最適化前の論理プラン)\n")
        f.write(q.explain(optimized=False))
        f.write("\n\n## optimized=True (最適化後: predicate/projection pushdownが適用された物理プラン)\n")
        f.write(q.explain(optimized=True))
        f.write("\n")
    print(f"wrote {out_path}")


def main():
    print(f"mode: {'SMOKE' if SMOKE else 'FULL'} NS={NS} reps={REPS}")
    generate(NS)  # 未生成のファイルだけ生成する(既存があればスキップ)

    rows = []
    for n in NS:
        for method in METHODS:
            data = run(method, n)
            rows.append(data)
            print(
                f"done: {method} n={n} -> median {data['median_sec']*1000:.3f} ms "
                f"(checksum={data['checksum_v0_mean_sum']}, rows={data['n_result_rows']})"
            )

    out_path = os.path.join(ROOT, "results", "exp_e.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["method", "n", "reps", "median_sec", "min_sec", "max_sec", "checksum_v0_mean_sum", "n_result_rows"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})
    print(f"wrote {out_path}")

    save_explain(NS[-1])


if __name__ == "__main__":
    main()
