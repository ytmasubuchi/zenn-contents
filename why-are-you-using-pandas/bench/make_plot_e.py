"""実験Eの結果からグラフを生成する。
pandas(即時評価: 全列読み込み / 手動列選択)と polars(即時評価 / 遅延評価)の
パイプライン(読み込み→列選択→行フィルタ→集計)所要時間を比較する。
pyarrow入り環境(Dockerfile.pyarrow)で実行すること。
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
IMAGES = os.path.join(ROOT, "images")
os.makedirs(IMAGES, exist_ok=True)

LABELS = {
    "pandas_naive": "pandas (read_parquet全列 → フィルタ → 集計)",
    "pandas_manual": "pandas (read_parquet(columns=)で手動最適化)",
    "polars_eager": "polars eager (read_parquet)",
    "polars_lazy": "polars lazy (scan_parquet, pushdown自動適用)",
}
# 既存グラフの配色規約を踏襲: 暖色系=pandas系(赤=素朴な条件, 橙=手動最適化した
# 条件)、寒色系=polars系(青=即時評価)。緑は他の実験でも「最適化が効いた
# 条件」を示す色として使っているため、polars_lazyにもそれを踏襲する。
COLORS = {
    "pandas_naive": "#d62728",
    "pandas_manual": "#ff7f0e",
    "polars_eager": "#1f77b4",
    "polars_lazy": "#2ca02c",
}
MARKERS = {
    "pandas_naive": "o",
    "pandas_manual": "s",
    "polars_eager": "D",
    "polars_lazy": "^",
}


def plot_exp_e():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "exp_e.csv"))))
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append((int(r["n"]), float(r["median_sec"]) * 1000))
    for v in by_method.values():
        v.sort()

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, pts in by_method.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker=MARKERS[method], color=COLORS[method], label=LABELS[method], linewidth=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("行数 N(対数軸)")
    ax.set_ylabel("読み込み→列選択→行フィルタ→集計の\n所要時間の中央値, ms(対数軸)")
    ax.set_title("即時評価 vs 遅延評価: predicate/projection pushdownの効果\n(40列のparquetから4列だけ使うクエリ)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(IMAGES, "exp_e_pipeline.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    plot_exp_e()
