"""実験G(ストリーミング/アウトオブコア実行のピークメモリ比較)の結果からグラフを生成する。

この実験の主指標はピークRSSであり、行数方向のスケーリングを見る実験ではない
(1回の計測=固定の行数Nに対する4条件の比較)ため、他実験のような折れ線ではなく、
ピークRSSの棒グラフ(左)と所要時間の棒グラフ(右)の2パネル構成にする。
"""
import csv
import os
import statistics

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

CONDITIONS = ["pandas_full", "pandas_chunked", "polars_eager", "polars_streaming"]
LABELS = {
    "pandas_full": "pandas\n(read_csv全件 → groupby)",
    "pandas_chunked": "pandas\n(read_csv(chunksize=)\n手動アウトオブコア集計)",
    "polars_eager": "polars eager\n(read_csv → group_by)",
    "polars_streaming": 'polars lazy\n(scan_csv →\ncollect(engine="streaming"))',
}
# 既存グラフの配色規約を踏襲: 暖色系=pandas系(赤=素朴な全件読み込み、橙=手動で
# out-of-core化した条件)、寒色系=polars系(青=即時評価)。緑は実験Eでも
# 「最適化(pushdown/streaming)が効いた条件」を示す色として使っており、ここでも踏襲する。
COLORS = {
    "pandas_full": "#d62728",
    "pandas_chunked": "#ff7f0e",
    "polars_eager": "#1f77b4",
    "polars_streaming": "#2ca02c",
}


def plot_exp_g():
    path = os.path.join(RESULTS, "exp_g.csv")
    if not os.path.exists(path):
        print(f"skip exp_g: {path} not found")
        return

    rows = list(csv.DictReader(open(path)))
    if not rows:
        print(f"skip exp_g: {path} is empty")
        return

    by_cond_rss = {}
    by_cond_time = {}
    for r in rows:
        by_cond_rss.setdefault(r["condition"], []).append(float(r["peak_rss_bytes"]) / 1024**2)
        by_cond_time.setdefault(r["condition"], []).append(float(r["elapsed_sec"]))

    conditions = [c for c in CONDITIONS if c in by_cond_rss]
    rss_vals = [statistics.median(by_cond_rss[c]) for c in conditions]
    time_vals = [statistics.median(by_cond_time[c]) for c in conditions]
    colors = [COLORS[c] for c in conditions]
    labels = [LABELS[c] for c in conditions]
    n_rows_label = rows[0]["rows"]
    reps = len(by_cond_rss[conditions[0]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    bars1 = ax1.bar(range(len(conditions)), rss_vals, color=colors)
    ax1.set_xticks(range(len(conditions)))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("ピークRSS(MiB, 中央値)")
    ax1.set_title("条件ごとのピークメモリ使用量")
    ax1.grid(True, axis="y", alpha=0.3)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(
            f"{h:.0f} MiB",
            (b.get_x() + b.get_width() / 2, h),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
        )

    bars2 = ax2.bar(range(len(conditions)), time_vals, color=colors)
    ax2.set_xticks(range(len(conditions)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("所要時間(秒, 中央値)")
    ax2.set_title("条件ごとの所要時間")
    ax2.grid(True, axis="y", alpha=0.3)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(
            f"{h:.2f} s",
            (b.get_x() + b.get_width() / 2, h),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
        )

    fig.suptitle(f"ストリーミング/アウトオブコア実行のピークメモリ比較(N={n_rows_label}行, {reps}回中央値)")
    fig.tight_layout()
    out = os.path.join(IMAGES, "exp_g_peak_memory.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    plot_exp_g()
