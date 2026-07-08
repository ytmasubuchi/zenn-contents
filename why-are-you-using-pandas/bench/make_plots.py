"""結果CSV/JSONからブログ掲載用のグラフ(PNG)を生成する。
ラベルは日本語フォント未同梱環境でも文字化けしないよう英語表記に統一する。
"""
import csv
import json
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

LIB_LABELS = {
    "pandas_object": "pandas (object dtype)",
    "pandas_arrow": "pandas (string[pyarrow])",
    "pandas_category": "pandas (category)",
    "polars": "polars (String)",
}
LIB_COLORS = {
    "pandas_object": "#d62728",
    "pandas_arrow": "#ff7f0e",
    "pandas_category": "#9467bd",
    "polars": "#1f77b4",
}
LIB_MARKERS = {
    "pandas_object": "o",
    "pandas_arrow": "s",
    "pandas_category": "^",
    "polars": "D",
}


def plot_exp_a():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "exp_a.csv"))))
    by_lib = {}
    for r in rows:
        by_lib.setdefault(r["lib"], []).append((int(r["length"]), float(r["api_bytes_per_elem"])))
    for v in by_lib.values():
        v.sort()

    fig, ax = plt.subplots(figsize=(7, 5))
    for lib, pts in by_lib.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker=LIB_MARKERS[lib], color=LIB_COLORS[lib], label=LIB_LABELS[lib], linewidth=2)

    ax.set_xlabel("文字列長 L(文字数)")
    ax.set_ylabel("1要素あたりのメモリ量(bytes)\n[API値: memory_usage(deep=True) / estimated_size]")
    ax.set_title("文字列格納方式ごとの1要素あたりのメモリコスト\n(N=200,000 のユニークなASCII文字列)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(IMAGES, "exp_a_memory.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def plot_exp_b1():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "exp_b1.csv"))))
    main = {"pandas": [], "polars": []}
    single_thread_point = None
    for r in rows:
        if r["lib"] in main:
            main[r["lib"]].append((int(r["n"]), float(r["median_sec"]) * 1000))
        elif r["lib"] == "polars_1thread":
            single_thread_point = (int(r["n"]), float(r["median_sec"]) * 1000)
    for v in main.values():
        v.sort()

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"pandas": "#d62728", "polars": "#1f77b4"}
    for lib, pts in main.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", color=colors[lib], label=lib, linewidth=2)

    if single_thread_point:
        ax.scatter(
            [single_thread_point[0]],
            [single_thread_point[1]],
            marker="x",
            color="#1f77b4",
            s=90,
            label="polars (POLARS_MAX_THREADS=1)",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("行数 N(対数軸)")
    ax.set_ylabel("101列float64のDataFrameから1列削除する\n所要時間の中央値, ms(対数軸)")
    ax.set_title("列削除コスト: pandasはO(N)のコピー、polarsはほぼO(1)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(IMAGES, "exp_b1_drop_scaling.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def plot_exp_b2():
    pandas_data = json.load(open(os.path.join(RESULTS, "exp_b2_pandas.json")))
    polars_data = json.load(open(os.path.join(RESULTS, "exp_b2_polars.json")))

    labels = ["pandas", "polars"]
    loop_ms = [pandas_data["loop_add_time_total_sec"] * 1000, polars_data["loop_add_time_total_sec"] * 1000]
    batch_ms = [pandas_data["batch_add_time_sec"] * 1000, polars_data["batch_add_time_sec"] * 1000]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar([i - width / 2 for i in x], loop_ms, width, label="150列を1列ずつ追加(ループ)", color="#d62728")
    bars2 = ax.bar([i + width / 2 for i in x], batch_ms, width, label="150列をまとめて追加(バッチconcat)", color="#1f77b4")

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.1f} ms", (b.get_x() + b.get_width() / 2, h), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("合計時間(ms)")
    ax.set_ylim(0, max(loop_ms) * 1.25)
    ax.set_title("ループ追加 vs バッチ追加\n(N=200,000行、float64の新規列を150列追加)")
    ax.legend(loc="upper center")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(IMAGES, "exp_b2_add_columns.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    plot_exp_a()
    plot_exp_b1()
    plot_exp_b2()
