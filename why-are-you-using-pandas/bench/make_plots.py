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
    "pandas_default": "pandas 3.0 default (str, storage=python)",
    "pandas_category": "pandas (category)",
    "pandas_pyarrow": "pandas 3.0 default (str, storage=pyarrow)",
    "polars": "polars (String)",
}
LIB_COLORS = {
    "pandas_object": "#d62728",
    "pandas_default": "#ff7f0e",
    "pandas_category": "#9467bd",
    "pandas_pyarrow": "#2ca02c",
    "polars": "#1f77b4",
}
LIB_MARKERS = {
    "pandas_object": "o",
    "pandas_default": "s",
    "pandas_category": "^",
    "pandas_pyarrow": "v",
    "polars": "D",
}


def plot_exp_a():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "exp_a.csv"))))
    by_lib = {}
    for r in rows:
        by_lib.setdefault(r["lib"], []).append((int(r["length"]), float(r["api_bytes_per_elem"])))

    # pandas 3.0 default (storage=pyarrow) はpyarrow導入済みの別イメージで計測しており、
    # 結果は別JSONに分けて保存されているのでここでマージする。
    pyarrow_path = os.path.join(RESULTS, "exp_a_pandas_pyarrow.json")
    if os.path.exists(pyarrow_path):
        pyarrow_rows = json.load(open(pyarrow_path))
        for r in pyarrow_rows:
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

    # pandas (ArrowDtype, float64[pyarrow]) はpyarrow導入済みの別イメージで
    # 計測しており、結果は別JSONに分けて保存されているのでここでマージする。
    pyarrow_path = os.path.join(RESULTS, "exp_b1_pandas_pyarrow.json")
    if os.path.exists(pyarrow_path):
        pyarrow_rows = json.load(open(pyarrow_path))
        main["pandas_pyarrow"] = [(int(r["n"]), float(r["median_sec"]) * 1000) for r in pyarrow_rows]

    for v in main.values():
        v.sort()

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"pandas": "#d62728", "polars": "#1f77b4", "pandas_pyarrow": "#2ca02c"}
    labels = {"pandas": "pandas", "polars": "polars", "pandas_pyarrow": "pandas (ArrowDtype)"}
    for lib, pts in main.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", color=colors[lib], label=labels[lib], linewidth=2)

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

    # pandas (ArrowDtype, float64[pyarrow]) はpyarrow導入済みの別イメージで
    # 計測しており、結果は別JSONに分けて保存されているのでここでマージする。
    pyarrow_path = os.path.join(RESULTS, "exp_b2_pandas_pyarrow.json")
    if os.path.exists(pyarrow_path):
        pyarrow_data = json.load(open(pyarrow_path))
        labels.append("pandas (ArrowDtype)")
        loop_ms.append(pyarrow_data["loop_add_time_total_sec"] * 1000)
        batch_ms.append(pyarrow_data["batch_add_time_sec"] * 1000)

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


def plot_exp_c():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "exp_c.csv"))))
    by_lib = {}
    for r in rows:
        by_lib.setdefault(r["lib"], []).append((int(r["n"]), float(r["median_sec"]) * 1e6))

    # pandas (ArrowDtype, float64[pyarrow]) はpyarrow導入済みの別イメージで
    # 計測しており、結果は別JSONに分けて保存されているのでここでマージする。
    pyarrow_path = os.path.join(RESULTS, "exp_c_pandas_pyarrow.json")
    if os.path.exists(pyarrow_path):
        pyarrow_rows = json.load(open(pyarrow_path))
        by_lib["pandas_pyarrow"] = [(int(r["n"]), float(r["median_sec"]) * 1e6) for r in pyarrow_rows]

    for v in by_lib.values():
        v.sort()

    labels = {
        "pandas": "pandas (.to_numpy())",
        "pandas_pyarrow": "pandas (ArrowDtype, no nulls)",
        "polars_single_chunk": "polars (欠損なし・単一チャンク)",
        "polars_multi_chunk": "polars (欠損なし・複数チャンク)",
        "polars_nulls": "polars (欠損あり)",
    }
    colors = {
        "pandas": "#d62728",
        "pandas_pyarrow": "#2ca02c",
        "polars_single_chunk": "#1f77b4",
        "polars_multi_chunk": "#ff7f0e",
        "polars_nulls": "#9467bd",
    }
    markers = {
        "pandas": "o",
        "pandas_pyarrow": "v",
        "polars_single_chunk": "D",
        "polars_multi_chunk": "^",
        "polars_nulls": "s",
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    for lib, pts in by_lib.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker=markers[lib], color=colors[lib], label=labels[lib], linewidth=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("行数 N(対数軸)")
    ax.set_ylabel(".to_numpy()の所要時間の中央値, us(対数軸)")
    ax.set_title("numpy変換コスト: ゼロコピー条件が崩れるとpolarsもO(N)になる")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(IMAGES, "exp_c_to_numpy.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def plot_exp_f():
    path = os.path.join(RESULTS, "exp_f.csv")
    if not os.path.exists(path):
        print(f"skip exp_f: {path} not found")
        return

    rows = list(csv.DictReader(open(path)))
    pandas_row = None
    polars_pts = []  # (threads, median_ms)
    for r in rows:
        if r["lib"] == "pandas":
            pandas_row = r
        elif r["lib"] == "polars":
            threads = int(r["threads_effective"] or r["threads_requested"])
            # 共有環境の制約で16/32スレッド条件はCPU競合により計測ノイズが大きく、
            # 信頼できないと判断したため不採用とし、8スレッドまでのみを採用する。
            if threads > 8:
                continue
            polars_pts.append((threads, float(r["median_sec"]) * 1000))
    polars_pts.sort()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    xs = [p[0] for p in polars_pts]
    ys = [p[1] for p in polars_pts]
    ax1.plot(xs, ys, marker="D", color="#1f77b4", label="polars (group_by + agg)", linewidth=2)
    if pandas_row:
        pandas_ms = float(pandas_row["median_sec"]) * 1000
        ax1.axhline(pandas_ms, color="#d62728", linestyle="--", linewidth=2,
                    label=f"pandas (groupby + agg, 1スレッド想定, {pandas_ms:.2f} ms)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(x) for x in xs])
    ax1.set_yscale("log")
    ax1.set_xlabel("POLARS_MAX_THREADS(スレッド数, 対数軸)")
    ax1.set_ylabel("groupby集計の所要時間の中央値, ms(対数軸)")
    ax1.set_title("スレッド数とgroupby集計の所要時間")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    if polars_pts:
        base_ms = dict(polars_pts).get(1, ys[0])
        speedup_xs = xs
        speedup_ys = [base_ms / p[1] for p in polars_pts]
        ax2.plot(speedup_xs, speedup_ys, marker="D", color="#1f77b4", label="polars 実測スピードアップ", linewidth=2)
        ax2.plot(speedup_xs, speedup_xs, color="#7f7f7f", linestyle=":", linewidth=1.5, label="理想的な線形スケーリング")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(x) for x in xs])
    ax2.set_xlabel("POLARS_MAX_THREADS(スレッド数, 対数軸)")
    ax2.set_ylabel("1スレッド比のスピードアップ倍率")
    ax2.set_title("polarsのスレッドスケーリング効率")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle("シングルスレッド vs マルチスレッド: groupby集計のスレッドスケーリング")
    fig.tight_layout()
    out = os.path.join(IMAGES, "exp_f_thread_scaling.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    plot_exp_a()
    plot_exp_b1()
    plot_exp_b2()
    plot_exp_c()
    plot_exp_f()
