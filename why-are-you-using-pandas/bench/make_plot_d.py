"""実験D単体のグラフ生成。pyarrow入りの専用イメージ(Dockerfile.pyarrow)内で実行する。"""
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

SERIES = [
    ("numeric", "False", "pandas(numpy backend)への変換 / 数値列", "#1f77b4", "o"),
    ("numeric", "True", "pandas(ArrowDtype)への変換 / 数値列", "#7fb3ff", "s"),
    ("string", "False", "pandas(numpy backend)への変換 / 文字列列", "#d62728", "^"),
    ("string", "True", "pandas(ArrowDtype)への変換 / 文字列列", "#ff9896", "D"),
]


def main():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "exp_d.csv"))))
    by_key = {}
    for r in rows:
        key = (r["cols"], r["use_pyarrow_ext"])
        by_key.setdefault(key, []).append((int(r["n"]), float(r["median_sec"]) * 1000))
    for v in by_key.values():
        v.sort()

    fig, ax = plt.subplots(figsize=(7, 5))
    for cols, flag, label, color, marker in SERIES:
        pts = by_key.get((cols, flag))
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker=marker, color=color, label=label, linewidth=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("行数 N(対数軸)")
    ax.set_ylabel("polars.DataFrame.to_pandas()の\n所要時間の中央値, ms(対数軸)")
    ax.set_title("polars→pandas変換コスト:\nuse_pyarrow_extension_array有無 × 数値列/文字列列")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(IMAGES, "exp_d_to_pandas.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
