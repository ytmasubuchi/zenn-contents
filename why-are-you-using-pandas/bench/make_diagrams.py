"""記事に挿入する概念図(データ実測ではなく構造の模式図)を生成する。
ラベルは英語で統一(日本語フォント未同梱環境での文字化け防止)。
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMAGES = os.path.join(ROOT, "images")
os.makedirs(IMAGES, exist_ok=True)

C_PANDAS = "#d62728"
C_ARROW = "#1f77b4"
C_BOX = "#f2f2f2"
C_EDGE = "#333333"


def box(ax, x, y, w, h, text, fc=C_BOX, ec=C_EDGE, fontsize=10, weight="normal", zorder=2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.3,
            edgecolor=ec,
            facecolor=fc,
            zorder=zorder,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight, zorder=zorder + 1)


def arrow(ax, xy_from, xy_to, color=C_EDGE, style="-|>", lw=1.4, connectionstyle="arc3,rad=0.0"):
    ax.add_patch(
        FancyArrowPatch(
            xy_from,
            xy_to,
            arrowstyle=style,
            mutation_scale=12,
            linewidth=lw,
            color=color,
            connectionstyle=connectionstyle,
            zorder=3,
        )
    )


def diagram_string_memory():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # ---- Left: pandas object dtype ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("pandas: object dtype\n(ポインタを格納するnumpy配列)", fontsize=12, weight="bold")
    ax.axis("off")

    ax.text(5, 9.3, 'Series(["a", "abc", "de"], dtype=object)', ha="center", fontsize=9, style="italic")

    # numpy array of pointers
    n_x = 1.5
    n_y = 6.8
    n_w = 7
    n_h = 1.0
    box(ax, n_x, n_y, n_w, n_h, "", fc="#ffffff")
    ax.text(n_x - 0.15, n_y + n_h + 0.35, "numpy ndarray(dtype=object): 各要素は固定長スロットに\"ポインタ\"を格納",
            ha="left", fontsize=9)
    cell_w = n_w / 3
    addrs = ["0x10", "0x256", "0x32"]
    for i, addr in enumerate(addrs):
        cx = n_x + i * cell_w
        ax.add_patch(Rectangle((cx, n_y), cell_w, n_h, fill=False, edgecolor=C_EDGE, linewidth=1.2, zorder=3))
        ax.text(cx + cell_w / 2, n_y + n_h / 2, f"ポインタ\n{addr}", ha="center", va="center", fontsize=9)

    # heap PyObjects scattered
    heap_items = [
        (0.8, 3.2, "a"),
        (4.0, 1.6, "abc"),
        (7.2, 3.6, "de"),
    ]
    for i, (hx, hy, s) in enumerate(heap_items):
        w, h = 2.0, 1.6
        box(ax, hx, hy, w, h, f"PyObject\n(ヘッダ 約49B)\n値=\"{s}\"", fc="#fdecea", fontsize=8)
        cx = n_x + i * cell_w + cell_w / 2
        arrow(ax, (cx, n_y), (hx + w / 2, hy + h), color=C_PANDAS, connectionstyle="arc3,rad=0.15")

    ax.text(5, 0.3, "→ 1要素あたりのコスト: ポインタ(8B) + PyObjectヘッダ(約49B) + 文字列バイト数\n"
                     "   ヒープ上に散らばって配置される(局所性が悪く、ボックス化のオーバーヘッド)",
            ha="center", fontsize=8.5, color="#555555")

    # ---- Right: Arrow layout ----
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Arrow形式\n(pandasのstring[pyarrow] / polars)", fontsize=12, weight="bold")
    ax.axis("off")

    ax.text(5, 9.3, 'Series(["a", "abc", "de"], dtype="string[pyarrow]")', ha="center", fontsize=9, style="italic")

    # contiguous data buffer
    data_y = 6.4
    data_h = 1.2
    segs = [("a", 1), ("abc", 3), ("de", 2)]
    total_chars = sum(n for _, n in segs)
    x0 = 1.0
    total_w = 8.0
    ax.text(x0, data_y + data_h + 0.35, "データバッファ: 全文字列のバイト列を連結して1つに格納(連続領域)", fontsize=9)
    cursor = x0
    colors = ["#e8f0fe", "#fdecea", "#eaf7ea"]
    offsets = [0]
    for i, (s, n) in enumerate(segs):
        w = total_w * n / total_chars
        box(ax, cursor, data_y, w, data_h, s, fc=colors[i], fontsize=11, weight="bold")
        cursor += w
        offsets.append(offsets[-1] + n)

    # offsets array
    off_y = 3.6
    off_h = 1.0
    ax.text(x0, off_y + off_h + 0.35, "オフセット配列(int32): 区切り位置 → [0, 1, 4, 6]", fontsize=9)
    off_w = total_w / 4
    for i, val in enumerate(offsets):
        cx = x0 + i * off_w
        box(ax, cx, off_y, off_w, off_h, str(val), fc="#ffffff", fontsize=10)

    # arrows from offsets to buffer boundaries
    cum = 0
    for i, (s, n) in enumerate(segs):
        w = total_w * n / total_chars
        bx = x0 + cum
        arrow(ax, (x0 + i * off_w + off_w / 2, off_y + off_h), (bx, data_y), color=C_ARROW,
              connectionstyle="arc3,rad=0.1")
        cum += w

    ax.text(5, 0.6, "→ 全体で1回のメモリ確保のみ、1要素あたりのコストは\n"
                     "   ほぼ文字列本体のバイト数のみ(ポインタやヘッダの追加コストなし)",
            ha="center", fontsize=8.5, color="#555555")

    fig.tight_layout()
    out = os.path.join(IMAGES, "diagram_string_memory.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def diagram_blockmanager():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # ---- Left: pandas BlockManager ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("pandas: BlockManager\n(同じdtypeの列は1つの配列にまとめられる)", fontsize=12, weight="bold")
    ax.axis("off")

    ax.text(5, 9.4, "df[['c0','c1','c2']]=float64, df['c3']=object", ha="center", fontsize=9, style="italic")

    # logical dataframe columns row
    col_names = ["c0\n(float64)", "c1\n(float64)", "c2\n(float64)", "c3\n(object)"]
    col_w = 2.0
    start_x = 1.0
    top_y = 7.7
    for i, name in enumerate(col_names):
        box(ax, start_x + i * col_w, top_y, col_w - 0.15, 0.9, name, fc="#ffffff", fontsize=8.5)

    # Block A: single contiguous 2D array for the 3 float64 columns
    blockA_x, blockA_y, blockA_w, blockA_h = 1.0, 4.6, 5.55, 1.8
    box(ax, blockA_x, blockA_y, blockA_w, blockA_h, "", fc="#e8f0fe")
    sub_w = blockA_w / 3
    for i in range(3):
        ax.add_patch(Rectangle((blockA_x + i * sub_w, blockA_y), sub_w, blockA_h, fill=False,
                                edgecolor=C_EDGE, linewidth=1.0, zorder=3))
    ax.text(blockA_x + blockA_w / 2, blockA_y - 0.25, "ブロックA: 2次元numpy配列1つ\n(c0,c1,c2をまとめて格納)",
            ha="center", va="top", fontsize=8.5)

    # Block B: object column, its own array
    blockB_x, blockB_y, blockB_w, blockB_h = 7.0, 4.6, 2.0, 1.8
    box(ax, blockB_x, blockB_y, blockB_w, blockB_h, "", fc="#fdecea")
    ax.text(blockB_x + blockB_w / 2, blockB_y - 0.25, "ブロックB:\nobject配列(c3)", ha="center", va="top", fontsize=8.5)

    for i in range(3):
        arrow(ax, (start_x + i * col_w + (col_w - 0.15) / 2, top_y), (blockA_x + i * sub_w + sub_w / 2, blockA_y + blockA_h))
    arrow(ax, (start_x + 3 * col_w + (col_w - 0.15) / 2, top_y), (blockB_x + blockB_w / 2, blockB_y + blockB_h))

    # consequence box
    box(ax, 1.0, 1.2, 8.0, 2.0,
        "列の追加・削除\n→ ブロックの再構築が必要(メモリ再確保+コピー)\n"
        "1列ずつ挿入を繰り返す → 小さなブロックが増え『断片化』",
        fc="#fff3cd", fontsize=9.5)
    arrow(ax, (5.0, 4.6), (5.0, 3.2), color=C_PANDAS, lw=2)

    # ---- Right: polars ----
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("polars: 列ごとに独立したバッファ", fontsize=12, weight="bold")
    ax.axis("off")

    ax.text(5, 9.4, "任意のdtypeの列 c0〜c3 を持つdf", ha="center", fontsize=9, style="italic")

    col_names2 = ["c0", "c1", "c2", "c3"]
    for i, name in enumerate(col_names2):
        box(ax, start_x + i * col_w, top_y, col_w - 0.15, 0.9, name, fc="#ffffff", fontsize=9)

    buf_y = 4.6
    buf_h = 1.8
    buf_colors = ["#e8f0fe", "#e8f0fe", "#e8f0fe", "#fdecea"]
    for i in range(4):
        bx = start_x + i * col_w
        box(ax, bx, buf_y, col_w - 0.15, buf_h, f"バッファ\n{col_names2[i]}", fc=buf_colors[i], fontsize=8.5)
        arrow(ax, (bx + (col_w - 0.15) / 2, top_y), (bx + (col_w - 0.15) / 2, buf_y + buf_h))

    box(ax, 1.0, 1.2, 8.0, 2.0,
        "列の追加・削除\n→ 列のリストにポインタを追加/削除するだけ\n"
        "他の列のバッファには影響しない(再確保なし)",
        fc="#e6f4ea", fontsize=9.5)
    arrow(ax, (5.0, 4.6), (5.0, 3.2), color=C_ARROW, lw=2)

    fig.tight_layout()
    out = os.path.join(IMAGES, "diagram_blockmanager.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    diagram_string_memory()
    diagram_blockmanager()
