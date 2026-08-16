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


def diagram_chunks():
    # 図1(文字列: 実データ+オフセット配列の2バッファ構成)と混同しないよう、
    # ここではベンチマークCと同じ数値型(float64)の例に絞り、1チャンク=1バッファを
    # 図1と同じ「セルを並べた箱」の描画スタイルで具体的に示す。
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    values_by_chunk = [["1.2", "5.0", "3.7", "9.1"], ["2.4", "6.6", "8.0", "4.3"], ["7.7", "1.9", "3.3", "5.5"]]

    # ---- Left: single chunk ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("単一チャンク\n(列全体が1つのArrow配列)", fontsize=12, weight="bold")
    ax.axis("off")

    ax.text(5, 9.4, "pl.Series([1.2, 5.0, 3.7, ..., 5.5])  # 1回でまとめて構築", ha="center", fontsize=9, style="italic")

    all_values = [v for chunk in values_by_chunk for v in chunk]
    n_x, n_y, n_w, n_h = 1.0, 6.4, 8.0, 1.6
    ax.text(n_x, n_y + n_h + 0.35, "チャンク0(Arrow配列 1つ): float64のデータバッファ1本に全12個の値を連続配置", fontsize=9)
    cell_w = n_w / len(all_values)
    for i, v in enumerate(all_values):
        cx = n_x + i * cell_w
        ax.add_patch(Rectangle((cx, n_y), cell_w, n_h, fill=True, facecolor="#e8f0fe",
                                edgecolor=C_EDGE, linewidth=1.2, zorder=3))
        ax.text(cx + cell_w / 2, n_y + n_h / 2, v, ha="center", va="center", fontsize=8, zorder=4)
    ax.text(5.0, 6.0, "(文字列型の場合の内部構造は図1を参照)", ha="center", va="top", fontsize=8.3, color="#777777")

    arrow(ax, (5.0, 5.1), (5.0, 3.8), color=C_ARROW, lw=2)
    box(ax, 2.0, 2.2, 6.0, 1.4, ".to_numpy()\nこのバッファをそのまま参照 → ゼロコピー", fc="#e6f4ea", fontsize=9.5)

    # ---- Right: multiple chunks ----
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("複数チャンク\n(concat等で複数のArrow配列のまま)", fontsize=12, weight="bold")
    ax.axis("off")

    ax.text(5, 9.4, "pl.concat([s1, s2, s3], rechunk=False)", ha="center", fontsize=9, style="italic")

    n_chunks = len(values_by_chunk)
    group_w = 8.0
    gap = 0.4
    chunk_w = (group_w - gap * (n_chunks - 1)) / n_chunks
    start_x = 1.0
    for ci, chunk_vals in enumerate(values_by_chunk):
        gx = start_x + ci * (chunk_w + gap)
        cell_w = chunk_w / len(chunk_vals)
        for i, v in enumerate(chunk_vals):
            cx = gx + i * cell_w
            ax.add_patch(Rectangle((cx, n_y), cell_w, n_h, fill=True, facecolor="#e8f0fe",
                                    edgecolor=C_EDGE, linewidth=1.2, zorder=3))
            ax.text(cx + cell_w / 2, n_y + n_h / 2, v, ha="center", va="center", fontsize=8, zorder=4)
        ax.text(gx + chunk_w / 2, n_y - 0.3, f"チャンク{ci}\n(s{ci+1}由来)",
                ha="center", va="top", fontsize=8.5, weight="bold")

    ax.text(5.0, 4.7, "それぞれ4個のfloat64値を持つ、独立したバッファ\n3つが論理的に1列としてつながっているだけ(仮想アドレス上でも別々の領域)",
            ha="center", va="top", fontsize=8.3, color="#555555")

    arrow(ax, (5.0, 3.7), (5.0, 2.8), color=C_PANDAS, lw=2)
    box(ax, 1.5, 1.2, 7.0, 1.4, ".to_numpy()\n3つのバッファを1本に連結 → コピー発生(rechunk)", fc="#fff3cd", fontsize=9.5)

    fig.tight_layout()
    out = os.path.join(IMAGES, "diagram_chunks.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def diagram_memory_layers():
    # 「チャンクの分断」がどのメモリレイヤで起きるかを示す3層図。
    # レイヤ2(仮想アドレス空間)の分断=チャンクであり、レイヤ3(物理ページ)の
    # 分散とは別の話であることを1枚で対比する。
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 13)
    ax.axis("off")

    C_CHUNK0 = "#e8f0fe"
    C_CHUNK1 = "#eaf7ea"

    def layer_band(y0, y1, title):
        ax.add_patch(Rectangle((0.2, y0), 11.6, y1 - y0, facecolor="#fafafa",
                               edgecolor="#cccccc", linewidth=1.0, zorder=1))
        ax.text(0.45, y1 - 0.15, title, ha="left", va="top", fontsize=10.5, weight="bold", zorder=2)

    # ---- Layer 1: logical ----
    layer_band(10.0, 12.6, "レイヤ1: 論理(DataFrameのSeries)")
    box(ax, 2.5, 10.4, 7.0, 1.0, 's = pl.concat([s1, s2], rechunk=False)\n→ DataFrame上では常に「1列」に見える', fc="#ffffff", fontsize=9.5)

    # ---- Layer 2: virtual address space ----
    layer_band(5.6, 9.4, "レイヤ2: 仮想アドレス空間 ← チャンクの分断はここで起きる")
    # chunk0: contiguous buffer (8 cells), chunk1: contiguous buffer (4 cells), separated
    c0_x, c0_w, c1_x, c1_w = 1.0, 5.2, 8.0, 2.8
    buf_y, buf_h = 6.6, 1.2
    for i in range(8):
        ax.add_patch(Rectangle((c0_x + i * c0_w / 8, buf_y), c0_w / 8, buf_h, facecolor=C_CHUNK0,
                               edgecolor=C_EDGE, linewidth=1.1, zorder=3))
    for i in range(4):
        ax.add_patch(Rectangle((c1_x + i * c1_w / 4, buf_y), c1_w / 4, buf_h, facecolor=C_CHUNK1,
                               edgecolor=C_EDGE, linewidth=1.1, zorder=3))
    ax.text(c0_x + c0_w / 2, buf_y - 0.3, "チャンク0(s1由来): 連続バッファ", ha="center", va="top", fontsize=9)
    ax.text(c1_x + c1_w / 2, buf_y - 0.3, "チャンク1(s2由来): 連続バッファ", ha="center", va="top", fontsize=9)
    ax.text((c0_x + c0_w + c1_x) / 2, buf_y + buf_h / 2, "別の\n確保領域", ha="center", va="center",
            fontsize=8, color="#777777")
    arrow(ax, (4.5, 10.4), (c0_x + c0_w / 2, buf_y + buf_h), connectionstyle="arc3,rad=0.1")
    arrow(ax, (7.5, 10.4), (c1_x + c1_w / 2, buf_y + buf_h), connectionstyle="arc3,rad=-0.1")

    # ---- Layer 3: physical memory ----
    layer_band(1.6, 5.0, "レイヤ3: 物理メモリ(OSがページ単位で管理)")
    page_y, page_h = 2.2, 1.2
    n_pages = 8
    page_w = 10.4 / n_pages
    # chunk0 -> pages 1, 4, 6 / chunk1 -> pages 2, 7 (scattered)
    page_colors = {1: C_CHUNK0, 4: C_CHUNK0, 6: C_CHUNK0, 2: C_CHUNK1, 7: C_CHUNK1}
    for i in range(n_pages):
        px = 0.8 + i * page_w
        ax.add_patch(Rectangle((px, page_y), page_w, page_h, facecolor=page_colors.get(i, "#ffffff"),
                               edgecolor=C_EDGE, linewidth=1.1, zorder=3))
        ax.text(px + page_w / 2, page_y + page_h / 2, f"ページ{i}", ha="center", va="center", fontsize=8.5, zorder=4)
    # arrows: virtual buffer thirds -> physical pages (crossing to show scatter)
    for frac, page in [(1 / 6, 4), (3 / 6, 1), (5 / 6, 6)]:
        arrow(ax, (c0_x + frac * c0_w, buf_y), (0.8 + page * page_w + page_w / 2, page_y + page_h),
              color=C_ARROW, connectionstyle="arc3,rad=0.12")
    for frac, page in [(1 / 4, 7), (3 / 4, 2)]:
        arrow(ax, (c1_x + frac * c1_w, buf_y), (0.8 + page * page_w + page_w / 2, page_y + page_h),
              color="#2e8b57", connectionstyle="arc3,rad=-0.12")
    ax.text(6.0, 1.75, "仮想アドレス上で連続なバッファでも、物理ページは分散しうる(プログラムからは見えない)",
            ha="center", va="bottom", fontsize=8.5, color="#555555")

    # ---- bottom note ----
    box(ax, 0.8, 0.1, 10.4, 1.1,
        ".to_numpy()のゼロコピー可否を決めるのはレイヤ2の分断(複数チャンク)だけ。\n"
        "レイヤ3の分散はOSが透過的に解決するため影響しない(numpyも仮想アドレス上で動くため)",
        fc="#fff3cd", fontsize=9.5)

    fig.tight_layout()
    out = os.path.join(IMAGES, "diagram_memory_layers.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    diagram_string_memory()
    diagram_blockmanager()
    diagram_chunks()
    diagram_memory_layers()
