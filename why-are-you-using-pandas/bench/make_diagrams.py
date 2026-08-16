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

# 図1〜4(diagram_string_memory 〜 diagram_memory_layers)で使ってきた塗り色を
# 定数化して再利用する(記事全体で「同じ意味には同じ色」を徹底するため)。
C_GOOD = "#e6f4ea"       # 良い結果・効率的な経路を示すナラティブボックス
C_WARN = "#fff3cd"       # 警告・コスト増を示すナラティブボックス
C_BAD_FILL = "#fdecea"   # pandas側の問題箇所(ボックス化/欠損/断片化など)の塗り
C_DATA_FILL = "#e8f0fe"  # 数値バッファ・Arrowバッファ等の塗り
C_ALT_FILL = "#eaf7ea"   # 3つ目の要素を区別するための塗り
C_ARROW2 = "#2e8b57"     # 2系統目を区別するための矢印色(seagreen)


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


def diagram_null_representation():
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.2))

    # ---- Left: numpy-based (pandas default) ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("numpy配列ベース(pandasのデフォルト)\n型ごとにNULLの表現方法がバラバラ", fontsize=12, weight="bold")
    ax.axis("off")

    x0 = 1.0

    # Row A: float64 with NaN embedded as a value
    ax.text(x0, 9.5, 'float64列: Series([1.0, NaN, 3.0, NaN])', ha="left", fontsize=9, style="italic")
    rowA_y = 8.3
    valsA = ["1.0", "NaN", "3.0", "NaN"]
    cw4 = 8.0 / 4
    for i, v in enumerate(valsA):
        cx = x0 + i * cw4
        fc = C_BAD_FILL if v == "NaN" else "#ffffff"
        ax.add_patch(Rectangle((cx, rowA_y), cw4, 0.9, facecolor=fc, edgecolor=C_EDGE, linewidth=1.2, zorder=3))
        ax.text(cx + cw4 / 2, rowA_y + 0.45, v, ha="center", va="center", fontsize=9.5, zorder=4)
    ax.text(x0, rowA_y - 0.25, "→ NaNは「欠損マーカー」であると同時に配列内の「値」そのもの\n"
                              "   (欠損専用の領域は無く、値の型(float64)でしか表現できない)",
            ha="left", va="top", fontsize=8.2, color="#555555")

    # Row B: int64 -> must upcast to float64 to hold a null
    ax.text(x0, 7.0, "int64列: Series([1, 2, 3]) に欠損を入れると…", ha="left", fontsize=9, style="italic")
    rowB_y = 5.8
    cw3 = 6.0 / 3
    for i, v in enumerate(["1", "2", "3"]):
        cx = x0 + i * cw3
        ax.add_patch(Rectangle((cx, rowB_y), cw3, 0.9, facecolor="#ffffff", edgecolor=C_EDGE, linewidth=1.2, zorder=3))
        ax.text(cx + cw3 / 2, rowB_y + 0.45, v, ha="center", va="center", fontsize=9.5, zorder=4)
    ax.text(x0 + 6.0 + 0.3, rowB_y + 0.45, "dtype:\nint64", ha="left", va="center", fontsize=8.5, color="#555555")

    arrow(ax, (x0 + 3.0, rowB_y), (x0 + 3.0, rowB_y - 1.0), color=C_PANDAS, lw=2)

    rowB2_y = rowB_y - 2.0
    valsB2 = ["1.0", "NaN", "3.0"]
    for i, v in enumerate(valsB2):
        cx = x0 + i * cw3
        fc = C_BAD_FILL if v == "NaN" else "#ffffff"
        ax.add_patch(Rectangle((cx, rowB2_y), cw3, 0.9, facecolor=fc, edgecolor=C_EDGE, linewidth=1.2, zorder=3))
        ax.text(cx + cw3 / 2, rowB2_y + 0.45, v, ha="center", va="center", fontsize=9.5, zorder=4)
    ax.text(x0 + 6.0 + 0.3, rowB2_y + 0.45, "dtype:\nfloat64", ha="left", va="center", fontsize=8.5, color="#555555")

    box(ax, x0, rowB2_y - 1.1, 8.0, 0.55,
        "int64 → float64へ自動的に昇格(値の型が変わる。巨大な整数では精度も変わりうる)",
        fc=C_WARN, fontsize=8.5)

    # Row C: object dtype -> Python None pointer
    ax.text(x0, rowB2_y - 1.9, 'object列: Series(["a", None, "c"], dtype=object)', ha="left", fontsize=9, style="italic")
    rowC_y = rowB2_y - 3.1
    cw3b = 8.0 / 3
    for i, v in enumerate(["ptr→\n\"a\"", "ptr→\nNone", "ptr→\n\"c\""]):
        cx = x0 + i * cw3b
        fc = C_BAD_FILL if "None" in v else "#ffffff"
        ax.add_patch(Rectangle((cx, rowC_y), cw3b, 0.9, facecolor=fc, edgecolor=C_EDGE, linewidth=1.2, zorder=3))
        ax.text(cx + cw3b / 2, rowC_y + 0.45, v, ha="center", va="center", fontsize=8.5, zorder=4)
    ax.text(x0, rowC_y - 0.25,
            "→ 欠損はPythonの唯一のNoneオブジェクトへの「ポインタ」として表現される\n"
            "   (型として統一されたNULLではなく、objectという万能dtypeの副産物)",
            ha="left", va="top", fontsize=8.2, color="#555555")

    # ---- Right: Arrow layout ----
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Arrow形式\n(pandasのpyarrowバックエンド/polars共通)", fontsize=12, weight="bold")
    ax.axis("off")

    ax.text(x0, 9.5, 'pl.Series([1, None, 3, None], dtype=pl.Int64)', ha="left", fontsize=9, style="italic")

    buf_y = 8.3
    valsBuf = ["1", "?", "3", "?"]
    for i, v in enumerate(valsBuf):
        cx = x0 + i * cw4
        box(ax, cx, buf_y, cw4, 0.9, v, fc=C_DATA_FILL, fontsize=9.5)
    ax.text(x0, buf_y - 0.25,
            "値バッファ(int64がそのまま): 全要素分の領域はあるが、\n"
            "NULL位置(\"?\")の値は無視される(不定値のままでもよい)",
            ha="left", va="top", fontsize=8.2, color="#555555")

    ax.text(x0, 6.7, "validity bitmap(1ビット/要素): 1=値あり(valid) / 0=NULL", ha="left", fontsize=9)
    bit_y = 5.5
    valsBit = ["1", "0", "1", "0"]
    for i, v in enumerate(valsBit):
        cx = x0 + i * cw4
        fc = C_GOOD if v == "1" else C_BAD_FILL
        box(ax, cx, bit_y, cw4, 0.9, v, fc=fc, fontsize=9.5)
        arrow(ax, (cx + cw4 / 2, bit_y + 0.9), (cx + cw4 / 2, buf_y), color=C_EDGE, lw=1.0,
              connectionstyle="arc3,rad=0.0")

    box(ax, x0, 3.8, 8.0, 1.0,
        "値とNULL情報が分離しているため、\nint64のままNULLを表現できる(pandasのような昇格は不要)",
        fc=C_GOOD, fontsize=9.5)

    ax.text(x0, 2.9,
            "→ この仕組みは数値・真偽値・文字列など、どの型でも共通\n"
            "   (文字列列の場合、この後ろに図1の実データバッファ+オフセット配列が続く)",
            ha="left", va="top", fontsize=8.3, color="#555555")

    fig.tight_layout()
    out = os.path.join(IMAGES, "diagram_null_representation.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def diagram_execution_model():
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.2))

    steps = ["読み込み\n(全列読込)", "select\n(列を絞る)", "filter\n(行を絞る)", "集計\n(sum等)"]
    n = 4
    box_w, gap = 1.75, 0.45
    total_w = n * box_w + (n - 1) * gap
    x0 = (10 - total_w) / 2

    # ---- Left: eager (pandas / polars eager API) ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("即時評価(pandas / polarsのeager API)\n1操作ごとに中間結果を実体化", fontsize=12, weight="bold")
    ax.axis("off")

    y_op, h_op = 8.3, 1.0
    xs = [x0 + i * (box_w + gap) for i in range(n)]
    for i, (x, label) in enumerate(zip(xs, steps)):
        box(ax, x, y_op, box_w, h_op, label, fc="#ffffff", fontsize=8.3)
        if i > 0:
            arrow(ax, (xs[i - 1] + box_w, y_op + h_op / 2), (x, y_op + h_op / 2))

    y_mat, h_mat = 6.2, 1.3
    mat_labels = ["全データが\nメモリ上に展開", "選んだ列だけの\n新規DataFrame", "絞った行だけの\n新規DataFrame", "最終結果"]
    for i, x in enumerate(xs):
        fc = C_BAD_FILL if i < n - 1 else C_GOOD
        box(ax, x, y_mat, box_w, h_mat, mat_labels[i], fc=fc, fontsize=7.8)
        arrow(ax, (x + box_w / 2, y_op), (x + box_w / 2, y_mat + h_mat),
              color=C_PANDAS if i < n - 1 else C_EDGE, lw=1.6)

    box(ax, x0, 3.6, total_w, 1.5,
        "各ステップは次のステップの都合を知らずに実行されるため、\n"
        "後で select/filter が減らす列・行も含めて、読み込み時点では全部読む\n"
        "→ ステップの数だけ中間結果分のメモリ確保とコピーが発生する",
        fc=C_WARN, fontsize=8.6)

    ax.text(5.0, 1.6, "(pandasは常にeager。polarsもeager API(pl.DataFrame)を使えば同様の挙動になる)",
            ha="center", fontsize=8.0, color="#777777")

    # ---- Right: lazy (polars lazy API) ----
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("遅延評価(polars lazy API)\n論理プラン構築 → 最適化 → 一括実行", fontsize=12, weight="bold")
    ax.axis("off")

    y_plan, h_plan = 8.5, 0.85
    for i, (x, label) in enumerate(zip(xs, steps)):
        box(ax, x, y_plan, box_w, h_plan, label, fc="#ffffff", fontsize=8.0)
        if i > 0:
            arrow(ax, (xs[i - 1] + box_w, y_plan + h_plan / 2), (x, y_plan + h_plan / 2))
    ax.add_patch(Rectangle((x0 - 0.25, y_plan - 0.25), total_w + 0.5, h_plan + 0.5, fill=False,
                            edgecolor="#888888", linewidth=1.2, linestyle="--", zorder=1))
    ax.text(x0 - 0.25, y_plan + h_plan + 0.35, "LazyFrame: 論理プラン(まだ実行されない)", fontsize=8.3, weight="bold")

    arrow(ax, (5.0, y_plan - 0.25), (5.0, 6.55), color=C_EDGE, lw=1.8)
    box(ax, x0, 5.6, total_w, 0.95,
        "オプティマイザ: predicate pushdown(filter条件を読込側へ) /\nprojection pushdown(必要な列だけ読込) 等",
        fc=C_DATA_FILL, fontsize=8.3)

    arrow(ax, (5.0, 5.6), (5.0, 4.85), color=C_ARROW, lw=1.8)
    box(ax, x0, 3.6, total_w, 1.15,
        "最適化済み実行プラン(例):\n「必要な列だけ・条件に合う行だけ」を読込段階から適用してスキャン\n"
        "→ 1回のクエリ実行で完了(中間DataFrameを作らない)",
        fc=C_GOOD, fontsize=8.3)

    ax.text(5.0, 2.9, "→ pushdownにより、そもそも読み込む/保持するデータ自体が減る",
            ha="center", fontsize=8.8, weight="bold", color="#1a6b3c")
    ax.text(5.0, 1.9, "(※ 全ての演算がpushdown対象とは限らず、対象外の演算はそのまま素通しされる)",
            ha="center", fontsize=8.0, color="#777777")

    fig.tight_layout()
    out = os.path.join(IMAGES, "diagram_execution_model.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def _draw_cores(ax, x0, y0, w, h, gap, n_cols, n_rows, active_flags, active_color, labels):
    cw = (w - gap * (n_cols - 1)) / n_cols
    ch = (h - gap * (n_rows - 1)) / n_rows
    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            x = x0 + c * (cw + gap)
            y = y0 + (n_rows - 1 - r) * (ch + gap)
            active = active_flags[idx]
            fc = active_color if active else "#eeeeee"
            ec = C_EDGE if active else "#aaaaaa"
            ax.add_patch(FancyBboxPatch((x, y), cw, ch, boxstyle="round,pad=0.015,rounding_size=0.03",
                                        linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=3))
            ax.text(x + cw / 2, y + ch / 2, labels[idx], ha="center", va="center",
                    fontsize=8, weight="bold" if active else "normal",
                    color="#ffffff" if active else "#999999", zorder=4)
            idx += 1


def diagram_thread_model():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4))

    n_cols, n_rows = 4, 2
    n_cores = n_cols * n_rows

    # ---- Left: pandas ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("pandas: Pythonから1操作ずつ\nnumpyカーネルを呼び出す(基本1コア)", fontsize=12, weight="bold")
    ax.axis("off")

    ax.text(5.0, 9.3, "df['c'] = df['a'] * 2 + df['b']  # 1操作ずつ順に実行", ha="center", fontsize=9, style="italic")

    flags = [True] + [False] * (n_cores - 1)
    labels = ["Core0\n100%"] + [f"Core{i}\n0%" for i in range(1, n_cores)]
    _draw_cores(ax, 1.0, 4.0, 8.0, 4.0, 0.35, n_cols, n_rows, flags, C_PANDAS, labels)

    box(ax, 1.0, 1.8, 8.0, 1.6,
        "Python層のループとGILの制約もあり、\n1回のnumpyカーネル呼び出しは通常1コアのみを使って処理する",
        fc=C_WARN, fontsize=9)
    ax.text(5.0, 0.9, "※ BLASを使う行列積など、一部のnumpy演算は内部で複数スレッドを使うこともある",
            ha="center", fontsize=8.0, color="#777777")

    # ---- Right: polars ----
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("polars: クエリ実行時に複数コアへ分割\n(morsel/パーティション単位で並列処理)", fontsize=12, weight="bold")
    ax.axis("off")

    ax.text(5.0, 9.3, "df.select((pl.col('a') * 2 + pl.col('b')).alias('c'))  # 1回のクエリ実行", ha="center",
            fontsize=8.7, style="italic")

    flags2 = [True] * n_cores
    util = [95, 88, 97, 90, 85, 93, 89, 96]
    labels2 = [f"Core{i}\n{u}%" for i, u in enumerate(util)]
    _draw_cores(ax, 1.0, 4.0, 8.0, 4.0, 0.35, n_cols, n_rows, flags2, C_ARROW, labels2)

    box(ax, 1.0, 1.8, 8.0, 1.6,
        "データを小さな単位(morsel)やパーティションに分割し、\n"
        "Rustのマルチスレッドランタイム上でコアごとに並列実行する",
        fc=C_GOOD, fontsize=9)
    ax.text(5.0, 0.9, "※ 使用コア数はデータ量・演算内容・環境(POLARS_MAX_THREADS等)に依存する",
            ha="center", fontsize=8.0, color="#777777")

    fig.tight_layout()
    out = os.path.join(IMAGES, "diagram_thread_model.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def diagram_kernel_fusion():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6))

    # ---- Left: pandas (numpy kernels called one by one) ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("pandas: numpyの汎用カーネルを\nPythonレイヤから逐次呼び出し", fontsize=12, weight="bold")
    ax.axis("off")

    ax.text(5.0, 9.4, "(df['a'] * 2 + df['b']).sum()", ha="center", fontsize=10, style="italic")

    kx, kw, kh = 0.6, 2.2, 0.85
    ky = 8.0
    box(ax, kx, ky, kw, kh, "numpyカーネル\nmultiply(a, 2)", fc="#ffffff", fontsize=8.3)
    box(ax, kx + kw + 1.0, ky, kw, kh, "numpyカーネル\nadd(_, b)", fc="#ffffff", fontsize=8.3)
    box(ax, kx + 2 * (kw + 1.0), ky, kw, kh, "numpyカーネル\nsum(_)", fc="#ffffff", fontsize=8.3)

    iy, ih, iw = 6.1, 1.1, kw
    box(ax, kx, iy, iw, ih, "中間配列1\n(N要素, 新規確保)", fc=C_BAD_FILL, fontsize=8.0)
    box(ax, kx + kw + 1.0, iy, iw, ih, "中間配列2\n(N要素, 新規確保)", fc=C_BAD_FILL, fontsize=8.0)
    box(ax, kx + 2 * (kw + 1.0), iy, iw, ih, "結果\n(スカラー)", fc=C_GOOD, fontsize=8.0)

    arrow(ax, (kx + kw / 2, ky), (kx + kw / 2, iy + ih), color=C_PANDAS, lw=1.6)
    arrow(ax, (kx + kw, iy + ih / 2), (kx + kw + 1.0, ky + kh / 2), color=C_EDGE,
          connectionstyle="arc3,rad=-0.2")
    arrow(ax, (kx + kw + 1.0 + kw / 2, ky), (kx + kw + 1.0 + kw / 2, iy + ih), color=C_PANDAS, lw=1.6)
    arrow(ax, (kx + kw + 1.0 + kw, iy + ih / 2), (kx + 2 * (kw + 1.0), ky + kh / 2), color=C_EDGE,
          connectionstyle="arc3,rad=-0.2")
    arrow(ax, (kx + 2 * (kw + 1.0) + kw / 2, ky), (kx + 2 * (kw + 1.0) + kw / 2, iy + ih), color=C_EDGE, lw=1.6)

    box(ax, 0.6, 3.8, 8.8, 1.4,
        "式ごとにカーネル呼び出しが分かれているため、演算のたびに\n"
        "N要素の中間配列を新規確保・書き込みする(この例で中間配列2個)",
        fc=C_WARN, fontsize=9)
    ax.text(5.0, 2.9, "→ データを複数回読み書きする(multiplyでN回書き込み→addでN回読み書き→sumでN回読み込み)",
            ha="center", fontsize=8.2, color="#555555")

    # ---- Right: polars (fused Rust kernel) ----
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("polars: Rust実装の専用カーネルで\n式を融合実行(fusion)", fontsize=12, weight="bold")
    ax.axis("off")

    ax.text(5.0, 9.4, "pl.col('a') * 2 + pl.col('b')).sum()  # 1つの実行式として評価", ha="center",
            fontsize=8.7, style="italic")

    box(ax, 1.5, 7.4, 7.0, 1.7,
        "融合カーネル(Rust, コンパイル済み)\nmultiply + add + sum を1つのループに融合し、SIMDでベクトル化",
        fc=C_DATA_FILL, fontsize=9.3)

    arrow(ax, (5.0, 7.4), (5.0, 6.3), color=C_ARROW, lw=1.8)
    box(ax, 3.0, 5.4, 4.0, 0.9, "結果\n(スカラー)", fc=C_GOOD, fontsize=8.5)

    box(ax, 0.6, 3.6, 8.8, 1.4,
        "式全体を1つのカーネルにまとめて実行するため、\n"
        "multiply/addの結果を中間配列として確保しない(中間配列0個)",
        fc=C_GOOD, fontsize=9)
    ax.text(5.0, 2.7, "→ 基本的にデータへの走査は1パスで済む(メモリ帯域・キャッシュ効率が有利)",
            ha="center", fontsize=8.2, color="#555555")
    ax.text(5.0, 1.8, "※ 融合の範囲は式の内容やクエリオプティマイザの判断に依存し、常に完全1パスとは限らない",
            ha="center", fontsize=8.0, color="#777777")

    fig.tight_layout()
    out = os.path.join(IMAGES, "diagram_kernel_fusion.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def diagram_streaming():
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.0))

    # ---- Left: full materialization ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("全データ実体化\n(pandas / polarsのeager実行)", fontsize=12, weight="bold")
    ax.axis("off")

    box(ax, 1.0, 6.4, 5.6, 2.6, "全データ(N行)を\nメモリ上に読み込んでから処理", fc=C_BAD_FILL, fontsize=10)
    arrow(ax, (3.8, 6.4), (3.8, 5.2), color=C_PANDAS, lw=1.8)
    box(ax, 1.0, 3.9, 5.6, 1.1, "集計処理を実行", fc="#ffffff", fontsize=9.5)
    arrow(ax, (3.8, 3.9), (3.8, 2.7), color=C_EDGE, lw=1.6)
    box(ax, 1.0, 1.7, 5.6, 0.9, "最終結果", fc=C_GOOD, fontsize=9.5)

    # peak-memory gauge
    gx, gy, gw, gh = 7.3, 1.7, 1.4, 7.3
    ax.add_patch(Rectangle((gx, gy), gw, gh, fill=False, edgecolor=C_EDGE, linewidth=1.3, zorder=2))
    ax.add_patch(Rectangle((gx, gy), gw, gh, facecolor=C_BAD_FILL, edgecolor="none", zorder=1))
    ax.text(gx + gw / 2, gy + gh + 0.3, "ピークメモリ", ha="center", fontsize=8.5, weight="bold")
    ax.text(gx + gw / 2, gy + gh / 2, "≈\nデータ\n全体", ha="center", va="center", fontsize=8.5, zorder=3)

    ax.text(5.0, 0.7, "処理全体を通じて、読み込んだデータをずっと保持し続ける", ha="center", fontsize=8.5, color="#555555")

    # ---- Right: batch pipeline (polars streaming) ----
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("バッチパイプライン\n(polars streaming engine)", fontsize=12, weight="bold")
    ax.axis("off")

    batch_y, batch_h, batch_w = 8.0, 1.1, 1.9
    batch_xs = [0.8, 3.3, 5.8]
    for i, bx in enumerate(batch_xs):
        fc = "#eeeeee" if i < 2 else C_DATA_FILL
        label = f"バッチ{i}\n(処理済・解放)" if i < 2 else f"バッチ{i}\n(処理中)"
        box(ax, bx, batch_y, batch_w, batch_h, label, fc=fc, fontsize=7.6)
        arrow(ax, (bx + batch_w / 2, batch_y), (bx + batch_w / 2, 6.5), color=C_ARROW, lw=1.4,
              connectionstyle="arc3,rad=0.0")

    box(ax, 0.8, 5.2, 6.9, 1.1, "集計状態(アキュムレータ): バッチが来るたびに更新、常に小さいサイズのまま",
        fc=C_ALT_FILL, fontsize=8.3)
    arrow(ax, (4.2, 5.2), (4.2, 4.1), color=C_EDGE, lw=1.6)
    box(ax, 1.5, 2.9, 5.4, 1.1, "最終結果", fc=C_GOOD, fontsize=9.5)

    gx2, gy2, gw2, gh2 = 8.3, 2.9, 1.1, 6.2
    peak_h = gh2 * 0.22
    ax.add_patch(Rectangle((gx2, gy2), gw2, gh2, fill=False, edgecolor=C_EDGE, linewidth=1.3, zorder=2))
    ax.add_patch(Rectangle((gx2, gy2), gw2, peak_h, facecolor=C_DATA_FILL, edgecolor="none", zorder=1))
    ax.text(gx2 + gw2 / 2, gy2 + gh2 + 0.3, "ピークメモリ", ha="center", fontsize=8.5, weight="bold")
    ax.text(gx2 + gw2 / 2, gy2 + gh2 * 0.45, "≈\nバッチ1つ\n+集計状態", ha="center", va="center", fontsize=7.8, zorder=3)

    ax.text(5.0, 1.8, "処理し終えたバッチは解放され、常時メモリに残るのはバッチ1つ分と集計状態だけ",
            ha="center", fontsize=8.3, color="#555555")
    ax.text(5.0, 0.9, "※ streamingは全演算に対応するわけではなく、非対応の演算は通常(非streaming)実行にフォールバックする",
            ha="center", fontsize=8.0, color="#777777")

    fig.tight_layout()
    out = os.path.join(IMAGES, "diagram_streaming.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    diagram_string_memory()
    diagram_blockmanager()
    diagram_chunks()
    diagram_memory_layers()
    diagram_null_representation()
    diagram_execution_model()
    diagram_thread_model()
    diagram_kernel_fusion()
    diagram_streaming()
