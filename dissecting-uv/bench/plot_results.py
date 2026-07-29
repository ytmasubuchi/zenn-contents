"""実験1(インストール速度ベンチマーク)の結果を横棒グラフのPNGに変換する。

数値は results/summary.md の実験1(中央値、Poetry 2.4.1が主数値)をそのまま使用している。
このスクリプトはDockerコンテナ内(python:3.12-slim + fonts-noto-cjk + matplotlib)で実行することを
前提としている。日本語ラベルを使うため、fonts-noto-cjkが入っていない環境では文字化け(豆腐)する。

色はdatavizスキルの参照パレット(light mode)から採用し、以下のコマンドで検証済み:
  python3 scripts/validate_palette.py "#2a78d6,#eb6834,#1baf7a" --mode light
  → Lightness band / Chroma floor / CVD separation / Normal-vision floor はPASS。
    Contrast vs surface はaqua(#1baf7a)のみ2.74:1でWARN(relief)だが、
    y軸のツールラベル(pip/poetry/uv)自体が識別を担うため色だけに依存しない設計にしている。

実行方法(リポジトリルートで、ホストは汚染しない):
  docker run --rm \
      -v "$(pwd)/dissecting-uv:/work/dissecting-uv" \
      -v "$(pwd)/images/dissecting-uv:/work/images/dissecting-uv" \
      -w /work python:3.12-slim bash -c "
        apt-get update && apt-get install -y --no-install-recommends fonts-noto-cjk \
          && pip install --no-cache-dir matplotlib \
          && python3 dissecting-uv/bench/plot_results.py"
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # dissecting-uv/
REPO_ROOT = os.path.dirname(ROOT)  # zenn-contents/
OUT_DIR = os.path.join(REPO_ROOT, "images", "dissecting-uv")
os.makedirs(OUT_DIR, exist_ok=True)

# dataviz参照パレット(light mode)由来。CVD/コントラストは上記コマンドで検証済み。
COLOR_LOCK = "#2a78d6"    # slot1 blue: 依存解決 (lock/resolve)
COLOR_INSTALL = "#eb6834"  # slot2 orange: 取得・展開 (install/sync)
COLOR_PIP = "#1baf7a"      # slot3 aqua: pip(一括処理、lock/installの区別なし)
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
SURFACE = "#fcfcfb"

# summary.md 実験1(中央値、Poetry 2.4.1主数値)より。単位は秒。
# (label, lock_or_none, install_or_none, total)
ROWS = [
    ("pip (cold)", None, None, 25.48),
    ("pip (warm)", None, None, 21.31),
    ("poetry (cold)", 5.73, 7.56, 13.29),
    ("poetry (warm)", 4.18, 5.72, 9.97),
    ("uv (cold)", 0.64, 2.15, 2.82),
    ("uv (warm)", 0.07, 0.14, 0.20),
]

# グループ間(pip/poetry/uv)に余白を入れたy座標。0が最上段になるよう後でinvert_yaxisする。
Y_POSITIONS = [0.0, 1.0, 2.8, 3.8, 5.6, 6.6]
BAR_HEIGHT = 0.62


def main():
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for y, (label, lock, install, total) in zip(Y_POSITIONS, ROWS):
        if lock is None:
            # pip: 単一バー(lock/installを分離計測していない)
            ax.barh(y, total, height=BAR_HEIGHT, color=COLOR_PIP, zorder=3)
        else:
            ax.barh(y, lock, height=BAR_HEIGHT, color=COLOR_LOCK, zorder=3)
            ax.barh(y, install, left=lock, height=BAR_HEIGHT, color=COLOR_INSTALL, zorder=3)
            # 積み上げ境界に2px相当のサーフェス色ギャップを入れる(枠線ではなく区切り線)
            ax.plot(
                [lock, lock],
                [y - BAR_HEIGHT / 2, y + BAR_HEIGHT / 2],
                color=SURFACE,
                linewidth=2,
                zorder=4,
                solid_capstyle="butt",
            )
        # 合計値ラベル(必ず表示。色は本文用のink、バーの色は使わない)
        ax.text(
            total + 0.35,
            y,
            f"{total:.2f}s",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color=INK_PRIMARY,
            zorder=5,
        )

    ax.set_yticks(Y_POSITIONS)
    ax.set_yticklabels([r[0] for r in ROWS], fontsize=10.5, color=INK_PRIMARY)
    ax.invert_yaxis()  # 先頭(pip cold)を最上段に

    ax.set_xlim(0, 29)
    ax.set_xlabel("所要時間（秒、中央値・線形軸）", fontsize=10.5, color=INK_MUTED)
    ax.set_title(
        "pip / poetry / uv のインストール時間比較\n(streamlit、cold=キャッシュなし、warm=キャッシュあり)",
        fontsize=12.5,
        color=INK_PRIMARY,
        pad=14,
    )

    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=9.5)

    legend_handles = [
        Patch(facecolor=COLOR_LOCK, label="依存解決 (lock/resolve)"),
        Patch(facecolor=COLOR_INSTALL, label="取得・展開 (install/sync)"),
        Patch(facecolor=COLOR_PIP, label="pip (一括処理、内訳非分離)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.99, 0.02),
    )

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "exp1_install_speed.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
