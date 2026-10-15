"""Task 16: 実測依存の図版(F4・F8)。

`results/report.json`(Task 11の比較器の出力)と`results/*__*__*.json`
(Task 10のランナーが書き出した生の結果ファイル)を読み、matplotlibで
F4(保証レベル×環境条件のマトリクス)とF8(LLMのmargin分布・温度別の
分岐位置)を描く。F9(判定フローチャート)はこのモジュールの対象外
(手書きSVG、`images_src/f9_flowchart.svg`)。

**このモジュールが「正」とするのは`results/`配下のファイルだけである**。
F4のセルの✓/×/△/n=1/–は、`reprobench.compare`が既に計算した
`report.json`の`agreement`/`groups`/`stability`フィールドを読んで判定
する――このモジュール自身はdigestの一致判定を一切再実装しない。1箇所
だけ、`report.json`が粒度を落としてしまっている内訳(`S8_curand_randn_
sizes`のサイズ別digest、`S9_logit_margin`のステップ別margin、`S9_
sampling_*`のトークン列)を生の結果ファイルから直接読む。これも
「読み込み元が増える」だけで、値を手で書き起こすことはしない。

**測定された相関だけを書く**規律(プロジェクト全体の方針)は図にも
適用される: ある条件で複数のプローブが食い違う判定を出したとき、
セルを単一の○×に均さず(ブリーフの指示)、`±`(混在)という専用の
判定を導入し、内訳は図下部の脚注に出す。
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from reprobench.compare import label_for

# --- フォント: base イメージに追加した Noto Sans CJK JP を明示指定する
# (Task 16 で Dockerfile.base に追加。matplotlib 自身のデフォルトフォント
# (DejaVu Sans) は日本語グリフを持たないため、指定しないと全角文字が
# 豆腐(tofu)になる。scripts/svg_to_png.sh が SVG 変換用の使い捨てイメージ
# に fonts-noto-cjk を入れているのと同じ対処をこちらにも行う)。
_CJK_FONT = "Noto Sans CJK JP"


def _configure_fonts() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    if _CJK_FONT in available:
        plt.rcParams["font.family"] = _CJK_FONT
    else:
        # CJKフォントが無い環境(例: コンテナ外でこのモジュールを素の
        # ホストpythonから試す場合)でも、文字化けで落ちるのではなく
        # 欠落グリフとして警告を出しつつ描画は続ける。
        import warnings

        warnings.warn(
            f"{_CJK_FONT!r} not found; Japanese text will render as tofu. "
            "Rebuild reprobench:base (Dockerfile.base installs fonts-noto-cjk)."
        )
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


# --- レベルの色(F3 images_src/f3_levels.svg と同じパレット) ---------

LEVEL_COLORS = {
    "L0": "#1b5e20",
    "L1": "#558b2f",
    "L2": "#f9a825",
    "L3": "#ef6c00",
    "L4": "#c62828",
}
LEVEL_TEXT_COLORS = {  # L2のchipは明色なので文字は濃色にする(F3と同じ)
    "L0": "#ffffff",
    "L1": "#ffffff",
    "L2": "#1f2937",
    "L3": "#ffffff",
    "L4": "#ffffff",
}
LEVEL_ORDER = ["L0", "L1", "L2", "L3", "L4"]
LEVEL_SHORT = {
    "L0": "規格",
    "L1": "明文の規約",
    "L2": "事実上の実装",
    "L3": "実行時環境",
    "L4": "保証なし",
}

# 判定記号の色(✓=緑/×=赤 は L0/L4 と同じ色相を流用し、「どちらに近いか」
# を直感的に対応させる。±=混在はL3のオレンジ=「環境次第」と同じ発想)。
VERDICT_COLOR = {
    "✓": "#1b5e20",
    "×": "#c62828",
    "±": "#ef6c00",
    "△": "#555f73",
    "n=1": "#9aa0a6",
    "–": "#c7ccd1",
}
VERDICT_FILL = {
    "✓": "#e8f5e9",
    "×": "#fdecea",
    "±": "#fff3e0",
    "△": "#eceff1",
    "n=1": "#f5f5f5",
    "–": "#fafafa",
}

GPU_HOSTS = ["anakin", "blade", "dgxspark"]
GPU_LABELS = [f"{h}/gpu/threadsmax" for h in GPU_HOSTS]


# --- データ読み込み ---------------------------------------------------


def load_report(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_raw_results(results_dir: Path) -> dict[str, dict]:
    """`results/*__*__*.json` を全部読み、`label_for(env)` をキーにした
    辞書で返す(`reprobench.compare.label_for` と同じラベル規則を使い、
    report.json のラベルとずれないようにする)。
    """
    out: dict[str, dict] = {}
    for path in sorted(glob.glob(str(results_dir / "*__*__*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        out[label_for(data.get("env", {}))] = data
    return out


def _raw_probe_meta(raw: dict[str, dict], label: str, probe: str) -> dict | None:
    """`raw[label]`の中から指定プローブの(スキップされていない)`meta`を
    探す。無ければ`None`。
    """
    result = raw.get(label)
    if result is None:
        return None
    for entry in result.get("probes", []):
        if entry.get("probe") == probe and "skipped" not in entry:
            return entry.get("meta", {})
    return None


# --- report.json から判定記号を引く汎用ヘルパー ------------------------
#
# これらの関数は digest の一致判定を一切再実装しない。既に`reprobench.
# compare.build_report`が計算した`agreement`/`groups`/`stability`を
# 読むだけである。


def _digest_group(groups: dict[str, list[str]], label: str) -> str | None:
    for dig, labels in groups.items():
        if label in labels:
            return dig
    return None


def pair_verdict(report: dict, probe: str, label_a: str, label_b: str, desc: str) -> tuple[str, str]:
    """2ラベル間でdigestが一致したか。"""
    row = report["rows"].get(probe)
    if row is None or "agreement" not in row:
        return "–", f"{probe}: {desc} — report.jsonに無い"
    groups = row["agreement"]["groups"]
    ga, gb = _digest_group(groups, label_a), _digest_group(groups, label_b)
    if ga is None or gb is None:
        missing = label_a if ga is None else label_b
        return "–", f"{probe}: {desc} — {missing} 欠損"
    if ga == gb:
        return "✓", f"{probe}: {desc} — 一致"
    return "×", f"{probe}: {desc} — 不一致"


def group_verdict(report: dict, probe: str, labels: list[str], desc: str) -> tuple[str, str]:
    """複数ラベルが互いに一致したか(観測1件ならn=1)。"""
    row = report["rows"].get(probe)
    if row is None or "agreement" not in row:
        return "–", f"{probe}: {desc} — report.jsonに無い"
    groups = row["agreement"]["groups"]
    present = [l for l in labels if _digest_group(groups, l) is not None]
    if not present:
        return "–", f"{probe}: {desc} — 観測なし"
    if len(present) == 1:
        return "n=1", f"{probe}: {desc} — 観測1件のみ({present[0]})"
    digs = {_digest_group(groups, l) for l in present}
    if len(digs) == 1:
        return "✓", f"{probe}: {desc} — {len(present)}件すべて一致"
    return "×", f"{probe}: {desc} — {len(present)}件中{len(digs)}種類に分裂"


def self_report_verdict(report: dict, probe: str, labels: list[str], desc: str) -> tuple[str, str]:
    """自己申告安定性(△)。プローブ自身が同一プロセス内で2回実行して
    比較した`stable_within_process`を読む(環境間のdigest不一致では
    ない)。
    """
    row = report["rows"].get(probe)
    if row is None or "stability" not in row:
        return "–", f"{probe}: {desc} — report.jsonに無い"
    stability = row["stability"]
    vals = {stability.get(l) for l in labels if l in stability}
    vals.discard(None)
    if not vals:
        return "–", f"{probe}: {desc} — 自己申告なし"
    if len(vals) > 1:
        return "△", f"{probe}: {desc} — 自己申告が割れた(unknown)"
    stable = next(iter(vals))
    return "△", f"{probe}: {desc} — {'安定' if stable else '不安定'}(自己申告)"


def grouped_value_verdict(report: dict, probe: str, group_value: str, desc: str) -> tuple[str, str]:
    """`kind=grouped`のプローブ(例: S5_kmeans_centers)について、ある
    `group_by_meta`の値1本ぶんの判定を引く。
    """
    row = report["rows"].get(probe)
    if row is None or "groups" not in row:
        return "–", f"{probe}[{group_value}]: {desc} — report.jsonに無い"
    sub = row["groups"].get(str(group_value))
    if sub is None:
        return "–", f"{probe}[{group_value}]: {desc} — 観測なし"
    present = sum(len(v) for v in sub["groups"].values())
    if present == 0:
        return "–", f"{probe}[{group_value}]: {desc} — 観測なし"
    if present == 1:
        return "n=1", f"{probe}[{group_value}]: {desc} — 観測1件のみ"
    return (
        ("✓", f"{probe}[{group_value}]: {desc} — {present}件すべて一致")
        if sub["all_equal"]
        else ("×", f"{probe}[{group_value}]: {desc} — {present}件中複数種類に分裂")
    )


def cross_probe_pair_verdict(report: dict, probe_a: str, probe_b: str, label: str, desc: str) -> tuple[str, str]:
    """同一ラベルについて、**2つの異なるプローブ**のdigestが一致したか
    (例: デフォルト経路 vs `use_deterministic_algorithms(True)`経路)。
    `report.json`は同じプローブ名同士しか比較しないため、この比較だけは
    2つの行から直接digestを引いて自前で突き合わせる(一致判定の再実装
    ではなく、report.jsonが既に持つdigestグループを読むだけ)。
    """
    row_a, row_b = report["rows"].get(probe_a), report["rows"].get(probe_b)
    if row_a is None or row_b is None or "agreement" not in row_a or "agreement" not in row_b:
        return "–", f"{probe_a} vs {probe_b}: {desc} — report.jsonに無い"
    ga = _digest_group(row_a["agreement"]["groups"], label)
    gb = _digest_group(row_b["agreement"]["groups"], label)
    if ga is None or gb is None:
        return "–", f"{probe_a} vs {probe_b}: {desc} — {label} 欠損"
    return ("✓", f"{probe_a} vs {probe_b}: {desc} — 一致") if ga == gb else ("×", f"{probe_a} vs {probe_b}: {desc} — 不一致")


def curand_size_threshold_items(raw: dict[str, dict]) -> list[tuple[str, str]]:
    """`S8_curand_randn_sizes`のサイズ別digest(`meta['digests']`)を
    3GPUホスト間で比較する。`report.json`はこのプローブ全体を1個の
    digestとして持つため(4サイズをまとめた1つの判定)、サイズごとの
    閾値効果(brief: n=1024/65536では一致、n=4194304/67108864では
    全GPUが異なる)はそこからは読めない――生の結果ファイルの`meta`を
    直接読む、このモジュール内で唯一の例外。
    """
    metas = {h: _raw_probe_meta(raw, label, "S8_curand_randn_sizes") for h, label in zip(GPU_HOSTS, GPU_LABELS)}
    metas = {h: m for h, m in metas.items() if m is not None}
    if len(metas) < 2:
        return [("–", "S8_curand_randn_sizes: 観測不足")]
    sizes = sorted({s for m in metas.values() for s in m.get("digests", {})}, key=int)
    items = []
    for size in sizes:
        digs = {h: m["digests"].get(size) for h, m in metas.items() if size in m.get("digests", {})}
        digs = {h: d for h, d in digs.items() if d is not None}
        if len(digs) < 2:
            items.append(("–", f"S8_curand_randn_sizes[n={size}]: 観測不足"))
            continue
        uniq = set(digs.values())
        if len(uniq) == 1:
            items.append(("✓", f"S8_curand_randn_sizes[n={size}]: {len(digs)}GPUすべて一致"))
        else:
            items.append(("×", f"S8_curand_randn_sizes[n={size}]: {len(digs)}GPU中{len(uniq)}種類に分裂"))
    return items


# --- F4: セルの集約 -----------------------------------------------------


def aggregate_cell(items: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """1セル(level×condition)ぶんの複数エビデンスを、単一の表示記号と
    脚注行のリストへまとめる。

    ✓と×が両方そろったら単純な多数決ではなく`±`(混在)にする――
    ブリーフの指示(「一致したセル」と「不一致のセルと一致のセルが
    混在する条件」を同じ記号に均さない)。
    """
    conclusive = [(s, d) for s, d in items if s in ("✓", "×")]
    self_reports = [(s, d) for s, d in items if s == "△"]
    singles = [(s, d) for s, d in items if s == "n=1"]
    if conclusive:
        uniq = {s for s, _ in conclusive}
        if uniq == {"✓"}:
            symbol = "✓"
        elif uniq == {"×"}:
            symbol = "×"
        else:
            symbol = "±"
        return symbol, [d for _, d in conclusive]
    if self_reports:
        return "△", [d for _, d in self_reports]
    if singles:
        return "n=1", [d for _, d in singles]
    return "–", []


# --- F4: 条件の定義 ------------------------------------------------------

# (キー, 列見出し(改行区切り), 所属バンド)。バンドの境目は design spec
# (docs/.../2026-10-14-randomness-reproducibility-design.md §4-2)の
# G1〜G4 と同じ区切り。
CONDITIONS: list[tuple[str, str, str]] = [
    ("1a", "同一プロセス\n内で2回", "G1 時間・コンテナ"),
    ("1b", "同一コンテナ\nを再起動", "G1 時間・コンテナ"),
    ("1c", "同一イメージ\nから\n別コンテナ", "G1 時間・コンテナ"),
    ("2a", "Python\nバージョン", "G2 ソフトウェア"),
    ("2b", "ライブラリ\nバージョン", "G2 ソフトウェア"),
    ("2c", "BLAS実装\n(OpenBLAS\n/MKL)", "G2 ソフトウェア"),
    ("2d", "ベース\nイメージ\nのlibc", "G2 ソフトウェア"),
    ("3a", "スレッド数", "G3 ハードウェア"),
    ("3b", "同アーキ別\nSIMD世代", "G3 ハードウェア"),
    ("3c", "別アーキ\n(x86_64/\naarch64)", "G3 ハードウェア"),
    ("3d", "SIMD明示\n無効化\n(対照実験)", "G3 ハードウェア"),
    ("4a", "CPU vs\nGPU", "G4 アクセラレータ"),
    ("4b", "GPU反復\n実行", "G4 アクセラレータ"),
    ("4c", "決定的\nモード", "G4 アクセラレータ"),
    ("4d", "dtype/\nTF32", "G4 アクセラレータ"),
    ("4e", "別GPU\nモデル", "G4 アクセラレータ"),
]


def build_evidence(report: dict, raw: dict[str, dict]) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """(level, condition_key) -> [(symbol, detail), ...] を組み立てる。

    ここにあるのは「どのプローブ・どのラベルを比べるか」という設計
    (条件軸は design spec の G1〜G4、レベル軸は F3)だけであり、
    ✓/×そのものはすべて上のヘルパー経由でreport.json/生の結果ファイル
    から計算する。
    """
    ev: dict[tuple[str, str], list[tuple[str, str]]] = {}

    def add(level: str, cond: str, item: tuple[str, str]) -> None:
        ev.setdefault((level, cond), []).append(item)

    # --- L0: 公開アルゴリズム仕様(生PRNGビット列) ---
    add("L0", "1b", pair_verdict(report, "S1_stdlib_mt19937", "anakin/base/hashseed0", "anakin/base/hashseedfree", "PYTHONHASHSEED 0 vs unset"))
    four_hosts_threadsmax = ["local/base/threadsmax", "anakin/base/threadsmax", "blade/base/threadsmax", "dgxspark/base/threadsmax"]
    add("L0", "1c", group_verdict(report, "S1_stdlib_mt19937", four_hosts_threadsmax, "4ホスト, 同imageから別コンテナ"))
    add("L0", "2a", pair_verdict(report, "S1_stdlib_mt19937", "blade/base/threadsmax", "blade/py313/threadsmax", "Python 3.11 vs 3.13"))
    add("L0", "2a", pair_verdict(report, "S1_stdlib_mt19937", "blade/base/threadsmax", "blade/oldpy/threadsmax", "Python 3.11 vs 3.10"))
    add("L0", "2b", pair_verdict(report, "S1_numpy_pcg64", "blade/base/threadsmax", "blade/oldlibs/threadsmax", "numpy 2.2.6 vs 1.21.6"))
    add("L0", "2c", pair_verdict(report, "S1_stdlib_mt19937", "blade/base/threadsmax", "blade/mkl/threadsmax", "OpenBLAS vs MKL"))
    add("L0", "2d", pair_verdict(report, "S1_stdlib_mt19937", "blade/base/threadsmax", "blade/alpine/threadsmax", "glibc vs musl"))
    add("L0", "3a", group_verdict(report, "S1_stdlib_mt19937", ["anakin/base/threads1", "anakin/base/threads8", "anakin/base/threadsmax"], "スレッド数 1/8/max"))
    add("L0", "3b", pair_verdict(report, "S1_stdlib_mt19937", "anakin/base/threadsmax", "blade/base/threadsmax", "AVX2(anakin) vs AVX-512(blade)"))
    add("L0", "3c", pair_verdict(report, "S1_stdlib_mt19937", "anakin/base/threadsmax", "dgxspark/base/threadsmax", "x86_64 vs aarch64"))
    add("L0", "3d", pair_verdict(report, "S1_stdlib_mt19937", "blade/base/nosimd512", "blade/base/threadsmax", "AVX-512無効化 前後"))
    add("L0", "4a", pair_verdict(report, "S1_stdlib_mt19937", "anakin/base/threadsmax", "anakin/gpu/threadsmax", "CPU専用image vs GPU image(同一ホスト)"))
    add("L0", "4e", group_verdict(report, "S1_stdlib_mt19937", GPU_LABELS, "3 GPUホスト(CPU側の生PRNGなのでGPUモデルは無関係のはず)"))

    # --- L1: ライブラリが文書で約束(NEP19, RandomStateの版間安定性) ---
    add("L1", "2b", pair_verdict(report, "S1_numpy_randomstate", "blade/base/threadsmax", "blade/oldlibs/threadsmax", "legacy RandomState, numpy 2.2.6 vs 1.21.6"))

    # --- L2: 文書化なし、実装がたまたま決めている(正規乱数の変換・np.sum) ---
    add("L2", "2a", pair_verdict(report, "S1_numpy_normal_generator", "blade/base/threadsmax", "blade/py313/threadsmax", "ziggurat変換, Python 3.11 vs 3.13"))
    add("L2", "2b", pair_verdict(report, "S1_generator_random_float32", "blade/base/threadsmax", "blade/oldlibs/threadsmax", "Generator float32一様分布(NEP19: 非保証), numpy 2.2.6 vs 1.21.6"))
    add("L2", "2b", pair_verdict(report, "S1_numpy_normal_generator", "blade/base/threadsmax", "blade/oldlibs/threadsmax", "ziggurat変換(無保証だが実際には不変), numpy 2.2.6 vs 1.21.6"))
    add("L2", "3b", pair_verdict(report, "S1_numpy_normal_randomstate", "anakin/base/threadsmax", "blade/base/threadsmax", "Box-Muller(polar)変換, AVX2 vs AVX-512"))
    add("L2", "3b", pair_verdict(report, "S1_numpy_normal_generator", "anakin/base/threadsmax", "blade/base/threadsmax", "ziggurat変換, AVX2 vs AVX-512"))

    # --- L3: 実行時環境が選ぶ(SIMDディスパッチ・BLASスレッド数・libm) ---
    add("L3", "2c", pair_verdict(report, "S3_matmul_float32", "blade/base/threadsmax", "blade/mkl/threadsmax", "A@B float32, OpenBLAS vs MKL"))
    add("L3", "2c", pair_verdict(report, "S3_matmul_float64", "blade/base/threadsmax", "blade/mkl/threadsmax", "A@B float64, OpenBLAS vs MKL"))
    add("L3", "2c", pair_verdict(report, "S4_exp_float64", "blade/base/threadsmax", "blade/mkl/threadsmax", "exp, OpenBLAS vs MKL"))
    add("L3", "2c", pair_verdict(report, "S2_sum_float64", "blade/base/threadsmax", "blade/mkl/threadsmax", "np.sum, OpenBLAS vs MKL"))
    add("L3", "2d", pair_verdict(report, "S4_sin_float64", "blade/base/threadsmax", "blade/alpine/threadsmax", "sin, glibc vs musl"))
    add("L3", "2d", pair_verdict(report, "S4_exp_float64", "blade/base/threadsmax", "blade/alpine/threadsmax", "exp, glibc vs musl"))
    add("L3", "2d", pair_verdict(report, "S4_log_float64", "blade/base/threadsmax", "blade/alpine/threadsmax", "log, glibc vs musl"))
    add("L3", "2d", pair_verdict(report, "S2_sum_float64", "blade/base/threadsmax", "blade/alpine/threadsmax", "np.sum, glibc vs musl"))
    threads3 = ["anakin/base/threads1", "anakin/base/threads8", "anakin/base/threadsmax"]
    add("L3", "3a", group_verdict(report, "S3_matmul_float32", threads3, "A@B float32, スレッド数 1/8/max"))
    add("L3", "3a", group_verdict(report, "S3_matmul_float64", threads3, "A@B float64, スレッド数 1/8/max"))
    add("L3", "3a", group_verdict(report, "S2_sum_float64", threads3, "np.sum, スレッド数 1/8/max"))
    add("L3", "3a", group_verdict(report, "S5_logreg_coef", threads3, "LogisticRegression係数, スレッド数 1/8/max"))
    add("L3", "3a", grouped_value_verdict(report, "S5_kmeans_centers", "1", "KMeans中心, effective_thread_count=1"))
    add("L3", "3a", grouped_value_verdict(report, "S5_kmeans_centers", "8", "KMeans中心, effective_thread_count=8"))
    add("L3", "3b", pair_verdict(report, "S4_exp_float64", "anakin/base/threadsmax", "blade/base/threadsmax", "exp, AVX2 vs AVX-512 (記事冒頭)"))
    add("L3", "3b", pair_verdict(report, "S4_log_float64", "anakin/base/threadsmax", "blade/base/threadsmax", "log, AVX2 vs AVX-512"))
    add("L3", "3b", pair_verdict(report, "S4_sin_float64", "anakin/base/threadsmax", "blade/base/threadsmax", "sin, AVX2 vs AVX-512"))
    add("L3", "3b", pair_verdict(report, "S2_sum_float32", "anakin/base/threadsmax", "blade/base/threadsmax", "np.sum float32, AVX2 vs AVX-512"))
    add("L3", "3c", pair_verdict(report, "S4_exp_float64", "anakin/base/threadsmax", "dgxspark/base/threadsmax", "exp, x86_64(AVX2) vs aarch64(SVE2)"))
    add("L3", "3c", pair_verdict(report, "S4_log_float64", "anakin/base/threadsmax", "dgxspark/base/threadsmax", "log, x86_64 vs aarch64"))
    add("L3", "3c", pair_verdict(report, "S4_sin_float64", "anakin/base/threadsmax", "dgxspark/base/threadsmax", "sin, x86_64 vs aarch64"))
    add("L3", "3c", pair_verdict(report, "S2_sum_float64", "anakin/base/threadsmax", "dgxspark/base/threadsmax", "np.sum, x86_64 vs aarch64"))
    add("L3", "3d", pair_verdict(report, "S4_exp_float64", "blade/base/nosimd512", "anakin/base/threadsmax", "exp, AVX-512無効化後のblade vs anakin"))
    add("L3", "3d", pair_verdict(report, "S4_log_float64", "blade/base/nosimd512", "anakin/base/threadsmax", "log, AVX-512無効化後のblade vs anakin"))
    add("L3", "3d", pair_verdict(report, "S4_sin_float64", "blade/base/nosimd512", "anakin/base/threadsmax", "sin, AVX-512無効化後のblade vs anakin"))
    add("L3", "4a", pair_verdict(report, "S2_sum_float64", "anakin/base/threadsmax", "anakin/gpu/threadsmax", "np.sum, CPU専用image vs GPU image"))
    add("L3", "4a", pair_verdict(report, "S4_exp_float64", "anakin/base/threadsmax", "anakin/gpu/threadsmax", "exp, CPU専用image vs GPU image"))
    add("L3", "4a", pair_verdict(report, "S7_mlp_forward", "anakin/base/threadsmax", "anakin/gpu/threadsmax", "torch CPU MLP forward, CPU専用image vs GPU image"))
    add("L3", "4a", pair_verdict(report, "S7_mlp_train10", "anakin/base/threadsmax", "anakin/gpu/threadsmax", "torch CPU MLP 10step学習, CPU専用image vs GPU image"))
    add("L3", "4d", group_verdict(report, "S9_greedy_fp32", GPU_LABELS, "LLM greedy生成トークン列, fp32, 3GPU"))
    add("L3", "4d", group_verdict(report, "S9_greedy_fp16", GPU_LABELS, "LLM greedy生成トークン列, fp16, 3GPU"))
    add("L3", "4d", group_verdict(report, "S9_greedy_bf16", GPU_LABELS, "LLM greedy生成トークン列, bf16, 3GPU"))
    add("L3", "4d", group_verdict(report, "S8_matmul_tf32_off", GPU_LABELS, "A@B float32, TF32 off, 3GPU"))
    add("L3", "4d", group_verdict(report, "S8_matmul_tf32_on", GPU_LABELS, "A@B float32, TF32 on, 3GPU"))
    add("L3", "4e", group_verdict(report, "S8_mlp_train10_cuda", GPU_LABELS, "torch CUDA MLP 10step学習, 3GPU"))
    for item in curand_size_threshold_items(raw):
        add("L3", "4e", item)

    # --- L4: 原理的に非決定・保証なし ---
    for host in ["anakin", "blade", "dgxspark", "local"]:
        add("L4", "1b", pair_verdict(report, "S0_hash_of_str", f"{host}/base/hashseed0", f"{host}/base/hashseedfree", f"hash('reproducibility'), {host}, PYTHONHASHSEED 0 vs unset"))
    add("L4", "1c", group_verdict(report, "S0_hash_of_str", four_hosts_threadsmax, "hash('reproducibility'), 4ホスト, 同imageから別コンテナ"))
    add("L4", "1a", self_report_verdict(report, "S0_parallel_completion_order", report["labels"], "並列ワーカーの完了順, 同一プロセス内で2回"))
    add("L4", "1a", self_report_verdict(report, "S0_glob_order", report["labels"], "glob列挙順, 同一プロセス内で2回"))
    add("L4", "1a", self_report_verdict(report, "S8_index_add_atomics", GPU_LABELS, "GPU index_add_(atomicAdd), 同一プロセス内で2回"))
    add("L4", "4b", self_report_verdict(report, "S8_index_add_atomics", GPU_LABELS, "GPU index_add_(atomicAdd), 同一プロセス内で2回"))
    # 4c 決定的モード: デフォルト経路 vs use_deterministic_algorithms(True)
    # 経路が同一ホストで一致するか(GEMM中心のMLPにはそもそも直すべき
    # 非決定性が無い、という実測)。cross_probe_pair_verdict は同一ラベル
    # について2つの異なるプローブ名のdigestを直接比較する(このモジュール
    # で2本だけの特例。他は全部同一プローブ・別ラベルの比較)。
    for label in GPU_LABELS:
        add("L4", "4c", cross_probe_pair_verdict(report, "S8_mlp_train10_cuda", "S8_mlp_train10_cuda_deterministic", label, f"MLP学習 {label}: デフォルト vs deterministic"))
    # atomicAddを使う経路(index_add_)はホスト間でも比較する: 決定的
    # モードを有効にしても、GPUモデルが違えば一致しない(anakin/bladeは
    # 一致、dgxsparkのみ異なる)。
    add("L4", "4c", group_verdict(report, "S8_index_add_atomics_deterministic", GPU_LABELS, "GPU index_add_(atomicAdd), deterministic, 3GPU"))

    return ev


# --- F4: 描画 ------------------------------------------------------------

# 単純な✓/×/△ではなく、脚注で内訳を見せたいセル(ブリーフ:
# 「均さない」)。±(混在)セルは常に脚注対象に自動で入るため、ここに
# 列挙するのはそれ以外で特に見せたい、対照実験・契約の分かれ目。
_FOOTNOTE_CELLS = {
    ("L0", "1c"),
    ("L0", "4e"),
    ("L1", "2b"),
    ("L3", "3b"),
    ("L3", "3d"),
    ("L4", "1b"),
    ("L4", "1c"),
    ("L4", "4c"),
}


def _wrap_footnote(idx: int, level: str, cond_label: str, details: list[str], width: int = 118) -> list[str]:
    import textwrap

    cond_oneline = cond_label.replace("\n", "")
    head = f"{idx}. [{level}×{cond_oneline}] " + " / ".join(details)
    return textwrap.wrap(head, width=width, subsequent_indent="    ")


def _text_width_data(fig, ax, renderer, s: str, fontsize: float, **kw) -> float:
    """文字列`s`をfontsize`fontsize`で描いたときの幅をデータ座標系の単位で
    返す(当たりをつけるための固定係数ではなく、実際にレンダラへ描かせて
    測る)。凡例の間隔・列見出しの折返し幅チェックに使う。
    """
    t = ax.text(0, 0, s, fontsize=fontsize, **kw)
    bbox = t.get_window_extent(renderer=renderer)
    (x0, _y0), (x1, _y1) = ax.transData.inverted().transform([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])
    t.remove()
    return abs(x1 - x0)


def render_f4(report: dict, raw: dict[str, dict], out_path: Path) -> None:
    evidence = build_evidence(report, raw)

    col_w, row_h = 1.7, 1.2
    band_gap = 0.5
    chip_w = 2.85
    header_h = 2.05
    group_h = 0.5
    title_h = 1.9

    col_x: list[float] = []
    x = chip_w + band_gap
    prev_band = None
    for _key, _label, band in CONDITIONS:
        if prev_band is not None and band != prev_band:
            x += band_gap
        col_x.append(x)
        x += col_w
        prev_band = band
    total_w = x + 0.2

    grid_top = title_h + group_h + header_h
    grid_bottom = grid_top + len(LEVEL_ORDER) * row_h

    footnotes: list[tuple[int, str, str, list[str]]] = []
    for level in LEVEL_ORDER:
        for key, label, _band in CONDITIONS:
            items = evidence.get((level, key), [])
            symbol, details = aggregate_cell(items)
            # ±(複数プローブ混在)とΔ(自己申告)は、記号だけでは「何が
            # 安定/不安定だったか」まで伝わらないため、常に脚注を付ける
            # (ブリーフの指示: 均さない)。それ以外に特に見せたい
            # ✓/✗の対照実験は _FOOTNOTE_CELLS で個別に指定する。
            if details and (symbol in ("±", "△") or (level, key) in _FOOTNOTE_CELLS):
                footnotes.append((len(footnotes) + 1, level, label, details))

    footnote_lines: list[str] = []
    for idx, level, cond_label, details in footnotes:
        footnote_lines.extend(_wrap_footnote(idx, level, cond_label, details))
    footnote_h = 0.30 * (len(footnote_lines) + 2) + 0.5

    total_h = grid_bottom + 0.3 + footnote_h

    fig = plt.figure(figsize=(total_w * 0.52, total_h * 0.52))
    ax = fig.add_axes((0.01, 0.01, 0.98, 0.98))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.invert_yaxis()
    ax.axis("off")
    fig.canvas.draw()  # レンダラを初期化する(テキスト幅の実測に使う)
    renderer = fig.canvas.get_renderer()

    # --- タイトル・凡例 ---
    ax.text(0.15, 0.55, "F4. 保証レベル × 環境条件", fontsize=22, fontweight="bold", va="center", color="#111827")
    ax.text(
        0.15,
        1.15,
        "results/report.json(本プロジェクトの実測、4ホスト・7イメージ・6構成、30ラン)から自動生成。"
        "セルは比較器(reprobench.compare)の判定をそのまま読む。",
        fontsize=10.5,
        va="center",
        color="#4b5563",
    )
    legend_items = [
        ("✓", "一致"),
        ("×", "不一致"),
        ("±", "複数プローブが混在(均さない)"),
        ("△", "自己申告安定性(環境間不一致ではない)"),
        ("n=1", "観測1件のみ"),
        ("–", "未測定/比較不能"),
    ]
    lx = 0.15
    ly = 1.65
    for sym, desc in legend_items:
        ax.add_patch(
            FancyBboxPatch(
                (lx, ly - 0.17),
                0.5,
                0.34,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                linewidth=1.0,
                edgecolor=VERDICT_COLOR[sym],
                facecolor=VERDICT_FILL[sym],
            )
        )
        ax.text(lx + 0.25, ly, sym, fontsize=10, ha="center", va="center", color=VERDICT_COLOR[sym], fontweight="bold")
        ax.text(lx + 0.58, ly, desc, fontsize=8.8, va="center", color="#374151")
        tw = _text_width_data(fig, ax, renderer, desc, fontsize=8.8)
        lx += 0.58 + tw + 0.5

    # --- バンド見出し(G1〜G4) ---
    band_spans: list[tuple[str, float, float]] = []
    for (_key, _label, band), cx in zip(CONDITIONS, col_x):
        if band_spans and band_spans[-1][0] == band:
            b, x0, _x1 = band_spans[-1]
            band_spans[-1] = (b, x0, cx + col_w)
        else:
            band_spans.append((band, cx, cx + col_w))
    band_y0 = title_h
    for band, x0, x1 in band_spans:
        ax.add_patch(
            FancyBboxPatch(
                (x0, band_y0 + 0.05),
                x1 - x0,
                group_h - 0.1,
                boxstyle="round,pad=0.01,rounding_size=0.06",
                linewidth=0.8,
                edgecolor="#9ca3af",
                facecolor="#f3f4f6",
            )
        )
        ax.text((x0 + x1) / 2, band_y0 + group_h / 2, band, fontsize=10.5, fontweight="bold", ha="center", va="center", color="#374151")

    # --- 列見出し(条件) ---
    header_fontsize = 9.3
    header_y0 = title_h + group_h
    for key, label, _band in CONDITIONS:
        for line in label.split("\n"):
            w = _text_width_data(fig, ax, renderer, line, fontsize=header_fontsize)
            if w > col_w - 0.1:
                import sys

                print(
                    f"reprobench.figures: WARNING column {key!r} line {line!r} width {w:.2f} "
                    f"exceeds column width {col_w - 0.1:.2f} (will overflow into the next column)",
                    file=sys.stderr,
                )
    for (_key, label, _band), cx in zip(CONDITIONS, col_x):
        ax.text(cx + col_w / 2, header_y0 + header_h / 2, label, fontsize=header_fontsize, ha="center", va="center", color="#111827")
    # 列の区切り線
    for (_key, _label, _band), cx in zip(CONDITIONS, col_x):
        ax.plot([cx, cx], [header_y0, grid_bottom], color="#e5e7eb", linewidth=0.6, zorder=0)

    # --- 行(レベル)チップ + セル ---
    footnote_lookup = {(lv, lb.replace("\n", "")): idx for idx, lv, lb, _ in footnotes}
    for ri, level in enumerate(LEVEL_ORDER):
        ry0 = grid_top + ri * row_h
        ax.add_patch(
            FancyBboxPatch(
                (0.15, ry0 + 0.08),
                chip_w - 0.3,
                row_h - 0.16,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                linewidth=0,
                facecolor=LEVEL_COLORS[level],
            )
        )
        ax.text(
            0.4,
            ry0 + row_h / 2 - 0.16,
            level,
            fontsize=15,
            fontweight="bold",
            va="center",
            color=LEVEL_TEXT_COLORS[level],
        )
        ax.text(
            0.4,
            ry0 + row_h / 2 + 0.22,
            LEVEL_SHORT[level],
            fontsize=8.6,
            va="center",
            color=LEVEL_TEXT_COLORS[level],
        )

        for (key, label, _band), cx in zip(CONDITIONS, col_x):
            items = evidence.get((level, key), [])
            symbol, _details = aggregate_cell(items)
            cy0 = ry0 + 0.08
            ax.add_patch(
                FancyBboxPatch(
                    (cx + 0.06, cy0),
                    col_w - 0.12,
                    row_h - 0.16,
                    boxstyle="round,pad=0.01,rounding_size=0.08",
                    linewidth=1.1,
                    edgecolor=VERDICT_COLOR[symbol],
                    facecolor=VERDICT_FILL[symbol],
                )
            )
            fontsize = 15 if symbol not in ("n=1", "–") else 10
            ax.text(
                cx + col_w / 2,
                ry0 + row_h / 2,
                symbol,
                fontsize=fontsize,
                fontweight="bold" if symbol in ("✓", "×", "±") else "normal",
                ha="center",
                va="center",
                color=VERDICT_COLOR[symbol],
            )
            idx = footnote_lookup.get((level, label.replace("\n", "")))
            if idx is not None:
                ax.text(
                    cx + col_w - 0.14,
                    ry0 + 0.2,
                    str(idx),
                    fontsize=7.5,
                    ha="right",
                    va="top",
                    color="#374151",
                    fontweight="bold",
                )

    # 行の区切り線
    for ri in range(len(LEVEL_ORDER) + 1):
        ry = grid_top + ri * row_h
        ax.plot([chip_w + band_gap - 0.1, total_w - 0.1], [ry, ry], color="#e5e7eb", linewidth=0.6, zorder=0)

    # --- 脚注 ---
    fy = grid_bottom + 0.45
    ax.text(0.15, fy, "脚注(混在セル±・主要な対照実験の内訳):", fontsize=9.5, fontweight="bold", color="#111827")
    fy += 0.34
    for line in footnote_lines:
        ax.text(0.15, fy, line, fontsize=8.3, color="#374151", va="top")
        fy += 0.30

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# --- F8: LLMのmargin分布と温度別の分岐位置 --------------------------------

_SAMPLING_CONDITIONS = [
    ("T=0.7\n(出荷既定)", "S9_sampling_t0_7_seed"),
    ("T=0.7\n(温度のみ単離)", "S9_sampling_t0_7_seed_isolated"),
    ("T=1.2\n(出荷既定)", "S9_sampling_t1_2_seed"),
    ("T=1.2\n(温度のみ単離)", "S9_sampling_t1_2_seed_isolated"),
]


def render_f8_margin(report: dict, raw: dict[str, dict], out_path: Path) -> None:
    margins = {}
    for host, label in zip(GPU_HOSTS, GPU_LABELS):
        meta = _raw_probe_meta(raw, label, "S9_logit_margin")
        if meta is not None:
            margins[host] = meta["margins"]

    row = report["rows"].get("S9_logit_margin", {})
    groups = row.get("agreement", {}).get("groups", {})
    all_token_match = len(groups) <= 1 and sum(len(v) for v in groups.values()) >= 2

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax_hist, ax_diff = axes

    colors = {"anakin": "#1565c0", "blade": "#6a1b9a", "dgxspark": "#ef6c00"}
    import numpy as np  # 局所import: このモジュールの他の場所は numpy を使わない

    bins = np.logspace(np.log10(1e-3), np.log10(20), 28)
    for host, vals in margins.items():
        ax_hist.hist(vals, bins=bins, histtype="step", linewidth=2.2, label=f"{host}", color=colors.get(host, "#333"))
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("top-1 と top-2 の logit 差(margin, 対数スケール)")
    ax_hist.set_ylabel("生成ステップ数(全84ステップ中)")
    ax_hist.set_title("(a) top1/top2 logit margin の分布(greedy生成、全ステップ)")
    ax_hist.legend(fontsize=9, title="GPU")
    ax_hist.grid(alpha=0.25)

    if margins.get("anakin") and margins.get("blade") and margins.get("dgxspark"):
        n = min(len(margins["anakin"]), len(margins["blade"]), len(margins["dgxspark"]))
        diff_ab = [abs(margins["anakin"][i] - margins["blade"][i]) for i in range(n)]
        diff_ad = [abs(margins["anakin"][i] - margins["dgxspark"][i]) for i in range(n)]
        steps = list(range(n))
        ax_diff.plot(steps, diff_ab, label="|anakin − blade|", color="#1565c0", linewidth=1.6)
        ax_diff.plot(steps, diff_ad, label="|anakin − dgxspark|", color="#ef6c00", linewidth=1.6)
        ax_diff.set_yscale("log")
        ax_diff.set_xlabel("生成ステップ(トークン位置)")
        ax_diff.set_ylabel("margin の環境間の絶対差(log scale)")
        ax_diff.set_title("(b) 同じトークン列でも margin 自体はビット一致しない")
        ax_diff.legend(fontsize=9)
        ax_diff.grid(alpha=0.25)

    match_note = (
        "3GPU(RTX 4090 / RTX PRO 6000 / GB10)とも生成トークン列はdigest一致"
        if all_token_match
        else "生成トークン列は環境間で一致しなかった(report.json参照)"
    )
    fig.suptitle(
        f"F8a. LLM greedy生成の logit margin — {match_note}。"
        "margin自体はfloat32の非結合性で環境ごとに最下位ビットが動く(今回はargmaxは動かなかった)。",
        fontsize=11.5,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_f8_divergence(report: dict, raw: dict[str, dict], out_path: Path) -> None:
    from reprobench.probes.s9_llm import first_divergence

    def tokens(probe: str, host: str) -> list[int] | None:
        meta = _raw_probe_meta(raw, f"{host}/gpu/threadsmax", probe)
        return None if meta is None else meta["token_ids"]

    greedy = {h: tokens("S9_greedy_batch1", h) for h in GPU_HOSTS}
    greedy_len = min(len(v) for v in greedy.values() if v is not None)
    greedy_div_ad = first_divergence(greedy["anakin"], greedy["dgxspark"])

    shipped_pts: list[tuple[float, int | None, int]] = []
    isolated_pts: list[tuple[float, int | None, int]] = []
    for label, probe in _SAMPLING_CONDITIONS:
        a = tokens(probe, "anakin")
        b = tokens(probe, "blade")
        d = tokens(probe, "dgxspark")
        if a is None or b is None or d is None:
            continue
        div_ad = first_divergence(a, d)
        temp = 0.7 if "0.7" in label else 1.2
        seq_len = min(len(a), len(b), len(d))
        target = isolated_pts if "単離" in label else shipped_pts
        target.append((temp, div_ad, seq_len))

    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    # greedy(T=0相当)は出荷既定/温度単離の区別が無い(do_sample=Falseでは
    # top_p/top_k/repetition_penaltyの単離有無がそもそも効かない)ので、
    # 両シリーズで共有する1点として1回だけ描く。
    if greedy_div_ad is None:
        ax.scatter([0.0], [greedy_len], marker="*", s=220, color="#2e7d32", zorder=3)
        ax.annotate(
            "分岐なし(生成末尾まで一致)\n— anakin/blade/dgxspark 共通",
            (0.0, greedy_len),
            textcoords="offset points",
            xytext=(12, 0),
            fontsize=8,
            color="#2e7d32",
            ha="left",
            va="center",
        )

    def plot_series(points, marker, color, series_label, label_dy):
        xs = [0.0] + [p[0] for p in points]
        ys_div = [greedy_div_ad] + [p[1] for p in points]
        for x, y in zip(xs, ys_div):
            if y is None:
                continue  # greedyの共有点は上で描画済み
            ax.scatter([x], [y], marker=marker, s=90, color=color, zorder=3)
            ax.annotate(
                f"{y}トークン目",
                (x, y),
                textcoords="offset points",
                xytext=(12, label_dy),
                fontsize=8,
                color=color,
                ha="left",
            )
        xs_line = [x for x, y in zip(xs, ys_div) if y is not None or x == 0.0]
        ys_line = [greedy_len if y is None else y for x, y in zip(xs, ys_div) if y is not None or x == 0.0]
        if len(xs_line) >= 2:
            ax.plot(xs_line, ys_line, color=color, linewidth=1.4, linestyle="--", zorder=2)
        ax.plot([], [], marker=marker, color=color, linestyle="--", label=series_label)

    plot_series(shipped_pts, "o", "#c62828", "出荷既定 generation_config\n(repetition_penalty等込み)", label_dy=14)
    plot_series(isolated_pts, "s", "#1565c0", "温度のみ単離\n(top_p=1, top_k=0, repetition_penalty=1.0)", label_dy=-18)

    ax.set_xticks([0.0, 0.7, 1.2])
    ax.set_xticklabels(["greedy\n(do_sample=False)", "T=0.7", "T=1.2"])
    ax.set_xlim(-0.15, 1.55)
    ax.set_ylim(-5, max(greedy_len, 95) * 1.12)
    ax.set_xlabel("temperature")
    ax.set_ylabel("anakin と dgxspark が最初に食い違うトークン位置")
    ax.set_title("F8b. 温度を上げるほど、GPUモデル間の分岐は早いトークンで起きる\n(anakin と blade は全温度でトークン列が完全一致)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.3, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --- エントリポイント -----------------------------------------------------


def main() -> None:
    results_dir = Path(os.environ.get("REPRO_RESULTS_DIR", "results"))
    images_dir = Path(os.environ.get("REPRO_IMAGES_DIR", "/images/reproducibility-of-randomness"))
    images_dir.mkdir(parents=True, exist_ok=True)

    _configure_fonts()

    report_path = results_dir / "report.json"
    if not report_path.exists():
        raise SystemExit(
            f"reprobench.figures: {report_path} not found -- run `make report` first "
            "(figures.py reads report.json, it does not build it)."
        )
    report = load_report(report_path)
    raw = load_raw_results(results_dir)

    render_f4(report, raw, images_dir / "f4_matrix.png")
    render_f8_margin(report, raw, images_dir / "f8_llm_margin.png")
    render_f8_divergence(report, raw, images_dir / "f8_llm_divergence.png")
    print(f"reprobench.figures: wrote f4_matrix.png, f8_llm_margin.png, f8_llm_divergence.png to {images_dir}")


if __name__ == "__main__":
    main()
