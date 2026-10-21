"""環境横断の結果比較器。

Task 10のランナーがホスト・イメージ・構成ごとに書き出したJSONファイル群を
読み込み、プローブごとに「どのラベルがどのラベルと一致したか」を判定し、
記事の付録表に貼れるMarkdownへ整形する。

このモジュールの出力は、記事が「これは再現する・これはしない」と述べる
唯一の根拠になる。したがって最優先の失敗モードは、一致していないものを
一致と報告すること・不一致ではないものを不一致と報告することであり、
とりわけ次の5点を誤ると実在しない知見を記事に刻んでしまう:

(1) そのラベルにこのプローブの結果が無い(欠損)ことを「一致」として
    数えてしまうこと(resolution 8)。
(2) `S0_parallel_completion_order`/`S0_glob_order`/`S8_index_add_atomics`/
    `S12_cudnn_benchmark_deterministic`/`S15_index_add_repeats`のように、
    プローブ自身が「この不安定さ自体が測定対象である」と自己申告している
    ものを、環境間の通常の不一致として報告してしまうこと(resolution 3)。
(3) `S5_kmeans_centers`/`S15_kmeans_repeats`のように、
    `meta["effective_thread_count"]`が違う2ランを比較してしまうこと
    (resolution 2)。スレッド数1・2・4では安定、8では不安定という相関が
    実測で確認されている(測定条件・実測値は`reprobench.probes.
    s5_sklearn`のモジュールdocstring参照)。このフィールドの値が特定
    できなかった(`None`)ラベル同士も、同じ"不明"というだけで同じ
    スレッド数だったとは限らないため、同様に比較してはならない
    (fix round 1 finding 1)。
(4) 比較対象の観測が1件しかないのに「全ラベルが一致した」と読める表示を
    すること(fix round 1 finding 0, critical)。1件の観測は一致の証拠に
    ならない。
(5) 同じラベル(同一`{host}/{image}/{config_label}`)に対応する結果
    ファイルが複数あるとき、後から読んだファイルで前のものを無言で
    上書きしてしまうこと(fix round 1 finding 3)。これは構成を再実行して
    結果が変わった、というファイル間再現性の失敗を握り潰すことになる。

このモジュールはこれらを「比較方針テーブル」(`_POLICY`)として宣言的に
持ち、各プローブがなぜ特別扱いされているかをソースだけでなく
`build_report`の出力・`to_markdown`の表の両方から読めるようにする。
方針テーブルに無いプローブには`_DEFAULT_POLICY`(既存のdigest完全一致
判定)が適用される。

このモジュールの責務は「読み込み・判定・Markdown整形」のみ。数値比較の
実装(ULP距離等)は持たず`reprobench.digest.compare_arrays`を使う。
"""
import itertools
import json
from dataclasses import dataclass
from typing import Iterable

from reprobench.digest import compare_arrays, decode_sample

# --- ラベル ---


def label_for(env: dict) -> str:
    """`env`から`{host}/{image}/{config_label}`形式のラベル文字列を作る。

    `host`/`image`/`config_label`のいずれかが`None`または空文字列の場合は
    その要素を省く(`envinfo.collect_env()`は`REPRO_IMAGE`/`REPRO_CONFIG`が
    未設定の環境ではimage/config_labelを`None`のまま返す契約であり、ここで
    "None"という文字列を結合へ混ぜ込むと、本当は同じ構成の2環境が見かけ上
    別ラベルになってしまう)。全要素が欠けている場合は"unknown"を返す
    (空文字列のラベルは比較結果の表の行として識別できず、空欄と区別が
    付かなくなるため)。
    """
    parts = [str(p) for p in (env.get("host"), env.get("image"), env.get("config_label")) if p]
    return "/".join(parts) if parts else "unknown"


# --- 読み込み ---


def load_results(paths: Iterable[str]) -> list[dict]:
    """結果JSONファイル群を読み込み、生の辞書のリストで返す。

    各ファイルの中身を変形・検証はしない(runner.pyの出力契約
    (`schema`/`env`/`probe_order`/`probes`)をそのまま前提とする)。
    """
    out = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def _probe_entry(result: dict, probe: str) -> dict | None:
    """1件の結果辞書から、指定したプローブ名のエントリ(結果 or スキップ)を
    探す。無ければ`None`。
    """
    for entry in result.get("probes", []):
        if entry.get("probe") == probe:
            return entry
    return None


def _occurrences_by_label(results: list[dict], probe: str) -> dict[str, list[dict]]:
    """1プローブについて、ラベルごとの全エントリ(結果 or スキップ)を
    出現順のリストで返す。

    同じラベル(同じ`{host}/{image}/{config_label}`)に対応する結果
    ファイルが複数あっても、後から読んだ方で前のものを無言で上書きしない
    (fix round 1 finding 3): 同一構成を2回計測してファイル間で結果が
    変わる、あるいはパス指定・globが同じファイルを2回拾う、のどちらも
    現実にあり得る。この関数はラベルごとに全エントリを保持するだけで、
    一致判定(`agreement`)・衝突検出は呼び出し側の責務とする。
    """
    occurrences: dict[str, list[dict]] = {}
    for result in results:
        label = label_for(result.get("env", {}))
        entry = _probe_entry(result, probe)
        if entry is not None:
            occurrences.setdefault(label, []).append(entry)
    return occurrences


def _ordered_labels(results: list[dict]) -> list[str]:
    """`results`に出現する順でラベルを重複無く返す。"""
    order: list[str] = []
    seen: set[str] = set()
    for result in results:
        label = label_for(result.get("env", {}))
        if label not in seen:
            seen.add(label)
            order.append(label)
    return order


# --- 素のdigest比較 ---


def agreement(results: list[dict], probe: str) -> dict:
    """指定した1プローブについて、`results`に含まれる各ラベルのdigestを
    突き合わせる。

    返り値: `{"groups": {digest: [labels]}, "all_equal": bool,
    "missing": [labels], "numeric": {(label_a, label_b): compare_arrays(...)},
    "label_conflicts": {label: [digest, ...]}}`。

    スキップされたラベル(理由を問わない)と、このプローブのエントリが
    そもそも無いラベルは、どちらも「比較に使えるdigestが無い」という点で
    同じなので`missing`にまとめる(欠損を一致として扱わない。resolution
    8)。スキップ理由ごとの区別(`missing requirement:`/`missing env var:`/
    `raised:`)は`build_report`が別途行う——`agreement`はこの関数単体の
    契約(ブリーフのテスト)通り、digestの一致判定だけに責務を絞る。

    `all_equal`は`missing`を除いた、実際にdigestを持つラベルだけで判定
    する。`numeric`は`sample_b64`を両方が持つラベルの組だけに計算する
    (resolution 9: `compare_arrays`を使い、ULP比較を再実装しない)。

    同じラベルに対応するエントリが複数あるとき(fix round 1 finding 3)、
    それらのdigestが一致していれば1つにまとめ、食い違っていれば
    `label_conflicts`に記録したうえで**両方の**digestをそのラベルの
    `groups`所属として残す(最後に読んだファイルで前のものを上書きしない)。
    その結果、1つのラベルが2つの`groups`キーにまたがって出現しうる——
    これは実在する食い違いを表しており、`len(groups) > 1`を通じて
    `all_equal=False`に自然に反映される。
    """
    occurrences = _occurrences_by_label(results, probe)
    order = _ordered_labels(results)

    present: dict[str, list[dict]] = {}
    missing: list[str] = []
    for label in order:
        entries = [e for e in occurrences.get(label, []) if "skipped" not in e]
        if entries:
            present[label] = entries
        else:
            missing.append(label)

    groups: dict[str, list[str]] = {}
    label_conflicts: dict[str, list[str]] = {}
    for label in order:
        if label not in present:
            continue
        digests = sorted({e["digest"] for e in present[label]})
        if len(digests) > 1:
            label_conflicts[label] = digests
        for d in digests:
            groups.setdefault(d, []).append(label)

    all_equal = len(groups) <= 1

    numeric: dict[tuple[str, str], dict] = {}
    sampled = [
        label
        for label in order
        if label in present and any(e.get("sample_b64") is not None for e in present[label])
    ]
    for a, b in itertools.combinations(sampled, 2):
        # fix round 1 finding 3の対象外: 同じラベルに複数ファイルがあり
        # なおかつサンプルを持つものが複数あるケースの数値比較までは扱わず
        # (由来の衝突は上の`label_conflicts`が既に捕捉している)、最初に
        # 見つかったサンプルを使う。(次回レビューで指摘された
        # `decode_sample`の重複呼び出しもここでは手を付けない——4ラベル
        # 程度の規模では無視できるため、別レビューに委ねる。)
        sample_a = next(e["sample_b64"] for e in present[a] if e.get("sample_b64") is not None)
        sample_b = next(e["sample_b64"] for e in present[b] if e.get("sample_b64") is not None)
        numeric[(a, b)] = compare_arrays(decode_sample(sample_a), decode_sample(sample_b))

    return {
        "groups": groups,
        "all_equal": all_equal,
        "missing": missing,
        "numeric": numeric,
        "label_conflicts": label_conflicts,
    }


def _json_safe_float(x: float):
    """非有限値(`inf`/`-inf`/`nan`)をJSON安全な文字列へ変換する。

    `compare_arrays`はshape/dtype不一致のとき`max_abs`/`max_rel`に
    `float("inf")`を返す(`reprobench.digest`参照)——これは`json.dumps`の
    既定設定では例外を送出せず非標準の`Infinity`トークンをそのまま書き
    出してしまう(`allow_nan=False`を付けて初めて検出できる。fix round 1
    small finding)。報告の境界でここを明示的に有限な文字列表現へ変換し、
    厳密なJSONパーサでも安全に読めるようにする。
    """
    if x != x:  # NaN
        return "nan"
    if x == float("inf"):
        return "inf"
    if x == float("-inf"):
        return "-inf"
    return x


def _json_safe_metrics(metrics: dict) -> dict:
    return {
        "bit_equal": metrics["bit_equal"],
        "max_abs": _json_safe_float(metrics["max_abs"]),
        "max_rel": _json_safe_float(metrics["max_rel"]),
        "max_ulp": metrics["max_ulp"],
    }


def _json_safe_agreement(agg: dict) -> dict:
    """`agreement()`の返り値をJSON安全な形に変換する。

    `numeric`のキーはラベルの2要素タプルであり、`json.dumps`はtupleを
    dictキーとして受け付けない(非文字列キーのうちstr/int/float/bool/
    Noneのみ自動的に文字列化される)。`"|"`区切りの文字列へ変換する
    (ラベル自体に`"|"`が混ざる運用は想定していない——ラベルは
    `{host}/{image}/{config_label}`の組み立てであり、現行の全ホスト名・
    イメージ名・構成ラベルに`"|"`を使わない)。`numeric`の各値も
    `_json_safe_metrics`で非有限値を文字列化する。
    """
    return {
        "groups": agg["groups"],
        "all_equal": agg["all_equal"],
        "missing": agg["missing"],
        "numeric": {f"{a}|{b}": _json_safe_metrics(v) for (a, b), v in agg["numeric"].items()},
        "label_conflicts": agg["label_conflicts"],
    }


def _linked_mismatch(results: list[dict], probe: str, other_probe: str) -> list[str]:
    """`probe`と`other_probe`が"linked"(構造的に同一の情報を運ぶはず)と
    宣言されているとき、実際に同じラベルでdigestが一致しているかを
    検証する(reviewerの偽陰性監査からの追加: 「構造的に同一」という
    前提は文書化するだけでなく検査する)。

    一致していないラベルの一覧を返す(空リストなら前提は保たれている)。
    どちらかのプローブがそのラベルで欠損/スキップの場合は、比較対象が
    無いのでそのラベルは検証対象から除く(欠損は別途`absent`/`skips`が
    扱う)。
    """
    occ_a = _occurrences_by_label(results, probe)
    occ_b = _occurrences_by_label(results, other_probe)
    mismatched = []
    for label, entries_a in occ_a.items():
        entries_b = occ_b.get(label)
        if not entries_b:
            continue
        digests_a = {e["digest"] for e in entries_a if "skipped" not in e}
        digests_b = {e["digest"] for e in entries_b if "skipped" not in e}
        if not digests_a or not digests_b:
            continue
        if digests_a != digests_b:
            mismatched.append(label)
    return sorted(mismatched)


def _round_digests(result: dict, probe: str) -> list[str | None] | None:
    """1件の結果ファイルについて、`rounds`(`--repeat`の記録)があれば
    各ラウンドでのこのプローブのdigestを順に並べて返す(スキップされた
    ラウンドは`None`)。`rounds`キー自体が無ければ`None`を返す——
    ラウンド間比較ができないことと、1ラウンドだけで安定していることは
    別の事実であり、前者を後者のように見せてはならない。
    """
    rounds = result.get("rounds")
    if not rounds:
        return None
    out: list[str | None] = []
    for round_probes in rounds:
        entry = next((e for e in round_probes if e.get("probe") == probe), None)
        out.append(None if entry is None or "skipped" in entry else entry["digest"])
    return out


def _round_info_by_label(results: list[dict], probe: str) -> dict[str, dict]:
    """ラベルごとのラウンド内安定性を計算する(fix round 1 finding 4:
    `--repeat`で記録した`rounds`は、これまで比較器のどこからも読まれて
    いなかった)。

    同じラベルに複数の結果ファイルがある場合は、`rounds`を持つ最初の
    1件だけを見る(ラウンド内安定性はそもそも「1プロセス内での繰り返し」
    の指標であり、複数ファイルをまたいだラウンドの合算は別の問いに
    なるため、このラウンドではスコープに含めない)。
    """
    info: dict[str, dict] = {}
    for result in results:
        label = label_for(result.get("env", {}))
        if label in info:
            continue
        digests = _round_digests(result, probe)
        if digests is None:
            continue
        present_digests = [d for d in digests if d is not None]
        unique = sorted(set(present_digests))
        info[label] = {
            "n_rounds": len(digests),
            "n_present": len(present_digests),
            "stable": (len(unique) <= 1) if present_digests else None,
            "digests": digests,
        }
    return info


# --- 比較方針テーブル ---


@dataclass(frozen=True)
class ProbePolicy:
    """1プローブに対する比較方針。判定ロジックそのものは持たない
    (判定は`_build_probe_row`が行う)。

    kind:
        "normal" — 既定の扱い。存在するラベルのdigestが全部一致するかで
            判定する。
        "grouped" — `group_by_meta`で指定したmetaフィールドの値ごとに
            ラベルをグループ化し、グループ内でのみ判定する
            (resolution 2)。値が特定できない(`None`)ラベルはどの
            グループにも属さず、比較不能として扱う(fix round 1
            finding 1)。
        "self_reported_stability" — プローブ自身が
            `meta["stable_within_process"]`を自己申告する。環境間の
            digest不一致を不一致として判定しない(resolution 3)。
        "linked" — 別の1プローブ(`linked_with`)と構造的に同一になる
            ことが分かっている。判定自体は"normal"と同じだが、その旨を
            注記し、かつ実際に両者のdigestが一致しているかを検査する
            (resolution 4、reviewer追加)。

    `note`には測定した事実のみを書き、仕組みの推測は書かない
    (プロジェクトの標準方針: 実測と食い違った仮説を過去に複数回
    書いてしまった反省から)。
    """

    kind: str
    note: str
    group_by_meta: str | None = None
    linked_with: str | None = None


_DEFAULT_POLICY = ProbePolicy(
    kind="normal",
    note="既定の扱い: 比較方針テーブルに個別の記載が無いプローブ。存在するラベル間でdigestが完全一致するかどうかのみで判定する。",
)

_POLICY: dict[str, ProbePolicy] = {
    "S5_kmeans_centers": ProbePolicy(
        kind="grouped",
        group_by_meta="effective_thread_count",
        note=(
            "測定(python:3.11-slim + scikit-learn==1.7.2、8CPUコンテナ、"
            "reprobench.probes.s5_sklearnモジュールdocstring参照): "
            "effective_thread_countが1・2・4のときは30フィット中digestが"
            "1種類に揃い、8のときは複数種類に割れた。異なる"
            "effective_thread_count同士を比較しても意味を持たないため、"
            "このフィールドの値ごとに分けて判定する。値が特定できなかった"
            "(None)ラベル同士も、同じ'不明'というだけで同じスレッド数"
            "だったとは限らないため比較しない。"
        ),
    ),
    "S0_parallel_completion_order": ProbePolicy(
        kind="self_reported_stability",
        note=(
            "プローブ自身が同一プロセス内で2回完了順を測り、"
            "meta['stable_within_process']に一致したかを自己申告する"
            "(reprobench.probes.s0_hidden_inputs参照)。環境間のdigest"
            "不一致はこのプローブの測定対象そのものであり、不一致として"
            "報告しない。"
        ),
    ),
    "S0_glob_order": ProbePolicy(
        kind="self_reported_stability",
        note=(
            "プローブ自身が同一プロセス内で2回`glob.glob`を呼び、"
            "meta['stable_within_process']に一致したかを自己申告する"
            "(reprobench.probes.s0_hidden_inputs参照)。環境間のdigest"
            "不一致はこのプローブの測定対象そのものであり、不一致として"
            "報告しない。"
        ),
    ),
    "S8_index_add_atomics": ProbePolicy(
        kind="self_reported_stability",
        note=(
            "プローブ自身が同一プロセス内で2回`index_add_`を実行し、"
            "meta['stable_within_process']に一致したかを自己申告する"
            "(reprobench.probes.s8_torch_gpu参照)。環境間のdigest不一致"
            "はこのプローブの測定対象そのものであり、不一致として報告"
            "しない。"
        ),
    ),
    "S8_index_add_atomics_deterministic": ProbePolicy(
        kind="normal",
        note=(
            "S8_index_add_atomicsの対照実験(`use_deterministic_algorithms"
            "(True)`)。測定: 座長の実機でS8_index_add_atomicsが不安定・"
            "こちらは安定という対比が確認されている(reprobench.probes."
            "s8_torch_gpu参照)。そのためこのプローブに限っては自己申告では"
            "なく通常どおり(kind=normal)不一致として扱う——ここでの"
            "予期しない不一致はそれ自体が実在の知見になる。"
        ),
    ),
    "S12_cudnn_benchmark_deterministic": ProbePolicy(
        kind="self_reported_stability",
        note=(
            "プローブ自身が各cudnn設定(benchmark x deterministicの2x2)で"
            "同一プロセス内で畳み込みの順伝播・逆伝播を複数回実行し、"
            "meta['stable_within_process']に一致したかを自己申告する"
            "(reprobench.probes.s12_cudnn参照)。benchmark=Trueは実行時の"
            "タイミング計測でアルゴリズムを選ぶため、環境間のdigest不一致"
            "はこのプローブの測定対象そのものであり、不一致として報告"
            "しない。"
        ),
    ),
    "S15_index_add_repeats": ProbePolicy(
        kind="self_reported_stability",
        note=(
            "S8_index_add_atomicsと同じ計算(同じseed・同じ`_index_add_once`)"
            "を同一プロセス内で`_N_ROUNDS`回繰り返し、"
            "meta['round_digests']に各回のdigestを、"
            "meta['stable_within_process']に全回が一致したかを自己申告する"
            "(reprobench.probes.s15_repeat_raw参照)。CUDAの`index_add_`は"
            "atomicAdd由来の非決定性が測定対象そのものであり、環境間の"
            "digest不一致を不一致として報告しない(S8_index_add_atomics"
            "と同じ扱い)。"
        ),
    ),
    "S15_kmeans_repeats": ProbePolicy(
        kind="grouped",
        group_by_meta="effective_thread_count",
        note=(
            "S5_kmeans_centersと同じデータ・同じパラメータでのKMeansフィット"
            "を同一プロセス内で`_N_ROUNDS`回繰り返し、"
            "meta['round_digests']に各回の(正準化した)クラスタ中心のdigest"
            "を記録する(reprobench.probes.s15_repeat_raw参照)。"
            "S5_kmeans_centersと同じ原因(有効スレッド数)で同じように"
            "不安定になりうるため、S5_kmeans_centersと同じ"
            "`kind='grouped'`(`effective_thread_count`ごとの比較)を適用"
            "する——'自己申告だから不一致を握り潰す'のではなく、スレッド数"
            "が同じ環境同士でだけ比較することで、1・2・4スレッド環境間の"
            "本物の一致/不一致は引き続き報告できるようにするため"
            "(S5_kmeans_centers自身もkind='self_reported_stability'では"
            "なくkind='grouped'を使っている)。"
        ),
    ),
    "S0_set_iteration_order": ProbePolicy(
        kind="linked",
        linked_with="S0_dict_from_set",
        note=(
            "S0_dict_from_setと同一のset(`_STRINGS`)を同一プロセス内で"
            "反復しているため、両プローブのdigestは構造的に一致する"
            "(辞書は挿入順を保持するため)。2件は1つの情報しか運ばないので、"
            "独立した2つの一致として読まない。"
        ),
    ),
    "S0_dict_from_set": ProbePolicy(
        kind="linked",
        linked_with="S0_set_iteration_order",
        note=(
            "S0_set_iteration_orderと同一のset(`_STRINGS`)を同一プロセス内"
            "で反復しているため、両プローブのdigestは構造的に一致する"
            "(辞書は挿入順を保持するため)。2件は1つの情報しか運ばないので、"
            "独立した2つの一致として読まない。"
        ),
    ),
}

#: 「同じプローブ名同士の比較」という`agreement`の枠組みの外にある注意点。
#: 指定した複数プローブは、互いの数値・トークン列を比較してはならない
#: (そもそも`agreement`が別名のプローブ同士を比較することは無いが、記事の
#: 読み手がmetaの類似フィールド(`train_idx_digest`等)を見て独自に比較
#: してしまわないよう、報告にもMarkdownにも明記する)。
_CROSS_PROBE_CAVEATS = (
    {
        "probes": ("S9_logit_margin", "S9_greedy_batch1"),
        "note": (
            "同じ生成runではない: S9_logit_marginはrepetition_penalty=1.0に"
            "単離した設定で生成し、S9_greedy_batch1はモデル出荷既定の"
            "generation_configをそのまま使って生成する"
            "(reprobench.probes.s9_llm参照)。トークン列・marginはどちらも"
            "互いの比較対象ではない。"
        ),
    },
    {
        "probes": (
            "S5_logreg_coef",
            "S5_random_forest_pred",
            "S5_train_test_split_index",
            "S6_lgbm_pred",
            "S6_lgbm_tree_structure",
        ),
        "note": (
            "S5は層化`train_test_split`、S6は`permutation`による分割であり"
            "(どちらも`meta['train_idx_digest']`を記録しているのは分割"
            "インデックスをダイジェストで残すためで、分割の作り方自体が"
            "違う)、S5側の指標とS6側の指標は互いの比較対象ではない。"
        ),
    },
)


def _skip_category(reason: str) -> str:
    """スキップ理由文字列の接頭辞から種別を判定する(runner.pyの契約:
    "missing requirement: "/"missing env var: "/"raised: "のいずれか)。
    契約に無い理由文字列は"unknown"になる——これは本来起こらないはずの
    状態であり、"missing requirement:"等と同じ「想定内で無害」な欠損
    としては扱わない(fix round 1 small finding)。
    """
    for prefix, category in (
        ("missing requirement: ", "missing_requirement"),
        ("missing env var: ", "missing_env_var"),
        ("raised: ", "raised"),
    ):
        if reason.startswith(prefix):
            return category
    return "unknown"


def _policy_dict(policy: ProbePolicy) -> dict:
    return {
        "kind": policy.kind,
        "note": policy.note,
        "group_by_meta": policy.group_by_meta,
        "linked_with": policy.linked_with,
    }


def _build_probe_row(results: list[dict], probe: str, labels: list[str]) -> dict:
    """1プローブぶんの報告行を組み立てる。

    `absent`/`skips`(missing_requirement/missing_env_var/raised/unknown)・
    `rounds`(ラウンド内安定性)は方針(kind)に関わらず常に計算する——
    自己申告系・グループ化系のプローブであっても、あるラベルで
    クラッシュ(`raised:`)していればそれは常に目立たせるべき実在の
    知見だから(resolution 7)。
    """
    policy = _POLICY.get(probe, _DEFAULT_POLICY)
    occurrences = _occurrences_by_label(results, probe)

    absent: list[str] = []
    skips: dict[str, dict] = {
        "missing_requirement": {},
        "missing_env_var": {},
        "raised": {},
        "unknown": {},
    }
    present: dict[str, list[dict]] = {}
    for label in labels:
        entries = occurrences.get(label, [])
        if not entries:
            absent.append(label)
            continue
        for entry in entries:
            if "skipped" in entry:
                skips.setdefault(_skip_category(entry["skipped"]), {})[label] = entry["skipped"]
        present_entries = [e for e in entries if "skipped" not in e]
        if present_entries:
            present[label] = present_entries

    row: dict = {
        "probe": probe,
        "policy": _policy_dict(policy),
        "absent": absent,
        "skips": skips,
        "rounds": _round_info_by_label(results, probe),
    }

    if policy.kind == "self_reported_stability":
        stability: dict[str, bool | None] = {}
        digests: dict[str, str | None] = {}
        label_conflicts: dict[str, list[str]] = {}
        for label, entries in present.items():
            unique_digests = sorted({e["digest"] for e in entries})
            if len(unique_digests) > 1:
                label_conflicts[label] = unique_digests
            digests[label] = unique_digests[0] if len(unique_digests) == 1 else None
            # 複数ファイルの自己申告が食い違った場合、どちらか一方を無言で
            # 選ばず"不明"(None)にする(fix round 1 finding 3と同じ流儀)。
            stabilities = {e.get("meta", {}).get("stable_within_process") for e in entries}
            stability[label] = next(iter(stabilities)) if len(stabilities) == 1 else None
        row["stability"] = stability
        row["digests"] = digests
        row["label_conflicts"] = label_conflicts
        return row

    if policy.kind == "grouped":
        field = policy.group_by_meta
        by_group: dict[str, list[str]] = {}
        ungroupable: list[str] = []
        for label, entries in present.items():
            # グループ化フィールドの値が複数ファイル間で食い違うケースまで
            # は扱わない(ここで見ているのは1プロセス内のmetaであり、その
            # 食い違いはdigest自体の食い違いとは別の軸——スコープ外として
            # 最初のエントリの値を使う)。値そのものが無い(None)場合は
            # 「不明」というだけで同じスレッド数だったとは限らないため、
            # どのグループにも入れない(fix round 1 finding 1, Important)。
            raw_value = entries[0].get("meta", {}).get(field)
            if raw_value is None:
                ungroupable.append(label)
                continue
            by_group.setdefault(str(raw_value), []).append(label)
        group_reports = {}
        for value, group_labels in by_group.items():
            sub_results = [r for r in results if label_for(r.get("env", {})) in group_labels]
            group_reports[value] = {
                **_json_safe_agreement(agreement(sub_results, probe)),
                "group_labels": sorted(group_labels),
            }
        row["group_by"] = field
        row["groups"] = group_reports
        row["ungroupable"] = sorted(ungroupable)
        return row

    # "normal" / "linked": 単一の agreement() で判定する。
    row["agreement"] = _json_safe_agreement(agreement(results, probe))
    if policy.kind == "linked" and policy.linked_with:
        row["linked_mismatch"] = _linked_mismatch(results, probe, policy.linked_with)
    return row


def build_report(results: list[dict]) -> dict:
    """全プローブ×全ラベルの比較報告を組み立てる。

    ラベルの集合は各`results[i]["env"]`から、プローブ名の集合は各
    `results[i]["probe_order"]`の和集合から作る(あるホストが選択
    実行した範囲の外のプローブは、そのホストでは単に`absent`になる)。
    """
    labels = _ordered_labels(results)

    probe_names: list[str] = sorted(
        {name for result in results for name in result.get("probe_order", [])}
    )

    rows = {probe: _build_probe_row(results, probe, labels) for probe in probe_names}

    return {
        "labels": labels,
        "probes": probe_names,
        "rows": rows,
        "cross_probe_caveats": [dict(c, probes=list(c["probes"])) for c in _CROSS_PROBE_CAVEATS],
    }


# --- Markdown整形 ---

_LEGEND = """## 記号の意味

- `✓` 比較対象の全ラベルでdigestが一致した(観測が2件以上ある)。
- `✗` 比較対象のラベル間でdigestが不一致だった。
- `n=1` 比較対象の観測が1件しか無い。一致・不一致のどちらも主張できない
  (1件だけでは「一致した」と読めないため`✓`とは区別する。fix round 1
  finding 0, critical)。
- `△` 自己申告された安定性でのみ判定する(プローブ自身が同一プロセス内で
  繰り返し実行した結果を比較し、`stable_within_process`として記録した値。
  環境間のdigest不一致はこのプローブの測定対象そのものであり不一致として
  扱わない)。`unknown`は自己申告自体が無い、または同じラベルの複数
  ファイル間で自己申告が食い違った場合。
- `–` 比較できるラベルが無い(全てのラベルで欠損/スキップ)。
- 「欠落」列: そのラベルにこの行の結果が無い(ファイルに無い、または
  `missing requirement:`/`missing env var:`でスキップされた——いずれも
  想定内で無害)。グループ化されたプローブでは、プローブ全体を通じての
  欠落をグループの先頭行にのみ表示する(同じ欠落を全グループ行へ
  繰り返さない)。
- 「クラッシュ」列(`⚠`): そのラベルで`raised:`によりスキップされた
  (計測ホスト上で実際に例外を送出した——調査が必要な実在の知見)、または
  runner.pyの契約に無い未知のスキップ理由(`unknown_skip:`)だった場合
  (これも想定外であり無害とは言えない)。
- 「ラベル衝突」列: 同じラベル(同一host/image/config_label)に対応する
  結果ファイルが複数あり、それらのdigestが互いに食い違った場合
  (fix round 1 finding 3)。最後に読んだファイルで前のものを無言で
  上書きしない——これ自体がファイル間(構成を再実行して結果が変わる)
  再現性の実在の知見になる。
- 「ラウンド内安定性」列: `--repeat`で複数ラウンドを記録したファイルに
  ついて、同一プロセス内の全ラウンドでdigestが一致したか
  (`stable`/`unstable`/`unknown`=ラウンドはあるが欠損だらけ、または
  このプローブがそのラウンドに無い)。`rounds`を持たないファイルの
  ラベルはこの列に出てこない(fix round 1 finding 4)。
- `[linked: ...]`: 構造的に同一になるはずの別プローブが注記されている
  (resolution 4)。実際に両者のdigestが一致しているかは検査済みであり、
  崩れていれば`⚠ LINKED PREMISE VIOLATED`がPolicy列の先頭に出る。
"""


def _fmt_labels(labels: list[str]) -> str:
    return ", ".join(labels) if labels else "–"


def _fmt_groups(groups: dict) -> str:
    if not groups:
        return "–"
    return "; ".join(f"{digest[:8]}:{','.join(labels)}" for digest, labels in groups.items())


def _fmt_missing(absent: list[str], skips: dict) -> str:
    parts = []
    if absent:
        parts.append("absent:" + ",".join(absent))
    if skips.get("missing_requirement"):
        parts.append("missing_requirement:" + ",".join(skips["missing_requirement"]))
    if skips.get("missing_env_var"):
        parts.append("missing_env_var:" + ",".join(skips["missing_env_var"]))
    return "; ".join(parts) if parts else "–"


def _fmt_raised(skips: dict) -> str:
    # "unknown"(runner.pyの契約に無いスキップ理由)も、想定内で無害な
    # missing_requirement/missing_env_varとは違い、ここに出す
    # (fix round 1 small finding: 欠落列に「いずれも想定内で無害」と
    # 書いておきながらunknownを紛れ込ませていたのを分離した)。
    parts = []
    raised = skips.get("raised") or {}
    if raised:
        parts.append("⚠ " + ", ".join(raised.keys()))
    unknown = skips.get("unknown") or {}
    if unknown:
        parts.append("⚠ unknown_skip: " + ", ".join(unknown.keys()))
    return "; ".join(parts) if parts else "–"


def _fmt_conflicts(conflicts: dict) -> str:
    if not conflicts:
        return "–"
    return "; ".join(
        f"{label}:{','.join(d[:8] for d in digests)}" for label, digests in conflicts.items()
    )


def _fmt_stable(ok) -> str:
    if ok is True:
        return "stable"
    if ok is False:
        return "unstable"
    return "unknown"  # 自己申告自体が無い、または複数ファイル間で食い違った


def _fmt_rounds(round_info: dict) -> str:
    if not round_info:
        return "–"
    parts = []
    for label, info in round_info.items():
        state = "unknown" if info["stable"] is None else ("stable" if info["stable"] else "unstable")
        parts.append(f"{label}:{state}(n={info['n_rounds']})")
    return "; ".join(parts)


def _verdict_symbol(agg: dict) -> str:
    """観測数に応じてverdictを出し分ける。

    `agg["groups"]`に含まれるラベルの総数(=比較できた観測の数。同じ
    ラベルが衝突で2つのdigestにまたがっていれば2としてカウントされ、
    その場合は`len(groups) > 1`により自然に`✗`になる)が0なら比較対象が
    無く、1なら観測が1件だけで一致・不一致のどちらも主張できない
    (fix round 1 finding 0, critical / finding 2, important)。2件以上
    そろって初めて`✓`/`✗`を主張する。
    """
    present_count = sum(len(v) for v in agg["groups"].values())
    if present_count == 0:
        return "–"
    if present_count == 1:
        return "n=1"
    return "✓" if agg["all_equal"] else "✗"


def _render_row(probe: str, row: dict) -> list[str]:
    policy = row["policy"]
    note = policy["note"]
    if policy["linked_with"]:
        prefix = f"[linked: {policy['linked_with']}] "
        if row.get("linked_mismatch"):
            # reviewerの偽陰性監査からの追加: 「構造的に同一」という前提が
            # 実際に崩れていたら、静かな[linked: ...]注記の裏に隠さず
            # 先頭で叫ぶ。
            prefix = (
                f"⚠ LINKED PREMISE VIOLATED for {', '.join(row['linked_mismatch'])} ⚠ " + prefix
            )
        note = prefix + note
    missing_col = _fmt_missing(row["absent"], row["skips"])
    raised_col = _fmt_raised(row["skips"])
    all_rounds = row.get("rounds", {})

    if policy["kind"] == "self_reported_stability":
        if row["stability"]:
            detail = "; ".join(f"{label}:{_fmt_stable(ok)}" for label, ok in row["stability"].items())
        else:
            detail = "–"
        conflict_col = _fmt_conflicts(row.get("label_conflicts", {}))
        rounds_col = _fmt_rounds(all_rounds)
        return [
            f"| {probe} | △ | {detail} | {missing_col} | {raised_col} | "
            f"{conflict_col} | {rounds_col} | {note} |"
        ]

    if policy["kind"] == "grouped":
        lines = []
        ungroupable = row.get("ungroupable") or []
        group_missing_col = missing_col
        if ungroupable:
            # fix round 1 finding 1: グループ化フィールドの値が特定できな
            # かった(None)ラベルは、グループ化もされず、かといって
            # "absent"でもないので、欠落列に別枠で明記する。
            extra = "no_group_value:" + ",".join(ungroupable)
            group_missing_col = extra if missing_col == "–" else f"{missing_col}; {extra}"
        # キーは辞書式順序で並べる(数値としての大小順ではない——プロジェクトの
        # 他箇所(reprobench.probes.s5_sklearnの`_thread_pool_info`)と同じ
        # 流儀: ここで求めているのは何らかの固定順序であって、2桁以上の値が
        # 混ざったときの数値順を保証するものではない)。
        for i, (group_value, agg) in enumerate(sorted(row["groups"].items())):
            verdict = _verdict_symbol(agg)
            detail = _fmt_groups(agg["groups"])
            conflict_col = _fmt_conflicts(agg.get("label_conflicts", {}))
            group_rounds = {
                label: info
                for label, info in all_rounds.items()
                if label in agg.get("group_labels", ())
            }
            rounds_col = _fmt_rounds(group_rounds)
            # fix round 1 small finding: 欠落・クラッシュはプローブ全体で
            # 1つの事実であり、グループの数だけ行を出しても繰り返し起きた
            # わけではない。先頭行にだけ出し、以降は空欄にする(linked-pair
            # 方針が二重計上を避けるのと同じ理由)。
            shown_missing = group_missing_col if i == 0 else ""
            shown_raised = raised_col if i == 0 else ""
            lines.append(
                f"| {probe} [{row['group_by']}={group_value}] | {verdict} | {detail} | "
                f"{shown_missing} | {shown_raised} | {conflict_col} | {rounds_col} | {note} |"
            )
        if not lines:
            lines.append(
                f"| {probe} | – | –(グループ無し) | {group_missing_col} | {raised_col} | "
                f"– | {_fmt_rounds(all_rounds)} | {note} |"
            )
        return lines

    # "normal" / "linked"
    agg = row["agreement"]
    verdict = _verdict_symbol(agg)
    detail = _fmt_groups(agg["groups"])
    conflict_col = _fmt_conflicts(agg.get("label_conflicts", {}))
    rounds_col = _fmt_rounds(all_rounds)
    return [
        f"| {probe} | {verdict} | {detail} | {missing_col} | {raised_col} | "
        f"{conflict_col} | {rounds_col} | {note} |"
    ]


def to_markdown(report: dict) -> str:
    """`build_report`の結果を、記事に貼れる○×表のMarkdownへ整形する。

    状態の区別(ドミナントな失敗モード対策として、以下は互いに潰れない
    別々の列・記号として出す): 全ラベルが一致/不一致/観測1件のみ(n=1)/
    自己申告安定性のみで判定(△)/一部ラベルで欠落/一部ラベルでクラッシュ
    (raised)/ラベル衝突/ラウンド内安定性。`S5_kmeans_centers`のように
    グループ化されたプローブは、単一のverdictではなくグループごとの行を
    出す(resolution 2)。
    """
    header = "| Probe | Verdict | Detail | 欠落 | クラッシュ | ラベル衝突 | ラウンド内安定性 | Policy |"
    lines = ["## 比較結果", "", _LEGEND, header, "|---|---|---|---|---|---|---|---|"]
    for probe in report["probes"]:
        lines.extend(_render_row(probe, report["rows"][probe]))

    if report.get("cross_probe_caveats"):
        lines.append("")
        lines.append("## 比較対象外の注記")
        for item in report["cross_probe_caveats"]:
            lines.append(f"- {', '.join(item['probes'])}: {item['note']}")

    return "\n".join(lines)
