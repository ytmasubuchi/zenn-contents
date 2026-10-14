"""環境横断の結果比較器。

Task 10のランナーがホスト・イメージ・構成ごとに書き出したJSONファイル群を
読み込み、プローブごとに「どのラベルがどのラベルと一致したか」を判定し、
記事の付録表に貼れるMarkdownへ整形する。

このモジュールの出力は、記事が「これは再現する・これはしない」と述べる
唯一の根拠になる。したがって最優先の失敗モードは、一致していないものを
一致と報告すること・不一致ではないものを不一致と報告することであり、
とりわけ次の3点を誤ると実在しない知見を記事に刻んでしまう:

(1) そのラベルにこのプローブの結果が無い(欠損)ことを「一致」として
    数えてしまうこと(resolution 8)。
(2) `S0_parallel_completion_order`/`S0_glob_order`/`S8_index_add_atomics`
    のように、プローブ自身が「この不安定さ自体が測定対象である」と
    自己申告しているものを、環境間の通常の不一致として報告してしまう
    こと(resolution 3)。
(3) `S5_kmeans_centers`のように、`meta["effective_thread_count"]`が
    違う2ランを比較してしまうこと(resolution 2)。スレッド数1・2・4
    では安定、8では不安定という相関が実測で確認されている(測定条件・
    実測値は`reprobench.probes.s5_sklearn`のモジュールdocstring参照)。

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


# --- 素のdigest比較 ---


def agreement(results: list[dict], probe: str) -> dict:
    """指定した1プローブについて、`results`に含まれる各ラベルのdigestを
    突き合わせる。

    返り値: `{"groups": {digest: [labels]}, "all_equal": bool,
    "missing": [labels], "numeric": {(label_a, label_b): compare_arrays(...)}}`。

    スキップされたラベル(理由を問わない)と、このプローブのエントリが
    そもそも無いラベルは、どちらも「比較に使えるdigestが無い」という点で
    同じなので`missing`にまとめる(欠損を一致として扱わない。resolution
    8)。スキップ理由ごとの区別(`missing requirement:`/`missing env var:`/
    `raised:`)は`build_report`が別途行う——`agreement`はこの関数単体の
    契約(ブリーフのテスト)通り、digestの一致判定だけに責務を絞る。

    `all_equal`は`missing`を除いた、実際にdigestを持つラベルだけで判定
    する。`numeric`は`sample_b64`を両方が持つラベルの組だけに計算する
    (resolution 9: `compare_arrays`を使い、ULP比較を再実装しない)。
    """
    present: dict[str, dict] = {}
    missing: list[str] = []
    order: list[str] = []
    for result in results:
        label = label_for(result.get("env", {}))
        order.append(label)
        entry = _probe_entry(result, probe)
        if entry is not None and "skipped" not in entry:
            present[label] = entry
        else:
            missing.append(label)

    groups: dict[str, list[str]] = {}
    for label in order:
        if label in present:
            groups.setdefault(present[label]["digest"], []).append(label)

    all_equal = len(groups) <= 1

    numeric: dict[tuple[str, str], dict] = {}
    sampled = [
        label
        for label in order
        if label in present and present[label].get("sample_b64") is not None
    ]
    for a, b in itertools.combinations(sampled, 2):
        arr_a = decode_sample(present[a]["sample_b64"])
        arr_b = decode_sample(present[b]["sample_b64"])
        numeric[(a, b)] = compare_arrays(arr_a, arr_b)

    return {"groups": groups, "all_equal": all_equal, "missing": missing, "numeric": numeric}


def _json_safe_agreement(agg: dict) -> dict:
    """`agreement()`の返り値をJSON安全な形に変換する。

    `numeric`のキーはラベルの2要素タプルであり、`json.dumps`はtupleを
    dictキーとして受け付けない(非文字列キーのうちstr/int/float/bool/
    Noneのみ自動的に文字列化される)。`"|"`区切りの文字列へ変換する
    (ラベル自体に`"|"`が混ざる運用は想定していない——ラベルは
    `{host}/{image}/{config_label}`の組み立てであり、現行の全ホスト名・
    イメージ名・構成ラベルに`"|"`を使わない)。
    """
    return {
        "groups": agg["groups"],
        "all_equal": agg["all_equal"],
        "missing": agg["missing"],
        "numeric": {f"{a}|{b}": v for (a, b), v in agg["numeric"].items()},
    }


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
            (resolution 2)。
        "self_reported_stability" — プローブ自身が
            `meta["stable_within_process"]`を自己申告する。環境間の
            digest不一致を不一致として判定しない(resolution 3)。
        "linked" — 別の1プローブ(`linked_with`)と構造的に同一になる
            ことが分かっている。判定自体は"normal"と同じだが、その旨を
            注記する(resolution 4)。

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
            "このフィールドの値ごとに分けて判定する。"
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
    """
    for prefix, category in (
        ("missing requirement: ", "missing_requirement"),
        ("missing env var: ", "missing_env_var"),
        ("raised: ", "raised"),
    ):
        if reason.startswith(prefix):
            return category
    return "unknown"  # runner.pyの契約外の理由文字列(本来起こらないはず)


def _policy_dict(policy: ProbePolicy) -> dict:
    return {
        "kind": policy.kind,
        "note": policy.note,
        "group_by_meta": policy.group_by_meta,
        "linked_with": policy.linked_with,
    }


def _build_probe_row(results: list[dict], probe: str, labels: list[str]) -> dict:
    """1プローブぶんの報告行を組み立てる。

    `absent`/`skips`(missing_requirement/missing_env_var/raised)は
    方針(kind)に関わらず常に計算する——自己申告系・グループ化系の
    プローブであっても、あるラベルでクラッシュ(`raised:`)していれば
    それは常に目立たせるべき実在の知見だから(resolution 7)。
    """
    policy = _POLICY.get(probe, _DEFAULT_POLICY)

    entries: dict[str, dict] = {}
    for result in results:
        label = label_for(result.get("env", {}))
        entry = _probe_entry(result, probe)
        if entry is not None:
            entries[label] = entry

    absent: list[str] = []
    skips: dict[str, dict] = {
        "missing_requirement": {},
        "missing_env_var": {},
        "raised": {},
        "unknown": {},
    }
    for label in labels:
        entry = entries.get(label)
        if entry is None:
            absent.append(label)
        elif "skipped" in entry:
            skips.setdefault(_skip_category(entry["skipped"]), {})[label] = entry["skipped"]

    row: dict = {
        "probe": probe,
        "policy": _policy_dict(policy),
        "absent": absent,
        "skips": skips,
    }

    if policy.kind == "self_reported_stability":
        stability = {}
        digests = {}
        for label, entry in entries.items():
            if "skipped" in entry:
                continue
            stability[label] = entry.get("meta", {}).get("stable_within_process")
            digests[label] = entry["digest"]
        row["stability"] = stability
        row["digests"] = digests
        return row

    if policy.kind == "grouped":
        field = policy.group_by_meta
        by_group: dict[str, list[str]] = {}
        for label, entry in entries.items():
            if "skipped" in entry:
                continue
            value = str(entry.get("meta", {}).get(field))
            by_group.setdefault(value, []).append(label)
        group_reports = {}
        for value, group_labels in by_group.items():
            sub_results = [r for r in results if label_for(r.get("env", {})) in group_labels]
            group_reports[value] = _json_safe_agreement(agreement(sub_results, probe))
        row["group_by"] = field
        row["groups"] = group_reports
        return row

    # "normal" / "linked": 単一の agreement() で判定する。
    row["agreement"] = _json_safe_agreement(agreement(results, probe))
    return row


def build_report(results: list[dict]) -> dict:
    """全プローブ×全ラベルの比較報告を組み立てる。

    ラベルの集合は各`results[i]["env"]`から、プローブ名の集合は各
    `results[i]["probe_order"]`の和集合から作る(あるホストが選択
    実行した範囲の外のプローブは、そのホストでは単に`absent`になる)。
    """
    labels: list[str] = []
    seen_labels: set[str] = set()
    for result in results:
        label = label_for(result.get("env", {}))
        if label not in seen_labels:
            seen_labels.add(label)
            labels.append(label)

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

- `✓` 比較対象の全ラベルでdigestが一致した。
- `✗` 比較対象のラベル間でdigestが不一致だった。
- `△` 自己申告された安定性でのみ判定する(プローブ自身が同一プロセス内で
  繰り返し実行した結果を比較し、`stable_within_process`として記録した値。
  環境間のdigest不一致はこのプローブの測定対象そのものであり不一致として
  扱わない)。
- `–` 比較できるラベルが無い(全てのラベルで欠損/スキップ)。
- 「欠落」列: そのラベルにこの行の結果が無い(ファイルに無い、または
  `missing requirement:`/`missing env var:`でスキップされた——いずれも
  想定内で無害)。
- 「クラッシュ」列(`⚠`): そのラベルで`raised:`によりスキップされた
  (計測ホスト上で実際に例外を送出した——調査が必要な実在の知見)。
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
    if skips.get("unknown"):
        parts.append("unknown_skip:" + ",".join(skips["unknown"]))
    return "; ".join(parts) if parts else "–"


def _fmt_raised(skips: dict) -> str:
    raised = skips.get("raised") or {}
    return ("⚠ " + ", ".join(raised.keys())) if raised else "–"


def _verdict_symbol(agg: dict) -> str:
    if not agg["groups"]:
        return "–"  # 比較できるラベルが1つも無い: "一致"と誤って読める表示はしない
    return "✓" if agg["all_equal"] else "✗"


def _render_row(probe: str, row: dict) -> list[str]:
    policy = row["policy"]
    note = policy["note"]
    if policy["linked_with"]:
        note = f"[linked: {policy['linked_with']}] " + note
    missing_col = _fmt_missing(row["absent"], row["skips"])
    raised_col = _fmt_raised(row["skips"])

    if policy["kind"] == "self_reported_stability":
        if row["stability"]:
            detail = "; ".join(
                f"{label}:{'stable' if ok else 'unstable'}" for label, ok in row["stability"].items()
            )
        else:
            detail = "–"
        return [f"| {probe} | △ | {detail} | {missing_col} | {raised_col} | {note} |"]

    if policy["kind"] == "grouped":
        lines = []
        # キーは辞書式順序で並べる(数値としての大小順ではない——プロジェクトの
        # 他箇所(reprobench.probes.s5_sklearnの`_thread_pool_info`)と同じ
        # 流儀: ここで求めているのは何らかの固定順序であって、2桁以上の値が
        # 混ざったときの数値順を保証するものではない)。
        for group_value, agg in sorted(row["groups"].items()):
            verdict = _verdict_symbol(agg)
            detail = _fmt_groups(agg["groups"])
            group_missing = _fmt_labels(agg["missing"])
            lines.append(
                f"| {probe} [{row['group_by']}={group_value}] | {verdict} | {detail} | "
                f"{group_missing} | {raised_col} | {note} |"
            )
        if not lines:
            lines.append(f"| {probe} | – | –(グループ無し) | {missing_col} | {raised_col} | {note} |")
        return lines

    # "normal" / "linked"
    agg = row["agreement"]
    verdict = _verdict_symbol(agg)
    detail = _fmt_groups(agg["groups"])
    return [f"| {probe} | {verdict} | {detail} | {missing_col} | {raised_col} | {note} |"]


def to_markdown(report: dict) -> str:
    """`build_report`の結果を、記事に貼れる○×表のMarkdownへ整形する。

    状態の区別(ドミナントな失敗モード対策として、以下は互いに潰れない
    別々の列・記号として出す): 全ラベルが一致/不一致/自己申告安定性のみで
    判定(△)/一部ラベルで欠落/一部ラベルでクラッシュ(raised)。
    `S5_kmeans_centers`のようにグループ化されたプローブは、単一のverdict
    ではなくグループごとの行を出す(resolution 2)。
    """
    lines = ["## 比較結果", "", _LEGEND, "| Probe | Verdict | Detail | 欠落 | クラッシュ | Policy |"]
    lines.append("|---|---|---|---|---|---|")
    for probe in report["probes"]:
        lines.extend(_render_row(probe, report["rows"][probe]))

    if report.get("cross_probe_caveats"):
        lines.append("")
        lines.append("## 比較対象外の注記")
        for item in report["cross_probe_caveats"]:
            lines.append(f"- {', '.join(item['probes'])}: {item['note']}")

    return "\n".join(lines)
