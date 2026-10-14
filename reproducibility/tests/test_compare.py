"""reprobench.compare のテスト: 環境横断の結果比較器を検証する。

このモジュールの出力(`build_report`/`to_markdown`)は記事の付録表そのもの
になる。したがって最優先で確認すべき失敗モードは「一致していないものを
一致と報告する」「不一致ではないものを不一致と報告する」こと――とりわけ
欠損(そのラベルにこのプローブの結果が無い)を一致扱いしてしまうこと、
自己申告された既知の不安定さ(S0_parallel_completion_order等)を環境間の
不一致として誤って報告すること、スレッド数が違うために本来比較できない
S5_kmeans_centersの2ラン を比較してしまうことの3つである。
"""
import json

import numpy as np
import pytest

from reprobench.digest import encode_sample
from reprobench.compare import (
    agreement,
    build_report,
    label_for,
    load_results,
    to_markdown,
)


def _res(label, digests=None, samples=None, meta=None, skips=None, image=None, config=None):
    """テスト用: `reprobench.runner.run()`が返すのと同じ形の辞書を組み立てる。

    `digests`は{プローブ名: digest文字列}、`samples`は{プローブ名:
    encode_sample済みのbase64文字列}、`meta`は{プローブ名: metaの辞書}、
    `skips`は{プローブ名: skip理由文字列(例: "raised: ValueError('x')")}。
    同じプローブ名を`digests`と`skips`の両方に入れてはならない(呼び出し側の
    責務とし、ここでは検証しない)。

    ラベルは`label_for`が`env`から`{host}/{image}/{config_label}`として
    組み立てる契約(compare.py参照)なので、`image`/`config`を省略すれば
    `host`だけがラベルになり、呼び出し側が渡した`label`文字列がそのまま
    ラベルになる。
    """
    digests = digests or {}
    samples = samples or {}
    meta = meta or {}
    skips = skips or {}
    probes = []
    for name, d in digests.items():
        probes.append(
            {
                "probe": name,
                "digest": d,
                "scalars": {},
                "sample_b64": samples.get(name),
                "meta": meta.get(name, {}),
            }
        )
    for name, reason in skips.items():
        probes.append({"probe": name, "skipped": reason, "wall_time_s": 0.001})
    return {
        "schema": 1,
        "env": {"host": label, "image": image, "config_label": config},
        "probe_order": list(digests.keys()) + list(skips.keys()),
        "probes": probes,
    }


def _b64(arr: np.ndarray) -> str:
    return encode_sample(arr)


# --- タスクブリーフ記載のテスト(そのまま) ---


def test_agreement_groups_identical_digests():
    rs = [
        _res("a", {"S2_sum_float32": "dead"}),
        _res("b", {"S2_sum_float32": "dead"}),
        _res("c", {"S2_sum_float32": "beef"}),
    ]
    g = agreement(rs, "S2_sum_float32")
    assert g["all_equal"] is False and sorted(map(len, g["groups"].values())) == [1, 2]


def test_missing_probe_is_reported_not_treated_as_match():
    rs = [_res("a", {"S8_x": "dead"}), _res("b", {})]
    g = agreement(rs, "S8_x")
    assert g["missing"] == ["b"] and g["all_equal"] is True  # 存在する2件だけで判定


def test_numeric_delta_computed_from_samples():
    rs = [
        _res("a", {"S2_sum_float64": "d1"}, samples={"S2_sum_float64": _b64(np.arange(8.0))}),
        _res(
            "b",
            {"S2_sum_float64": "d2"},
            samples={"S2_sum_float64": _b64(np.nextafter(np.arange(8.0), 9.0))},
        ),
    ]
    g = agreement(rs, "S2_sum_float64")
    assert g["numeric"][("a", "b")]["max_ulp"] == 1


def test_to_markdown_renders_one_row_per_probe():
    md = to_markdown(build_report([_res("a", {"S1_x": "d"}), _res("b", {"S1_x": "d"})]))
    assert md.count("\n| S1_x") == 1


# --- 解決事項に対する追加テスト ---


def test_label_for_joins_host_image_config():
    # ラベルは env から {host}/{image}/{config_label} で作る(インターフェース
    # 契約そのもの)。
    env = {"host": "blade", "image": "cuda12.4", "config_label": "threads8"}
    assert label_for(env) == "blade/cuda12.4/threads8"


def test_label_for_omits_missing_parts():
    # 解決事項の前提: envinfo.collect_env()はREPRO_IMAGE/REPRO_CONFIGが
    # 未設定の環境ではimage/config_labelをNoneのまま返す契約(envinfo.py
    # 参照)。Noneを文字列"None"として混ぜ込むと、本当は同じ構成の2環境が
    # 見かけ上別ラベルになってしまう。
    env = {"host": "macbook", "image": None, "config_label": None}
    assert label_for(env) == "macbook"


def test_agreement_numeric_only_when_both_sides_have_samples():
    # 解決事項9: sample_b64が無い側がいればそのペアはnumericに出さない。
    rs = [
        _res("a", {"S2_sum_float64": "d1"}, samples={"S2_sum_float64": _b64(np.arange(8.0))}),
        _res("b", {"S2_sum_float64": "d1"}),  # sample_b64無し
    ]
    g = agreement(rs, "S2_sum_float64")
    assert g["numeric"] == {}


def test_kmeans_grouped_by_effective_thread_count():
    # 解決事項2: effective_thread_countが一致するラベル同士でのみ判定し、
    # 異なるスレッド数同士は混ぜない。
    samples = {
        "a1": _b64(np.array([1.0, 2.0, 3.0])),
        "a2": _b64(np.array([1.0, 2.0, 3.0])),
        "b1": _b64(np.array([9.0, 9.0, 9.0])),
    }
    rs = [
        _res(
            "a1",
            {"S5_kmeans_centers": "same"},
            samples={"S5_kmeans_centers": samples["a1"]},
            meta={"S5_kmeans_centers": {"effective_thread_count": 1}},
        ),
        _res(
            "a2",
            {"S5_kmeans_centers": "same"},
            samples={"S5_kmeans_centers": samples["a2"]},
            meta={"S5_kmeans_centers": {"effective_thread_count": 1}},
        ),
        _res(
            "b1",
            {"S5_kmeans_centers": "different"},
            samples={"S5_kmeans_centers": samples["b1"]},
            meta={"S5_kmeans_centers": {"effective_thread_count": 8}},
        ),
    ]
    report = build_report(rs)
    row = report["rows"]["S5_kmeans_centers"]
    assert row["policy"]["kind"] == "grouped"
    assert row["group_by"] == "effective_thread_count"
    groups = row["groups"]
    assert groups["1"]["all_equal"] is True
    assert sorted(sum(groups["1"]["groups"].values(), [])) == ["a1", "a2"]
    # 1スレッド側のグループには8スレッド側のラベルが絶対に混ざらない。
    assert "b1" not in sum(groups["1"]["groups"].values(), [])
    assert groups["8"]["all_equal"] is True
    assert sum(groups["8"]["groups"].values(), []) == ["b1"]


def test_self_reported_stability_probe_is_not_rendered_as_disagreement():
    # 解決事項3: S0_parallel_completion_order/S0_glob_order/
    # S8_index_add_atomicsは環境間でdigestが食い違っても、それ自体は
    # 「不一致」ではない(プローブ自身が自己申告するstable_within_process
    # を見る)。
    rs = [
        _res(
            "a",
            {"S0_glob_order": "d1"},
            meta={"S0_glob_order": {"stable_within_process": True}},
        ),
        _res(
            "b",
            {"S0_glob_order": "d2"},
            meta={"S0_glob_order": {"stable_within_process": True}},
        ),
    ]
    report = build_report(rs)
    row = report["rows"]["S0_glob_order"]
    assert row["policy"]["kind"] == "self_reported_stability"
    assert "agreement" not in row  # 通常のdigest一致判定はそもそも行わない
    assert row["stability"] == {"a": True, "b": True}


def test_s8_index_add_atomics_deterministic_is_treated_normally():
    # 解決事項3の後半: 対照群は自己申告ではなく通常どおり不一致として扱う。
    rs = [
        _res(
            "a",
            {"S8_index_add_atomics_deterministic": "d1"},
            meta={"S8_index_add_atomics_deterministic": {"stable_within_process": True}},
        ),
        _res(
            "b",
            {"S8_index_add_atomics_deterministic": "d2"},
            meta={"S8_index_add_atomics_deterministic": {"stable_within_process": True}},
        ),
    ]
    report = build_report(rs)
    row = report["rows"]["S8_index_add_atomics_deterministic"]
    assert row["policy"]["kind"] == "normal"
    assert row["agreement"]["all_equal"] is False


def test_linked_pair_is_annotated_both_ways():
    # 解決事項4: S0_dict_from_set/S0_set_iteration_orderは構造的に同一の
    # 情報を2回運ぶだけなので、互いにlinked_with注記を持つ。
    rs = [
        _res("a", {"S0_dict_from_set": "d", "S0_set_iteration_order": "d"}),
        _res("b", {"S0_dict_from_set": "d", "S0_set_iteration_order": "d"}),
    ]
    report = build_report(rs)
    assert report["rows"]["S0_dict_from_set"]["policy"]["linked_with"] == "S0_set_iteration_order"
    assert report["rows"]["S0_set_iteration_order"]["policy"]["linked_with"] == "S0_dict_from_set"


def test_raised_skip_is_separated_from_missing_requirement_skip():
    # 解決事項7: "raised: "は欠損(missing requirement/missing env var)と
    # 混ぜずに別枠で目立たせる。
    rs = [
        _res("a", {}, skips={"S7_x": "raised: ValueError('boom')"}),
        _res("b", {}, skips={"S7_x": "missing requirement: torch"}),
        _res("c", {"S7_x": "ok"}),
    ]
    report = build_report(rs)
    row = report["rows"]["S7_x"]
    assert row["skips"]["raised"] == {"a": "raised: ValueError('boom')"}
    assert row["skips"]["missing_requirement"] == {"b": "missing requirement: torch"}
    assert "a" not in row["absent"] and "b" not in row["absent"]


def test_default_policy_applies_to_unlisted_probe():
    # 解決事項1: 表に無いプローブにも既定の方針が適用され、その旨が読める。
    report = build_report([_res("a", {"S2_sum_float32": "d"})])
    row = report["rows"]["S2_sum_float32"]
    assert row["policy"]["kind"] == "normal"
    assert row["policy"]["note"]  # 既定の扱いである旨が空文字列ではない


def test_build_report_is_json_safe():
    # 解決事項10: このプロジェクトはシリアライズでしか落ちない値に二度
    # 刺されているため、実際のdictから組み立てたreportをjson.dumpsに
    # 通して確認する(tupleキーの取り回しなど)。
    rs = [
        _res(
            "a",
            {"S2_sum_float64": "d1"},
            samples={"S2_sum_float64": _b64(np.arange(8.0))},
        ),
        _res(
            "b",
            {"S2_sum_float64": "d2"},
            samples={"S2_sum_float64": _b64(np.nextafter(np.arange(8.0), 9.0))},
        ),
        _res(
            "c",
            {
                "S5_kmeans_centers": "d3",
                "S0_glob_order": "d4",
                "S0_dict_from_set": "d5",
                "S0_set_iteration_order": "d5",
            },
            meta={
                "S5_kmeans_centers": {"effective_thread_count": 4},
                "S0_glob_order": {"stable_within_process": False},
            },
            skips={"S7_x": "raised: ValueError('boom')"},
        ),
    ]
    report = build_report(rs)
    json.dumps(report)  # 例外を送出しないこと自体が検証


def test_build_report_is_json_safe_from_real_probe_output():
    # 解決事項10の実物版: 合成フィクスチャではなく本物のrun()出力で確認する。
    import reprobench.probes  # noqa: F401  (副作用: 全プローブを登録する)
    from reprobench.runner import run

    out_a = run(selected=["S0", "S1", "S2"])
    out_b = run(selected=["S0", "S1", "S2"])
    out_a["env"]["host"] = "real-a"
    out_b["env"]["host"] = "real-b"
    report = build_report([out_a, out_b])
    json.dumps(report)


def test_load_results_reads_files(tmp_path):
    p1 = tmp_path / "one.json"
    p2 = tmp_path / "two.json"
    p1.write_text(json.dumps(_res("a", {"S1_x": "d"})))
    p2.write_text(json.dumps(_res("b", {"S1_x": "d"})))
    results = load_results([str(p1), str(p2)])
    assert len(results) == 2
    assert {label_for(r["env"]) for r in results} == {"a", "b"}


def test_cross_probe_caveats_mention_s9_logit_margin_and_s9_greedy_batch1():
    # 解決事項5: 別ラン(repetition_penaltyが違う)なので比較対象ではない旨の
    # 注記が報告に含まれる。
    report = build_report([_res("a", {"S9_logit_margin": "d"})])
    caveats = " ".join(c["note"] + " ".join(c["probes"]) for c in report["cross_probe_caveats"])
    assert "S9_logit_margin" in caveats and "S9_greedy_batch1" in caveats


def test_to_markdown_distinguishes_missing_from_raised():
    rs = [
        _res("a", {}, skips={"S7_x": "raised: ValueError('boom')"}),
        _res("b", {}, skips={"S7_x": "missing requirement: torch"}),
        _res("c", {"S7_x": "ok"}),
    ]
    md = to_markdown(build_report(rs))
    lines = [line for line in md.splitlines() if line.startswith("| S7_x")]
    assert len(lines) == 1
    line = lines[0]
    assert "a" in line and "b" in line
    # raisedの列とmissingの列は別の表セルであり、同じ文字列に潰れていない
    # こと(resolution 7)を、ラベル"a"(raised)が独立した位置に出ることで
    # 確認する。
    cells = [c.strip() for c in line.split("|")]
    assert any("a" in cell for cell in cells if "b" not in cell)


def test_to_markdown_shows_groups_for_grouped_probe():
    # 解決事項2+レンダリング要件: グループ化されたプローブは単一の verdict
    # ではなく、グループごとに行を出す。
    rs = [
        _res(
            "a1",
            {"S5_kmeans_centers": "same"},
            samples={"S5_kmeans_centers": _b64(np.array([1.0]))},
            meta={"S5_kmeans_centers": {"effective_thread_count": 1}},
        ),
        _res(
            "b1",
            {"S5_kmeans_centers": "different"},
            samples={"S5_kmeans_centers": _b64(np.array([9.0]))},
            meta={"S5_kmeans_centers": {"effective_thread_count": 8}},
        ),
    ]
    md = to_markdown(build_report(rs))
    assert "effective_thread_count=1" in md
    assert "effective_thread_count=8" in md


def test_to_markdown_legend_defines_symbols():
    md = to_markdown(build_report([_res("a", {"S1_x": "d"})]))
    assert "✓" in md and "✗" in md and "△" in md


def test_all_missing_probe_does_not_render_as_agreement():
    # ドミナントな失敗モード対策: 比較できるラベルが1つも無いときに
    # "✓ 一致"と誤って読める表示をしない。
    rs = [_res("a", {}, skips={"S7_x": "missing requirement: torch"})]
    md = to_markdown(build_report(rs))
    line = next(line for line in md.splitlines() if line.startswith("| S7_x"))
    cells = [c.strip() for c in line.split("|")]
    assert cells[2] != "✓"
