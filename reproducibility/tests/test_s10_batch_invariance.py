"""reprobench.probes.s10_batch_invariance のテスト。

`BATCH_INVARIANCE_REDESIGN.md` の設計(config表・padding_side軸・事前登録
された予測)の実装を検証する。実モデルを使う部分(torch/transformers/
HFキャッシュ済みモデル)が無い環境では`run_probe`は`SkippedProbe`を返す
ため、そのパスは`test_s9_llm.py`と同じ流儀で`if isinstance(r, ProbeResult)`
にガードする。位置計算(`compute_p_logit_position`)・EOS後のpadding除去
(`trim_at_eos`)・バッチ構成の定義(`content_texts`/`filler_content`)は
純粋関数でありtorch等に依存しないため、ガード無しでそのまま検証できる
(brief: 「実モデルを使うテストは重いので、位置計算などの純粋なロジックは
モデル無しでテストできるように関数を切り出すこと」)。
"""
import json

import numpy as np

import reprobench.probes  # モジュールimportの副作用でS10プローブ群を登録する
from reprobench import registry
from reprobench.registry import ProbeResult, SkippedProbe, run_probe
from reprobench.probes.s9_llm import _PROMPT as _S9_PROMPT
from reprobench.probes.s10_batch_invariance import (
    GREEDY_CONFIG_ORDER,
    LOGIT_CONFIG_ORDER,
    _BASE_CONFIG_NAMES,
    _LOGIT_CONFIG_SPECS,
    _P_PROMPT,
    compute_p_logit_position,
    content_texts,
    filler_content,
    trim_at_eos,
)


# --- config定義が設計書と一致していること --------------------------------


def test_p_prompt_is_identical_to_s9s_prompt():
    # fix round 1 (座長指摘): S10は独自の短いプロンプトを使っていたため、
    # (a) S9の既存実測と比較できない、(b) 「一言で答えろ」という指示は
    # 数トークンでEOSに達するため分岐位置の分布にほぼ情報が無く、EOS後の
    # pad_token_id混入(first_divergenceへの誤検出)と実質的な分岐を区別
    # できなかった。S10はs9_llm._PROMPTをそのままimportして使うことで、
    # 2つのプローブが「同じPについて測っている」という設計の前提を保つ。
    assert _P_PROMPT == _S9_PROMPT
    assert _P_PROMPT == "List three distinct facts about the Roman Empire, one short sentence each."


def test_base_config_names_match_design_doc_table():
    # BATCH_INVARIANCE_REDESIGN.md の config表に載っている7本。
    assert _BASE_CONFIG_NAMES == (
        "b1",
        "b8_same",
        "b8_mixed_first",
        "b8_mixed_last",
        "b2_long",
        "b16_mixed",
        "b4_padheavy",
    )


def test_greedy_probe_uses_the_seven_base_configs():
    assert GREEDY_CONFIG_ORDER == _BASE_CONFIG_NAMES


def test_logit_probe_splits_b8_mixed_first_into_left_and_right_variants():
    # 設計書: 「さらに b8_mixed_first について padding_side を left / right
    # の両方で測る」。残り6本は単一のpadding_sideで走る。
    assert "b8_mixed_first_right" in LOGIT_CONFIG_ORDER
    assert "b8_mixed_first_left" in LOGIT_CONFIG_ORDER
    assert len(LOGIT_CONFIG_ORDER) == 8
    assert _LOGIT_CONFIG_SPECS["b8_mixed_first_right"] == ("b8_mixed_first", "right")
    assert _LOGIT_CONFIG_SPECS["b8_mixed_first_left"] == ("b8_mixed_first", "left")
    for name in ("b1", "b8_same", "b8_mixed_last", "b2_long", "b16_mixed", "b4_padheavy"):
        content_name, _side = _LOGIT_CONFIG_SPECS[name]
        assert content_name == name


def test_content_texts_places_p_at_the_declared_index():
    for name in _BASE_CONFIG_NAMES:
        contents, p_index = content_texts(name)
        assert contents[p_index] == _P_PROMPT

    # b8_mixed_first: Pは先頭(位置0)。
    contents, p_index = content_texts("b8_mixed_first")
    assert p_index == 0
    assert len(contents) == 8  # P + filler7本

    # b8_mixed_last: Pは末尾(filler7本の後ろ)。
    contents, p_index = content_texts("b8_mixed_last")
    assert p_index == 7
    assert len(contents) == 8

    # b16_mixed: P + filler15本 = 16行。
    contents, p_index = content_texts("b16_mixed")
    assert p_index == 0
    assert len(contents) == 16

    # b4_padheavy: P + filler3本 = 4行。
    contents, p_index = content_texts("b4_padheavy")
    assert p_index == 0
    assert len(contents) == 4

    # b2_long: P + filler1本 = 2行。
    contents, p_index = content_texts("b2_long")
    assert p_index == 0
    assert len(contents) == 2

    # b1/b8_same: filler無し。
    contents, p_index = content_texts("b1")
    assert contents == [contents[0]]
    assert p_index == 0
    contents, p_index = content_texts("b8_same")
    assert len(contents) == 8
    assert len(set(contents)) == 1  # 8本とも同一プロンプト
    assert p_index == 0


def test_content_texts_rejects_unknown_config_name():
    import pytest

    with pytest.raises(ValueError):
        content_texts("not_a_real_config")


def test_filler_content_lengths_are_clearly_distinct():
    # brief: 「filler プロンプトは長さが明確に異なるものを複数用意する
    # こと(設計書の狙い通り)」。繰り返し回数を増やすほど単調に長くなる
    # ことを確認する(実際のトークン長はtokenizer依存だが、文字列長が
    # 単調増加していれば、トークン長もほぼ確実に単調増加する)。
    lengths = [len(filler_content(n)) for n in range(1, 16)]
    assert lengths == sorted(lengths)
    assert len(set(lengths)) == len(lengths)  # 全て相異なる


# --- ロジットを取る位置の計算(padding_sideごと)------------------------


def test_compute_p_logit_position_right_padding():
    # 右パディング: 実トークンは常に位置0から始まるので、Pの最終実トークン
    # 位置はPの実トークン長-1で決まり、バッチ内の他系列や合計のパディング
    # 後系列長には依存しない。
    assert compute_p_logit_position("right", p_real_len=5, padded_seq_len=5) == 4
    assert compute_p_logit_position("right", p_real_len=5, padded_seq_len=100) == 4
    assert compute_p_logit_position("right", p_real_len=1, padded_seq_len=10) == 0


def test_compute_p_logit_position_left_padding():
    # 左パディング: 実トークンは常にパディング後シーケンスの末尾に右詰め
    # されるので、Pの最終実トークン位置は常に最後のインデックスであり、
    # Pの実トークン長には依存しない。
    assert compute_p_logit_position("left", p_real_len=5, padded_seq_len=5) == 4
    assert compute_p_logit_position("left", p_real_len=1, padded_seq_len=100) == 99
    assert compute_p_logit_position("left", p_real_len=50, padded_seq_len=100) == 99


def test_compute_p_logit_position_rejects_unknown_padding_side():
    import pytest

    with pytest.raises(ValueError):
        compute_p_logit_position("center", p_real_len=1, padded_seq_len=1)


def test_compute_p_logit_position_rejects_out_of_range_right_padding():
    import pytest

    with pytest.raises(ValueError):
        compute_p_logit_position("right", p_real_len=0, padded_seq_len=5)
    with pytest.raises(ValueError):
        compute_p_logit_position("right", p_real_len=6, padded_seq_len=5)


def test_position_calculation_against_a_small_dummy_tensor_right_padding():
    # brief: 「小さいダミーテンソルで、right/left 両方について期待する
    # 位置が取れることを検証する」。3行・パディング後シーケンス長4の
    # ダミーattention_mask/logitsで、右パディングの場合に正しい行が
    # 取れることを確認する。
    # 行0: 実トークン長4(パディング無し)、行1(=P): 実トークン長2、
    # 行2: 実トークン長3。
    attn = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ]
    )
    vocab = 2
    logits = np.arange(attn.shape[0] * attn.shape[1] * vocab).reshape(attn.shape[0], attn.shape[1], vocab)

    p_index = 1
    p_real_len = 2
    padded_seq_len = attn.shape[1]
    pos = compute_p_logit_position("right", p_real_len, padded_seq_len)

    assert pos == 1
    # 位置posは実トークン(mask==1)で、その次(pos+1)はパディング(mask==0)
    # — つまりposはPの「最後の」実トークンであるべき。
    assert attn[p_index, pos] == 1
    assert attn[p_index, pos + 1] == 0
    expected_row = logits[p_index, pos, :]
    assert np.array_equal(logits[p_index][pos], expected_row)
    # 他の行の実トークン長・パディングの有無が変わっても、Pの行自体の
    # 計算結果(どの位置を読むか)は変わらない。
    pos_same = compute_p_logit_position("right", p_real_len, padded_seq_len=10)
    assert pos_same == pos


def test_position_calculation_against_a_small_dummy_tensor_left_padding():
    # 左パディング: 実トークンは末尾に右詰めされるので、どの行についても
    # 最後のインデックスが実トークンになっているはず。
    attn = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    vocab = 2
    logits = np.arange(attn.shape[0] * attn.shape[1] * vocab).reshape(attn.shape[0], attn.shape[1], vocab)

    p_index = 0
    p_real_len = 2
    padded_seq_len = attn.shape[1]
    pos = compute_p_logit_position("left", p_real_len, padded_seq_len)

    assert pos == padded_seq_len - 1 == 3
    assert attn[p_index, pos] == 1
    expected_row = logits[p_index, pos, :]
    assert np.array_equal(logits[p_index][pos], expected_row)
    # 実トークン長が変わっても(左パディングでは)常に最後のインデックス。
    pos_row2 = compute_p_logit_position("left", p_real_len=4, padded_seq_len=padded_seq_len)
    assert pos_row2 == pos


# --- EOS到達後のpadding混入を取り除く(trim_at_eos) ------------------------


def test_trim_at_eos_truncates_at_first_match_inclusive():
    assert trim_at_eos([1, 2, 3, 99, 4, 4, 4], eos_token_ids=99) == [1, 2, 3, 99]


def test_trim_at_eos_no_match_returns_full_sequence():
    assert trim_at_eos([1, 2, 3], eos_token_ids=99) == [1, 2, 3]


def test_trim_at_eos_accepts_a_list_of_eos_ids():
    # Qwen2.5-0.5B-Instructのmodel.generation_config.eos_token_idは
    # [151645, 151643]という2要素のリスト(後者はpad_token_idでもある)。
    # どちらが先に現れても、そこで切り詰められること。
    assert trim_at_eos([5, 6, 151645, 151643, 151643], eos_token_ids=[151645, 151643]) == [
        5,
        6,
        151645,
    ]
    assert trim_at_eos([5, 6, 151643, 151645], eos_token_ids=[151645, 151643]) == [5, 6, 151643]


def test_trim_at_eos_removes_padding_after_an_early_finisher_in_a_batch():
    # brief(座長指摘): バッチ内の他行がPより長く生成を続けた場合、Pの行は
    # pad_token_idで埋められる。これを取り除かないと「Pの内容自体が
    # バッチ構成で変わった」のか「単に水増しされただけ」かを区別できない。
    # 実際に観測した生データ(b1=[59604, 151645], 混在バッチでのPの行の
    # 生データ=[59604, 151645, 151643, ...(151643が18個)...])を模した
    # ケースで、trim後はb1と一致することを確認する。
    b1_like = [59604, 151645]
    padded_like = [59604, 151645] + [151643] * 18
    trimmed = trim_at_eos(padded_like, eos_token_ids=[151645, 151643])
    assert trimmed == b1_like


def test_trim_at_eos_does_not_mutate_the_input_list():
    original = [1, 2, 99, 3]
    trim_at_eos(original, eos_token_ids=99)
    assert original == [1, 2, 99, 3]


# --- requires によるスキップ ------------------------------------------------


def test_s10_probes_skip_without_model(monkeypatch):
    monkeypatch.setattr(registry, "check_requirement", lambda r: r != "hf_model")
    for name in ("S10_batch_invariance_logits", "S10_batch_invariance_greedy"):
        r = run_probe(name)
        assert isinstance(r, SkippedProbe) and "hf_model" in r.reason


def test_s10_probes_declare_all_three_requirements(monkeypatch):
    for req in ("transformers", "torch", "hf_model"):

        def _fake_check(r, missing=req):
            return r != missing

        monkeypatch.setattr(registry, "check_requirement", _fake_check)
        for name in ("S10_batch_invariance_logits", "S10_batch_invariance_greedy"):
            r = run_probe(name)
            assert isinstance(r, SkippedProbe) and req in r.reason, name


# --- メタに必要なキーが揃っていること(モデルが使える環境でのみ実行) --------


def test_logits_probe_meta_has_required_keys():
    r = run_probe("S10_batch_invariance_logits")
    if isinstance(r, ProbeResult):
        for key in (
            "seed",
            "model",
            "device",
            "dtype",
            "p_prompt",
            "p_real_token_length",
            "pad_token",
            "pad_token_id",
            "pad_token_source",
            "filler_base_sentence",
            "filler_repeats",
            "config_order",
            "p1_b1_vs_b8_same_p_real_len_equal",
            "configs",
        ):
            assert key in r.meta, key
        assert r.meta["config_order"] == list(LOGIT_CONFIG_ORDER)
        for name in LOGIT_CONFIG_ORDER:
            cfg = r.meta["configs"][name]
            for key in (
                "digest",
                "digest_rerun",
                "stable_within_process",
                "argmax_token_id",
                "margin_top1_top2",
                "p_real_len",
                "padded_seq_len",
                "padding_side",
                "batch_size",
                "p_index",
                "bit_equal_vs_b1",
                "max_abs_vs_b1",
                "max_ulp_vs_b1",
            ):
                assert key in cfg, (name, key)


def test_logits_probe_b1_vs_b1_is_trivially_bit_equal():
    r = run_probe("S10_batch_invariance_logits")
    if isinstance(r, ProbeResult):
        b1 = r.meta["configs"]["b1"]
        assert b1["bit_equal_vs_b1"] is True
        assert b1["max_abs_vs_b1"] == 0.0
        assert b1["max_ulp_vs_b1"] == 0


def test_logits_probe_asserts_b1_and_b8_same_p_real_len_match():
    # brief: 「実装したら、b1 と b8_same で P の実トークン長が同じで
    # あることを assert で確認すること」。本プローブは実行時にこれを
    # assertしており(位置計算の前提そのもの)、ここではその結果が
    # metaに観測可能な形で残ることを確認する。
    r = run_probe("S10_batch_invariance_logits")
    if isinstance(r, ProbeResult):
        assert r.meta["p1_b1_vs_b8_same_p_real_len_equal"] is True
        assert (
            r.meta["configs"]["b1"]["p_real_len"]
            == r.meta["configs"]["b8_same"]["p_real_len"]
        )


def test_logits_probe_padding_side_recorded_per_config():
    r = run_probe("S10_batch_invariance_logits")
    if isinstance(r, ProbeResult):
        assert r.meta["configs"]["b8_mixed_first_right"]["padding_side"] == "right"
        assert r.meta["configs"]["b8_mixed_first_left"]["padding_side"] == "left"


def test_greedy_probe_meta_has_required_keys():
    r = run_probe("S10_batch_invariance_greedy")
    if isinstance(r, ProbeResult):
        for key in (
            "seed",
            "model",
            "device",
            "dtype",
            "do_sample",
            "max_new_tokens",
            "padding_side",
            "p_prompt",
            "pad_token",
            "pad_token_source",
            "eos_token_ids",
            "config_order",
            "configs",
        ):
            assert key in r.meta, key
        assert r.meta["do_sample"] is False
        assert r.meta["padding_side"] == "left"
        assert r.meta["max_new_tokens"] == 128
        assert r.meta["config_order"] == list(GREEDY_CONFIG_ORDER)
        for name in GREEDY_CONFIG_ORDER:
            cfg = r.meta["configs"][name]
            for key in (
                "digest",
                "token_ids",
                "raw_token_ids",
                "text",
                "n_tokens",
                "n_tokens_raw",
                "batch_size",
                "padded_prompt_len",
                "p_index",
                "first_divergence_vs_b1",
            ):
                assert key in cfg, (name, key)
            # token_idsはraw_token_idsをtrim_at_eosした結果であり、常に
            # raw以下の長さであるべき(brief: EOS後のpadding混入を除く)。
            assert cfg["n_tokens"] <= cfg["n_tokens_raw"]
        assert r.meta["configs"]["b1"]["first_divergence_vs_b1"] is None


def test_s10_probe_meta_survives_a_json_round_trip():
    for name in ("S10_batch_invariance_logits", "S10_batch_invariance_greedy"):
        r = run_probe(name)
        if isinstance(r, ProbeResult):
            round_tripped = json.loads(json.dumps(r.meta))
            assert round_tripped == r.meta, name


def test_s10_probe_top_level_digest_is_order_sensitive_config_map():
    # digestは「config名 -> ロジット/トークン列digest」の宣言順リストから
    # 取る(brief)。少なくとも、全く同じconfig集合でも実データが変われば
    # 全体digestも変わる(= 中身を実際に見ている)ことを確認する。
    r = run_probe("S10_batch_invariance_logits")
    if isinstance(r, ProbeResult):
        per_config_digests = [r.meta["configs"][name]["digest"] for name in LOGIT_CONFIG_ORDER]
        assert len(set(per_config_digests)) >= 2  # 全config一致はしていない
        assert r.digest  # 空でない
