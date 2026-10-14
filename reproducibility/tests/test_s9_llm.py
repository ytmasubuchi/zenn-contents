"""reprobench.probes.s9_llm のテスト: LLM推論とtemperatureの意味論を検証する。

transformers/torch/HFキャッシュ済みモデルはこのテストコンテナ
(python:3.11-slim + numpy + pytest)には無いため、`run_probe`は常に
`SkippedProbe`を返す。`first_divergence`は純粋関数でありtorch等に依存
しないため、ガード無しでそのまま実行できる。
"""
import json

import reprobench.probes  # モジュールimportの副作用でS9プローブ群を登録する
from reprobench import registry
from reprobench.registry import ProbeResult, SkippedProbe, run_probe
from reprobench.probes.s9_llm import first_divergence


# --- タスクブリーフ記載のテスト(そのまま) ---


def test_first_divergence_finds_position():
    assert first_divergence([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2], [1, 2, 3]) == 2


def test_llm_probes_skip_without_model(monkeypatch):
    monkeypatch.setattr(registry, "check_requirement", lambda r: r != "hf_model")
    r = run_probe("S9_greedy_batch1")
    assert isinstance(r, SkippedProbe) and "hf_model" in r.reason


def test_llm_probe_meta_records_tokens_and_margin_when_available():
    r = run_probe("S9_logit_margin")
    if isinstance(r, ProbeResult):
        assert len(r.meta["margins"]) == r.meta["n_tokens"]


# --- 曖昧さの解決事項に対する追加テスト ---


def test_temperature_zero_probe_skips_cleanly_without_model(monkeypatch):
    monkeypatch.setattr(registry, "check_requirement", lambda r: r != "hf_model")
    r = run_probe("S9_temperature_zero_api_semantics")
    assert isinstance(r, SkippedProbe) and "hf_model" in r.reason


def test_llm_probes_declare_all_three_requirements(monkeypatch):
    # transformers/torch/hf_modelのどれか1つでも欠けていればスキップ
    # しなければならない(resolution 2: requiresの宣言漏れはraisedに
    # 化けてしまい、設計通りのスキップと区別できなくなる)。
    for req in ("transformers", "torch", "hf_model"):

        def _fake_check(r, missing=req):
            return r != missing

        monkeypatch.setattr(registry, "check_requirement", _fake_check)
        r = run_probe("S9_greedy_batch1")
        assert isinstance(r, SkippedProbe) and req in r.reason


def test_greedy_probes_use_do_sample_false_not_temperature_zero():
    # temperature=0はtransformersに存在しないため、greedyはdo_sample=Falseで
    # 行う(temperature=0 + do_sample=Trueはstrictly positive要求で弾かれる)。
    r = run_probe("S9_greedy_batch1")
    if isinstance(r, ProbeResult):
        assert r.meta["do_sample"] is False
        assert "temperature" not in r.meta or r.meta.get("temperature") is None


# --- fix round 1 (coordinator review) ---


def test_every_s9_probe_records_generation_config_and_device():
    # finding 4: generation_config.json ships top_p/top_k/repetition_penalty
    # that are not neutral; recording it is what lets a reader notice.
    # finding 7: the model must run on CUDA when available, and every probe
    # must say which device it actually used.
    names = [
        "S9_greedy_batch1", "S9_greedy_batch8", "S9_greedy_fp32",
        "S9_greedy_bf16", "S9_greedy_fp16", "S9_temperature_zero_api_semantics",
        "S9_sampling_t0_7_seed", "S9_sampling_t0_7_seed_isolated",
        "S9_sampling_t1_2_seed", "S9_sampling_t1_2_seed_isolated",
        "S9_logit_margin",
    ]
    for name in names:
        r = run_probe(name)
        if isinstance(r, ProbeResult):
            assert "generation_config" in r.meta, name
            assert "device" in r.meta, name


def test_every_s9_probe_meta_survives_a_json_round_trip():
    # fix round 2 small finding: generation_config.to_dict() is produced by
    # a library we do not control, and nothing previously checked that the
    # dict it returns is actually JSON-serializable. This is exactly the
    # class of bug finding 9 (int-keyed meta["digests"] in S8) was -- a meta
    # field that looks fine in-process but breaks (or silently changes
    # shape) the moment a later task's results.json round-trips it.
    names = [
        "S9_greedy_batch1", "S9_greedy_batch8", "S9_greedy_fp32",
        "S9_greedy_bf16", "S9_greedy_fp16", "S9_temperature_zero_api_semantics",
        "S9_sampling_t0_7_seed", "S9_sampling_t0_7_seed_isolated",
        "S9_sampling_t1_2_seed", "S9_sampling_t1_2_seed_isolated",
        "S9_logit_margin",
    ]
    for name in names:
        r = run_probe(name)
        if isinstance(r, ProbeResult):
            round_tripped = json.loads(json.dumps(r.meta))
            assert round_tripped["generation_config"] == r.meta["generation_config"], name


def test_sampling_probes_have_isolated_and_realistic_variants():
    # finding 4: an isolating variant with top_p=1.0/top_k=0/repetition_penalty=1.0
    # (temperature is the only active parameter) alongside a realistic
    # variant using the model's shipped generation_config defaults, labeled
    # for what each is.
    pairs = [
        ("S9_sampling_t0_7_seed", "S9_sampling_t0_7_seed_isolated"),
        ("S9_sampling_t1_2_seed", "S9_sampling_t1_2_seed_isolated"),
    ]
    for realistic_name, isolated_name in pairs:
        realistic = run_probe(realistic_name)
        isolated = run_probe(isolated_name)
        if isinstance(realistic, ProbeResult) and isinstance(isolated, ProbeResult):
            assert realistic.meta["isolation"] == "model_shipped_defaults"
            assert isolated.meta["isolation"] == "temperature_only"
            assert isolated.meta["isolating_kwargs"] == {
                "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
            }


def test_logit_margin_uses_isolating_settings():
    r = run_probe("S9_logit_margin")
    if isinstance(r, ProbeResult):
        assert r.meta["isolation"] == "temperature_only"
        assert r.meta["isolating_kwargs"]["repetition_penalty"] == 1.0


def test_isolated_sampling_probes_skip_without_model(monkeypatch):
    monkeypatch.setattr(registry, "check_requirement", lambda r: r != "hf_model")
    for name in ("S9_sampling_t0_7_seed_isolated", "S9_sampling_t1_2_seed_isolated"):
        r = run_probe(name)
        assert isinstance(r, SkippedProbe) and "hf_model" in r.reason
