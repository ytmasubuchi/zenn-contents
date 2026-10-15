"""reprobench.registry のテスト: プローブ登録と実行契約を検証する。"""
import importlib.util
import sys
import types

import pytest

from reprobench import registry
from reprobench.registry import ProbeResult, SkippedProbe, all_probes, register, run_probe


@pytest.fixture(autouse=True)
def _isolated_registry():
    """テストが登録するダミープローブが他のテストへ漏れないようにする。

    `_REGISTRY` はモジュールグローバルな辞書であり、本番プローブ(Task4
    以降がモジュールimport時に登録するもの)もここに載る。このテスト
    ファイルは実行のたびに `@register("T_xxx")` で使い捨てのダミーを
    登録するため、前後でスナップショットを取り復元することで、実行順序
    やリラン(pytest-rerunfailures等)によって同名プローブの二重登録
    (`register` が送出するはずの `ValueError`)が誤って発火しないように
    する。
    """
    snapshot = dict(registry._REGISTRY)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(snapshot)


# --- タスクブリーフ記載のテスト(そのまま) ---


def test_register_adds_probe_to_registry():
    @register("T_dummy")
    def _p():
        return ProbeResult("T_dummy", "d", {}, None, {})

    assert "T_dummy" in all_probes()


def test_run_probe_skips_when_requirement_absent(monkeypatch):
    monkeypatch.setattr(registry, "check_requirement", lambda r: False)

    @register("T_needs_cuda", requires=("cuda",))
    def _p():
        raise AssertionError("must not run")

    r = run_probe("T_needs_cuda")
    assert isinstance(r, SkippedProbe) and "cuda" in r.reason


def test_run_probe_converts_exception_to_skip():
    @register("T_boom")
    def _p():
        raise RuntimeError("boom")

    r = run_probe("T_boom")
    assert isinstance(r, SkippedProbe) and "boom" in r.reason


def test_one_failing_probe_does_not_affect_others():
    @register("T_ok")
    def _ok():
        return ProbeResult("T_ok", "d", {}, None, {})

    @register("T_bad")
    def _bad():
        raise ImportError("no module named torch")

    assert isinstance(run_probe("T_bad"), SkippedProbe)
    assert isinstance(run_probe("T_ok"), ProbeResult)


# --- 曖昧さの解決事項に対する追加テスト ---


def test_run_probe_dispatches_through_module_global_check_requirement(monkeypatch):
    # 解決事項1: `run_probe` は `check_requirement` をモジュールグローバル
    # 経由で呼ばなければならない(ローカルimportやキャプチャされた参照では
    # なく)。このテストコンテナにはtorchが無いため、本物の
    # `check_requirement("torch")` は何もmonkeypatchしなくても`False`を
    # 返す — つまり「常にFalseを返すよう差し替える」テストだけでは、
    # `run_probe`がパッチ後の関数ではなく差し替え前の参照を握っていても
    # 同じ結果になり、検出できない(他のmonkeypatchテストが実際にそう)。
    # ここでは逆に`check_requirement`を常に`True`を返すよう差し替え、
    # 「本来ならスキップされるはずの要求」のプローブが実行されることを
    # 確認する。`run_probe`がモジュールグローバル経由で呼んでいなければ、
    # 差し替え前の(torch不在でFalseを返す)本物の関数が使われ続けて
    # プローブはスキップされ、このテストは失敗する。
    monkeypatch.setattr(registry, "check_requirement", lambda r: True)

    @register("T_patched_true", requires=("torch",))
    def _p():
        return ProbeResult("T_patched_true", "d", {}, None, {})

    assert isinstance(run_probe("T_patched_true"), ProbeResult)


def test_run_probe_distinguishes_missing_requirement_from_raised_reason(monkeypatch):
    # 解決事項2: 「設計通りのスキップ」と「バグによる例外」は理由文字列の
    # 接頭辞で区別できなければ、数ヶ月後に結果JSONを読む人が両者を
    # 見分けられない。
    monkeypatch.setattr(registry, "check_requirement", lambda r: False)

    @register("T_missing_req", requires=("torch",))
    def _missing():
        raise AssertionError("must not run")

    @register("T_raises")
    def _raises():
        raise RuntimeError("kaboom")

    missing = run_probe("T_missing_req")
    raised = run_probe("T_raises")
    assert missing.reason.startswith("missing requirement: ")
    assert raised.reason.startswith("raised: ")


def test_register_duplicate_name_raises_immediately():
    # 解決事項3: コピペ事故で同名プローブが後勝ちで無言に隠れるのを防ぐ。
    @register("T_dup")
    def _first():
        return ProbeResult("T_dup", "d", {}, None, {})

    with pytest.raises(ValueError):

        @register("T_dup")
        def _second():
            return ProbeResult("T_dup", "d", {}, None, {})


def test_all_probes_returns_entries_sorted_by_name():
    # 解決事項4: 実行順序がモジュールのimport順に依存してはいけない。
    @register("T_zzz_last")
    def _z():
        return ProbeResult("T_zzz_last", "d", {}, None, {})

    @register("T_aaa_first")
    def _a():
        return ProbeResult("T_aaa_first", "d", {}, None, {})

    names = list(all_probes())
    assert names == sorted(names)
    assert names.index("T_aaa_first") < names.index("T_zzz_last")


def test_register_unknown_requirement_raises_immediately():
    # 解決事項5: requiresのタイプミスは、常にスキップされ続ける
    # もっともらしい理由文字列になりがちなので、登録時に弾く。
    with pytest.raises(ValueError):

        @register("T_typo_requirement", requires=("cudaa",))
        def _p():
            return ProbeResult("T_typo_requirement", "d", {}, None, {})


def test_check_requirement_raises_for_unknown_requirement_string():
    with pytest.raises(ValueError):
        registry.check_requirement("numpy")


def test_check_requirement_true_when_module_importable(monkeypatch):
    # 解決事項6: torch/lightgbm/sklearn/transformers は importlib.util.find_spec
    # による存在確認のみで判定する(実際にimportしない)。
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert registry.check_requirement("lightgbm") is True
    assert registry.check_requirement("sklearn") is True
    assert registry.check_requirement("transformers") is True
    assert registry.check_requirement("torch") is True


def test_check_requirement_false_when_module_not_importable(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert registry.check_requirement("lightgbm") is False


def test_check_requirement_cuda_true_when_torch_and_cuda_available(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert registry.check_requirement("cuda") is True


def test_check_requirement_cuda_false_when_torch_missing(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert registry.check_requirement("cuda") is False


def test_check_requirement_cuda_false_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert registry.check_requirement("cuda") is False


def test_check_requirement_hf_model_true_when_cached(monkeypatch):
    fake_hub = types.SimpleNamespace(
        try_to_load_from_cache=lambda repo_id, filename: (
            "/fake/cache/path"
            if repo_id == registry.HF_MODEL_ID and filename == "config.json"
            else None
        )
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    assert registry.check_requirement("hf_model") is True


def test_check_requirement_hf_model_false_when_not_cached(monkeypatch):
    fake_hub = types.SimpleNamespace(try_to_load_from_cache=lambda repo_id, filename: None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    assert registry.check_requirement("hf_model") is False


def test_check_requirement_hf_model_false_when_huggingface_hub_missing():
    # このテストコンテナにはhuggingface_hubをインストールしていないため、
    # import は実際に(monkeypatchなしで)ModuleNotFoundErrorになる。
    assert registry.check_requirement("hf_model") is False


def test_check_requirement_returns_false_when_check_itself_raises(monkeypatch):
    def boom(name):
        raise RuntimeError("corrupted import machinery")

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    assert registry.check_requirement("torch") is False
