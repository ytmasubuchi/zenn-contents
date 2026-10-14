"""reprobench.probes.s0_hidden_inputs のテスト:
「インプットが全部同じ」という前提が静かに崩れるケースを検証する。
"""
import reprobench.probes  # モジュールimportの副作用でS0プローブ群を登録する
from reprobench.registry import ProbeResult, run_probe


# --- タスクブリーフ記載のテスト(そのまま) ---


def test_set_iteration_probe_records_hash_seed():
    r = run_probe("S0_set_iteration_order")
    assert "pythonhashseed" in r.meta


def test_glob_order_probe_is_sorted_flag_recorded():
    r = run_probe("S0_glob_order")
    assert "sorted_equals_raw" in r.meta  # OSによって一致したりしなかったりする


def test_parallel_completion_order_probe_returns_result():
    assert isinstance(run_probe("S0_parallel_completion_order"), ProbeResult)
