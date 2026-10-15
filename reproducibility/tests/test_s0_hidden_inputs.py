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


# --- 曖昧さの解決事項に対する追加テスト(fix round 1) ---


def test_glob_order_probe_records_in_process_stability():
    # 呼び出しのたびに新しいディレクトリを作ると、「環境間の不一致」と
    # 「このプローブがそもそも再現しない」を区別できなくなる(finding 7)。
    r = run_probe("S0_glob_order")
    assert "stable_within_process" in r.meta


def test_parallel_completion_order_probe_records_deterministic_job_shape():
    # 作業量はjob idだけで決まる決定的な値でなければならない(ランダムな
    # 遅延で完了順を人為的に揺らがせると、プローブがホストについて何も
    # 語らなくなる。finding 1)。
    r = run_probe("S0_parallel_completion_order")
    assert "work_sizes" in r.meta
    assert len(r.meta["work_sizes"]) == 32
    # 同じ入力から呼べば常に同じ作業量の列になる(非決定的なのは
    # 「完了順」であって「各ジョブに与える作業量」ではない)。
    r2 = run_probe("S0_parallel_completion_order")
    assert r.meta["work_sizes"] == r2.meta["work_sizes"]
