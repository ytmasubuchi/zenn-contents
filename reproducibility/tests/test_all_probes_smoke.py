"""登録済み全プローブに対するスモークテスト(fix round 1, finding 4)。

`test_s1_prng.py`/`test_numeric_probes.py`/`test_s0_hidden_inputs.py` は
各ブリーフに明記された代表的なプローブだけを呼ぶため、実装当時の22個中
8個(`S1_numpy_normal_randomstate`, `S1_numpy_normal_generator`,
`S1_generator_random_float32`, `S2_mean_float32`, `S4_log_float64`,
`S4_sin_float64`, `S0_dict_from_set`, `S0_hash_of_str`)はどのテストからも
一度も実行されていなかった。`run_probe` は例外を`SkippedProbe`に丸めて
しまうため、未実行のプローブが計測ホスト上で初めて壊れても、結果JSONから
そのプローブが静かに消えるだけで誰も気づけない。

ここでは `all_probes()` が返す全プローブについて、(1) クラッシュせず
`ProbeResult` を返すか、既知の`requires`未充足による`SkippedProbe`である
こと、(2) 完了順が本質的に非決定な`S0_parallel_completion_order`を除き、
同一プロセス内で2回呼んでもdigestが一致すること、を一括で検証する。
"""
import reprobench.probes  # モジュールimportの副作用で全プローブを登録する
from reprobench.registry import ProbeResult, SkippedProbe, all_probes, run_probe

# 完了順そのものが測定対象であるため、同一プロセス内の2回の呼び出しが
# 一致することを要求できない(brief/task-6-report参照)。
_NON_DETERMINISTIC = {"S0_parallel_completion_order"}


def test_every_registered_probe_runs_or_skips_cleanly_and_is_stable():
    for name in all_probes():
        first = run_probe(name)
        if isinstance(first, SkippedProbe):
            assert first.reason.startswith("missing requirement: "), (
                f"{name}: skipped for an unexpected reason: {first.reason!r}"
            )
            continue
        assert isinstance(first, ProbeResult), f"{name}: {first!r}"

        if name in _NON_DETERMINISTIC:
            continue

        second = run_probe(name)
        assert isinstance(second, ProbeResult), f"{name}: {second!r}"
        assert first.digest == second.digest, (
            f"{name}: digest differs across two in-process calls"
        )


def test_generator_random_float32_probe_uses_native_float32_dtype():
    # NEP 19のビット修正検出プローブの価値は、float32のまま(float64へ
    # 昇格する前に)digestを取っていることに丸ごと依存する。
    r = run_probe("S1_generator_random_float32")
    assert r.meta["dtype"] == "float32"
