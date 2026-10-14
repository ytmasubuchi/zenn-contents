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
こと、(2) そのプローブが本質的に非決定(`_SELF_REPORTED_STABILITY`)な
ものを除き、同一プロセス内で2回呼んでもdigestが一致すること、を一括で
検証する。

fix round 2 finding D: `S0_glob_order` は呼び出しのたびに新しい
`TemporaryDirectory` を作るため、2回の`run_probe`呼び出しは**別々の
ディレクトリ**を列挙した結果であり、digestの一致は構造的に保証されない
(ext4/XFS/tmpfsでは経験的に一致しても、保証ではない)。このプローブ自身が
同一ディレクトリに対する2回の`glob.glob`呼び出しを内部で行い、その一致を
`meta["stable_within_process"]`として既に記録しているので、スモーク
テストではdigestの一致ではなく、このキーが存在することだけを確認する。

Task 7 fix(fix round 1で再測定・訂正): `S5_kmeans_centers`も同じ扱いに
加えた。スレッド数を明示的に固定しない既定状態では、プロセス内の2回の
呼び出しで`cluster_centers_`のdigestが一致しない。`OMP_NUM_THREADS`を
固定値(1でも4でも)に設定すれば消えるため、原因はKMeansが別の局所
最適解を選ぶことではなく(真のクラスタ構造を持つデータセットで再測定
した結果、収束反復数は常に同一だった)、スレッド数固定の有無による
縮約の浮動小数点加算順序の揺れである(詳細は
`reprobench/probes/s5_sklearn.py`のモジュールdocstring参照)。このプローブ
自身が内部で2回fitして一致を`meta["stable_within_process"]`に記録している。

Task 8 fix round 1 finding 0: `S8_index_add_atomics`も同じ扱いに加えた。
`atomicAdd`を使うCUDAの`index_add_`はスレッド実行順に加算順序が依存する
真に非決定的な演算であり(cuBLAS GEMMのみで構成される他のS8学習系
プローブとは異なり、座長の実機測定でも`use_deterministic_algorithms`
無しでは不安定だった)、このプローブ自身が内部で2回実行して
`meta["stable_within_process"]`に記録している
(`S8_index_add_atomics_deterministic`は対照実験で、安定することが
期待されるため`_SELF_REPORTED_STABILITY`には含めない)。
"""
import reprobench.probes  # モジュールimportの副作用で全プローブを登録する
from reprobench.registry import ProbeResult, SkippedProbe, all_probes, run_probe

# これらのプローブは「同一プロセス内の2回のrun_probe呼び出しでdigestが
# 一致すること」を要求できない: S0_parallel_completion_orderは完了順
# そのものが測定対象であり、S0_glob_orderは呼び出しのたびに別の
# TemporaryDirectoryを列挙するため2回のrun_probe呼び出しが別物を比べる
# ことになる。どちらもプローブ自身が内部で2回実行して
# `meta["stable_within_process"]`に安定性を記録しているので、ここでは
# それが存在することだけを確認する。
_SELF_REPORTED_STABILITY = {
    "S0_parallel_completion_order",
    "S0_glob_order",
    "S5_kmeans_centers",
    "S8_index_add_atomics",
}

# fix round 1 finding 5: レジストリの設計通りのスキップには2種類ある。
# (1) `requires=`で宣言した依存が無い("missing requirement: ")、
# (2) プローブ自身が判定する環境変数などの前提条件が無い
# ("missing env var: " -- 例: S8_mlp_train10_cuda_deterministicは
# CUBLAS_WORKSPACE_CONFIGがプロセス起動時に設定されていなければ、
# プロセス起動後に設定しても手遅れなのでここでスキップを返す)。
# どちらも「バグではない」設計通りのスキップであり、理由文字列の接頭辞で
# 区別できる(接頭辞が無ければ`raised: ...`であり、バグの疑いがある側)。
_EXPECTED_SKIP_PREFIXES = ("missing requirement: ", "missing env var: ")


def test_every_registered_probe_runs_or_skips_cleanly_and_is_stable():
    for name in all_probes():
        first = run_probe(name)
        if isinstance(first, SkippedProbe):
            assert first.reason.startswith(_EXPECTED_SKIP_PREFIXES), (
                f"{name}: skipped for an unexpected reason: {first.reason!r}"
            )
            continue
        assert isinstance(first, ProbeResult), f"{name}: {first!r}"

        if name in _SELF_REPORTED_STABILITY:
            assert "stable_within_process" in first.meta, (
                f"{name}: expected to self-report in-process stability"
            )
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
