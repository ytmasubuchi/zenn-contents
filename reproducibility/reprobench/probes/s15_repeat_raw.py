"""S15: 反復実行の生データ記録(`S15_index_add_repeats` / `S15_kmeans_repeats`)。

記事は2箇所で「生データは`results/`には無く、プロジェクトの判断記録にのみ
残っている」と自認している。既存の`S8_index_add_atomics`/`S5_kmeans_centers`
は「同一プロセス内で2回実行して一致したか」という集約値
(`meta["stable_within_process"]`)しか記録しておらず、各回の生digestそのもの
は結果JSONに残らない。本モジュールは同じ計算を複数回(`_N_ROUNDS`回)繰り
返し、各回のdigestを`meta["round_digests"]`に**順序付きの列**として記録
することで、この穴を塞ぐ。

重い依存(`torch`/`sklearn`)はalpine等に無いことがあるため、関数本体の
内部でimportする(モジュール先頭には置かない。パッケージdocstring参照)。

## `S15_index_add_repeats`

`s8_torch_gpu.S8_index_add_atomics`と**同じ計算**(同じseed・同じ
`_index_add_once`)を使う。既存のS8プローブ自体は変更せず、その非公開
ヘルパー(`_index_add_once`と関連定数)をそのままimportして再利用する
(`s10_batch_invariance.py`がS9の`_PROMPT`/`first_divergence`を同じ流儀で
importしているのと同じパターン)。計算式を独自に書き写すと、将来どちらか
だけが変更されて気づかれずに乖離するおそれがあるため、「別の機会に測った
同じ現象」であることをコードレベルで保証する。

CUDAの`index_add_`は`atomicAdd`で実装されており、加算順序がスレッド実行順
依存であるため既定では非決定的である(`s8_torch_gpu`のモジュールdocstring
参照)。したがって`round_digests`が5回とも一致しないことは**期待された
挙動**であり、異常ではない——このモジュールのテストは個々のdigestが
一致するか否かを断定しない。

このプローブ自身が自己申告する不安定さであるため、`S8_index_add_atomics`
と同じ扱いとして、`tests/test_all_probes_smoke.py`の
`_SELF_REPORTED_STABILITY`と`reprobench/compare.py`の`_POLICY`
(`kind="self_reported_stability"`)の両方に登録する。片方だけに登録すると、
同一プロセス内の2回の`run_probe`呼び出し(スモークテスト)でdigestが食い違い
テストが落ちる、あるいは環境間比較で本来報告すべきでない「不一致」が記事に
混入する、のどちらかが起こる(過去タスクで実際に問題になった失敗モード)。

## `S15_kmeans_repeats`

`s5_sklearn.S5_kmeans_centers`と**同じデータ**(`_kmeans_dataset`)・
**同じパラメータ**(`n_clusters=4`、`n_init=10`、`random_state`は
`s5_sklearn._SEED`)を使う。こちらも非公開ヘルパー(`_kmeans_dataset`/
`_canonical_centers`/`_effective_thread_count`/`_thread_pool_info`)を
そのままimportして再利用し、同じ理由で計算の乖離を防ぐ。

`S5_kmeans_centers`のモジュールdocstringが実測した通り、このKMeansフィット
は有効スレッド数(`meta["effective_thread_count"]`)が1・2・4のときは安定、
8(測定時のコンテナのnproc)のときは不安定という相関がある。
`S15_kmeans_repeats`は同じデータ・同じパラメータを使うため、同じ原因
(環境のスレッド数)で同じように不安定になりうる。そのため同一プロセス内の
2回の`run_probe`呼び出し比較(スモークテスト)については`S5_kmeans_centers`
と同じ理由で`_SELF_REPORTED_STABILITY`に登録する。一方、環境横断の比較
(`reprobench/compare.py`)については`S5_kmeans_centers`自身も
`kind="self_reported_stability"`ではなく`kind="grouped"`
(`group_by_meta="effective_thread_count"`)を使っている——「自己申告だから
不一致を無条件に握り潰す」のではなく、「スレッド数が同じ環境同士でだけ
比較する」ことで、1・2・4スレッド環境間の本物の一致/不一致を引き続き
報告できるようにするためである。`S15_kmeans_repeats`も同じ`meta`
フィールドを記録するため、同じ`kind="grouped"`を適用する。

いずれのプローブも、安定/不安定の判定そのものを握り潰さない——
集約した真偽値だけでなく、各回のdigestをそのまま列として残すことが本
プローブの目的である。
"""
import os

import numpy as np

from reprobench.digest import digest, encode_sample
from reprobench.probes.s5_sklearn import (
    _SEED as _S5_SEED,
    _canonical_centers,
    _effective_thread_count,
    _kmeans_dataset,
    _thread_pool_info,
)
from reprobench.probes.s8_torch_gpu import (
    _ATOMIC_BINS,
    _ATOMIC_N,
    _SEED as _S8_SEED,
    _device_meta,
    _index_add_once,
)
from reprobench.registry import ProbeResult, register

#: 両プローブが繰り返す回数。brief Step3が指定する「5回実行」をそのまま
#: 両プローブに適用する(S15_kmeans_repeatsの回数は brief では明示されて
#: いないが、同じ定数を使うことで「n_rounds>=5」という要件を両方満たす)。
_N_ROUNDS = 5


def round_digests_and_stability(arrays: list[np.ndarray]) -> tuple[list[str], bool]:
    """配列の列から各回のdigestの列と、全回が一致したかの真偽値を返す純粋関数。

    `digest`はnumpy専用の正準バイト表現を取るだけでtorch/sklearnに依存
    しないため、GPU・sklearn無しの環境でもそのまま単体テストできる。
    """
    digests = [digest(a) for a in arrays]
    return digests, len(set(digests)) <= 1


def max_abs_diffs_from_first_round(arrays: list[np.ndarray]) -> list[float]:
    """各回の配列を1回目と比較した要素ごとの最大絶対差の列を返す純粋関数。

    先頭の要素(1回目と1回目自身の差)は常に`0.0`になる。
    """
    first = arrays[0]
    return [float(np.max(np.abs(a - first))) for a in arrays]


@register("S15_index_add_repeats", requires=("torch", "cuda"))
def _index_add_repeats() -> ProbeResult:
    """重複indexへの`index_add_`(`S8_index_add_atomics`と同じ計算)を
    `_N_ROUNDS`回繰り返し、各回のdigestと、1回目を基準とした最大絶対差を
    記録する。
    """
    import torch

    arrays: list[np.ndarray] = []
    for _ in range(_N_ROUNDS):
        cpu_gen = torch.Generator(device="cpu").manual_seed(_S8_SEED)
        arr, _input_digests = _index_add_once(torch, cpu_gen)
        arrays.append(arr)

    round_digests, stable = round_digests_and_stability(arrays)
    max_abs_diffs = max_abs_diffs_from_first_round(arrays)

    scalars = {"max_abs_diff_from_first_round": max(max_abs_diffs)}
    meta = {
        "seed": _S8_SEED,
        "torch_version": torch.__version__,
        "n": _ATOMIC_N,
        "bins": _ATOMIC_BINS,
        "n_rounds": _N_ROUNDS,
        "round_digests": round_digests,
        "max_abs_diff_from_first_round_per_round": max_abs_diffs,
        "stable_within_process": stable,
        **_device_meta(torch),
    }
    for key, value in scalars.items():
        meta[f"{key}_hex"] = value.hex()
    return ProbeResult(
        probe="S15_index_add_repeats",
        digest=digest(round_digests),
        scalars=scalars,
        sample_b64=encode_sample(arrays[0]),
        meta=meta,
    )


@register("S15_kmeans_repeats", requires=("sklearn",))
def _kmeans_repeats() -> ProbeResult:
    """KMeansのフィット(`S5_kmeans_centers`と同じデータ・同じパラメータ)を
    `_N_ROUNDS`回繰り返し、各回の(正準化した)クラスタ中心のdigestを記録する。
    """
    from sklearn.cluster import KMeans

    n_clusters = 4
    n_init = 10
    X = _kmeans_dataset()

    centers_list: list[np.ndarray] = []
    inertias: list[float] = []
    for _ in range(_N_ROUNDS):
        model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=_S5_SEED)
        model.fit(X)
        centers_list.append(_canonical_centers(model.cluster_centers_.astype(np.float64)))
        inertias.append(float(model.inertia_))

    round_digests, stable = round_digests_and_stability(centers_list)

    scalars = {"inertia_first_round": inertias[0]}
    meta = {
        "seed": _S5_SEED,
        "n_clusters": n_clusters,
        "n_init": n_init,
        "n_rounds": _N_ROUNDS,
        "round_digests": round_digests,
        "inertia_per_round": inertias,
        "stable_within_process": stable,
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "effective_thread_count": _effective_thread_count(),
        "thread_pool_info": _thread_pool_info(),
        "centers_canonicalized": "lexsort_rows",
    }
    for key, value in scalars.items():
        meta[f"{key}_hex"] = value.hex()
    return ProbeResult(
        probe="S15_kmeans_repeats",
        digest=digest(round_digests),
        scalars=scalars,
        sample_b64=encode_sample(centers_list[0]),
        meta=meta,
    )
