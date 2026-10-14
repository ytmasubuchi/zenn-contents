"""S0: 「インプットが全部同じ」という前提が静かに崩れるケース。

比較対象の2環境に「全く同じコード・同じ入力」を与えたつもりでも、言語・OS・
ランタイムが暗黙に持つ非決定性のせいで観測値が変わることがある:
`set`の反復順(ハッシュランダム化)、ファイルシステムの列挙順(`glob`)、
並列ワーカーの完了順。これらはPRNGの非決定性とは別系統の「隠れた入力の
ドリフト」であり、比較器が環境間の不一致を報告したとき、まずこちらを
疑えるよう独立に記録しておく。
"""
import glob
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from reprobench.digest import digest
from reprobench.registry import ProbeResult, register

_STRINGS = [f"key_{i:02d}" for i in range(20)]
_N_FILES = 100
_N_JOBS = 32


@register("S0_set_iteration_order")
def _set_iteration_order() -> ProbeResult:
    order = list(set(_STRINGS))
    return ProbeResult(
        probe="S0_set_iteration_order",
        digest=digest(order),
        scalars={},
        sample_b64=None,
        meta={
            "n": len(_STRINGS),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "order": order,
        },
    )


@register("S0_dict_from_set")
def _dict_from_set() -> ProbeResult:
    # 辞書は挿入順を保持する(Python 3.7+)ため、setを反復して作った辞書の
    # キー順はそのsetの反復順をそのまま引き継ぐ。
    d = {key: i for i, key in enumerate(set(_STRINGS))}
    order = list(d.keys())
    return ProbeResult(
        probe="S0_dict_from_set",
        digest=digest(order),
        scalars={},
        sample_b64=None,
        meta={
            "n": len(_STRINGS),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "order": order,
        },
    )


@register("S0_glob_order")
def _glob_order() -> ProbeResult:
    # 同一ディレクトリに対して同一プロセス内で2回 `glob.glob` を呼び、
    # 列挙順が安定しているかを `stable_within_process` に記録する。
    # 呼び出しのたびに新しいTemporaryDirectoryを作ると、2つの異なる
    # ディレクトリを比べることになり、「環境間の不一致」と「このプローブが
    # そもそも再現しない」を区別できなくなる(並列プローブと同じ理由)。
    with tempfile.TemporaryDirectory(prefix="reprobench_s0_glob_") as tmp:
        names = [f"file_{i:03d}.txt" for i in range(_N_FILES)]
        for name in names:
            open(os.path.join(tmp, name), "w").close()
        pattern = os.path.join(tmp, "*.txt")
        first = [os.path.basename(p) for p in glob.glob(pattern)]
        second = [os.path.basename(p) for p in glob.glob(pattern)]
        # with文を抜ける時点でTemporaryDirectoryが一時ファイル・ディレクトリ
        # 自体を削除するため、ここでの明示的な後始末は不要。
    sorted_raw = sorted(first)
    return ProbeResult(
        probe="S0_glob_order",
        digest=digest(first),
        scalars={},
        sample_b64=None,
        meta={
            "n": _N_FILES,
            "sorted_equals_raw": first == sorted_raw,
            "order": first,
            "stable_within_process": first == second,
        },
    )


@register("S0_hash_of_str")
def _hash_of_str() -> ProbeResult:
    value = hash("reproducibility")
    return ProbeResult(
        probe="S0_hash_of_str",
        digest=digest(value),
        scalars={},
        sample_b64=None,
        meta={"pythonhashseed": os.environ.get("PYTHONHASHSEED"), "value": value},
    )


_WORK_BASE = 50_000


def _work_size(i: int) -> int:
    """ジョブ `i` に与える作業量(ビジーループの反復回数)。

    idに対して単調な作業量(例: `i`に比例)だと、完了順はほぼ常に
    「作業量が少ない順」= 「id順」に落ち着いてしまい、スケジューラ由来の
    揺らぎが見えなくなる。ここでは係数を `(i*7+5) % 12 + 1` という非単調な
    式で決め、複数のジョブが同程度の作業量を持つようにして、それらの間の
    完了順が実際のスレッドスケジューリングに委ねられるようにする。
    """
    coefficient = (i * 7 + 5) % 12 + 1
    return _WORK_BASE * coefficient


def _job(i: int) -> int:
    """`_work_size(i)` 回分の実際のCPU作業を行ってから `i` を返す。

    作業量はidだけで決まる決定的な値であり、乱数由来の遅延は一切使わない
    (乱数で遅延を散らすと、どの2つの実行・どの2台のホストを比べても
    digestが**構造的に**一致しなくなり、プローブがホストについて何も
    語らなくなる)。完了順が揺らぐとすれば、それは本物のOSスケジューラ・
    GIL切り替えタイミングに由来するものであり、揺らがずに毎回同じ順序に
    なるなら、それもまた正直な測定結果である。
    """
    total = 0
    for _ in range(_work_size(i)):
        total += 1
    return i


def _completion_order() -> list:
    with ThreadPoolExecutor(max_workers=_N_JOBS) as ex:
        futures = {ex.submit(_job, i): i for i in range(_N_JOBS)}
        return [futures[fut] for fut in as_completed(futures)]


@register("S0_parallel_completion_order")
def _parallel_completion_order() -> ProbeResult:
    # この値はランごとに変わりうるのが正しい挙動(揺らがず安定するなら、
    # それも正直な測定結果)。同一プロセス内で2回実行して一致したかどうかを
    # `stable_within_process` に記録する(G1-1aの材料)。
    first = _completion_order()
    second = _completion_order()
    return ProbeResult(
        probe="S0_parallel_completion_order",
        digest=digest(first),
        scalars={},
        sample_b64=None,
        meta={
            "n_jobs": _N_JOBS,
            "work_sizes": [_work_size(i) for i in range(_N_JOBS)],
            "order": first,
            "stable_within_process": first == second,
        },
    )
