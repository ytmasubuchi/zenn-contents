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
import random
import tempfile
import time
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
    with tempfile.TemporaryDirectory(prefix="reprobench_s0_glob_") as tmp:
        names = [f"file_{i:03d}.txt" for i in range(_N_FILES)]
        for name in names:
            open(os.path.join(tmp, name), "w").close()
        raw = [os.path.basename(p) for p in glob.glob(os.path.join(tmp, "*.txt"))]
        # with文を抜ける時点でTemporaryDirectoryが一時ファイル・ディレクトリ
        # 自体を削除するため、ここでの明示的な後始末は不要。
    sorted_raw = sorted(raw)
    return ProbeResult(
        probe="S0_glob_order",
        digest=digest(raw),
        scalars={},
        sample_b64=None,
        meta={
            "n": _N_FILES,
            "sorted_equals_raw": raw == sorted_raw,
            "order": raw,
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


def _job(i: int) -> int:
    """微小なランダム遅延のあと `i` を返す。

    遅延はOSエントロピーでseedしたローカルな乱数源(`random.Random()`、
    引数なし)を使う。他プローブが使うPRNGの状態や、本プロセス内の他の
    呼び出しからも独立させるためで、遅延を入れないとGIL/
    ThreadPoolExecutorの実装次第で完了順が提出順とほぼ常に一致してしまい、
    「完了順は揺らぐ」という本プローブの主張を実証できなくなる。
    """
    time.sleep(random.Random().random() * 0.001)
    return i


def _completion_order() -> list:
    with ThreadPoolExecutor(max_workers=_N_JOBS) as ex:
        futures = {ex.submit(_job, i): i for i in range(_N_JOBS)}
        return [futures[fut] for fut in as_completed(futures)]


@register("S0_parallel_completion_order")
def _parallel_completion_order() -> ProbeResult:
    # この値はランごとに変わるのが正しい挙動。同一プロセス内で2回実行して
    # 一致したかどうかを `stable_within_process` に記録する(G1-1aの材料)。
    first = _completion_order()
    second = _completion_order()
    return ProbeResult(
        probe="S0_parallel_completion_order",
        digest=digest(first),
        scalars={},
        sample_b64=None,
        meta={
            "n_jobs": _N_JOBS,
            "order": first,
            "stable_within_process": first == second,
        },
    )
