"""プローブの登録と実行契約。

プローブとは、引数を取らず `ProbeResult` を返す関数のことである。
Task 4以降はこのモジュールに約30個のプローブを登録し、Task 10の
ランナーが `all_probes()` で列挙して `run_probe()` で1つずつ実行する。

このモジュールが保証する唯一の性質は「ある環境に何が欠けていても、
実行できるプローブは実行され、結果が記録される」ことである。alpineには
PyTorchがなく、CPUイメージにはCUDAがなく、一部のホストは言語モデルの
取得に失敗するかもしれない。それでも走れるプローブは走らねばならず、
1個のプローブの欠落・例外が他のプローブを巻き込んではならない。

この契約のためプローブ本体には1つの規律を課す: **重い依存の import は
プローブ関数の内部で行うこと**。モジュール先頭で `import torch` を
書くと、torch の無い alpine イメージでそのプローブモジュール自体が
import できなくなり、同じモジュール内の他のプローブまで道連れになる。

このモジュールの責務は「プローブの登録・列挙・実行(と欠落/例外の
吸収)」のみ。プローブそのもの(Task 4以降)や実行順序の決定・結果の
永続化(Task 10)はここでは扱わない。
"""
import importlib.util
import sys
from dataclasses import dataclass
from typing import Callable

#: 本プロジェクトが計測対象とする言語モデル。`"hf_model"` requirement は
#: このモデルがHuggingFaceキャッシュに存在するかどうかを見る。Task 9は
#: この定数を直接importして使うため、測定対象モデルの単一の正とする。
HF_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

#: `register`/`check_requirement` が受け付ける requirement 識別子。
#: 未知の識別子は(タイプミスが常にスキップという紛らわしい結果になる
#: のを避けるため)登録時・問い合わせ時のどちらでも即座に拒否する。
_SUPPORTED_REQUIREMENTS = ("torch", "cuda", "lightgbm", "sklearn", "transformers", "hf_model")


@dataclass(frozen=True)
class ProbeResult:
    probe: str
    digest: str
    scalars: dict[str, float]
    sample_b64: str | None
    meta: dict


@dataclass(frozen=True)
class SkippedProbe:
    probe: str
    reason: str


@dataclass(frozen=True)
class Probe:
    """登録済みプローブ1件(名前・本体関数・前提条件)。"""

    name: str
    func: Callable[[], ProbeResult]
    requires: tuple[str, ...] = ()


_REGISTRY: dict[str, Probe] = {}


def register(name: str, *, requires: tuple[str, ...] = ()) -> Callable:
    """プローブ関数を名前付きで登録するデコレータを返す。

    `requires` に未知の識別子が1つでも混ざっていれば、ここで即座に
    `ValueError` を送出する(デコレータを適用するまで待つと、タイプミスが
    実行時まで気づかれず、そのプローブが常にスキップされ続けてしまう)。

    同名のプローブが既に登録済みの場合も、デコレータ適用時点
    (= 登録のタイミング)で `ValueError` を送出する。Task 4〜9は約30個の
    プローブを複数モジュールにまたがって追加するため、コピペ事故で
    片方が無言で隠れる事態を防ぐ。
    """
    for req in requires:
        if req not in _SUPPORTED_REQUIREMENTS:
            raise ValueError(f"register: unknown requirement {req!r} for probe {name!r}")

    def decorator(fn: Callable[[], ProbeResult]) -> Callable[[], ProbeResult]:
        if name in _REGISTRY:
            raise ValueError(f"register: duplicate probe name {name!r}")
        _REGISTRY[name] = Probe(name=name, func=fn, requires=requires)
        return fn

    return decorator


def all_probes() -> dict[str, Probe]:
    """登録済みの全プローブを名前順に並べて返す。

    実行順序がモジュールのimport順(≒ファイルシステムの列挙順)に依存
    すると、別マシンでの再実行時にプローブの順序だけが変わってしまい、
    本来の計測対象ではない差分が結果JSONの比較に紛れ込む。
    """
    return {name: _REGISTRY[name] for name in sorted(_REGISTRY)}


def check_requirement(req: str) -> bool:
    """プローブの前提条件が現在の環境で満たされているかを調べる。

    `req` は `_SUPPORTED_REQUIREMENTS` のいずれかでなければならない
    (`register` が登録時に弾いているため、通常ここに未知の値が来るのは
    このモジュールを直接呼ぶ側に限られるが、ここでも同様に拒否する)。

    torch/lightgbm/sklearn/transformers は `importlib.util.find_spec` に
    よる存在確認のみで判定し、実際にはimportしない(importそのものが
    重い、あるいはイメージによってはimport自体が失敗の原因になりうる
    ため)。"cuda" はtorchがimport可能であることに加え
    `torch.cuda.is_available()` を見る。"hf_model" はHuggingFaceキャッシュ
    に `HF_MODEL_ID` の設定ファイルが存在するかを見る。

    判定処理そのものが例外を送出した場合(壊れたインストール等)は、
    envinfo.py の `_safe` と同じ流儀でFalseに丸めるが、原因を追えるよう
    requirement名と例外を先にstderrへ出力する。
    """
    if req not in _SUPPORTED_REQUIREMENTS:
        raise ValueError(f"check_requirement: unknown requirement {req!r}")

    try:
        if req == "cuda":
            if importlib.util.find_spec("torch") is None:
                return False
            import torch

            return bool(torch.cuda.is_available())
        if req == "hf_model":
            from huggingface_hub import try_to_load_from_cache

            return try_to_load_from_cache(HF_MODEL_ID, "config.json") is not None
        return importlib.util.find_spec(req) is not None
    except Exception as exc:
        print(f"registry: requirement {req!r} check failed: {exc!r}", file=sys.stderr)
        return False


def run_probe(name: str) -> ProbeResult | SkippedProbe:
    """名前でプローブを1件実行する。

    `requires` を満たさない場合は本体を呼ばずに
    `SkippedProbe(reason="missing requirement: <req>")` を返す。これは
    設計通りのスキップであり、調査上は無害である。

    本体が例外を送出した場合は `SkippedProbe(reason="raised: <repr>")`
    を返す。こちらはバグの可能性があるため、理由文字列の接頭辞で両者を
    区別できるようにしてある。どちらの場合も例外は外へ伝播しない —
    1個のプローブの欠落・バグが他のプローブの実行を道連れにしないため。
    """
    probe = _REGISTRY[name]
    for req in probe.requires:
        if not check_requirement(req):
            return SkippedProbe(probe=name, reason=f"missing requirement: {req}")
    try:
        return probe.func()
    except Exception as exc:
        return SkippedProbe(probe=name, reason=f"raised: {exc!r}")
