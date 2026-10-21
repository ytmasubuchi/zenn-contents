"""S13: `torch.use_deterministic_algorithms(True)`下でのCUBLAS workspace依存。

重い依存(`torch`)はalpine等に無いことがあるため、関数本体の内部で
importする(モジュール先頭には置かない。パッケージdocstring参照)。

## 題材

PyTorch公式ドキュメント(`torch.use_deterministic_algorithms`)は、
決定的アルゴリズムを有効にした状態でCUDA上の一部のcuBLAS演算(行列積を
含む)を使うには、環境変数`CUBLAS_WORKSPACE_CONFIG`を`:4096:8`または
`:16:8`にプロセス起動時点で設定しておく必要があり、未設定だと
`RuntimeError`が送出されると述べている。本プローブは`CUBLAS_OPS`に
挙げた複数の行列積系演算それぞれについて、実際に例外が出るかどうかを
演算ごとに実測する。

**`CUBLAS_WORKSPACE_CONFIG`はこのプローブ自身が設定しない。** cuBLASは
プロセス起動時にこの環境変数を読むため、プローブ内で`os.environ`へ
書き込んでも手遅れである(S8の`S8_mlp_train10_cuda_deterministic`と同じ
事情)。設定するかどうかはconfig側(本番環境ではIMAGE=gpuのpractice
configが意図的に未設定のままにする)が決める。本プローブは
`os.environ.get("CUBLAS_WORKSPACE_CONFIG")`で現在値を読んで`meta`に
記録するだけである。

## 例外が出ることは期待挙動であり、握りつぶさない

**演算ごとに`try`/`except`で囲み、例外が出たらその例外クラス名と
メッセージを`raised`に記録し、出なければ`digest`を記録する。** 例外を
吸収して正常終了したように見せてはならない(Review Focus 3)。また、
**「どの演算が実際に例外を送出するか」は事前登録された予測の対象であり、
このモジュール自身が決め打ちしてはならない**——`CUBLAS_OPS`はどの演算を
測るかの列挙に過ぎず、結果は`tests/test_s13_cublas_workspace.py`も
`S13_cublas_workspace_config`の`meta["ops"]`も決め打ちしない。
"""
import os

import numpy as np

from reprobench.digest import digest
from reprobench.registry import ProbeResult, register

_SEED = 20261014
_MATMUL_SIZE = 512
_BATCH = 32
_IN_FEATURES = 128
_OUT_FEATURES = 64

#: 決定的アルゴリズム下でCUBLAS workspaceを要求しうる、行列積を含む複数の
#: 演算の名前。どれが実際に例外を送出するかはこのモジュールの時点では
#: 未定(モジュールdocstring参照)。
CUBLAS_OPS: tuple[str, ...] = ("mm", "bmm", "addmm", "linear_forward_backward")


def record_outcome(value_digest: str | None, exc: Exception | None) -> dict:
    """演算1回の実行結果を`meta["ops"][op]`のセル辞書に変換する純粋関数。

    `exc`が与えられれば(例外が送出された場合)、その例外オブジェクトを
    握りつぶさず`raised`にクラス名とメッセージをそのまま記録する。`exc`が
    `None`であれば(例外が出なかった場合)、`value_digest`を`digest`として
    記録する。どちらか一方だけを渡す契約で、両方同時に埋めることはない。
    torch等への依存が無いため、組み込みの`Exception`サブクラスを使って
    GPU無し環境でもそのまま単体テストできる。
    """
    if exc is not None:
        return {"raised": {"type": type(exc).__name__, "message": str(exc)}}
    return {"digest": value_digest}


def _run_mm(torch, cpu_gen) -> np.ndarray:
    a = torch.randn(_MATMUL_SIZE, _MATMUL_SIZE, generator=cpu_gen).to("cuda")
    b = torch.randn(_MATMUL_SIZE, _MATMUL_SIZE, generator=cpu_gen).to("cuda")
    c = torch.mm(a, b)
    return c.cpu().numpy().astype(np.float64)


def _run_bmm(torch, cpu_gen) -> np.ndarray:
    side = _MATMUL_SIZE // 4
    a = torch.randn(_BATCH, side, side, generator=cpu_gen).to("cuda")
    b = torch.randn(_BATCH, side, side, generator=cpu_gen).to("cuda")
    c = torch.bmm(a, b)
    return c.cpu().numpy().astype(np.float64)


def _run_addmm(torch, cpu_gen) -> np.ndarray:
    bias = torch.randn(_MATMUL_SIZE, generator=cpu_gen).to("cuda")
    a = torch.randn(_MATMUL_SIZE, _MATMUL_SIZE, generator=cpu_gen).to("cuda")
    b = torch.randn(_MATMUL_SIZE, _MATMUL_SIZE, generator=cpu_gen).to("cuda")
    c = torch.addmm(bias, a, b)
    return c.cpu().numpy().astype(np.float64)


def _run_linear_forward_backward(torch, cpu_gen) -> np.ndarray:
    import torch.nn as nn

    x = torch.randn(_BATCH, _IN_FEATURES, generator=cpu_gen).to("cuda")
    linear = nn.Linear(_IN_FEATURES, _OUT_FEATURES).to("cuda")
    out = linear(x)
    loss = out.sum()
    loss.backward()
    return linear.weight.grad.detach().cpu().numpy().astype(np.float64)


_OP_FUNCS = {
    "mm": _run_mm,
    "bmm": _run_bmm,
    "addmm": _run_addmm,
    "linear_forward_backward": _run_linear_forward_backward,
}


@register("S13_cublas_workspace_config", requires=("torch", "cuda"))
def _cublas_workspace_config() -> ProbeResult:
    import torch

    cublas_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    orig_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        ops: dict[str, dict] = {}
        for op_name in CUBLAS_OPS:
            cpu_gen = torch.Generator(device="cpu").manual_seed(_SEED)
            try:
                arr = _OP_FUNCS[op_name](torch, cpu_gen)
            except Exception as exc:  # noqa: BLE001 - 例外そのものが測定対象(docstring参照)
                ops[op_name] = record_outcome(None, exc)
            else:
                ops[op_name] = record_outcome(digest(arr), None)
    finally:
        torch.use_deterministic_algorithms(orig_deterministic)

    meta = {
        "seed": _SEED,
        "torch_version": torch.__version__,
        "cublas_workspace_config": cublas_cfg,
        "op_names": list(CUBLAS_OPS),
        "ops": ops,
    }
    return ProbeResult(
        probe="S13_cublas_workspace_config",
        digest=digest(ops),
        scalars={},
        sample_b64=None,
        meta=meta,
    )
