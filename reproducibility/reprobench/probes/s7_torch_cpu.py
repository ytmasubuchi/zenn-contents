"""S7: PyTorch(CPU)の3層MLPの順伝播・10ステップ学習。

重い依存(`torch`)はalpine等に無いことがあるため、関数本体の内部で
importする(モジュール先頭には置かない。パッケージdocstring参照)。

モデルは`Linear(256,128)->ReLU->Linear(128,64)->ReLU->Linear(64,10)`の
3層MLP。重み初期化・入力生成はどちらも`torch.manual_seed(20261014)`が
シードするグローバルRNGを消費するため、プローブ関数の**先頭で毎回
reseedする**(resolution 3)。これにより、重み初期化→入力生成→(学習が
あれば)学習データ生成、という同じ消費順序が毎回再現され、同一プロセス
内の2回の呼び出しでbit単位の同じ結果になる。

`S7_mlp_forward`は推論のみ(勾配無し)、`S7_mlp_train10`はMSE損失・SGDで
10ステップ学習し、学習後の全パラメータを連結した1本のベクトルを
ダイジェスト化する(損失の推移も`meta["losses"]`に残す)。

**fix round 1 finding 7**: `meta["dtype"]`には**計算が実際に行われた
dtype**(`nn.Linear`/`torch.randn`の既定であるfloat32)を記録する。
以前は`digest`/`encode_sample`用にfloat64へ後から昇格した**numpy配列**の
dtypeをそのまま`str(arr.dtype)`していたため、常に`"float64"`になり、
実際の計算精度を正しく表していなかった(s4_libm.py等の既存プローブが
計算と同じdtypeの配列をそのままdigestに使うのとは事情が違う——ここでは
比較しやすさのためfloat64へ昇格してからdigestを取っているため、計算
dtypeとdigest対象dtypeが分かれる)。`S7_mlp_train10`には`dtype`が
そもそも記録されていなかったため、併せて追加した。
"""
import numpy as np

from reprobench.digest import digest, encode_sample
from reprobench.registry import ProbeResult, register

_SEED = 20261014
_BATCH = 4096
_IN_FEATURES = 256
_HIDDEN1 = 128
_HIDDEN2 = 64
_OUT_FEATURES = 10
_STEPS = 10
_LR = 0.01


def _build_mlp():
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(_IN_FEATURES, _HIDDEN1),
        nn.ReLU(),
        nn.Linear(_HIDDEN1, _HIDDEN2),
        nn.ReLU(),
        nn.Linear(_HIDDEN2, _OUT_FEATURES),
    )


def _common_meta(torch, extra: dict) -> dict:
    return {
        "seed": _SEED,
        "torch_version": torch.__version__,
        "threads": torch.get_num_threads(),
        "layer_sizes": [_IN_FEATURES, _HIDDEN1, _HIDDEN2, _OUT_FEATURES],
        **extra,
    }


@register("S7_mlp_forward", requires=("torch",))
def _mlp_forward() -> ProbeResult:
    import torch

    torch.manual_seed(_SEED)
    model = _build_mlp()
    x = torch.randn(_BATCH, _IN_FEATURES)
    with torch.no_grad():
        y = model(x)
    compute_dtype = str(y.dtype).removeprefix("torch.")  # fix round 1 finding 7
    arr = y.numpy().astype(np.float64)

    scalars = {"mean": float(arr.mean()), "std": float(arr.std())}
    meta = _common_meta(
        torch,
        {
            "shape": list(arr.shape),
            "dtype": compute_dtype,
            "digest_dtype": str(arr.dtype),
            "batch": _BATCH,
        },
    )
    for key, value in scalars.items():
        meta[f"{key}_hex"] = value.hex()
    return ProbeResult(
        probe="S7_mlp_forward",
        digest=digest(arr),
        scalars=scalars,
        sample_b64=encode_sample(arr),
        meta=meta,
    )


@register("S7_mlp_train10", requires=("torch",))
def _mlp_train10() -> ProbeResult:
    import torch
    import torch.nn as nn

    torch.manual_seed(_SEED)
    model = _build_mlp()
    x = torch.randn(_BATCH, _IN_FEATURES)
    target = torch.randn(_BATCH, _OUT_FEATURES)
    optimizer = torch.optim.SGD(model.parameters(), lr=_LR)
    loss_fn = nn.MSELoss()

    losses = []
    for _ in range(_STEPS):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    params = torch.cat([p.detach().flatten() for p in model.parameters()])
    compute_dtype = str(params.dtype).removeprefix("torch.")  # fix round 1 finding 7
    arr = params.numpy().astype(np.float64)

    scalars = {"first_loss": losses[0], "final_loss": losses[-1]}
    meta = _common_meta(
        torch,
        {
            "batch": _BATCH,
            "steps": _STEPS,
            "lr": _LR,
            "n_params": len(arr),
            "losses": losses,
            "dtype": compute_dtype,
            "digest_dtype": str(arr.dtype),
        },
    )
    for key, value in scalars.items():
        meta[f"{key}_hex"] = value.hex()
    return ProbeResult(
        probe="S7_mlp_train10",
        digest=digest(arr),
        scalars=scalars,
        sample_b64=encode_sample(arr),
        meta=meta,
    )
