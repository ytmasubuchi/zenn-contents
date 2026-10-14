"""S8: PyTorch(CUDA)のMLP学習・TF32・curandのGPUモデル依存性。

重い依存(`torch`)はalpine等に無いことがあるため、関数本体の内部で
importする(モジュール先頭には置かない。パッケージdocstring参照)。

全S8プローブの`meta`には必ず次を記録する(`_device_meta`):
- `matmul_allow_tf32` / `cudnn_allow_tf32`: 実際のフラグ値。両者の既定値は
  非対称(matmul側はPyTorch 1.12以降False、cudnn側はTrueのまま)なので、
  「今回の実行で実際に何だったか」を毎回記録しないとTF32由来の差分を
  誤読する。
- `multi_processor_count`: GPUのSM数。`S8_curand_randn_sizes`の結果を
  解釈するのに必須(SM数がスレッド→要素割り当てを左右するため)。

TF32を切り替える2本(`S8_matmul_tf32_on`/`_off`)は`torch.backends.cuda.
matmul.allow_tf32`と`torch.backends.cudnn.allow_tf32`の両方を切り替える。
フラグはプロセスグローバルな可変状態であり、切り替えたまま放置すると
後続の他プローブ(同一プロセス内で`all_probes()`を回すランナーや
スモークテスト)が意図せず変化後の値を引き継いでしまう。そのため
`_tf32_flags`コンテキストマネージャで元の値を保存し、必ず`finally`で
復元する。

`S8_mlp_train10_cuda_deterministic`は`CUBLAS_WORKSPACE_CONFIG`がプロセス
起動時に設定されていなければ`SkippedProbe`を返す(cuBLASはプロセス起動
後にこの環境変数を読むため、プローブ内で`os.environ`へ書き込んでも
手遅れ)。

`S8_curand_randn_sizes`はこのバッチで最も重要なプローブである。PyTorchの
CUDA RNGは`curand_init`がPhiloxをスレッドindexでseedするため、スレッド→
要素の割り当てがGPUのSM数に依存し、要素数がある閾値を超えると**同じ
seed・同じPhiloxでもGPUモデルが違えば生成される乱数そのものが変わる**
(PyTorch issue #194676)。4サイズ(1024/65536/4194304/67108864)それぞれを
`torch.manual_seed(20261014)`で**独立に**(ストリームを継続させず)
reseedしてから生成し、サイズごとに別々のダイジェストを
`meta["digests"]`に記録する。2026-10-14時点でこの挙動を公式ドキュメントに
追記するPR(#196733)はapprove済み・未マージであり、本プローブの実測が
一次情報になる。
"""
import os
from contextlib import contextmanager

import numpy as np

from reprobench.digest import digest, encode_sample
from reprobench.registry import ProbeResult, SkippedProbe, register

_SEED = 20261014
_BATCH = 4096
_IN_FEATURES = 256
_HIDDEN1 = 128
_HIDDEN2 = 64
_OUT_FEATURES = 10
_STEPS = 10
_LR = 0.01
_CURAND_SIZES = (1024, 65536, 4194304, 67108864)
_CUBLAS_WORKSPACE_CONFIGS = (":4096:8", ":16:8")
_MATMUL_SIZE = 2048


def _device_meta(torch) -> dict:
    props = torch.cuda.get_device_properties(0)
    return {
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "multi_processor_count": int(props.multi_processor_count),
        "device_name": props.name,
    }


@contextmanager
def _tf32_flags(torch, matmul: bool, cudnn: bool):
    orig_matmul = torch.backends.cuda.matmul.allow_tf32
    orig_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = orig_matmul
        torch.backends.cudnn.allow_tf32 = orig_cudnn


def _build_mlp():
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(_IN_FEATURES, _HIDDEN1),
        nn.ReLU(),
        nn.Linear(_HIDDEN1, _HIDDEN2),
        nn.ReLU(),
        nn.Linear(_HIDDEN2, _OUT_FEATURES),
    )


def _train10_cuda(torch, dtype=None):
    """3層MLPをCUDA上で10ステップ学習し、学習後の全パラメータ・損失推移を
    返す。`dtype`を渡すとモデル・データをそのdtypeへキャストする
    (`S8_mlp_train10_cuda_bf16`用)。
    """
    import torch.nn as nn

    torch.manual_seed(_SEED)
    model = _build_mlp().to("cuda")
    x = torch.randn(_BATCH, _IN_FEATURES, device="cuda")
    target = torch.randn(_BATCH, _OUT_FEATURES, device="cuda")
    if dtype is not None:
        model = model.to(dtype)
        x = x.to(dtype)
        target = target.to(dtype)
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

    params = torch.cat([p.detach().flatten().float() for p in model.parameters()])
    arr = params.cpu().numpy().astype(np.float64)
    return arr, losses


def _train10_result(name: str, torch, dtype_name: str, dtype=None) -> ProbeResult:
    arr, losses = _train10_cuda(torch, dtype)
    scalars = {"first_loss": losses[0], "final_loss": losses[-1]}
    meta = {
        "seed": _SEED,
        "torch_version": torch.__version__,
        "batch": _BATCH,
        "steps": _STEPS,
        "lr": _LR,
        "n_params": len(arr),
        "losses": losses,
        "dtype": dtype_name,
        **_device_meta(torch),
    }
    for key, value in scalars.items():
        meta[f"{key}_hex"] = value.hex()
    return ProbeResult(
        probe=name,
        digest=digest(arr),
        scalars=scalars,
        sample_b64=encode_sample(arr),
        meta=meta,
    )


@register("S8_mlp_train10_cuda", requires=("torch", "cuda"))
def _mlp_train10_cuda() -> ProbeResult:
    import torch

    return _train10_result("S8_mlp_train10_cuda", torch, "float32")


@register("S8_mlp_train10_cuda_bf16", requires=("torch", "cuda"))
def _mlp_train10_cuda_bf16() -> ProbeResult:
    import torch

    return _train10_result(
        "S8_mlp_train10_cuda_bf16", torch, "bfloat16", dtype=torch.bfloat16
    )


@register("S8_mlp_train10_cuda_deterministic", requires=("torch", "cuda"))
def _mlp_train10_cuda_deterministic() -> ProbeResult | SkippedProbe:
    # CUBLAS_WORKSPACE_CONFIGはcuBLASがプロセス起動時に読む環境変数で
    # あり、ここで os.environ に書き込んでも手遅れ(モジュールdocstring
    # 参照)。未設定ならば`torch.use_deterministic_algorithms(True)`下での
    # CUDA matmulが例外を送出するため、それを「raised」として記録する
    # のではなく、ここで素直にスキップする。`import torch`より前に
    # チェックすることで、torch不在の環境でもこのチェック自体は検証できる。
    cublas_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas_cfg not in _CUBLAS_WORKSPACE_CONFIGS:
        return SkippedProbe(
            probe="S8_mlp_train10_cuda_deterministic",
            reason=(
                "missing env var: CUBLAS_WORKSPACE_CONFIG must be set to one of "
                f"{_CUBLAS_WORKSPACE_CONFIGS!r} at process start for "
                f"torch.use_deterministic_algorithms(True) to work on CUDA matmul; "
                f"got {cublas_cfg!r}"
            ),
        )

    import torch

    torch.use_deterministic_algorithms(True)
    try:
        arr, losses = _train10_cuda(torch)
    finally:
        torch.use_deterministic_algorithms(False)

    scalars = {"first_loss": losses[0], "final_loss": losses[-1]}
    meta = {
        "seed": _SEED,
        "torch_version": torch.__version__,
        "batch": _BATCH,
        "steps": _STEPS,
        "lr": _LR,
        "n_params": len(arr),
        "losses": losses,
        "dtype": "float32",
        "cublas_workspace_config": cublas_cfg,
        "deterministic_algorithms": True,
        **_device_meta(torch),
    }
    for key, value in scalars.items():
        meta[f"{key}_hex"] = value.hex()
    return ProbeResult(
        probe="S8_mlp_train10_cuda_deterministic",
        digest=digest(arr),
        scalars=scalars,
        sample_b64=encode_sample(arr),
        meta=meta,
    )


def _tf32_result(name: str, tf32: bool) -> ProbeResult:
    import torch

    with _tf32_flags(torch, tf32, tf32):
        torch.manual_seed(_SEED)
        a = torch.randn(_MATMUL_SIZE, _MATMUL_SIZE, device="cuda", dtype=torch.float32)
        b = torch.randn(_MATMUL_SIZE, _MATMUL_SIZE, device="cuda", dtype=torch.float32)
        c = a @ b
        arr = c.cpu().numpy().astype(np.float64)

        scalars = {"trace": float(np.trace(arr)), "fro_norm": float(np.linalg.norm(arr))}
        meta = {
            "seed": _SEED,
            "torch_version": torch.__version__,
            "shape": list(arr.shape),
            "tf32": tf32,
            **_device_meta(torch),
        }
        for key, value in scalars.items():
            meta[f"{key}_hex"] = value.hex()
        return ProbeResult(
            probe=name,
            digest=digest(arr),
            scalars=scalars,
            sample_b64=encode_sample(arr),
            meta=meta,
        )


@register("S8_matmul_tf32_on", requires=("torch", "cuda"))
def _matmul_tf32_on() -> ProbeResult:
    return _tf32_result("S8_matmul_tf32_on", True)


@register("S8_matmul_tf32_off", requires=("torch", "cuda"))
def _matmul_tf32_off() -> ProbeResult:
    return _tf32_result("S8_matmul_tf32_off", False)


@register("S8_curand_randn_sizes", requires=("torch", "cuda"))
def _curand_randn_sizes() -> ProbeResult:
    import torch

    digests: dict[int, str] = {}
    scalars: dict[str, float] = {}
    for n in _CURAND_SIZES:
        # 各サイズごとに独立してreseedする(ストリームを継続させない)。
        # 「同じseed・同じPhiloxで生成した乱数そのものがGPUモデルで
        # 変わるか」を見るプローブなので、サイズ間でRNG状態を共有すると
        # 問いがぼやける(モジュールdocstring参照)。
        torch.manual_seed(_SEED)
        x = torch.randn(n, device="cuda")
        arr = x.cpu().numpy().astype(np.float64)
        digests[n] = digest(arr)
        scalars[f"mean_{n}"] = float(arr.mean())

    meta = {
        "seed": _SEED,
        "torch_version": torch.__version__,
        "sizes": list(_CURAND_SIZES),
        "digests": digests,
        **_device_meta(torch),
    }
    for key, value in scalars.items():
        meta[f"{key}_hex"] = value.hex()
    return ProbeResult(
        probe="S8_curand_randn_sizes",
        digest=digest(list(digests.values())),
        scalars=scalars,
        sample_b64=None,
        meta=meta,
    )
