"""S8: PyTorch(CUDA)のMLP学習・TF32・atomic演算・curandのGPUモデル依存性。

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
復元する。`torch.use_deterministic_algorithms`も同じプロセスグローバルな
可変状態であるため、`_deterministic_algorithms`コンテキストマネージャで
同様に元の値(`torch.are_deterministic_algorithms_enabled()`)を保存して
復元する(fix round 1 finding 6: 旧版は`finally`で無条件に`False`へ戻して
おり、元が`True`だった場合に後続プローブへ意図せず伝播していた)。

`S8_mlp_train10_cuda_deterministic`は`CUBLAS_WORKSPACE_CONFIG`がプロセス
起動時に設定されていなければ`SkippedProbe`を返す(cuBLASはプロセス起動
後にこの環境変数を読むため、プローブ内で`os.environ`へ書き込んでも
手遅れ)。これは`"missing requirement: "`とは別の、プローブ自身が判定する
「環境変数の前提条件」によるスキップであり、理由文字列は
`"missing env var: "`で始める。`tests/test_all_probes_smoke.py`はこの
接頭辞も設計通りのスキップとして認識する(fix round 1 finding 5)。

**fix round 1 finding 3(重要)**: 行列積・MLP学習系プローブの入力は、
本バッチが測定したいGPU依存curandの影響そのものを受けないよう、**CPU側の
ジェネレータで生成してから`.to("cuda")`で転送する**(`_cpu_randn`)。
以前は`torch.randn(..., device="cuda")`で直接CUDA上に生成しており、
2048×2048=4,194,304要素は`S8_curand_randn_sizes`が測る4サイズのうちの
1つと厳密に一致していた——つまり「2つのGPUでTF32 on/offの行列積が違う」
という観測が、行列積アルゴリズムの差ではなく入力そのものの違い(curandの
GPUモデル依存)によるものである可能性を排除できていなかった。入力を
CPU生成に切り替え、`digest(a)`/`digest(b)`(行列積系)・`digest(x)`/
`digest(target)`(学習系)を`meta`に記録することで、「入力が一致していた
かどうか」をソースを読まずとも記録から直接確認できるようにする。

`S8_curand_randn_sizes`はこのバッチで最も重要なプローブである。PyTorchの
CUDA RNGは`curand_init`がPhiloxをスレッドindexでseedするため、スレッド→
要素の割り当てがGPUのSM数に依存し、要素数がある閾値を超えると**同じ
seed・同じPhiloxでもGPUモデルが違えば生成される乱数そのものが変わる**
(PyTorch issue #194676)。4サイズ(1024/65536/4194304/67108864)それぞれを
`torch.manual_seed(20261014)`で**独立に**(ストリームを継続させず)
reseedしてから生成し、サイズごとに別々のダイジェストを
`meta["digests"]`に記録する(キーは文字列——fix round 1 finding 9:
JSONはdictのキーを文字列化するため、int キーのままでは比較器が読む
`results.json`との整合性が無い)。2026-10-14時点でこの挙動を公式
ドキュメントに追記するPR(#196733)はapprove済み・未マージであり、
本プローブの実測が一次情報になる。

**fix round 1 finding 0(重要、座長の実測で追加)**: `S8_mlp_train10_cuda`
系は`Linear`+`ReLU`+`MSELoss`のみで構成されており、これはCUDA上で
決定的な演算(cuBLASのGEMMはシングルストリーム・固定workspace下では
run-to-runで決定的)である。座長がRTX 4090実機で測定した結果、
`S8_mlp_train10_cuda`と`S8_mlp_train10_cuda_deterministic`は**同一
digest**になった——つまりこのMLPには「決定的アルゴリズムフラグが
修正すべき非決定性」がそもそも存在しない。GPUの非決定性は
`atomicAdd`を使う演算(`index_add_`/`scatter_add_`/embedding backward等)
に由来するため、`S8_index_add_atomics`(既定、重複indexへの
`index_add_`)と`S8_index_add_atomics_deterministic`
(`use_deterministic_algorithms(True)`下)を追加する。前者は
プロセス内で安定しないことが期待される自己申告プローブ
(`_SELF_REPORTED_STABILITY`入り)、後者は決定的アルゴリズムが実際に
その非決定性を消すことを示す対照実験である。
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
_ATOMIC_N = 1_000_000
_ATOMIC_BINS = 64


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


@contextmanager
def _deterministic_algorithms(torch, enabled: bool):
    # fix round 1 finding 6: 元の値を保存して復元する。以前は`finally`で
    # 無条件に`False`に戻しており、プロセス起動時に`True`だった場合に
    # 後続の他プローブへ意図せず伝播していた。
    orig = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(enabled)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(orig)


def _cpu_randn(*shape, generator, dtype=None) -> tuple:
    """CPU側のジェネレータで乱数を生成してからCUDAへ転送する。

    `torch.randn(..., device="cuda")`で直接生成すると、本バッチが測定
    したいcurandのGPUモデル依存性そのものが行列積・学習系プローブの
    入力に混入してしまう(fix round 1 finding 3、モジュールdocstring
    参照)。CPU生成の値をそのまま返すので、呼び出し側でCPU側配列の
    digestを記録できる。
    """
    import torch

    t_cpu = torch.randn(*shape, generator=generator)
    if dtype is not None:
        t_cpu = t_cpu.to(dtype)
    return t_cpu, t_cpu.to("cuda")


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
    """3層MLPをCUDA上で10ステップ学習し、学習後の全パラメータ・損失推移・
    入力(x/target、CPU生成)のdigestを返す。`dtype`を渡すとモデル・データ
    をそのdtypeへキャストする(`S8_mlp_train10_cuda_bf16`用)。
    """
    import torch.nn as nn

    torch.manual_seed(_SEED)  # モデルの重み初期化(CPU側のデフォルトRNG)を再現可能にする
    model = _build_mlp().to("cuda")

    cpu_gen = torch.Generator(device="cpu").manual_seed(_SEED)
    x_cpu, x = _cpu_randn(_BATCH, _IN_FEATURES, generator=cpu_gen)
    target_cpu, target = _cpu_randn(_BATCH, _OUT_FEATURES, generator=cpu_gen)
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
    input_digests = {
        "x": digest(x_cpu.numpy().astype(np.float64)),
        "target": digest(target_cpu.numpy().astype(np.float64)),
    }
    return arr, losses, input_digests


def _train10_result(name: str, torch, dtype_name: str, dtype=None) -> ProbeResult:
    arr, losses, input_digests = _train10_cuda(torch, dtype)
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
        "input_digests": input_digests,
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

    with _deterministic_algorithms(torch, True):
        arr, losses, input_digests = _train10_cuda(torch)

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
        "input_digests": input_digests,
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
        cpu_gen = torch.Generator(device="cpu").manual_seed(_SEED)
        a_cpu, a = _cpu_randn(_MATMUL_SIZE, _MATMUL_SIZE, generator=cpu_gen, dtype=torch.float32)
        b_cpu, b = _cpu_randn(_MATMUL_SIZE, _MATMUL_SIZE, generator=cpu_gen, dtype=torch.float32)
        c = a @ b
        arr = c.cpu().numpy().astype(np.float64)

        scalars = {"trace": float(np.trace(arr)), "fro_norm": float(np.linalg.norm(arr))}
        meta = {
            "seed": _SEED,
            "torch_version": torch.__version__,
            "shape": list(arr.shape),
            "tf32": tf32,
            "input_a_digest": digest(a_cpu.numpy().astype(np.float64)),
            "input_b_digest": digest(b_cpu.numpy().astype(np.float64)),
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

    digests: dict[str, str] = {}
    scalars: dict[str, float] = {}
    for n in _CURAND_SIZES:
        # 各サイズごとに独立してreseedする(ストリームを継続させない)。
        # 「同じseed・同じPhiloxで生成した乱数そのものがGPUモデルで
        # 変わるか」を見るプローブなので、サイズ間でRNG状態を共有すると
        # 問いがぼやける(モジュールdocstring参照)。このプローブ**だけ**は
        # 意図的に`device="cuda"`で直接生成する——curandのGPUモデル依存性
        # そのものが測定対象であるため。
        torch.manual_seed(_SEED)
        x = torch.randn(n, device="cuda")
        arr = x.cpu().numpy().astype(np.float64)
        # fix round 1 finding 9: JSONはdictのキーを文字列化するため、
        # intキーのままではresults.jsonとin-processの値が食い違う。
        digests[str(n)] = digest(arr)
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


def _index_add_once(torch, cpu_gen):
    """重複indexへの`index_add_`を1回実行し、結果と入力のdigestを返す。

    `index_add_`はCUDA上で複数スレッドが同じ出力要素へ`atomicAdd`する
    実装であり、加算順序がスレッド実行順(スケジューリングのタイミング)
    に依存する——`Linear`/`ReLU`/`MSELoss`のみのMLP(cuBLAS GEMM、
    run-to-runで決定的)とは異なり、こちらは本当にGPU上で非決定的な
    演算である(モジュールdocstring finding 0参照)。
    """
    values_cpu, values = _cpu_randn(_ATOMIC_N, generator=cpu_gen)
    index_cpu = torch.randint(0, _ATOMIC_BINS, (_ATOMIC_N,), generator=cpu_gen)
    index = index_cpu.to("cuda")
    out = torch.zeros(_ATOMIC_BINS, device="cuda")
    out.index_add_(0, index, values)
    arr = out.cpu().numpy().astype(np.float64)
    input_digests = {
        "values": digest(values_cpu.numpy().astype(np.float64)),
        "index": digest(index_cpu.numpy().astype(np.int64)),
    }
    return arr, input_digests


@register("S8_index_add_atomics", requires=("torch", "cuda"))
def _index_add_atomics() -> ProbeResult:
    import torch

    cpu_gen = torch.Generator(device="cpu").manual_seed(_SEED)
    first, input_digests = _index_add_once(torch, cpu_gen)
    cpu_gen2 = torch.Generator(device="cpu").manual_seed(_SEED)
    second, _ = _index_add_once(torch, cpu_gen2)
    stable = bool(np.array_equal(first, second))

    scalars = {"sum": float(first.sum())}
    meta = {
        "seed": _SEED,
        "torch_version": torch.__version__,
        "n": _ATOMIC_N,
        "bins": _ATOMIC_BINS,
        "deterministic_algorithms": False,
        "stable_within_process": stable,
        "input_digests": input_digests,
        **_device_meta(torch),
    }
    for key, value in scalars.items():
        meta[f"{key}_hex"] = value.hex()
    return ProbeResult(
        probe="S8_index_add_atomics",
        digest=digest(first),
        scalars=scalars,
        sample_b64=encode_sample(first),
        meta=meta,
    )


@register("S8_index_add_atomics_deterministic", requires=("torch", "cuda"))
def _index_add_atomics_deterministic() -> ProbeResult:
    import torch

    with _deterministic_algorithms(torch, True):
        cpu_gen = torch.Generator(device="cpu").manual_seed(_SEED)
        first, input_digests = _index_add_once(torch, cpu_gen)
        cpu_gen2 = torch.Generator(device="cpu").manual_seed(_SEED)
        second, _ = _index_add_once(torch, cpu_gen2)
    stable = bool(np.array_equal(first, second))

    scalars = {"sum": float(first.sum())}
    meta = {
        "seed": _SEED,
        "torch_version": torch.__version__,
        "n": _ATOMIC_N,
        "bins": _ATOMIC_BINS,
        "deterministic_algorithms": True,
        "stable_within_process": stable,
        "input_digests": input_digests,
        **_device_meta(torch),
    }
    for key, value in scalars.items():
        meta[f"{key}_hex"] = value.hex()
    return ProbeResult(
        probe="S8_index_add_atomics_deterministic",
        digest=digest(first),
        scalars=scalars,
        sample_b64=encode_sample(first),
        meta=meta,
    )
