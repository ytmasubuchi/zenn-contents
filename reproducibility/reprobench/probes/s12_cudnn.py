"""S12: cuDNNの`benchmark`/`deterministic`設定と、プロセス内反復での安定性。

重い依存(`torch`)はalpine等に無いことがあるため、関数本体の内部で
importする(モジュール先頭には置かない。パッケージdocstring参照)。

## 題材

畳み込み層を2枚含む小さなモデルの順伝播・逆伝播を、
`torch.backends.cudnn.benchmark` × `torch.backends.cudnn.deterministic`の
2×2(`CUDNN_SETTINGS`)それぞれで実行する。モデルの重み・入力・targetは
設定ごとに同じseedから再構築する(= 設定間で入力を揃える)。

## 本プローブの要点: プロセス内反復

`torch.backends.cudnn.benchmark = True`は、cuDNNに複数の畳み込み
アルゴリズムを**実行時にタイミング計測させ**、最速のものを選ばせる
(PyTorch公式ドキュメント「Reproducibility」が明記する通り、この
ベンチマークは実行ごとに異なるアルゴリズムを選びうる非決定的な処理
である)。選ばれるアルゴリズムの中には`atomicAdd`などスレッド実行順に
依存する非決定的な実装を含むものもあるため、**同一プロセス内で同じ
設定を繰り返し実行するだけでも、計算結果そのものが変わりうる**。これが
本プローブの存在意義であり、1回だけ実行して記録しても測れない。その
ため各設定について`_N_ITERS`回のプロセス内反復を行い、反復間でdigestが
全て一致したかどうかを`stable_within_process`として記録する
(`deterministic=True`側は安定するはずだという**予測**はあるが、
S11/S14と同様にテストでこの実測結果を決め打ちしない)。
"""
import numpy as np

from reprobench.digest import digest, encode_sample
from reprobench.registry import ProbeResult, register

_SEED = 20261014
_N_ITERS = 3
_BATCH = 32
_IN_CHANNELS = 3
_HIDDEN_CHANNELS = 16
_KERNEL_SIZE = 3
_SPATIAL = 16

#: cuDNN設定の2×2グリッド。要素は`(benchmark, deterministic)`。
CUDNN_SETTINGS: tuple[tuple[bool, bool], ...] = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)


def result_key(benchmark: bool, deterministic: bool) -> str:
    """`(benchmark, deterministic)`から`meta["results"]`のキー文字列を作る純粋関数。"""
    return f"benchmark={benchmark},deterministic={deterministic}"


def stable_within_process(digests: list[str]) -> bool:
    """同一設定でのプロセス内反復のdigest列が全て一致するかを返す純粋関数。

    torch/numpyへの依存は一切無いため、GPU無し環境でもそのまま単体
    テストできる。空リスト・要素数1のリストは(比較対象が無い/1つしか
    無いため)安定とみなす——ただし呼び出し側は常に`_N_ITERS >= 2`で
    呼び出す。
    """
    return len(set(digests)) <= 1


def _build_conv_model():
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(_IN_CHANNELS, _HIDDEN_CHANNELS, _KERNEL_SIZE, padding=1),
        nn.ReLU(),
        nn.Conv2d(_HIDDEN_CHANNELS, _HIDDEN_CHANNELS, _KERNEL_SIZE, padding=1),
    )


def _run_setting(torch, benchmark: bool, deterministic: bool) -> dict:
    """1つの`(benchmark, deterministic)`設定で、順伝播・逆伝播を`_N_ITERS`回
    プロセス内反復し、セル辞書を返す。

    `torch.backends.cudnn.benchmark`/`.deterministic`はプロセスグローバルな
    可変状態であるため、`try`/`finally`で必ず元の値へ復元する(S8の
    `_tf32_flags`と同じ流儀。他の後続プローブへ意図せず伝播させないため)。
    入力はCPU側のジェネレータで生成してから`.to("cuda")`で転送する
    (S8の`_cpu_randn`と同じ流儀。本プローブが測りたいのはcudnn設定の
    影響であり、curandのGPUモデル依存性を混入させたくないため)。モデルの
    重みは反復間で変えない(optimizer.stepを呼ばない)ため、反復間の差分は
    cudnn設定由来の非決定性にのみ帰着する。
    """
    import torch.nn.functional as F

    orig_benchmark = torch.backends.cudnn.benchmark
    orig_deterministic = torch.backends.cudnn.deterministic
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
    try:
        torch.manual_seed(_SEED)
        model = _build_conv_model().to("cuda")
        cpu_gen = torch.Generator(device="cpu").manual_seed(_SEED)
        x = torch.randn(
            _BATCH, _IN_CHANNELS, _SPATIAL, _SPATIAL, generator=cpu_gen
        ).to("cuda")
        target = torch.randn(
            _BATCH, _HIDDEN_CHANNELS, _SPATIAL, _SPATIAL, generator=cpu_gen
        ).to("cuda")

        iteration_digests: list[str] = []
        last_arr: np.ndarray | None = None
        for _ in range(_N_ITERS):
            model.zero_grad()
            out = model(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            grads = torch.cat([p.grad.detach().flatten().float() for p in model.parameters()])
            arr = grads.cpu().numpy().astype(np.float64)
            iteration_digests.append(digest(arr))
            last_arr = arr
    finally:
        torch.backends.cudnn.benchmark = orig_benchmark
        torch.backends.cudnn.deterministic = orig_deterministic

    return {
        "digest": iteration_digests[-1],
        "iteration_digests": iteration_digests,
        "stable_within_process": stable_within_process(iteration_digests),
        "sample_b64": encode_sample(last_arr),
    }


@register("S12_cudnn_benchmark_deterministic", requires=("torch", "cuda"))
def _cudnn_benchmark_deterministic() -> ProbeResult:
    import torch

    results: dict[str, dict] = {}
    for benchmark, deterministic in CUDNN_SETTINGS:
        results[result_key(benchmark, deterministic)] = _run_setting(
            torch, benchmark, deterministic
        )

    # プローブ全体のdigestは(benchmark, deterministic) -> digest の順序付き
    # マッピングから取る。宣言順(CUDNN_SETTINGS)を保持するため、dictではなく
    # list-of-tuplesにする(s11_rng_order.py/s14_dataloader_seed.pyと同じ理由:
    # `canonical_bytes`のdict実装はキーでソートし直すため宣言順を失う)。
    digest_map_ordered = [
        (result_key(b, d), results[result_key(b, d)]["digest"]) for b, d in CUDNN_SETTINGS
    ]

    meta = {
        "seed": _SEED,
        "torch_version": torch.__version__,
        "n_iters": _N_ITERS,
        "cudnn_settings": [list(s) for s in CUDNN_SETTINGS],
        "results": results,
        # 全4設定のstable_within_processをANDで畳んだ、トップレベルの
        # 自己申告フラグ。tests/test_all_probes_smoke.pyの
        # `_SELF_REPORTED_STABILITY`はこのキーが`meta`直下に存在することを
        # 一律に見るため(`benchmark=True`側の個別セルの値は
        # `meta["results"][...]["stable_within_process"]`に別途記録済み)。
        "stable_within_process": all(
            cell["stable_within_process"] for cell in results.values()
        ),
    }
    return ProbeResult(
        probe="S12_cudnn_benchmark_deterministic",
        digest=digest(digest_map_ordered),
        scalars={},
        sample_b64=None,
        meta=meta,
    )
