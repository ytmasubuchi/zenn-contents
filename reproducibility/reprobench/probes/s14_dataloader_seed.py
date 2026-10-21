"""S14: DataLoaderのworker seeding——「インプットが全部同じ」という前提の破れ。

PyTorchで最も有名な再現性の落とし穴を実測する。`torch.utils.data.DataLoader`
を`num_workers > 0`で使うと、各worker(子プロセス)は起動時に
`torch.manual_seed(base_seed + worker_id)`で**torchのグローバルRNG**を
worker毎に異なる値で再シードする(`torch/utils/data/_utils/worker.py`の
`_worker_loop`)。同じ箇所は`numpy.random.seed(...)`も呼ぶため、実は
**numpyのモジュールレベルのグローバル関数**(`numpy.random.rand()`等)も
worker毎に再シードされる(この自動再シード自体はこのdocstring執筆時点の
PyTorchに実際に存在する――本プローブは一度これを素朴な実装
(`np.random.rand()`呼び出し)で確かめた上で下記の構成に書き換えた)。

**依然として再シードされない**のは、`Dataset.__init__`(= 親プロセス、
fork前)で一度だけ作られ、インスタンス変数として保持された
`numpy.random.Generator`/`RandomState`である。`torch.manual_seed`も
`np.random.seed`も「グローバルな既定の乱数源」しか触らず、ユーザーが
自分で保持しているGeneratorインスタンスの内部状態には関与しない。
`worker_init_fn`を書かない限り、全workerはforkした瞬間の親プロセスの
メモリ(=このGeneratorインスタンスの内部状態)をそのまま物理的に共有
するため、同じ乱数列を消費することになる。これはPyTorch公式ドキュメント
(Randomness in multi-process data loading)が明示的に警告している、
現在でも有効な落とし穴である。

## 題材

サンプルのインデックスに依存しない1個の乱数値をデータ拡張の代わりに
引く`Dataset`を、`rng_source`(`"torch"`/`"numpy"`)ごとに用意する。

- `torch`: `__getitem__`の中で`torch.rand(1)`(グローバルな既定の
  torch RNG)を直接呼ぶ。workerごとに自動で異なる値へ再シードされる
  側の対照実験。
- `numpy`: `__init__`(= 親プロセス側)で`np.random.default_rng(seed)`を
  1回だけ作り、インスタンス変数に保持する。`__getitem__`はこの
  インスタンスから引く。自動再シードの対象外になる側。

インデックスに依存しないのは意図的: あるworkerが処理するk番目のサンプル
と、別のworkerが処理するk番目のサンプルが**同じ乱数源の状態から**値を
引けば、それらは全く別のサンプルであるにもかかわらず同じ値になる
(= データ拡張が重複する)。これを観測値列の重複として数えたものが
`duplicate_sample_count`である。

`torch`版は各workerが異なるシードで再シードされるため重複は(統計的に)
起きないはずで、`numpy`版は全workerが同一のGeneratorインスタンスの状態を
物理的に共有するため重複が起きるはずだ、というのがこのプローブの検証
対象の予測である。ただし「numpy版が壊れる」という結果そのものをテストで
固定してはならない(`tests/test_s14_dataloader_seed.py`参照) — 実測で
確かめる対象である。

## `worker_init_fn`を指定しない理由

`worker_init_fn`で`numpy.random.seed(...)`をworkerごとに呼べば、この
重複は消える。しかしそれこそが今回**測りたい現象**なので、本プローブ
では意図的に`worker_init_fn`を指定しない(指定すると、再現性の罠その
ものが観測できなくなる)。

## worker起動の失敗について

`num_workers > 0`はコンテナの共有メモリ不足などで失敗しうる。測定全体を
止めないため、`DataLoader`の構築・反復を条件ごとに`try/except`で囲み、
失敗した場合は例外を握りつぶさず`worker_start_failed`にその内容を記録
して次の条件へ進む。
"""
import numpy as np

from reprobench.digest import digest
from reprobench.registry import ProbeResult, register

#: データ拡張の乱数源2通り。宣言順が`meta["results"]`のトップレベルキー
#: の順序、およびプローブ全体のdigestの順序付きマッピングの順序になる。
RNG_SOURCES: tuple[str, ...] = ("torch", "numpy")

#: 検証するworker数3通り。0は「workerを使わない(= 比較対象の素のベース
#: ライン)」ケース。
WORKER_COUNTS: tuple[int, ...] = (0, 2, 4)

_SEED = 20261014
_DATASET_SIZE = 16


def duplicate_sample_count(values: list[float]) -> int:
    """観測値列のうち、重複している要素の個数を返す純粋関数。

    「ユニークな値の個数」と「値の総数」の差分として数える
    (= 2回以上出現する値がある場合、その超過分の総和)。torch/numpyへの
    依存は無いため、モデルや外部状態なしで単体テストできる。
    """
    return len(values) - len(set(values))


def _build_dataset(rng_source: str):
    """`rng_source`に応じたデータ拡張を行う`torch.utils.data.Dataset`を返す。

    重い依存(`torch`)はここ(関数本体)でimportする(パッケージdocstring
    参照)。`__getitem__`はインデックス`idx`を一切使わず、乱数源から
    1個の値を引くだけである(モジュールdocstringの「題材」参照:
    インデックス非依存であることが、worker間の重複を観測可能にする
    設計そのもの)。

    `torch`版は`__getitem__`の中で都度グローバルな既定RNGを呼ぶ(= worker
    毎の自動再シードの対象)。`numpy`版は`__init__`(= 親プロセス、fork前)
    で一度だけ`np.random.default_rng`を作りインスタンス変数に保持する
    (= 自動再シードの対象外。モジュールdocstring参照)。
    """
    import torch
    from torch.utils.data import Dataset

    if rng_source == "torch":

        class _TorchAugmentedDataset(Dataset):
            def __len__(self) -> int:
                return _DATASET_SIZE

            def __getitem__(self, idx: int) -> float:
                return float(torch.rand(1).item())

        return _TorchAugmentedDataset()
    if rng_source == "numpy":

        class _NumpyAugmentedDataset(Dataset):
            def __init__(self) -> None:
                # 親プロセスで一度だけ作るGeneratorインスタンス。
                # worker起動時の`np.random.seed(...)`(モジュールレベルの
                # グローバルRandomState向け)の対象にはならない。
                self._rng = np.random.default_rng(_SEED)

            def __len__(self) -> int:
                return _DATASET_SIZE

            def __getitem__(self, idx: int) -> float:
                return float(self._rng.random())

        return _NumpyAugmentedDataset()
    raise ValueError(f"_build_dataset: unknown rng_source {rng_source!r}")


def _run_cell(rng_source: str, num_workers: int) -> dict:
    """1つの`(rng_source, num_workers)`条件を実行し、セル辞書を返す。

    毎回`torch.manual_seed`でtorchのグローバルRNGを固定シードへreseed
    してから`DataLoader`を構築する(S7と同じ、呼び出し順序への依存を
    断つための流儀)。`numpy`版のシードは`_build_dataset`が作る
    `Generator`インスタンス自身に対して固定されるため、ここでグローバル
    numpy状態を触る必要はない。`worker_init_fn`は指定しない(測定対象
    そのものを消してしまうため、モジュールdocstring参照)。

    worker起動・反復が失敗した場合は例外を握りつぶさず、`digest`には
    空リストのdigest、`worker_start_failed`には例外の内容を記録して
    返す(測定全体を止めない。brief Review Focus 4)。
    """
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(_SEED)
    dataset = _build_dataset(rng_source)

    try:
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
        )
        values = [float(batch.item()) for batch in loader]
    except Exception as exc:  # worker起動失敗(共有メモリ不足等)を含む
        return {"digest": digest([]), "worker_start_failed": repr(exc)}

    return {
        "digest": digest(values),
        "n_samples": len(values),
        "duplicate_sample_count": duplicate_sample_count(values),
    }


@register("S14_dataloader_worker_seed", requires=("torch",))
def _dataloader_worker_seed() -> ProbeResult:
    """`RNG_SOURCES` x `WORKER_COUNTS`の全組み合わせを実行する。

    `meta["results"]`は`{rng_source: {str(num_workers): cell}}`という
    構造を持つ。worker数はJSONの整数キー非対応のため文字列化する。
    """
    results: dict[str, dict[str, dict]] = {}
    for rng_source in RNG_SOURCES:
        results[rng_source] = {}
        for num_workers in WORKER_COUNTS:
            results[rng_source][str(num_workers)] = _run_cell(rng_source, num_workers)

    # プローブ全体のdigestは(rng_source, num_workers) -> digest の順序付き
    # マッピングから取る。宣言順(RNG_SOURCES x WORKER_COUNTS)を保持する
    # ため、dictではなくlist-of-tuplesにする(s11_rng_order.pyと同じ理由:
    # `canonical_bytes`のdict実装はキーでソートし直すため宣言順を失う)。
    digest_map_ordered = [
        ((rng_source, num_workers), results[rng_source][str(num_workers)]["digest"])
        for rng_source in RNG_SOURCES
        for num_workers in WORKER_COUNTS
    ]

    meta = {
        "seed": _SEED,
        "dataset_size": _DATASET_SIZE,
        "rng_sources": list(RNG_SOURCES),
        "worker_counts": list(WORKER_COUNTS),
        "results": results,
    }
    return ProbeResult(
        probe="S14_dataloader_worker_seed",
        digest=digest(digest_map_ordered),
        scalars={},
        sample_b64=None,
        meta=meta,
    )
