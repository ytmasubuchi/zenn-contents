"""S5: scikit-learnの推定器(ロジスティック回帰/ランダムフォレスト/k-means)
と、その前段の`train_test_split`。

重い依存(`sklearn`)はalpine等に無いことがあるため、関数本体の内部で
importする(モジュール先頭には置かない。パッケージdocstring参照)。

データはsklearnの`make_classification`(内部でlegacy `RandomState`を使う)
ではなく、`np.random.default_rng(20261014)`で**直接**生成する。本プロジェクト
のL0(PRNGの生ビット列比較、S1参照)はPCG64/MT19937などのビットジェネレータの
話であり、sklearnの乱数生成経路(legacy `RandomState`)を経由すると、
「sklearn推定器の非決定性」と「データ生成そのものの非決定性」という2つの
独立変数が混ざってしまう。`default_rng`で決定的に生成した固定データセット
(20,000サンプル×30特徴量の二値分類)を全S5プローブが共有することで、
差分の原因を「推定器の挙動」だけに絞り込む。

`train_test_split`・各推定器の`random_state`はすべて`20261014`で固定する
(データ生成のseedと数値は同じだが、役割は別——前者はnumpyのPCG64を、
後者はsklearn/各推定器内部の`RandomState`をシードする)。

**実測の記録(fix round 2で再訂正。以下は測定した相関のみを述べる——
推定した仕組みは述べない。fix round 1はここで二度、測定していない仕組みを
書いて二度とも誤りだったため、このモジュールでは今後そうしない)**:

`_kmeans_dataset()`(4つの分離されたクラスタ)を使い、30回ずつフィットして
`_canonical_centers()`適用後のdigestの一致率・`n_iter_`・`inertia_`の相対
ばらつきを測定した(`python:3.11-slim` + `scikit-learn==1.7.2`、8 CPU
コンテナ)。**次の表は1回の30フィット実行の記録であり、「30回中の重複
digest数」自体は固定値ではない**——この数自体が不安定な過程の集計値で
あるため、独立に30回フィットし直すたびに違う数になりうる(実際、fix
round 3でコーディネーターが独立に再実行したところ、下表の16/15/17では
なく13/15/19になり、「8固定」行と「未設定」行の`inertia_`相対ばらつきの
大小関係も入れ替わった——両実行を通じて共通していたのは、安定/不安定の
定性的なパターンと、不安定側の相対ばらつきが高々3.9e-16程度に収まる
という上限のみ)。したがって以下の個々の数値を再現可能な定数として
引用してはならない。1回の実行の記録として:

| 設定 | 30回中の重複digest数(この実行限りの値) | `n_iter_` | `inertia_`相対ばらつき(この実行限りの値) |
|---|---|---|---|
| `OMP_NUM_THREADS=1` | 1種類(30/30一致) | 常に2 | 0 |
| `OMP_NUM_THREADS=2` | 1種類(30/30一致) | 常に2 | 0 |
| `OMP_NUM_THREADS=4` | 1種類(30/30一致) | 常に2 | 0 |
| `OMP_NUM_THREADS=8`(=nproc固定) | 16種類 | 常に2 | 約3.9e-16 |
| 未設定 + `OMP_DYNAMIC=FALSE` | 15種類 | 常に2 | 約3.9e-16 |
| 未設定(既定) | 17種類 | 常に2 | 約1.9e-16 |

再現性のある形で測定できた相関は次だけである: **スレッド数1・2・4では
安定し、8(=このコンテナのnproc)では不安定**(この定性的なパターン
自体は独立な2回の実行で一致した)。`OMP_DYNAMIC=FALSE`を足しても
不安定さは変わらない。`n_iter_`は両実行とも全フィットで2のまま一度も
変わらず、`inertia_`の相対ばらつきは両実行を通じて高々3.9e-16
(float64で1〜2ULP相当)に留まる——ただしどちらの設定がその上限に近い
値を取るかは実行ごとに入れ替わりうる。さらに、スレッド数1・2・4のそれぞれは
**それ自体は安定している(2回の独立実行で確認済み)が、互いに異なる**
digestを返す(1→`972e0934…`、2→`42e031c0…`、4→`034d84cd…`、どちらの
実行でも同一)。

**ここから先、仕組み(なぜ8だけ不安定か、`cluster_centers_`のどの演算が
揺れているか等)は一切主張しない**——測定していないため。測定した事実
だけから言えるのは次の3点のみである: (1)スレッド数が1・2・4で安定し8で
不安定という相関が確認されたこと。(2)この不安定さは収束先のクラスタ
割り当てそのものの違いではないこと——`n_iter_`が揺れないこと自体は
(2回で同じ反復数に収束する別解がありうる以上)それだけでは証拠になら
ないが、`inertia_`の相対ばらつきが高々1〜2ULP、`_kmeans_centers`の
`meta["max_abs_center_delta"]`(2回の内部フィット間のクラスタ中心座標の
最大絶対差)がある実行ではsub-ULPの`6.9e-18`だったことは、2回の結果が
数値として極めて近いという直接の測定であり、これが根拠である。
(3)そのためスレッド数の違う2ホストの`S5_kmeans_centers`を比較する際は
`meta["effective_thread_count"]`が一致していることを前提にすべきこと。
"""
import os

import numpy as np

from reprobench.digest import digest, encode_sample
from reprobench.registry import ProbeResult, register

_SEED = 20261014
_N_SAMPLES = 20_000
_N_FEATURES = 30
_TEST_SIZE = 0.2


def _n_jobs() -> int:
    """`REPRO_THREADS`環境変数を並列数に反映する(並列数で結果が変わるかを
    見るため)。未設定時は`1`(シングルスレッド)にフォールバックする。
    """
    raw = os.environ.get("REPRO_THREADS")
    return int(raw) if raw else 1


def _thread_pool_info() -> list[dict]:
    """`threadpoolctl.threadpool_info()`の全エントリを決定的な順序で返す。

    `threadpoolctl.threadpool_info()`自体のリスト順は実行ごとに変わりうる
    (fix round 2で指摘: 同じ環境で`[openmp, blas, blas]`の順になることも
    `[blas, blas, openmp]`の順になることもある)。`_effective_thread_count`
    が単一の値に潰す前の生データをそのまま残し、`(user_api, internal_api,
    num_threads)`の辞書式順序で並べることで、潰す前の情報も失わず、選び方
    自体の非決定性も混入させない(`num_threads`も`str()`で文字列化してから
    比較するため、この3つ目のキーの大小は数値順ではなく辞書式——"10"は
    "2"より前に来る——である。ここで求めているのは何らかの固定順序であって
    数値としての大小順ではないため、これでも要件は満たす)。ただしこのキーは
    `(user_api, internal_api, num_threads)`が完全に一致する2つ以上の
    エントリ(区別不能なほど同一設定のOpenMPランタイムが複数同時に読み込
    まれる、という縮退したケース)までは順序を一意に決めない——その場合は
    どれを選んでも値は同じなので実害は無い(fix round 3で指摘:
    `num_threads`を欠いた2要素キーのままでは、同じ`user_api`/
    `internal_api`だが`num_threads`が異なる2つのOpenMPランタイムという、
    まさにこの関数が区別したいはずのケースで順序を一意に決められなかった)。
    """
    import threadpoolctl

    info = threadpoolctl.threadpool_info()
    return sorted(
        (
            {
                "user_api": i.get("user_api"),
                "internal_api": i.get("internal_api"),
                "num_threads": i.get("num_threads"),
            }
            for i in info
        ),
        key=lambda i: (str(i["user_api"]), str(i["internal_api"]), str(i["num_threads"])),
    )


def _effective_thread_count() -> int | None:
    """OpenMP/BLASが実際に使うスレッド数を`threadpoolctl`経由で読む。

    `REPRO_THREADS`/`OMP_NUM_THREADS`は未設定のことが多く(fix round 1
    finding 7)、その場合「環境変数の値」を記録しても`None`にしかならず、
    KMeansの不安定さとスレッド数の相関を見る以上、実際に有効なスレッド数を
    記録しないと議論ができない。threadpoolctlはscikit-learnの必須依存で
    あるため、sklearnが使える環境では常にimportできる。

    `_thread_pool_info()`で順序を固定した後に"openmp"エントリを選ぶため、
    1つのOpenMPランタイムしか無い環境(検証コンテナで確認したのはこれの
    み——4台の実測ホストでは未確認)では常に同じ値を返す。2つ以上の
    OpenMPランタイムが同時に読み込まれる環境では、ここで選ばれなかった
    方の値は`thread_pool_info`フィールド(`_kmeans_centers`のmeta参照)に
    残る。
    """
    info = _thread_pool_info()
    openmp = [i["num_threads"] for i in info if i.get("user_api") == "openmp"]
    if openmp:
        return int(openmp[0])
    if info:
        return int(info[0]["num_threads"])
    return None


def _dataset() -> tuple[np.ndarray, np.ndarray]:
    """`default_rng(_SEED)`で決定的に生成した二値分類データセット。

    特徴量は標準正規分布、ラベルは真の線形結合をロジスティック関数に通した
    確率でベルヌーイサンプリングする(sklearnの`make_classification`は
    使わない——モジュールdocstring参照)。

    **`S5_kmeans_centers`はこのデータセットを使わない**(下の`_kmeans_dataset`
    参照)。この単一の等方ガウス分布は二値分類のために構築したものであり、
    クラスタリング用には構築していない——別途クラスタリング用のデータ
    セットを用意した設計上の理由は`_kmeans_dataset`参照。
    """
    rng = np.random.default_rng(_SEED)
    X = rng.standard_normal((_N_SAMPLES, _N_FEATURES))
    true_coef = rng.standard_normal(_N_FEATURES)
    logits = X @ true_coef
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=_N_SAMPLES) < probs).astype(np.int64)
    return X, y


def _kmeans_dataset() -> np.ndarray:
    """`S5_kmeans_centers`専用: 真のクラスタ構造を持つ、十分に分離された
    4つの等方ガウス分布の混合(fix round 1 finding 2)。

    分類用`_dataset()`(構造の無い単一の等方ガウス)はクラスタリング用には
    構築されていない——`_dataset()`のdocstring参照。`S5_kmeans_centers`には
    代わりに、設計上明確なクラスタ構造を持つこのデータセットを使う:
    標準偏差1の点群を4つ、原点から10(=10標準偏差)離した互いに直交する
    方向に配置する(クラスタ間距離がクラスタ内分散に対して圧倒的に大きい
    という設計上の性質であり、実際にこのデータセットで`n_iter_`が全フィット
    で2のまま一度も変わらなかったこと——モジュールdocstring参照——は、この
    設計の帰結と整合する実測結果である)。`sklearn.datasets.make_blobs`は
    使わない(内部でlegacy `RandomState`を使うため、モジュールdocstring
    冒頭の理由がそのまま当てはまる)。
    """
    rng = np.random.default_rng(_SEED)
    n_per_cluster = _N_SAMPLES // 4
    centers = np.zeros((4, _N_FEATURES))
    centers[0, 0] = 10.0
    centers[1, 0] = -10.0
    centers[2, 1] = 10.0
    centers[3, 1] = -10.0
    blocks = [
        centers[i] + rng.standard_normal((n_per_cluster, _N_FEATURES))
        for i in range(4)
    ]
    return np.concatenate(blocks, axis=0)


def _canonical_centers(centers: np.ndarray) -> np.ndarray:
    """クラスタ中心の行順を正規化する(fix round 1 finding 2)。

    KMeansはクラスタに安定したラベル・順序を割り当てない(どの行がどの
    クラスタかは初期化の乱数依存)。2つの環境が**同一の**クラスタリングを
    見つけても、行の並びが違うだけでdigestが一致しなくなり、比較器が
    実在しない不一致を報告してしまう。各行を辞書式順序(最初の特徴量を
    主キー)で並べ替えてからdigestを取ることで、この見かけ上の不一致を
    除去する。
    """
    order = np.lexsort(centers.T[::-1])
    return centers[order]


def _split(y: np.ndarray):
    """全S5プローブが共有する train/test インデックス分割。"""
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(y))
    return train_test_split(
        indices, test_size=_TEST_SIZE, random_state=_SEED, stratify=y
    )


def _hexed(meta: dict, scalars: dict) -> dict:
    """JSONのfloat表現はfloat64の全ビットを往復させる保証が無いため、
    scalarsの各値のhex表現をmetaへ残す(s4_libm.pyと同じ理由)。
    """
    for key, value in scalars.items():
        meta[f"{key}_hex"] = value.hex()
    return meta


@register("S5_logreg_coef", requires=("sklearn",))
def _logreg_coef() -> ProbeResult:
    from sklearn.linear_model import LogisticRegression

    X, y = _dataset()
    train_idx, test_idx = _split(y)
    model = LogisticRegression(solver="lbfgs", max_iter=200, random_state=_SEED)
    model.fit(X[train_idx], y[train_idx])

    coef = model.coef_.astype(np.float64).ravel()
    intercept = float(model.intercept_[0])
    arr = np.concatenate([coef, [intercept]])
    scalars = {
        "intercept": intercept,
        "coef_l2_norm": float(np.linalg.norm(coef)),
    }
    meta = _hexed(
        {
            "seed": _SEED,
            "n_features": _N_FEATURES,
            "n_train": len(train_idx),
            "solver": "lbfgs",
            "max_iter": 200,
            "n_iter": int(np.asarray(model.n_iter_).ravel()[0]),
            # fix round 1 finding 10: S5/S6は同じ生成式のデータセットを
            # 使うが独立に分割するため、分割自体が食い違っていないかを
            # 比較器が直接確認できるよう、使った分割のdigestを残す。
            "train_idx_digest": digest(train_idx.astype(np.int64)),
        },
        scalars,
    )
    return ProbeResult(
        probe="S5_logreg_coef",
        digest=digest(arr),
        scalars=scalars,
        sample_b64=encode_sample(arr),
        meta=meta,
    )


@register("S5_random_forest_pred", requires=("sklearn",))
def _random_forest_pred() -> ProbeResult:
    from sklearn.ensemble import RandomForestClassifier

    X, y = _dataset()
    train_idx, test_idx = _split(y)
    n_jobs = _n_jobs()
    model = RandomForestClassifier(
        n_estimators=100, random_state=_SEED, n_jobs=n_jobs
    )
    model.fit(X[train_idx], y[train_idx])
    proba = model.predict_proba(X[test_idx])[:, 1].astype(np.float64)

    scalars = {"mean_proba": float(proba.mean())}
    meta = _hexed(
        {
            "seed": _SEED,
            "n_estimators": 100,
            "n_jobs": n_jobs,
            "repro_threads_env": os.environ.get("REPRO_THREADS"),
            "n_test": len(test_idx),
            "train_idx_digest": digest(train_idx.astype(np.int64)),
        },
        scalars,
    )
    return ProbeResult(
        probe="S5_random_forest_pred",
        digest=digest(proba),
        scalars=scalars,
        sample_b64=encode_sample(proba),
        meta=meta,
    )


@register("S5_kmeans_centers", requires=("sklearn",))
def _kmeans_centers() -> ProbeResult:
    # 測定した相関のみ(モジュールdocstring参照、fix round 2): このプロセス
    # が使う有効スレッド数(`meta["effective_thread_count"]`)が1・2・4の
    # ときは安定、8(=コンテナのnproc)のときは不安定、という相関を30回×6
    # 設定の実測で確認した。`n_iter_`は全180回のフィットで一度も変わって
    # いない。これ以上の仕組み(なぜ8だけか等)は測定していないので書かない。
    # 本プローブはこの不安定さを握り潰す方向(強制的にスレッド数を固定
    # する)には倒さず、S0_parallel_completion_order/S0_glob_orderと同じ
    # 流儀で「プローブ自身が2回実行して一致したかを自己申告する」形に
    # する(resolution 3)。
    from sklearn.cluster import KMeans

    X = _kmeans_dataset()
    n_clusters = 4

    def _fit_once():
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=_SEED)
        model.fit(X)
        return model

    first = _fit_once()
    second = _fit_once()
    first_centers = _canonical_centers(first.cluster_centers_.astype(np.float64))
    second_centers = _canonical_centers(second.cluster_centers_.astype(np.float64))
    stable = bool(
        np.array_equal(first_centers, second_centers)
        and first.inertia_ == second.inertia_
    )
    # fix round 2 finding I1: `stable`は真偽値1個に潰れてしまい、
    # `stable_within_process=False`が返ってきたとき、記録からは「n_iter_が
    # 変わったのか(収束先そのものが違う)」と「inertia_/中心座標だけが
    # ULP程度揺れたのか(縮約の丸め)」を区別できない。測定ホストでの
    # 1回限りの実行になるため、2回目のfitの値も生で残す。
    max_abs_center_delta = float(np.max(np.abs(first_centers - second_centers)))

    scalars = {"inertia": float(first.inertia_)}
    meta = _hexed(
        {
            "seed": _SEED,
            "n_clusters": n_clusters,
            "n_init": 10,
            "n_iter": int(first.n_iter_),
            "n_iter_second": int(second.n_iter_),
            "inertia_second": float(second.inertia_),
            "inertia_second_hex": float(second.inertia_).hex(),
            "max_abs_center_delta": max_abs_center_delta,
            "max_abs_center_delta_hex": max_abs_center_delta.hex(),
            "stable_within_process": stable,
            "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
            "effective_thread_count": _effective_thread_count(),
            "thread_pool_info": _thread_pool_info(),
            "centers_canonicalized": "lexsort_rows",
        },
        scalars,
    )
    return ProbeResult(
        probe="S5_kmeans_centers",
        digest=digest(first_centers),
        scalars=scalars,
        sample_b64=encode_sample(first_centers),
        meta=meta,
    )


@register("S5_train_test_split_index", requires=("sklearn",))
def _train_test_split_index() -> ProbeResult:
    # このプローブは推定器を一切呼ばない(L0: train_test_split自身の
    # 挙動——sklearn内部のRandomStateのshuffleアルゴリズム——だけを
    # 単離するため)。
    _X, y = _dataset()
    train_idx, test_idx = _split(y)
    arr = np.concatenate([train_idx, test_idx]).astype(np.int64)
    meta = {
        "seed": _SEED,
        "test_size": _TEST_SIZE,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "stratify": True,
        "train_idx_digest": digest(train_idx.astype(np.int64)),
    }
    return ProbeResult(
        probe="S5_train_test_split_index",
        digest=digest(arr),
        scalars={},
        sample_b64=encode_sample(arr.astype(np.float64)),
        meta=meta,
    )
