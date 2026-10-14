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

**実測で判明した事実(fix round 1で訂正)**: `S5_kmeans_centers`は、スレッド
数を明示的に固定しない(=環境変数未設定の)既定状態では、同一プロセス内の
2回の呼び出しで`cluster_centers_`のdigestが一致しない。`OMP_NUM_THREADS`を
**固定値(1でも4でも)に明示的に設定すると、5回連続で完全に一致**する
——固定前の版のコメントは「1スレッドに固定しても再現した」と書いていたが、
これは誤りであり(fix round 1 finding 1で指摘された)、実測の転記ミスだった。
正しくは「スレッド数を固定すれば安定し、固定しなければ安定しない」である。

さらに、最初の実装は`_dataset()`(構造の無い単一の等方ガウス分布、分類用)を
そのままKMeansに通しており、k=4の局所最適解がほぼ無数に近接して並ぶ
退化したケースを測っていた(fix round 1 finding 2)。`S5_kmeans_centers`
専用に真のクラスタ構造を持つデータセット(`_kmeans_dataset`)を用意して
20回連続でフィットし直した結果、**収束反復数(`n_iter_`)は20回とも完全に
同一(常に2)**——つまり「どの局所最適解を選ぶか」が揺れているのではなく、
収束経路そのものは毎回同一である。一方`inertia_`自体は20回中4回、
相対誤差にしておよそ2e-16(=float64の機械イプシロン相当、1ULP)だけ
他と異なる値を取った。これは「違う解に収束した」のではなく、同じ解を
表現する浮動小数点の最終縮約(距離二乗和・クラスタごとの点群平均)の
加算順序がわずかに揺れているだけだと読める——n_iter_が一度も揺れていない
ことがその根拠である。つまり本当の原因は「KMeansが別の局所最適解に落ちる」
ことではなく、S2/S3と同じ「縮約の浮動小数点非結合性」がKMeansの内部実装
(OpenMP並列化されたE-step)を経由して現れているだけであり、スレッド数を
固定すれば消える(`_kmeans_centers`のコメント参照)。
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


def _effective_thread_count() -> int | None:
    """OpenMP/BLASが実際に使うスレッド数を`threadpoolctl`経由で読む。

    `REPRO_THREADS`/`OMP_NUM_THREADS`は未設定のことが多く(fix round 1
    finding 7)、その場合「環境変数の値」を記録しても`None`にしかならず、
    KMeansの非決定性の原因がスレッド数だと主張する以上、実際に有効な
    スレッド数を記録しないと議論ができない。threadpoolctlはscikit-learnの
    必須依存であるため、sklearnが使える環境では常にimportできる。
    """
    import threadpoolctl

    info = threadpoolctl.threadpool_info()
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
    参照)。単一の等方ガウス分布にはクラスタ構造が無く、k=4のクラスタリング
    には無数の近接した局所最適解が存在してしまうため。
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

    分類用`_dataset()`(構造の無い単一の等方ガウス)をそのままKMeansに
    かけると、k=4の「正解」が存在しない退化したケースになり、`n_init=10`
    のベスト選択がほぼ無限に近い数の近接した局所最適解の間でどれを選ぶか
    というノイズを増幅してしまう。ここでは標準偏差1の点群を4つ、原点から
    10(=10標準偏差)離した互いに直交する方向に配置する——クラスタ間距離が
    クラスタ内分散に対して圧倒的に大きいため、"正しい"分割は実質的に一意。
    `sklearn.datasets.make_blobs`は使わない(内部でlegacy `RandomState`を
    使うため、モジュールdocstring冒頭の理由がそのまま当てはまる)。
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
    # fix round 1で再測定・訂正(モジュールdocstring参照): 真のクラスタ
    # 構造を持つ`_kmeans_dataset()`を使うと、20回連続フィットで`n_iter_`は
    # 常に2(=KMeansが別の局所最適解を選んでいるのではなく、収束経路自体は
    # 毎回同一)。`inertia_`自体はまれに1ULP相当(相対誤差約2e-16)だけ
    # 揺れることがあるが、n_iter_が揺れないことから「別の解」ではなく
    # 最終縮約の丸めの揺れだと判断できる。`cluster_centers_`(クラスタ
    # ごとの点群の平均という縮約演算)のdigestは、スレッド数を明示的に
    # 固定しない既定状態では同一プロセス内の呼び出し間でもっとはっきり
    # 一致しない。`OMP_NUM_THREADS`を1にも4にも固定すると5回とも完全一致
    # するため、原因は「スレッド数そのもの」ではなく「スレッド数が固定
    # されていないこと」——プロセス内の呼び出しごとに縮約の浮動小数点
    # 加算順序が変わりうることだと判断できる(S2/S3が検出する非結合性と
    # 同じ現象が、sklearnのOpenMP並列E-stepという別の経路で現れている
    # だけ)。本プローブはこの効果を握り潰す方向(強制的にスレッド数を
    # 固定する)には倒さず、S0_parallel_completion_order/S0_glob_orderと
    # 同じ流儀で「プローブ自身が2回実行して一致したかを自己申告する」形に
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

    scalars = {"inertia": float(first.inertia_)}
    meta = _hexed(
        {
            "seed": _SEED,
            "n_clusters": n_clusters,
            "n_init": 10,
            "n_iter": int(first.n_iter_),
            "stable_within_process": stable,
            "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
            "effective_thread_count": _effective_thread_count(),
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
