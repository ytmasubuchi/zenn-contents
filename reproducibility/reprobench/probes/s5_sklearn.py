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

**実測で判明した事実**: `S5_kmeans_centers`は`random_state`を固定しても
同一プロセス内の2回の呼び出しで結果が一致しない(`_kmeans_centers`の
コメント参照)。`random_state`と`n_jobs`/スレッド数を固定しても再現する
ため、これは本記事が検出したい非決定性の一種として自己申告する
(resolution 3、`S0_parallel_completion_order`と同じ扱い)。
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


def _dataset() -> tuple[np.ndarray, np.ndarray]:
    """`default_rng(_SEED)`で決定的に生成した二値分類データセット。

    特徴量は標準正規分布、ラベルは真の線形結合をロジスティック関数に通した
    確率でベルヌーイサンプリングする(sklearnの`make_classification`は
    使わない——モジュールdocstring参照)。
    """
    rng = np.random.default_rng(_SEED)
    X = rng.standard_normal((_N_SAMPLES, _N_FEATURES))
    true_coef = rng.standard_normal(_N_FEATURES)
    logits = X @ true_coef
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=_N_SAMPLES) < probs).astype(np.int64)
    return X, y


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
    # 実測で判明した事実: KMeansは`random_state`を固定しても**同一プロセス
    # 内で2回呼んでも結果が一致しない**。sklearnのLloyd/Elkan E-stepは
    # OpenMPで並列化されており、スレッドのスケジューリングタイミングに
    # 応じて浮動小数点の縮約順序がわずかに(ULPオーダーで)揺れる。
    # `n_init=10`はそのわずかな揺れを増幅する: 10回の独立な初期化のうち
    # inertiaが僅差でほぼ並ぶ複数の局所最適解が存在する場合、「どれが
    # bestか」の選択自体がこのULPジッタで入れ替わりうるため、最終的な
    # `cluster_centers_`がプロセス内の呼び出しごとに**別の局所最適解**
    # に揺れる(実機で`OMP_NUM_THREADS=1`に固定しても再現したため、
    # スレッド数そのものではなく、並列実行のタイミング依存であることを
    # 確認済み)。これは本プロジェクトが検出したい非決定性の一種である
    # ため、ここでそれを握り潰す方向(例えば強制的にシングルスレッド化
    # する)には倒さず、S0_parallel_completion_order/S0_glob_orderと同じ
    # 流儀で「プローブ自身が2回実行して一致したかを自己申告する」形に
    # する(resolution 3)。
    from sklearn.cluster import KMeans

    X, _y = _dataset()
    n_clusters = 4

    def _fit_once():
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=_SEED)
        model.fit(X)
        return model

    first = _fit_once()
    second = _fit_once()
    stable = bool(
        np.array_equal(first.cluster_centers_, second.cluster_centers_)
        and first.inertia_ == second.inertia_
    )

    centers = first.cluster_centers_.astype(np.float64)
    scalars = {"inertia": float(first.inertia_)}
    meta = _hexed(
        {
            "seed": _SEED,
            "n_clusters": n_clusters,
            "n_init": 10,
            "n_iter": int(first.n_iter_),
            "stable_within_process": stable,
        },
        scalars,
    )
    return ProbeResult(
        probe="S5_kmeans_centers",
        digest=digest(centers),
        scalars=scalars,
        sample_b64=encode_sample(centers),
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
    }
    return ProbeResult(
        probe="S5_train_test_split_index",
        digest=digest(arr),
        scalars={},
        sample_b64=encode_sample(arr.astype(np.float64)),
        meta=meta,
    )
