"""S6: LightGBMの予測値と木構造そのもののスレッド数依存性。

重い依存(`lightgbm`)はalpine等に無いことがあるため、関数本体の内部で
importする(モジュール先頭には置かない。パッケージdocstring参照)。

データセットはS5(`s5_sklearn.py`)と**同じ生成式**(`default_rng(20261014)`、
20,000サンプル×30特徴量)を使うが、コードはこのモジュール内に複製して
自己完結させる。`s5_sklearn`から関数をimportしないのは、S6が宣言する
`requires`は`("lightgbm",)`だけであり、万一sklearnが欠けた環境では
train/testの分割にsklearnの`train_test_split`を使ってしまうと、本来
「lightgbm不在」の設計通りスキップであるべきものが「sklearn不在」による
`raised: ...`(バグの疑いがある側)に化けてしまうため。分割自体も
`default_rng`のpermutationのみで行い、sklearnに一切依存しない。

`num_threads`は環境変数`REPRO_THREADS`に従わせる。意図的に
`"deterministic": True`(LightGBMのヒストグラム構築を強制的に決定的にする
オプション)は**設定しない**——それを設定すると、まさに測定したい
「スレッド数でヒストグラム構築の浮動小数点加算順序が変わるか」という
効果そのものを握り潰してしまう。測定したい非決定性を仮説に合わせて
消してしまわないため。
"""
import json
import os

import numpy as np

from reprobench.digest import digest, encode_sample
from reprobench.registry import ProbeResult, register

_SEED = 20261014
_N_SAMPLES = 20_000
_N_FEATURES = 30
_TEST_SIZE = 0.2
_NUM_BOOST_ROUND = 50
_NUM_LEAVES = 31
_LEARNING_RATE = 0.1


def _num_threads() -> int:
    raw = os.environ.get("REPRO_THREADS")
    return int(raw) if raw else 1


def _dataset() -> tuple[np.ndarray, np.ndarray]:
    # S5(`s5_sklearn._dataset`)と同じ生成式。コードを複製する理由は
    # モジュールdocstring参照。
    rng = np.random.default_rng(_SEED)
    X = rng.standard_normal((_N_SAMPLES, _N_FEATURES))
    true_coef = rng.standard_normal(_N_FEATURES)
    logits = X @ true_coef
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=_N_SAMPLES) < probs).astype(np.int64)
    return X, y


def _split(y: np.ndarray):
    # sklearnのtrain_test_splitには依存しない(モジュールdocstring参照)。
    # 同一seedのpermutationで先頭をtestに切り出すだけの素朴な分割。
    rng = np.random.default_rng(_SEED)
    shuffled = rng.permutation(len(y))
    n_test = int(round(len(y) * _TEST_SIZE))
    test_idx = shuffled[:n_test]
    train_idx = shuffled[n_test:]
    return train_idx, test_idx


def _params(num_threads: int) -> dict:
    return {
        "objective": "binary",
        "num_leaves": _NUM_LEAVES,
        "learning_rate": _LEARNING_RATE,
        "seed": _SEED,
        "num_threads": num_threads,
        "verbosity": -1,
    }


def _train():
    import lightgbm as lgb

    X, y = _dataset()
    train_idx, test_idx = _split(y)
    num_threads = _num_threads()
    params = _params(num_threads)
    dtrain = lgb.Dataset(X[train_idx], label=y[train_idx])
    booster = lgb.train(params, dtrain, num_boost_round=_NUM_BOOST_ROUND)
    return booster, X, test_idx, params


@register("S6_lgbm_pred", requires=("lightgbm",))
def _lgbm_pred() -> ProbeResult:
    booster, X, test_idx, params = _train()
    pred = booster.predict(X[test_idx]).astype(np.float64)

    scalars = {"mean_pred": float(pred.mean())}
    meta = {
        "seed": _SEED,
        "num_threads": params["num_threads"],
        "repro_threads_env": os.environ.get("REPRO_THREADS"),
        "num_boost_round": _NUM_BOOST_ROUND,
        "num_leaves": _NUM_LEAVES,
        "learning_rate": _LEARNING_RATE,
        "n_test": len(test_idx),
        "mean_pred_hex": scalars["mean_pred"].hex(),
    }
    return ProbeResult(
        probe="S6_lgbm_pred",
        digest=digest(pred),
        scalars=scalars,
        sample_b64=encode_sample(pred),
        meta=meta,
    )


@register("S6_lgbm_tree_structure", requires=("lightgbm",))
def _lgbm_tree_structure() -> ProbeResult:
    booster, _X, test_idx, params = _train()
    model_json = booster.dump_model()
    # dict のキー順はJSON化の実装都合で揺れうる(本物の木構造の差では
    # ない)ため、sort_keys=Trueで正規化してからダイジェストを取る。
    # リスト(子ノード等)の順序はそのまま——あれはノード配置という
    # 本物の構造情報であり、潰してはいけない。
    canonical = json.dumps(model_json, sort_keys=True)

    meta = {
        "seed": _SEED,
        "num_threads": params["num_threads"],
        "repro_threads_env": os.environ.get("REPRO_THREADS"),
        "num_boost_round": _NUM_BOOST_ROUND,
        "num_leaves": _NUM_LEAVES,
        "learning_rate": _LEARNING_RATE,
        "num_trees": booster.num_trees(),
        "normalization": "json.dumps(sort_keys=True)",
    }
    return ProbeResult(
        probe="S6_lgbm_tree_structure",
        digest=digest(canonical),
        scalars={},
        sample_b64=None,
        meta=meta,
    )
