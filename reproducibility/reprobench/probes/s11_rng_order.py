"""S11: 乱数消費順序へのリファクタ耐性。

S0〜S10が検証してきたのは一貫して「同じコード・異なる環境」という軸
での非決定性だった。本プローブはその軸を直交させる: 環境を完全に固定
しても(同一プロセス・同一ホスト・同一seed)、**乱数の消費順序を変える
だけの、値の意味を変えないつもりのリファクタ**が最終的な数値列を壊し
うることを示す。「壊れる」ことを示すだけでなく、「どう書けば壊れない
か」(`spawn`実装)を同時に測る。

## 題材

3個の「成分」(インデックス0,1,2)それぞれについて、2つの意味のある値
`a`(primary)・`b`(secondary)と、1つの「出力に使われない値」`dead`
(leftover debug codeや未使用の検証用drawを模す。常に成分内の最後に
描画する)を乱数から引く。最終的な観測値列は常に
`[a0, b0, a1, b1, a2, b2]`という6要素のリストであり、**この値の並びと
意味はどのパターンでも変えない** — 変えるのは「乱数発生器への呼び出し
をどの順序・何回行うか」だけである(値そのものの意味を変える変更は
リファクタではないので避ける、という要件通り)。

## 2つの実装(`IMPL_NAMES`)

- `naive`: 3成分すべてが単一の`numpy.random.Generator`を使い回す
  (多くの素朴な実装はこうなる: 1個のグローバルな`rng`を使い回すか、
  関数に素通しする)。
- `spawn`: `numpy.random.SeedSequence(seed).spawn(3)`で成分ごとに
  独立したストリームを持つ`Generator`を作る。

## 4つのリファクタパターン(`REFACTOR_PATTERNS`、`baseline`に対する変更)

`baseline`: 各成分について`a = draw(); b = draw(); dead = draw()`の順で
呼ぶ処理を、1個のループで成分順(0,1,2)に行う。

- `loop_split`: 単一ループを2つに分ける。1本目のループで全成分の`a`を
  先に描画し、2本目のループで全成分の`b`→`dead`を描画する。**各成分
  自身の呼び出し順序(a→b→dead)は変えていない** — 変わるのは「他の
  成分と比べてどのタイミングで実行されるか」だけの、成分**間**の
  並び替え。
- `swap_order`: 各成分の中で`a`と`b`の呼び出し順序を入れ替える
  (`b = draw(); a = draw(); dead = draw()`)。`loop_split`とは異なり、
  成分**内**の呼び出し順序そのものを変える。
- `drop_dead_call`: 使っていない`dead`の呼び出しを消す
  (`a = draw(); b = draw()`のみ)。典型的な「デッドコード削除」の
  リファクタ。
- `early_return`: 条件分岐(成分インデックスが0かどうか)によって、
  一部の成分だけ`dead`呼び出しをスキップする。`drop_dead_call`が
  全成分に対して一律に消すのに対し、こちらは実行時の条件によって
  消費数が成分ごとに(かつ非一律に)変わる。

## 予測についての注意

`naive`がどのパターンで壊れ、`spawn`がどのパターンで生き延びるかは、
このdocstringの時点では**予測**に過ぎず、断定ではない。例えば
`loop_split`は成分間の並び替えのみなので`spawn`(成分ごとに独立した
ストリーム)は無傷だろうと予想できるが、`swap_order`は成分**内**の
呼び出し順序を変えるため`spawn`であっても同じ成分の`a`/`b`の値そのもの
が入れ替わり、生き延びないだろうと予想できる — つまり「成分ごとに
ストリームを分ける」ことは万能ではなく、各成分**自身**の内部の呼び出し
順序・回数は依然として安定させる責任が呼び出し側に残る、という点を
示すのが本プローブの狙いである。しかし予測と実測は別物であり、実測
結果は`S11_rng_consumption_order`の`meta["results"]`に記録する。
テスト(`test_s11_rng_order.py`)はこの予測のどちらが正しいかを決め打ち
しない(「頑健版は壊れない」をassertしない)。
"""
import numpy as np

from reprobench.digest import digest
from reprobench.registry import ProbeResult, register

#: 4つのリファクタパターン+参照実装`baseline`。宣言順がプローブの
#: digestの順序付きマッピングの順序そのものになる(モジュールdocstring
#: 末尾の注意、および`reprobench.digest.canonical_bytes`のlist実装が
#: 宣言順を保持することに依拠する)。
REFACTOR_PATTERNS: tuple[str, ...] = (
    "baseline",
    "loop_split",
    "swap_order",
    "drop_dead_call",
    "early_return",
)

#: 2つの実装。`naive`は単一の共有Generator、`spawn`は成分ごとに独立した
#: Generator(`SeedSequence.spawn`)。
IMPL_NAMES: tuple[str, ...] = ("naive", "spawn")

_N_COMPONENTS = 3
_SEED = 20261014


def _make_generators(impl: str, seed: int) -> list[np.random.Generator]:
    """`impl`に応じて成分ごとのGeneratorのリスト(長さ`_N_COMPONENTS`)を返す。

    `naive`は同一の`Generator`インスタンスへの参照を成分数ぶん複製した
    リストであり、全成分が物理的に同じ乱数ストリームを共有する。
    `spawn`は`SeedSequence(seed).spawn(_N_COMPONENTS)`で作った独立した
    子`SeedSequence`それぞれから`default_rng`する、成分ごとに別個の
    `Generator`インスタンスのリストである。
    """
    if impl == "naive":
        rng = np.random.default_rng(seed)
        return [rng] * _N_COMPONENTS
    if impl == "spawn":
        children = np.random.SeedSequence(seed).spawn(_N_COMPONENTS)
        return [np.random.default_rng(child) for child in children]
    raise ValueError(f"_make_generators: unknown impl {impl!r}")


def generate(impl: str, pattern: str, seed: int) -> tuple[list[float], int]:
    """(impl, pattern, seed)から観測値列と消費した乱数の個数を生成する純粋関数。

    観測値列は常に`[a0, b0, a1, b1, a2, b2]`という6要素(成分数×2)の
    `float`のリストである。`dead`(出力に使われない値)の描画も乱数を
    消費するため`n_consumed`には数えるが、観測値列そのものには含めない。
    numpy以外の重い依存には一切触れないため、モデルや外部状態なしで
    単体テストできる。
    """
    if impl not in IMPL_NAMES:
        raise ValueError(f"generate: unknown impl {impl!r}")
    if pattern not in REFACTOR_PATTERNS:
        raise ValueError(f"generate: unknown pattern {pattern!r}")

    generators = _make_generators(impl, seed)
    n_consumed = 0

    def draw(i: int) -> float:
        nonlocal n_consumed
        n_consumed += 1
        return float(generators[i].standard_normal())

    a: list[float] = [0.0] * _N_COMPONENTS
    b: list[float] = [0.0] * _N_COMPONENTS

    if pattern == "baseline":
        for i in range(_N_COMPONENTS):
            a[i] = draw(i)
            b[i] = draw(i)
            draw(i)  # dead: 出力に使われない値(leftover debug codeを模する)
    elif pattern == "loop_split":
        # 単一ループを2つに分ける: 1本目で全成分のa、2本目で全成分の
        # b→deadを描画する。成分自身の内部呼び出し順序(a→b→dead)は
        # baselineと変えていない(=成分間の並び替えのみ)。
        for i in range(_N_COMPONENTS):
            a[i] = draw(i)
        for i in range(_N_COMPONENTS):
            b[i] = draw(i)
            draw(i)  # dead
    elif pattern == "swap_order":
        # 成分の中でaとbの呼び出し順序を入れ替える(deadは最後のまま)。
        # loop_splitとは異なり、成分内の呼び出し順序そのものを変える。
        for i in range(_N_COMPONENTS):
            b[i] = draw(i)
            a[i] = draw(i)
            draw(i)  # dead
    elif pattern == "drop_dead_call":
        # 使っていないdeadの呼び出しを全成分一律に消す。
        for i in range(_N_COMPONENTS):
            a[i] = draw(i)
            b[i] = draw(i)
    elif pattern == "early_return":
        # 条件分岐(成分インデックスが0)によって、成分0だけdead呼び出し
        # を早期リターンでスキップする。drop_dead_callと異なり全成分
        # 一律ではなく、消費数が成分ごとに変わりうる。
        for i in range(_N_COMPONENTS):
            a[i] = draw(i)
            b[i] = draw(i)
            if i != 0:
                draw(i)  # dead(成分0だけスキップ)
    else:  # pragma: no cover - REFACTOR_PATTERNSの契約で到達しないはず
        raise AssertionError(f"generate: unhandled pattern {pattern!r}")

    values: list[float] = []
    for i in range(_N_COMPONENTS):
        values.append(a[i])
        values.append(b[i])
    return values, n_consumed


@register("S11_rng_consumption_order", requires=())
def _rng_consumption_order() -> ProbeResult:
    """各(impl, pattern)の組み合わせを実行し、同じimplのbaselineとの
    一致/不一致を記録する。

    比較は常に同じ実装の`baseline`パターンとの間で行う。`naive`と
    `spawn`の値そのものは比較しない(別々の乱数ストリームを使うため、
    一致する数学的な理由が無い)。
    """
    results: dict[str, dict[str, dict]] = {}

    for impl in IMPL_NAMES:
        baseline_values, _baseline_n = generate(impl, "baseline", _SEED)
        results[impl] = {}
        for pattern in REFACTOR_PATTERNS:
            values, n_consumed = generate(impl, pattern, _SEED)
            results[impl][pattern] = {
                "digest": digest(values),
                "n_consumed": n_consumed,
                "matches_baseline": values == baseline_values,
            }

    # プローブ全体のdigestは(impl, pattern) -> digest の順序付きマッピング
    # から取る。宣言順(IMPL_NAMES x REFACTOR_PATTERNS)をそのまま保持する
    # ため、dictではなくlist-of-tuplesにする(`canonical_bytes`のdict実装は
    # キーでソートし直すため、宣言順を失う。s10_batch_invariance.pyと同じ
    # 理由)。
    digest_map_ordered = [
        ((impl, pattern), results[impl][pattern]["digest"])
        for impl in IMPL_NAMES
        for pattern in REFACTOR_PATTERNS
    ]

    meta = {
        "seed": _SEED,
        "n_components": _N_COMPONENTS,
        "impl_names": list(IMPL_NAMES),
        "refactor_patterns": list(REFACTOR_PATTERNS),
        "results": results,
    }
    return ProbeResult(
        probe="S11_rng_consumption_order",
        digest=digest(digest_map_ordered),
        scalars={},
        sample_b64=None,
        meta=meta,
    )
