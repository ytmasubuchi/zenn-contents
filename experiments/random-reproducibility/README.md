# random-reproducibility

Python の DS/ML における乱数の再現性は「どのレイヤーまで」保証されるのかを、
実際にホスト上で計測した裏どり実験一式。記事本文の主張はすべてここでの実測結果
に基づく。

## 目的

`random.seed(42)` や `np.random.seed(42)`、`random_state=42` を固定しても、
以下のような境界を越えると本当に結果が変わらないのか/変わるのかを、
**print の丸めではなく float64 のビット表現の SHA256 ハッシュ**で厳密に確認する。

- 同一プロセス内・プロセス再実行 (E1)
- `PYTHONHASHSEED` 未固定による `hash()` / `set` 順序の非決定性 (E2)
- 同一 Docker イメージの別コンテナ間 (E3)
- 異なるライブラリバージョン(numpy/scikit-learn)間 (E4)
- 異なる CPU アーキテクチャ (x86_64 vs arm64, QEMU) 間 (E5)
- 同一マシン内のスレッド数・並列度の違い (E6)

## 実行方針(重要)

**すべての Python 実行は Docker コンテナ内で行う。** ホスト上で直接
`python` / `uv run` 等を実行しない。`uv.lock` の生成自体も、
`python:3.12-slim` イメージをマウントしたコンテナ内で `uv lock` を実行して
生成している(後述)。ホストで直接実行するのは `docker build` / `docker run`
などの Docker CLI コマンドのみ。

## 環境

- Host: Linux x86_64, Intel Xeon w7-3565X, Docker 29.2.1
- QEMU/binfmt: `tonistiigi/binfmt --install arm64` を実行してホストに
  arm64 の binfmt_misc ハンドラを登録済み。以後 `docker run --platform
  linux/arm64` / `docker build --platform linux/arm64` が動作する。
- GPU 実験は対象外(ドライバ不整合のため未実施)。

## 実験マトリクス

| # | 内容 | 比較対象 | スクリプト |
|---|------|----------|-----------|
| E1 | 同一プロセス内/プロセス再実行の再現性(ベースライン) | 同一イメージを3回 `docker run` | `scripts/e1_baseline.py` |
| E2 | `PYTHONHASHSEED` の影響 | 未設定3回 vs `PYTHONHASHSEED=0` 3回 | `scripts/e2_hashseed.py` |
| E3 | 同一イメージ・別コンテナインスタンス | E1と同じスクリプトを3コンテナで実行 | `scripts/e1_baseline.py`(E1と共通) |
| E4 | 同一アーキ・異なるライブラリバージョン | old image (py3.11+numpy1.26+sklearn1.3) vs new/base image (py3.12+numpy2.1+sklearn1.5) | `scripts/e4_version_stream.py` |
| E5 | 異アーキ (x86_64 vs arm64 via QEMU) | base image を `--platform linux/amd64` と `linux/arm64` で実行 | `scripts/e5_arch_compare.py` |
| E6 | 並列・スレッドによる非決定性 | `OMP_NUM_THREADS=1` vs `8`、torch thread数、sklearn `n_jobs` | `scripts/e6_parallel_nondeterminism.py` |

厳密比較のルール: すべての乱数列・モデル出力は
`numpy.ndarray.astype(np.float64).tobytes()` (または int dtype) の生バイト列を
SHA256 でハッシュ化して比較する(`scripts/_common.py::sha256_of_array`)。
`print()` した値やその丸めでは比較しない。

## ディレクトリ構成

```
experiments/random-reproducibility/
├── README.md              このファイル
├── pyproject.toml          依存関係の定義(uv管理)
├── uv.lock                 ロックファイル(コンテナ内で `uv lock` して生成)
├── docker/
│   ├── Dockerfile.base     python:3.12-slim + numpy2.1.3/scipy1.14.1/sklearn1.5.2/torch2.5.1(cpu)
│   │                       -- E1,E2,E3,E6 と E4の"new"側、E5のamd64/arm64両方に使用
│   └── Dockerfile.old      python:3.11-slim + numpy1.26.4/scipy1.11.4/sklearn1.3.2
│                           -- E4の"old"側専用
├── scripts/
│   ├── _common.py          共通ヘルパー(SHA256ハッシュ、メタデータ収集)
│   ├── compare.py           2つの結果JSONを再帰的に比較し一致/不一致を報告するツール
│   ├── e1_baseline.py       E1(およびE3で再利用)
│   ├── e2_hashseed.py       E2
│   ├── e4_version_stream.py E4
│   ├── e5_arch_compare.py   E5
│   └── e6_parallel_nondeterminism.py E6
├── run_all.sh              全実験を実行し results/ に保存する
└── results/                 実測結果(JSON)。本リポジトリに実際にコミットされた実測データ
```

## 再現手順

前提: Docker がホストで動作していること。arm64 QEMU が未セットアップの場合は
自動インストールを試みる(`docker run --privileged --rm tonistiigi/binfmt
--install arm64`)。失敗した場合 E5 はスキップされる。

```bash
cd experiments/random-reproducibility
./run_all.sh
```

`results/` に生JSON(`e*_*.json`)と比較レポート(`e*_compare*.json`)が出力される。

`uv.lock` を再生成する場合(ホストで直接 uv/python を実行しないため、
コンテナ内で実行する):

```bash
docker run --rm -v "$PWD":/work -w /work python:3.12-slim \
  bash -c "pip install -q uv==0.5.11 && uv lock"
```

## 実測結果サマリ

実行環境: Linux x86_64 (Intel Xeon w7-3565X), Docker 29.2.1、
Python 3.12.13 (base image) / 3.11.15 (old image)、
numpy 2.1.3 (base) / 1.26.4 (old)、scikit-learn 1.5.2 (base) / 1.3.2 (old)、
torch 2.5.1+cpu。

### E1: 同一プロセス内・プロセス再実行の再現性

`e1_baseline.py` を同一イメージから3回、別コンテナとして実行
(`results/e1_run1.json` 〜 `e1_run3.json`)。`stdlib random`、
`numpy.random.RandomState` (MT19937)、`np.random.seed()` によるグローバル
レガシー状態、`np.random.default_rng()` (PCG64)、`torch.manual_seed`
(CPU)、および `train_test_split` + `RandomForestClassifier` の予測値・
`feature_importances_` まで含めた **全15項目のSHA256ハッシュが3回とも完全
一致**(`results/e1_compare_1v2.json`, `e1_compare_1v3.json`: `n_mismatches:
0`)。固定シードでの再現性はベースラインとして完全に成立する。

### E2: PYTHONHASHSEED の影響

`e2_hashseed.py` を `PYTHONHASHSEED` 未設定で3回、`PYTHONHASHSEED=0` 固定
で3回実行(`results/e2_unset_run*.json`, `e2_seed0_run*.json`)。

- 未設定: `hash("hello")` は3回とも異なる値(実測:
  `-3694237083992383962` → `4393989760295717189` → `449626946617816297`)。
  `list(set(feature_columns))` の順序も毎回異なり、元のカラム順と一致しな
  かった(`order_matches_original: false` が3回とも)。これが
  「setを経由した特徴量カラム順が実行ごとに変わる」DSあるあるの実体。
- `PYTHONHASHSEED=0` 固定: `hash("hello")` は3回とも
  `-2096571579003691106` で完全一致、`set` 経由の順序ハッシュも3回とも
  一致。

### E3: 同一イメージ・別コンテナインスタンス

E1と同じスクリプトを3つの独立した `docker run`(別コンテナ、同一イメージ
`random-repro:base-amd64`)で実行(`results/e3_container1.json` 〜
`container3.json`)。**全15項目が完全一致**
(`e3_compare_1v2.json`, `e3_compare_1v3.json`: `n_mismatches: 0`)。
コンテナの再起動そのものは乱数再現性に影響しない。

### E4: 同一アーキ・異なるライブラリバージョン

`e4_version_stream.py` を old image (numpy 1.26.4 / sklearn 1.3.2) と new
image (numpy 2.1.3 / sklearn 1.5.2) で実行し比較
(`results/e4_compare.json`: `n_matches: 12`, `n_mismatches: 2`)。

- **一致**: `RandomState` (MT19937) の raw uint32 stream / uniform /
  normal、および `Generator` (PCG64) の raw int64 / uniform / normal ―
  numpy のメジャーバージョンを跨いでも **ビット単位で完全一致**。
- **一致**: `RandomForestClassifier(random_state=42)` の予測値 **および**
  `feature_importances_` ― sklearn 1.3.2 → 1.5.2 でも完全一致。
- **不一致**: `LogisticRegression(random_state=42)` の `coef_` のみ不一致
  (予測クラスラベル自体は一致するが係数の浮動小数点値が変わる ―
  ソルバーの収束経路がバージョン間で変化したため)。

### E5: 異アーキテクチャ (x86_64 vs arm64, QEMU)

`e5_arch_compare.py` を `random-repro:base-amd64`(`--platform
linux/amd64`)と `random-repro:base-arm64`(`--platform linux/arm64`、QEMU
エミュレーション)で実行し比較(`results/e5_compare.json`: `n_matches: 16`,
`n_mismatches: 7`)。

**一致した項目**:
- `RandomState` (MT19937) の raw int / uniform / normal stream ― 完全一致。
- `Generator` (PCG64) の raw int / uniform / normal stream ― 完全一致。
- `np.sin` ― 完全一致(意外にも一致した。下記「予想外だった結果」参照)。
- 2000万要素の `np.sum` リダクション ― 完全一致。
- `RandomForestClassifier` / `LogisticRegression` の **predictions**(クラ
  スラベル)― 完全一致。
- `RandomForestClassifier` の学習に使う `sklearn_version` 文字列、
  torch の `final_loss_hex`(最終ステップの損失値そのもの)、
  `num_threads`(=1指定)― 完全一致。

**一致しなかった項目**:
- `np.exp` ― 不一致(超越関数の実装がアーキ依存)。
- 512×512 行列積(BLAS `matmul`)― 不一致。
- `RandomForestClassifier.feature_importances_` ― predictions は一致した
  にもかかわらず重要度の浮動小数点値は不一致(木の分割で使う不純度計算
  が BLAS/libm 由来の微小な誤差の影響を受けたと考えられる)。
- `LogisticRegression.coef_` ― 不一致。
- torch MLP の `loss_curve_sha256`(全50ステップの損失列)と
  `final_params_sha256`(最終パラメータ)― 不一致。
- `torch.__version__` 文字列 ― amd64 は `2.5.1+cpu`、arm64 は `2.5.1`
  (aarch64 用の PyPI ホイールには `+cpu` ローカルバージョンタグが付かない
  ため。**計算結果の差ではなく単なるパッケージング上の表記差**)。

QEMU(arm64 エミュレーション)でのフルスクリプト実行は約100秒
(x86_64ネイティブでは数秒)。

### E6: 並列・スレッドによる非決定性

`e6_parallel_nondeterminism.py` を `OMP_NUM_THREADS=1` と `8` で実行し比較
(`results/e6_compare_omp.json`: `n_matches: 34`, `n_mismatches: 2` ―
不一致はスレッド数の環境変数値そのものと `threadpool_info` のみで、
**計算結果自体は一致**)。

- **非結合性の実証**: `(1.0 + 1e16) + (-1e16) = 0.0` だが
  `1.0 + (1e16 + (-1e16)) = 1.0` ― 3値の最小例で完全に異なる結果。
  20万要素の配列でも forward sum / reversed sum / naive sequential sum が
  それぞれ異なるビットパターンになった。
- **BLAS thread数**: `OMP_NUM_THREADS=1` → `8` で `threadpoolctl` 上は
  実際に OpenBLAS のスレッド数が 1→8 に変化したことを確認したが、
  1024×1024 行列積と 2000万要素の `np.sum` の結果は **ビット単位で一致し
  た**(想定と異なる ― 詳細は「予想外だった結果」を参照)。
- **torch CPU thread数**: `torch.set_num_threads(1)` vs `8` でも、
  この規模の小さいMLP(10→16→1、200サンプル、50 step)では損失曲線の
  ハッシュが完全一致した。
- **sklearn `n_jobs`**: `n_jobs=1` vs `n_jobs=-1` の
  `RandomForestClassifier(random_state=42)` は予測値・
  `feature_importances_` ともに完全一致 ― 各木に決定的にシードが配布され
  る実装のため、想定どおり再現した。

## 予想外だった結果

1. **E4: RandomForest がバージョンを跨いで完全一致した。**
   sklearn の実装はマイナー/メジャーバージョン間で分割探索のロジックが
   変わることがあると想定していたが、`random_state=42` 固定・
   `n_estimators=100` で predictions と feature_importances_ まで
   ビット完全一致した。一方 `LogisticRegression` は係数が変化した
   (ソルバーの収束路の違い)。「木ベースモデルは安定、勾配ベースの
   最適化を伴うモデルは不安定」という非対称性が実測で確認できた。
2. **E6: BLAS/torch のスレッド数を変えても、この規模では結果が変わらな
   かった。** `threadpoolctl` で実際にスレッド数が変化したことを確認し
   た上でなお、1024×1024 の `np.sum`/行列積、および小さいMLPの学習では
   ビット単位で結果が一致した。「スレッド数を変えれば浮動小数点結果は
   変わりうる」は一般には真だが、**すべての演算サイズで必ず変わるわけ
   ではない**(reduction/GEMM の実装がスレッド数に関わらず同じ演算順序
   にfall backするサイズがある)ことが実測できた。記事では「変わりうる
   が保証された非決定性ではない」という書き方に留める。
3. **numpy の RandomState/Generator の raw stream は、メジャーバージョン
   (1.26→2.1)を跨いでも完全に一致した。** 期待どおりではあるが、
   `Generator`(PCG64)側は仕様上ストリーム安定性の「保証」はないため、
   実測で確認できたのは価値がある。
4. **E5: `np.sin` はアーキ間で完全一致したが `np.exp` は不一致だった。**
   同じ「超越関数」でも、libm の実装がSIMD経路に落ちるかどうかや
   glibc内部のアルゴリズム選択によって、アーキ間の一致/不一致が関数ごと
   に変わることが実測できた。「超越関数はアーキ間で一致しない」と一括り
   にはできない。
5. **E5: RandomForest の predictions は一致したが feature_importances_ は
   不一致だった。** クラス分類という離散的な出力は微小な浮動小数点差の
   影響を受けにくく吸収してしまうが、連続値である重要度スコアはその差が
   そのまま表面化する。「モデルの最終出力(離散)」と「モデルの内部状態
   (連続値)」で再現性の頑健さが異なることが実測で確認できた。
6. **E5: torch MLPは学習過程(loss_curve)は不一致だが最終lossの値
   (final_loss_hex)だけは一致した。** 50ステップの学習を通して積算された
   誤差が、たまたま最後のステップで同じfloat32表現に収束したケース。
   1点だけの比較では「再現した」と誤認しかねないことを示す実例であり、
   本記事で「複数指標・複数ステップを見ないと再現性の評価を誤る」という
   注意点の根拠にしている。

## 実行できなかった項目・理由

- GPU 関連の実験はすべて対象外(ホストのGPUドライバ不整合により実行不可、
  タスク要件でも明示的に除外)。
- 上記以外はすべて実行済み。E5 (arm64/QEMU) を含め、E1〜E6 すべてが完走し
  `results/` に実測JSONが残っている。
