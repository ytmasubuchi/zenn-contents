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
- E1〜E6(CPU実験)の時点では GPU 実験は対象外だった(ホストのGPUドライバ
  不整合のため未実施)。後日、別ホストで GPU (NVIDIA RTX 4090) が使える
  環境が用意されたため、E8 として GPU 実験を追加した。詳細は
  「GPU実験(E8系)」の節を参照。

## 実験マトリクス

| # | 内容 | 比較対象 | スクリプト |
|---|------|----------|-----------|
| E1 | 同一プロセス内/プロセス再実行の再現性(ベースライン) | 同一イメージを3回 `docker run` | `scripts/e1_baseline.py` |
| E2 | `PYTHONHASHSEED` の影響 | 未設定3回 vs `PYTHONHASHSEED=0` 3回 | `scripts/e2_hashseed.py` |
| E3 | 同一イメージ・別コンテナインスタンス | E1と同じスクリプトを3コンテナで実行 | `scripts/e1_baseline.py`(E1と共通) |
| E4 | 同一アーキ・異なるライブラリバージョン | old image (py3.11+numpy1.26+sklearn1.3) vs new/base image (py3.12+numpy2.1+sklearn1.5) | `scripts/e4_version_stream.py` |
| E5 | 異アーキ (x86_64 vs arm64 via QEMU) | base image を `--platform linux/amd64` と `linux/arm64` で実行 | `scripts/e5_arch_compare.py` |
| E6 | 並列・スレッドによる非決定性 | `OMP_NUM_THREADS=1` vs `8`、torch thread数、sklearn `n_jobs` | `scripts/e6_parallel_nondeterminism.py` |

GPU実験(E8系、CUDA環境のみで実行、詳細は下の節):

| # | 内容 | 比較対象 | スクリプト |
|---|------|----------|-----------|
| E8a | GPU反復再現性(デフォルト設定) | 同一イメージを3回 `docker run --gpus all` | `scripts/e8a_gpu_repeat.py` |
| E8b | GPU反復再現性(`use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG`) | E8aと同じ測定を決定的モードで3回 | `scripts/e8b_gpu_deterministic.py` |
| E8c | CPU vs GPU(同一シード・同一入力) | 同一プロセス内でCPU側とGPU側を計算し比較 | `scripts/e8c_cpu_vs_gpu.py` |
| E8d | TF32 有効/無効の影響 | 同一入力の行列積をTF32 on/offで計算し比較 | `scripts/e8d_tf32.py` |
| E8e | E8a〜E8dの不一致項目のULP/絶対誤差/相対誤差 | E7と同形式のdiff計算 | `scripts/e8e_ulp_diff.py` |

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
│   ├── e6_parallel_nondeterminism.py E6
│   ├── e8a_gpu_repeat.py    E8a(デフォルト設定でのGPU反復再現性)
│   ├── e8b_gpu_deterministic.py E8b(deterministicモードでのGPU反復再現性)
│   ├── e8c_cpu_vs_gpu.py    E8c(CPU vs GPU比較)
│   ├── e8d_tf32.py          E8d(TF32有効/無効の影響)
│   └── e8e_ulp_diff.py      E8e(E8a〜E8dの不一致項目のULP/誤差diff)
├── run_all.sh              CPU実験(E1〜E6)を実行し results/ に保存する
├── run_gpu.sh              GPU実験(E8系)を実行し results/ に保存する(`docker run --gpus all` が必要)
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

GPU実験(E8系)を再現する場合は、NVIDIA GPU + `nvidia-container-runtime` が
使えるホストで以下を実行する(`docker run --gpus all` が必要。ホストの
デフォルト runtime が `runc` のままでも `--gpus all` を明示すれば動く):

```bash
docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
cd experiments/random-reproducibility
./run_gpu.sh
```

`results/e8*.json` と `results/e8_raw/` (ULP diff用の生配列 `.npz`) が出力される。

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

## GPU実験(E8系)

記事本文の7章「GPUの世界(理論編)」は、執筆時にGPUが使えず公式ドキュメント
の裏どりのみだった。後日、別ホストで実機のGPUが使える環境が用意されたため、
E8として実測を追加した。

### 検証環境

- GPU: NVIDIA GeForce RTX 4090(Ada Lovelace, compute capability 8.9)
- ドライバ: 580.173.02(`nvidia-smi` 表示の CUDA Version: 13.0)
- Docker: nvidia runtime あり(デフォルト runtime は `runc` のため、すべての
  `docker run` に `--gpus all` を明示)
- イメージ: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`
  (torch 2.5.1+cu124、CPU実験と同じ torch メジャー/マイナーバージョン)
- コンテナ内で確認できた値: `torch.version.cuda = "12.4"`、
  `torch.backends.cudnn.version() = 90100`(cuDNN 9.1.0)、
  `torch.backends.cuda.matmul.allow_tf32` のデフォルトは **`False`**、
  `torch.backends.cudnn.allow_tf32` のデフォルトは **`True`**
  (同じ「TF32」でも matmul 経路と cuDNN 経路でデフォルトが違う。
  `torch.backends.cudnn.benchmark` / `torch.backends.cudnn.deterministic`
  のデフォルトはいずれも `False`)。
- **重要な制約**: このホストのGPUは常駐の vLLM サーバとVRAMを共有しており、
  実験開始時点で 24564 MiB 中 23768 MiB が使用済み、空きはわずか **305 MiB**
  だった。既存プロセスは一切停止せず、この空きVRAMの範囲内で実験する方針
  で臨んだ。

### 実測結果: GPU実験はすべてCUDA OOMでブロックされた

結論から言うと、E8a〜E8dのGPU計算を伴う測定は**1つも成功しなかった**。
原因はドライバやツールキットの不整合ではなく、**空きVRAMが少なすぎて
CUDAコンテキストそのものが作れない**という、実行前に確認した通りの
資源枯渇だった。

根拠(`results/e8_environment_probe.json` に詳細を記録):

1. PyTorchを使わない素のCUDA Cプログラム(`nvidia/cuda:13.0.0-devel`
   イメージでビルド)で `cudaSetDevice(0)` → `cudaMemGetInfo` →
   `cudaMalloc(1MB)` を実行しても、**`cudaSetDevice` の時点で
   `cudaErrorMemoryAllocation`(code=2, "out of memory")** が返る。
   3回試行してすべて同じ結果。PyTorch固有の問題ではなく、CUDAドライバの
   レベルで新規コンテキストを作る余地がないことが確認できた。
2. PyTorch側でも、`torch.cuda.is_available()` は `True` を返す(デバイスの
   可視性を見るだけなので軽量)一方、`torch.zeros(1, device='cuda')`
   のような **1要素だけのテンソル確保ですら** `RuntimeError: CUDA error:
   out of memory` になる。`torch.cuda.is_available() == True` は
   「GPU計算が実際に走る」ことの十分条件ではない、という実務上の教訓。
3. `run_with_fallback`(`scripts/_common.py`)により、512×512 → 128×128 →
   32×32 → 8×8(matmulの場合)のようにサイズを段階的に下げて再試行したが、
   最小サイズでも同一のエラーで失敗した。テンソルサイズの問題ではなく、
   コンテキスト生成そのものが通らない以上、どれだけ小さくしても解決しない。

この失敗自体は3コンテナ×2設定(E8a/E8b)すべてで完全に再現した
(`results/e8a_compare_1v2.json` / `e8a_compare_1v3.json` /
`e8b_compare_1v2.json` / `e8b_compare_1v3.json`: いずれも `n_mismatches: 0`
― エラーメッセージ文字列まで含めて全項目が一致)。つまり**「失敗する」
という結果自体は完全に再現性があった**。

一方で、CUDAコンテキストや確保を必要としない軽量なメタデータ取得
(GPU名、compute capability、cuDNNバージョン、TF32/cudnn.benchmark/
cudnn.deterministic の各デフォルト値、`nvidia-smi` 経由のVRAM使用量)は
すべてのJSON (`results/e8a_run*.json` 等の `metadata.gpu`) に正しく記録
できている。

### 実行できなかった項目

- E8a: CUDA上のPhilox rand/randn、512×512 matmul(cuBLAS)、Conv2d
  forward/backward(cuDNN)、MLP学習、`index_add_`/`scatter_add_` の
  atomicAdd非決定性(同一プロセス内20回反復)、`embedding_bag`、`cumsum`
  ― すべてCUDA OOMで未実行(エラーメッセージは記録済み)。
- E8b: 上記と同じ測定を `use_deterministic_algorithms(True)` +
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` で試みたが、同じくCUDA OOMで未実行。
  そのため「`index_add_` のCUDA実装がdeterministicモードでエラーを送出
  するか」という記事の主張自体は、**この環境では検証できなかった**
  (エラーは出たが、それはOOMであってdeterminism起因のエラーではない)。
- E8c: CPU側の値(`torch.rand`、512×512 matmul、MLP学習)は正常に取得
  できたが、GPU側がすべてOOMのため比較不能。
- E8d: TF32のデフォルト値そのものは記録できたが、on/offでの実際の行列積
  比較・誤差実測はGPU側がOOMのため不可能。
- E8e: 上記の理由により、diffを取る生データが存在せず
  (`results/e8_ulp_diff.json` は全項目 `"reason": "insufficient_data"`)。

### 予想外だった点(記事の素材として)

1. **「GPUドライバの不整合」だった当初の障壁が解消しても、別の障壁
   (VRAM枯渇)に置き換わっただけで、結局GPU実験は今回も実行できなかっ
   た。** GPU上の再現性の議論以前に、「共有GPU環境ではそもそも実験が
   走らない」という、地味だが実務でよく遭遇する制約こそが最初のハード
   ルだった、という点は理論編には出てこない実測ならではの発見。
2. **`torch.cuda.is_available()` が `True` を返しても実際にGPU計算が
   できるとは限らない。** 空きVRAMが極端に少ないと、デバイスの可視性
   チェックは通るのに、最小の1要素テンソル確保でもCUDAコンテキスト生成
   自体が失敗する。ヘルスチェックとして `is_available()` だけを見るのは
   不十分。
3. **TF32のデフォルト値は `torch.backends.cuda.matmul.allow_tf32`
   (`False`)と `torch.backends.cudnn.allow_tf32`(`True`)で異なる**
   (torch 2.5.1実測)。「TF32は既定でオンだったかオフだったか」という
   問いに単一の答えはなく、経路(cuBLAS matmul経由かcuDNN経由か)ごとに
   確認する必要がある。
4. **失敗そのものが完全に再現した。** 3つの独立したコンテナで、同じ
   エラーメッセージ文字列まで含めてビット単位で一致(`n_mismatches: 0`)
   ― 「常に同じ理由で失敗する」こと自体も一種の再現性であり、
   `compare.py` の仕組みがそのまま使えた。

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

- E1〜E6(CPU実験)はすべて実行済み。E5 (arm64/QEMU) を含め、E1〜E6 すべてが
  完走し `results/` に実測JSONが残っている。
- GPU実験(E8系)は、当初(この節を最初に書いた時点)はホストのGPUドライバ
  不整合により実行対象外だった。後日、別ホストでGPUが使える環境が用意され
  スクリプト自体は実装・実行したが、**そのホストのGPUが常駐の別プロセス
  (vLLM)とVRAMを共有しており空きVRAMが約305MiBしかなかったため、CUDA
  コンテキストの生成自体が失敗し、GPU計算を伴う測定はすべて未実行に終わっ
  た**。詳細・根拠は「GPU実験(E8系)」の節および
  `results/e8_environment_probe.json` を参照。GPUメタデータ(GPU名・
  cuDNNバージョン・TF32デフォルト値など)自体は正常に取得できている。
