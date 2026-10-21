# 実務4軸(S11〜S15)の予測 — 事前登録

記録日時: 2026-10-21
**本ファイルは実測(Task 7)を走らせる前にコミットされる。** 予測の事前性は git のコミット履歴(タイムスタンプ)で担保する。

対象: `reproducibility/results/*__*__practice.json`(config `practice`、S11〜S15 を走らせる)。
GPU実測環境: `torch==2.14.1+cu130`(`docker/Dockerfile.gpu` に固定)。

## 経緯: 計画上は3点を事前登録する予定だったが、1点を撤回した

計画(`docs/superpowers/plans/2026-10-21-reproducibility-article-revision.md` Task 6)は当初、
以下3点を事前登録するとしていた。

1. `SeedSequence.spawn()` を使った頑健版(S11)が、リファクタ4パターン全てを生き延びるか
2. `cudnn.benchmark=True`(S12)が同一プロセス内の反復で実際に揺れるか
3. `CUBLAS_WORKSPACE_CONFIG`(S13)未設定時に例外が出る演算の範囲

**予測1(S11)は統括者の裁定により撤回した。** S11 は CPU だけで動くプローブであり、
Task 2 で実装してテストを走らせた時点で答えが判明している(実測済み、以下にそのまま記録する)。

```
naive : baseline=一致 / loop_split=不一致 / swap_order=不一致 / drop_dead_call=不一致 / early_return=不一致
spawn : baseline=一致 / loop_split=一致   / swap_order=不一致 / drop_dead_call=一致   / early_return=一致
```

既に答えを知っているものを「予測」と称して事前登録するのは、事前登録という手続きの意味を失わせる。
これは計画のタスク順序(Task 2-5で実装 → Task 6で予測 → Task 7で実測)を設計した統括者自身の欠陥であり、
実装者や記事執筆者の責任ではない。**この制約(CPU専用プローブは原理的に事前登録できない)自体が、
方法論上の論点として記事に値する。** 事前登録という語は「結果を知らない状態で予測をコミットし、
あとから答え合わせをする」ことで初めて意味を持つ。実装とテストが同一マシンで完結するプローブは、
実装者が(たとえ記録に残していなくても)実行した瞬間に答えを得てしまうため、後から「予測」を
書いても事後確証にしかならない。これは構造的な制約であり、個々の几帳面さの問題ではない。

同じ理由で **S14(DataLoader worker seeding)の結果も事前登録しない。** S14 も CPU だけで動く
(torch・numpyともにCUDA不要)。加えて、Task 3 で実装者が実測済みであり、かつその挙動(torchは
workerごとに`np.random.seed`も再シードする一方、fork前に保持したnumpyの`Generator`インスタンスは
再シードされない)は PyTorch 公式ドキュメント「Randomness in multi-process data loading」から
文書上予想がつく範囲でもある。実測結果(そのまま記録する):

```
torch : workers 0/2/4 とも duplicate_sample_count=0
numpy : workers 0 で 0、2 で 8、4 で 12 (n_samples=16)
```

**事前登録しない他の2点(自明なので記事にもその理由を明記する):**
- 素朴版(S11 naive)がリファクタで壊れること — 単一Generatorの共有ストリームを前提にした時点で
  消費順序への依存は避けられず、結果は文書から予想がつく。
- `S15_index_add_repeats` が非決定であること — `index_add_`のCUDA実装がatomicAddを使うことは
  PyTorch公式ドキュメントが明記する既知の事実であり、本プロジェクト自身が既存の`S8_index_add_atomics`
  で実測済みである。

残るのが、**GPU必須であり実装環境(Mac、CUDA無し)では`SkippedProbe`になるため、
執筆時点で誰も答えを知らない**以下の2点である。これが本ファイルの事前登録の対象である。

---

## 予測A: S12 `cudnn.benchmark` × `cudnn.deterministic` のプロセス内安定性

### 題材(`reprobench/probes/s12_cudnn.py`)

2層のCNN(Conv2d×2)の順伝播・逆伝播を、`(benchmark, deterministic)` の2×2グリッド
(`CUDNN_SETTINGS`)それぞれについて、同一プロセス内で`_N_ITERS=3`回反復する。
入力・重み・targetは各設定内で固定(iterごとに再生成しない)。
各設定について、3回の反復の勾配digestが全て一致するかを`stable_within_process`として記録する。

### 機構の仮説(根拠: PyTorch公式ドキュメント)

PyTorch公式の reproducibility notes(`https://docs.pytorch.org/docs/stable/notes/randomness.html`、
「CUDA convolution benchmarking」「CUDA convolution determinism」の2節)から、原文を引用する。

> "When a cuDNN convolution is called with a new set of size parameters, an optional feature
> can run multiple convolution algorithms, benchmarking them to find the fastest one. **Then,
> the fastest algorithm will be used consistently during the rest of the process for the
> corresponding set of size parameters.**"

> "While disabling CUDA convolution benchmarking (discussed above) ensures that CUDA selects
> the same algorithm each time an application is run, **that algorithm itself may be
> nondeterministic, unless either `torch.use_deterministic_algorithms(True)` or
> `torch.backends.cudnn.deterministic = True` is set.**"

この2つの原文から導かれる機構は、**S12プローブ自身のdocstringが暗に前提としている
「`benchmark=True`が揺れの主因」という読みとは異なる。** 要点は以下の通り。

1. `benchmark`は「どのアルゴリズムを選ぶか」を制御するが、**同一プロセス内では、
   同じ入力サイズに対して一度選んだアルゴリズムを使い回す**(上記引用の太字部分)。
   S12は`_N_ITERS`回とも同じ`_BATCH`/`_SPATIAL`/`_KERNEL_SIZE`を使うため、`benchmark=True`で
   あっても2回目以降のタイミング再計測は原理的に発生しない。したがって「`benchmark`が
   プロセス内反復のたびにアルゴリズムを選び直すから揺れる」という説明は機構として誤りである。
2. 揺れの実際の原因は**選ばれたアルゴリズム自体が非決定的かどうか**であり、これを制御するのは
   `deterministic`フラグである(上記2つ目の引用)。`deterministic=False`では、`benchmark`の
   値に関わらず、cuDNNがスレッド実行順依存(atomicAddなど)の非決定的な畳み込みアルゴリズムを
   選びうる。`deterministic=True`はこれを禁止する。

### 予測(実測前に記録)

| benchmark | deterministic | `stable_within_process` の予測 |
|---|---|---|
| False | False | **不安定** |
| False | True  | **安定** |
| True  | False | **不安定** |
| True  | True  | **安定** |

すなわち、**安定性を決めるのは`deterministic`であって`benchmark`ではない**、というのが
本予測の核心である。`benchmark`の値(2設定)は`deterministic=True`側では両方とも安定、
`deterministic=False`側では両方とも不安定になると予測する。

### 外れた場合に何が言えるか

- **`(False, False)`または`(True, False)`が安定していた場合**: このプローブの畳み込み形状
  (`_BATCH=32, _IN_CHANNELS=3, _HIDDEN_CHANNELS=16, _KERNEL_SIZE=3, _SPATIAL=16`)に対して、
  cuDNNが(たまたま)非atomicAddベースの決定的な実装を選んでいることを意味する。これは
  「`deterministic=False`は必ず非決定的アルゴリズムを選ぶ」という読みが誤りで、**非決定性は
  入力形状・GPUアーキテクチャに条件付けられる**(本プロジェクトのS5 kmeans/S8 index_add_が
  既に示している構造と同型)ことの新たな実例になる。
- **`(False, True)`または`(True, True)`が不安定だった場合**: これは単なる誤差ではなく、
  **PyTorch公式ドキュメントの主張そのものが実測で崩れる**という、本記事にとって最重量級の結果に
  なる。`torch.backends.cudnn.deterministic = True`が謳い文句通りの保証を提供しない具体例として、
  独立した節を割いて報告する価値がある。
- **`benchmark`の値だけで安定性が分かれた場合**(例: `deterministic`を揃えても`benchmark=True`側だけ
  不安定): 本予測の核心(「安定性を決めるのは`benchmark`ではなく`deterministic`」)が外れたことになる。
  この場合は「同一サイズでも`benchmark=True`は反復ごとに再ベンチマークしうる」という、
  公式ドキュメントの説明(プロセス内でキャッシュされる)と矛盾する挙動が観測されたことを意味し、
  それ自体が報告に値する発見になる。

---

## 予測B: S13 `CUBLAS_WORKSPACE_CONFIG` 未設定時に例外が出る演算の範囲

### 題材(`reprobench/probes/s13_cublas_workspace.py`)

`torch.use_deterministic_algorithms(True)`の下で、`CUBLAS_OPS = ("mm", "bmm", "addmm",
"linear_forward_backward")`の4演算を順に実行する。config `practice`は`CUBLAS_WORKSPACE_CONFIG`を
意図的に設定しない。演算ごとに例外の有無を`ops[op_name]`に記録する(例外クラス名とメッセージ、
または成功時はdigest)。

S13モジュール自身のdocstringは「PyTorch公式ドキュメントが、決定的アルゴリズムを有効にした状態で
CUDA上の一部のcuBLAS演算を使うには`CUBLAS_WORKSPACE_CONFIG`の設定が必要で、未設定だと
`RuntimeError`が送出されると述べている」と書いている。**しかし、この記述は現在の実測環境
(`torch==2.14.1+cu130`)には当てはまらない可能性が高い。** 以下、一次資料で確認した内容を示す。

### 機構の調査(一次資料: PyTorchのC++ソースとGitHubのPR)

`CUBLAS_WORKSPACE_CONFIG`が未設定時に`RuntimeError`を送出する仕組みは、
`aten/src/ATen/Context.cpp`の`Context::alertCuBLASConfigNotDeterministic()`であり、
これは`aten/src/ATen/cuda/CUDABlas.cpp`内の`gemm_internal_cublas<T>`(`mm`/`addmm`/
`nn.Linear`の経路)と`bgemm_internal_cublas<T>`(`bmm`の経路)の両方から、float/double/complex/
half/bfloat16の各型について呼ばれている(PyTorch v2.6.0ソースで確認)。これはS13の4演算
全てがこの同一のチェックを通ることを意味する。

しかし、**このチェックはPyTorch 2.10.0で削除されている。** v2.9.0のソースには
`alertCuBLASConfigNotDeterministic`・`checkCuBLASConfigDeterministic`・
`cublas_config_var_name`が`Context.cpp`に存在するが、v2.10.0ではこれらが**全て消えている**
(実際にGitHubから両バージョンのソースを取得して確認済み)。該当するPRは

> **PR #161749「[cuBLAS] update cuBLAS determinism docs, remove workspace requirement
> checks」**(2025-10-03マージ、v2.10.0から含まれる)
> 本文: "Since CUDA 11.x we've been allocating cuBLAS workspaces explicitly per
> handle/stream combination … According to the cuBLAS documentation, this appears to be
> sufficient for determinism without any explicit workspace requirements to e.g.,
> `:4096:8` or `:16:8` as was previously expressed in PyTorch docs."

削除後の`aten/src/ATen/cuda/CublasHandlePool.cpp`(v2.14.1)では、`CUBLAS_WORKSPACE_CONFIG`は
`parseChosenWorkspaceSize()`内で読まれるが、これは**workspaceのサイズを決めるためだけ**に
使われ、未設定時は単にデフォルトサイズ(非Hopper/Blackwell系で`:4096:2:16:8`相当)にフォール
バックするだけで、例外もワーニングも出さない(ワーニングが出るのは値の**書式が壊れている**場合
のみ)。v2.14.1の`CUDABlas.cpp`(2840行)全体を検索しても`determinist`という文字列は一つも
現れない(CUDAビルドにおいて)。

現在の公式ドキュメント(`https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html`、
v2.14系)の「will throw a RuntimeError」リストにも、`torch.mm`/`torch.bmm`/`torch.addmm`/
`torch.nn.Linear`は含まれておらず、`CUBLAS_WORKSPACE_CONFIG`という文字列自体がページ中に
一度も出現しない。これは上記のソースコード調査と整合する。

**つまりS13モジュールのdocstringが引用する「PyTorch公式ドキュメントの主張」は、
(a)おそらく2.9.0以前のドキュメントを指しており、(b)本プロジェクトが実際に使う
`torch==2.14.1+cu130`にはもはや当てはまらない。**

### 予測(実測前に記録)

**4演算(`mm`, `bmm`, `addmm`, `linear_forward_backward`)のいずれも`RuntimeError`を
送出せず、全て正常に`digest`を記録する。** 「どの演算で例外が出て、どの演算で出ないか」という
問いに対する答えは「**全演算で出ない**」である。

根拠: 上記の通り、例外を送出していたコードパス(`alertCuBLASConfigNotDeterministic`と、
それを呼んでいた`gemm_internal_cublas`/`bgemm_internal_cublas`内の呼び出し)がPyTorch 2.10.0で
完全に削除されており、`torch==2.14.1+cu130`はその削除後のバージョンに当たる。

### 外れた場合に何が言えるか

- **1演算でも`RuntimeError`(またはそれに類する例外)が出た場合**: 以下のいずれかを意味する。
  1. 本プロジェクトが参照したPyTorchのソース(GitHub上のタグ`v2.9.0`/`v2.10.0`/`v2.14.1`)と、
     実際にインストールされる`torch==2.14.1+cu130`のビルドに差異がある(例えば、cuBLASの
     バージョンやCUDA 13.0固有の事情で、削除されたはずのチェックとは別の経路で例外が
     再発している)。
  2. 本調査が見落とした別のコードパス(例えば`cublasLt`経由のGEMM、またはTF32関連の
     分岐)が、今回確認した`gemm_internal_cublas`/`bgemm_internal_cublas`とは別に
     決定性チェックを持っている。
  — いずれにせよ、**どの演算で例外が出たか**を実測結果からそのまま記録し、ソースを再調査する。
    「出ないはず」という予測が外れること自体が、CUDA 13.0世代での cuBLAS 決定性の仕様を
    明らかにする価値を持つ。
- **4演算とも例外が出なかった場合**: 予測通りであり、**「`CUBLAS_WORKSPACE_CONFIG`を設定しないと
  例外になる」という、広く流布した(そして本プロジェクトのS13自身のdocstringにも書かれていた)
  知識が、少なくともPyTorch 2.10以降では成立しないこと**を実測で確認したことになる。
  記事では、この知識がいつ(PR #161749、2025-10-03)・なぜ(CUDA 11.x以降cuBLASの
  workspaceがhandle/stream単位で明示的に確保されるようになったため)陳腐化したかを、
  S13の節で明記する。

---

## 検証方法(Task 7で行う)

`results/*__*__practice.json`の`S12_cudnn_benchmark_deterministic`と
`S13_cublas_workspace_config`のmetaを読み、本ファイルの表・予測と突き合わせる。
**予測に合わせて結果を解釈しない。外れた予測は外れたまま記録する。** 判定結果は
本ファイルに「実測結果」節として追記し、実測で使われたコミットとは別のコミットとして記録する。
