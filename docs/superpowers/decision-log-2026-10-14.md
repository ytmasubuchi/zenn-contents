# SDD ledger — plan: docs/superpowers/plans/2026-10-14-randomness-reproducibility.md

Spec: docs/superpowers/specs/2026-10-14-randomness-reproducibility-design.md (read, binding authority)
Worktree: /Users/ljmb20260017/Documents/zenn-contents-wt/randomness-reproducibility
Branch: feature/randomness-reproducibility (base 74ad013 = local main)

## Pre-flight scan — cross-task pairs sharing a file or interface

| Pair | Produced | Consumed | Finding |
|---|---|---|---|
| T1 → T4,T5 | digest, encode_sample | probe digests/samples | OK |
| T1 → T11 | decode_sample, compare_arrays | numeric delta between results | OK |
| T2 → T10 | collect_env() keys | runner "env" block | OK |
| T3 → T4..T9 | register / ProbeResult / SkippedProbe / run_probe | every probe module | OK |
| T3 → T10 | all_probes(), run_probe() | runner selection + skip recording | OK |
| T4..T9 → T10 | probe names "S<n>_*" | `--probes S1` prefix selection | OK — all probe names start with S<n>_ |
| T10 → T11 | JSON {schema, env, probes[], rounds[]} | load_results / label {host}/{image}/{config_label} | OK — host, image, config_label all present in T2 env keys |
| T10 → T14 | `--repeat` → rounds[] | G1-1a (same-process twice) | OK (fixed in self-review: repeat added to run() signature) |
| T12 → T13,T14,T16 | Makefile targets | run / report / figures | OK (fixed in self-review: report, figures, build-all-cpu added) |
| T12 → T8,T9 | alpine excludes torch | S7/S8/S9 requires-skip | OK — consistent |
| T13 → T14 | config names threads1/threads8/threadsmax/nosimd512/hashseed0/hashseedfree | measurement runs | OK |
| T15 → T16 | images/reproducibility-of-randomness/ | both write PNGs into it | OK — disjoint filenames |
| T12 (gpu image) → T14 Step 3 | nvidia/cuda:12.6.0 | GPU runs incl. dgxspark | **CONFLICT — see Ruling 1** |
| R1 research output → T17 | reports/ | article source material | **CONFLICT — see Ruling 5** |

## Pre-flight scan — per-task self-consistency

| Task | Finding |
|---|---|
| T1 | OK — every symbol the tests use is in Interfaces; Step 2 carries an inline fallback container command since Dockerfile.base does not exist yet |
| T2 | OK |
| T3 | OK — tests monkeypatch `registry.check_requirement`, which requires run_probe to call it at module level, not via local import; implementer is told |
| T4 | **DEFECT — test name promises a known-value check its body does not perform. See Ruling 4** |
| T5 | OK |
| T6 | OK — parallel-completion probe deliberately does not assert stability |
| T7 | OK |
| T8, T9 | Conditional asserts (skip silently when CUDA/model absent). Plan-mandated and correct for environment-dependent probes; flagged here so the task review does not treat it as a vacuous test |
| T10 | OK |
| T11 | `_res` / `_b64` are test fixtures the plan leaves to the implementer. Acceptable |
| T12–T18 | OK — prose/execution tasks with explicit verification criteria |

## Rulings

Ruling 1: GPU イメージを `nvidia/cuda:12.6.0-runtime-ubuntu22.04` から **CUDA 13 系 + torch cu130** に変更する — dgxspark の GB10 は compute capability (12,1) で CUDA 12.6 のビルドでは叩けず、3ホストともドライバ 580.x / CUDA 13.0 で実測確認済み(vllm イメージで torch 2.13+cu130、experiments-gpu:local で torch 2.14+cu130 が GB10 上で動作)。コスト: CUDA 13 のイメージが anakin/blade の古いドライバで動かない場合 GPU 実測が止まる → その場合 spec のリスク欄どおり GPU を2台に縮退。

Ruling 2: ワークツリーのベースを origin/main (3ea1353) ではなくローカル HEAD (74ad013) にした — ローカル main が既にマージ済みの作業を持っており origin が遅れているため。コスト: PR のベースがリモートとずれる → PR 作成前に rebase すれば解消。

Ruling 3: ワークツリーをリポジトリ外(`../zenn-contents-wt/`)に置いた — `.claude/worktrees` が ignore されておらず、main に .gitignore コミットを足すのを避けるため。コスト: EnterWorktree の path 切替制約を受ける → 本作業は単一ワークツリーで完結するため影響なし。

Ruling 4: T4 の `test_stdlib_mt19937_matches_known_values` を2つに分割する — (a) `test_stdlib_mt19937_matches_published_value`: `random.seed(0); random.random()` が MT19937 の公開仕様から決まる既知値 `0.8444218515250481` に一致することを assert する(これが L0「規格レベル」の真の証明になる)、(b) `test_prng_probe_records_seed_and_count`: meta の seed/n を assert する。元のテストは名前が約束する検証を本体が行っておらず、レビュー規約上の欠陥だったため。コスト: 既知値が誤っていれば即座にテストが落ちて判明する。

Ruling 5: deep-research の出力 `reports/` はメインチェックアウト側に生成される。Task 17 の直前にワークツリーへコピーしてこのブランチでコミットする。`research_notes/`(生ノート)は .gitignore 済みで git に入れない。コスト: コピーを忘れると記事の一次情報が PR に含まれない → Task 17 の Step 1 で必須手順として扱う。

Ruling 6: `.gitignore` に `.superpowers/` と `research_notes/` を追加した(Task 12 Step 4 の前倒し) — SDD ワークスペースが初回コミットに混入するのを防ぐため。コスト: なし。

## Progress

Task 1: dispatched (implementer, sonnet) — BASE 3ded887 — brief task-1-brief.md, report task-1-report.md

Ruling 7: Task 15(原理図 SVG)を Task 1 と並行させず、理論調査7本の報告が届いてから別ワークツリーで着手する — F1(エントロピー階層)・F6(SIMD縮約木)・F7(GPU縮約)は、まさに今調査中の事実をそのまま図にするものであり、手持ちの理解で先に描くと描き直しになる公算が高いため。SDD の「実装サブエージェントは同時に1体」の規則も同一ワークツリー内では維持する。コスト: 直列化でウォールクロックが延びる → 調査着地後に T5〜T10 と重ねて回収する。

Plan/Spec updated: Ruling 1(GPU イメージ CUDA 13 化)と Ruling 4(T4 テスト分割)を両ファイルに反映済み。全18タスクの brief を事前生成済み。

Ruling 5 (改訂): 調査サブエージェントはワークツリー外に書けないため、research_notes/ は**ワークツリー内**に生成されている(`research_notes/乱数の再現性と保証レベル/`)。.gitignore 済みなので git には入らない。Task 17 の材料は同一ワークツリー内にあり、メインチェックアウトからのコピーは不要になった。`reports/`(最終レポート)も同様にワークツリー内に出る想定。

調査着地 1/7: simd_blas_parallel.md (59KB)。記事の筋に影響する重要な指摘あり —
- NumPy の AVX-512 と非 AVX-512 で結果が最大 ~4 ULP 違うことは **sin/cos/exp/log について公式に確認済み**(NumPy issue #23523、メンテナが "speed-vs-consistency tradeoff" と明言)。
- 一方 **`np.sum` 自体が AVX2 機と AVX-512 機で違うという文書化された報告は見つかっていない**(調査側が inference として明示的にフラグ)。
→ 記事冒頭(3b: anakin AVX2 vs blade AVX-512)の目玉は、S2(np.sum)ではなく **S4(exp/log/sin)の方が先に当たる公算が高い**。両方測った上で、実際に差が出た方を冒頭に置く(Global Constraints の「実測が仮説と違ったら記事の筋を実測に合わせる」をそのまま適用)。S2 で差が出なければ、それ自体が「縮約は意外と頑健」という記事の知見になる。
- OpenMP 仕様(5.0/5.1)は reduction の結合順を unspecified と明記し、**同一スレッド数でもビット一致を保証しないと明言**している。これは L3/L4 の一次情報として強い。
- Intel MKL の CNR(MKL_CBWR)は再現性を買える唯一の文書化された仕組みだが、スレッド数固定が条件で最大50%超の性能低下。OpenBLAS に同等機能は見つからず。

調査着地 2/7: reproducibility_terminology.md —
- ACM Artifact Review and Badging は **v1.0(2016)と v1.1(2020)で Reproducibility と Replicability の意味を入れ替えている**(ACM 自身が NISO の勧告に従った旨を脚注で明示)。記事で定義を引くときは v1.1 であることを必ず明記しないと、古い定義を覚えている読者と食い違う。両版の verbatim を確保済み。
- ACM の許容誤差に関する文言を一次情報として確保: 「exact replication or reproduction of results is not required... results must be in agreement to within a tolerance deemed acceptable... differences... should not change the main claims」。**第8章の納品文言テンプレの直接の根拠**として使える(権威ある出典が「完全一致を要求するな、許容誤差を書け」と言っている)。
- VIM(JCGM 200:2012)には repeatability / reproducibility の定義はあるが **replicability の定義は存在しない** — ACM/NISO が計量学の語彙に足した概念。
- Reproducible Builds(ビット単位同一のビルド成果物)と数値計算の再現性は別概念、という切り分け段落も確保済み。記事で混同を避ける注記に使う。

調査着地 3/7: os_hardware_entropy.md — 第3章(階層降下)と F1 の素材が揃った。
- HW: Intel DRNG は熱雑音由来の全デジタル回路(生 ~3 Gbit/s)を AES-CBC-MAC で conditioning。**RDRAND は SP 800-90A の DRBG から読み、RDSEED は 90B/C 準拠の conditioned entropy を直接読む**。DRBG は1シードあたり最大1022出力で再シード。ARM は RNDR / RNDRRS が同じ構図(RNDRRS は読むたびに強制再シード)。
- OS: Linux 5.17/5.18 の書き直しで SHA-1 → **BLAKE2s(入力プール)+ ChaCha20(出力段、5.18 以降 CPU ごと)**。RDRAND/RDSEED は混ぜるが単独では信用しない設計。**5.6 以降 /dev/random は起動後ブロックしない**(man7 random(4) の原文確保済み)。「/dev/random はエントロピー枯渇でブロックする」という通説は現行カーネルでは誤り — 記事で明示的に訂正する価値あり。
- Python: `os.urandom()` は Linux で getrandom(2)、macOS で getentropy()/CCRandomGenerateBytes。`random.seed(None)` は OS 乱数、無ければシステム時刻(docs 原文確保)。
- **PYTHONHASHSEED: PEP 456 の SipHash24、動機は hash-flooding DoS(CVE-2012-1150)。dict は 3.7 以降 挿入順保証で PYTHONHASHSEED の影響を受けないが、set はハッシュスロット依存なので影響を受ける** — S0 プローブの設計(set と dict を分けて測る)がこれで裏付けられた。
- 統合: **seed 固定が切るのは `random.seed(None)` が呼ぶはずだった os.urandom() ただ1箇所**。カーネル CSPRNG も CPU TRNG も PYTHONHASHSEED の SipHash 鍵も止まらない。これが F1 の核心メッセージ。
- 弱い主張として要注意: 「コンテナは**ホストのカーネル CSPRNG を共有する**」は一次情報が見つからず二次情報のみ。記事では断定せず、Mac(Docker Desktop = Linux VM 経由なのでホスト macOS のカーネルではない)との対比として扱う。

調査着地 4/7: llm_inference_temperature.md — 第7章の骨格が固まった。プローブ設計の誤りも1件発見。

Ruling 8: Task 9 の greedy プローブを `temperature=0` ではなく **`do_sample=False`** で行うよう修正(plan / task-9-brief 反映済み)。理由: **transformers に「temperature=0」という決定的モードは存在しない**。公式の決定的パスは `do_sample=False` であり、`temperature=0` + `do_sample=True` は strictly positive 要求で弾かれる。「temperature=0」は OpenAI/vLLM 系 API の慣習であって HF の概念ではない。あわせて `S9_temperature_zero_api_semantics` プローブを追加し、transformers に temperature=0 を渡すと実際に何が起きるか(例外型/警告文/無視)を実測記録する。コスト: 誤ったままだと記事第7章の中心的な実験が「存在しないモード」を測ることになり、全体が崩れていた。

一次情報の質に関する注意(記事で必ず区別する):
- Thinking Machines "Defeating Nondeterminism in LLM Inference"(Horace He, 2025-09-10)は **査読論文ではなくエンジニアリングブログ**。報告値: Qwen3-235B の greedy 1000回で unique completion が 80 → 1、matmul ~20% 低下 / end-to-end 1.6〜2.1倍の低下。`batch_invariant_ops` を OSS 公開、vLLM(issue #27433)と SGLang が取り込み中。
- 分岐率の定量値(decode step の 0.3〜1.3% で batch 起因の flip など)は**すべて 2026 年の arXiv プレプリントで未査読**。記事では必ず「プレプリント」と明記する。なお **我々の実測は一次データなのでこれより強い**。
- 機構: Split-K(matmul の K 次元分割)と Split-KV / FlashDecoding 系の分割数が batch サイズや KV 長で動的に変わる → 部分和の結合順が変わる。Stream-K についての一次情報は見つからず(記事で Stream-K には言及しない)。
- vLLM: `temperature<1e-5` で greedy 分岐、`0<t<0.01` は 0.01 にクランプして警告。**実装ごとに「temperature=0」の意味が違う**こと自体が記事のネタになる。
- OpenAI: seed は "best effort" で保証なし、system_fingerprint 一致でも差が出る可能性ありと明記(原文確保済み)。
- dtype: bf16 = 指数8 + 仮数7、fp16 = 指数5 + 仮数10、fp32 = 指数8 + 仮数23。**bf16 は fp16 より仮数が3bit少なく、flip しやすい**。ただし現ノートは二次情報のクロスチェックなので、公開前に IEEE 754 / Google bfloat16 仕様で裏を取る(Task 18)。

調査着地 5/7: ieee754_floating_point.md — 第4章 L0 の根拠が固まった。記事の言い回しに関わる重要な制約が2つ。

制約A(執筆時に必ず守る): **IEEE 754 の細かい subclause 番号(5.4.1 等)を記事に書かない。** 調査側が primary PDF からテキスト抽出できず、番号は二次情報由来だと明示的にフラグを立てている。「clause 5 の必須演算」「clause 9.2 の推奨演算」程度の粒度に留めるか、Task 18 で番号を裏取りしてから書く。

制約B: 「IEEE 754 が保証するのは1演算であって演算列ではない」は**規格本文に否認文があるわけではなく、スコープからの推論**。ただし**代わりに使える一次引用が見つかった** — IEEE 754 ワーキンググループ自身の "Differences Among IEEE 754 Implementations"(grouper.ieee.org)に「**the IEEE standard does not guarantee that the same program will deliver identical results on all conforming systems**」。しかも同ページは「差が出たらソフトのバグだ」という俗説を明示的に否定している。**第4章の冒頭はこの引用で始める**のが最も強い。

その他の収穫:
- 必須(正しく丸められる): add / subtract / multiply / divide / squareRoot / fusedMultiplyAdd / remainder / 形式変換。
- 推奨どまりの理由が明確になった: **Table Maker's Dilemma**(Kahan の用語)。任意の超越関数を任意の点で正しく丸めるのに何ビット余分に要るかを一般的に抑える手法・証明が無い。算術プリミティブはガードビットの上界が証明できるので「ただで」正しい丸めが手に入る。主要な libm で clause 9.2 を正しく丸めているものは無い。
- **glibc は現行マニュアルからアーキ別 ULP 表を削除していた**(現在は x86_64 限定の外部論文 Gladman et al. 2021 を参照するのみ。その論文は取得失敗)。ただし **glibc 2.27 のアーカイブ版マニュアルの生HTMLを直接パースして確認: `log2`(double)は AArch64 で 1 ULP、x86_64 で 2 ULP** — 同一関数の実測クロスアーキ差。我々の x86_64 vs aarch64 実測(3c)の理論的裏づけとしてそのまま使える。
- musl: 正しく丸める関数を明示列挙(floor/ceil/fma/sqrt 等の形状系が中心)、他は <1.5 ULP を目標。
- Intel MKL VML は HA <1.0 ULP / LA ≤4.0 ULP と公式に階層化。SVML の ~4 ULP は NumPy メーリングリスト経由のみ(低信頼)。
- **Apple の Accelerate/vecLib には ULP の公式文書が存在しない**(調査側が明言)。Mac の libm 差について記事で断定しない。
- コンパイラ: `-ffp-contract={off,on,fast}`(GCC/Clang 原文確保)、Clang は既定 on。C99 の `#pragma STDC FP_CONTRACT OFF` は GCC が歴史的に無視してきたので、**信頼できるレバーは pragma ではなくフラグ**。x87 の 80bit 余剰精度はレジスタ溢れ時に予測不能に切り捨てられる。
- 再現的総和の文献: Kahan(1965)/Neumaier(1974)の補償総和は**固定順序での精度**を直す。Demmel/Nguyen/Ahrens の reproducible summation と ReproBLAS は別問題 —**異なる並列縮約順序・プロセッサ数を跨いでビット一致させる**ために pre-rounding で固定小数点的に振る舞わせる。

調査着地 6/7: gpu_cuda_pytorch.md — 記事の目玉が1つ増えた。

Ruling 9: Task 8 に **`S8_curand_randn_sizes`** を追加(plan / task-8-brief 反映済み)。`torch.manual_seed` 固定のまま `torch.randn(n, device="cuda")` を n=1024/65536/4194304/67108864 の4サイズで生成し、サイズごとにダイジェストを取る。根拠: PyTorch の CUDA RNG は `curand_init` が Philox を**スレッド index** で seed するため、スレッド→要素の割り当てが **SM 数に依存**する。要素数が閾値を超えると、**同じ seed・同じ Philox でも GPU モデルが違えば乱数の値そのものが変わる**(PyTorch issue #194676、T4 vs L4 の実数例と計算式あり)。これは「丸めがズレる」ではなく「**乱数が違う**」ケースで、第6章 G4 の最強の実例。**この挙動を公式 docs に追記する PR #196733 は approve 済み・未マージ**(2026-10-14 時点)= 現時点で公式には未文書化であり、我々の実測が一次情報になる。あわせて全 S8 プローブの meta に TF32 両フラグの実値と SM 数を記録させる。コスト: 閾値が我々の4サイズの外にあれば差が出ない → その場合はサイズを増やして再測。

一次引用(記事でそのまま使う):
- PyTorch 公式: 「**Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. ... results may not be reproducible between CPU and GPU executions, even when using identical seeds.**」
- cuBLAS: ビット単位の再現性が成り立つのは「**when a single CUDA stream is active**」のみ。さらに**同一アーキかつ同一 SM 数**が条件。
- cuDNN: 「**no cuDNN routines guarantee bitwise reproducibility across different architectures**」(Volta vs Turing を例示)。
- TF32 の既定値は非対称: `matmul.allow_tf32` は PyTorch 1.12 以降 **False**、`cudnn.allow_tf32` は **True** のまま。2.9+ で `fp32_precision` 文字列 API が追加。
- `use_deterministic_algorithms`: 決定的代替があるもの(Conv*, scatter_add_, index_add, Embedding 等)と、**決定的実装が無くエラーになるもの**(AvgPool3d, AdaptiveAvgPool*, MaxUnpool*, NLLLoss, CTCLoss, grid_sample 等)の公式リストを確保。
- SDPA: FlashAttention / Efficient / cuDNN の backward は atomics か key 分割により既定で非決定。**常に決定的なのは MATH backend のみ**。
- `torch.manual_seed` は CPU と全 CUDA デバイスを seed するが、**NumPy と Python の random は別途 seed が要る**(公式明記)。

弱い箇所(記事で断定しない): CUDA Programming Guide の atomicAdd 原文は取得失敗で未検証。Hopper/Blackwell の TF32 アキュムレータ bit 幅も未検証。

調査着地 7/7(完了): prng_specs_and_policies.md — **L0 の定義に関わる訂正**が1件。

Ruling 10: Task 4 に `S1_numpy_mt19937_generator`(= `Generator(MT19937(seed))`)と `S1_generator_random_float32` を追加(plan / brief 反映済み)。理由: **MT19937 の仕様が固定しているのは state→出力の twist+tempering であって、seed→state の写像ではない。** CPython `random` は参照実装の `init_by_array`、NumPy `RandomState` は独自の `mt19937_seed`、`Generator(MT19937)` は `SeedSequence` — **同一 seed・同一 MT19937 で3つの異なるストリーム**になる(調査側がソース差分で確認、ただし単一の公式文書に書かれた記述ではないと明示フラグ)。当初「MT19937 は L0」と単純化するつもりだったが、**L0 なのは core だけで seeding は L2** が正しい。記事の第4章はこの区別を必ず書く。コスト: 3本が実は一致したらテストが落ちて即判明する。

Ruling 11: `oldlibs` イメージを **python:3.10-slim + numpy 1.21.6** に変更し、**対照イメージ `oldpy`(python:3.10-slim + numpy 2.2.6)を追加**(plan / briefs 反映済み)。理由2つ。(1) 当初の numpy 1.26.4 では **NEP 19 のストリーム破壊を1つも跨げない** — 文書化された破壊点は 1.19.0(Dirichlet 等)と 1.22.0(float32 uniform 最下位ビット)であり、1.26→2.2 では何も起きない公算が高い。1.21.6 なら 1.22.0 の変化を跨ぐ。(2) numpy 1.21.6 は Python 3.10 までしか対応しないため Python も下げる必要があり、そのままでは **2a(Python版)と 2b(ライブラリ版)が交絡する**。`oldpy` を挟むことで base vs oldpy = Python のみ、oldpy vs oldlibs = numpy のみ、と分離できる。コスト: イメージが1つ増え blade のランが1本増える(数分)。

一次引用と事実(記事でそのまま使う):
- NEP 19(Status: Final, 2018-05-24)原文: RandomState は「Bug fixes that change the stream SHALL NOT be made to RandomState」、Generator は「breaking stream-compatibility in order to introduce new features or improve performance will be allowed with caution ... on X.Y releases, never X.Y.Z」。**NEP 19 は "frozen" という語を一度も使っていない** — 記事で「凍結」と書くなら言い換えだと分かる書き方にする。
- numpy.org の compatibility ページは上記に加え、**同一ビルド・同一 seed でも CPU の浮動小数差やリンクされた LAPACK 版の違いで再現が壊れうる**と明記。NumPy 自身が L3 を認めている一次情報。
- 分布変換: legacy `standard_normal` は polar 法(ソースのコメントに明記)、`Generator.standard_normal` は 256段 ziggurat。NumPy 公式「it is not possible to reproduce the exact random values using Generator for these distributions」。
- ストリーム破壊の実例(release note 原文確保): 1.19.0(Dirichlet、SeedSequence+spawn の padding、MT19937.jumped)、1.22.0(float32 uniform 最下位ビット)。
- PCG64DXSM は **1.21.0(2021)** で追加(1.25 ではない)。**`default_rng()` は今も素の PCG64 が既定**で、DXSM 推奨は「数百万規模の並列ストリーム」限定の狭いスコープ。
- NumPy が同梱するのは **Philox のみ。Threefry は無い**(JAX と randomgen)。JAX の `jax_threefry_partitionable` は Threefry の内部を意図的に変え、**同名のまま Random123 参照実装とのビット互換を壊している** — 「仕様名が同じでも実装が違う」の好例。
- SeedSequence の目的は (a) 質の悪い seed(MT19937 はゼロ語の多い state で劣化)と (b) 連番 seed の相関、への対策。**O'Neill の std::seed_seq 代替が元で SplitMix 論文ではない**(調査側が誤帰属を能動的に否定)。
- scikit-learn: 「バージョン間で保証しない」という形の文は**サイト上に存在しない**(調査側が「無い」と明示)。実際の免責は backwards compatibility 用語集側の「bug fixes may change the behavior of estimators, including the predictions of an estimator trained on the same data and random_state」。**存在しない文を要約してはいけない。**
- SciPy には NumPy / scikit-learn のようなストリーム破壊の明示的 release note が見つからない — **開示が無いことは破壊が無いことを意味しない**、という非対称性として書く価値あり。

Task 1: implementer DONE — commit 7f706d4、16 tests passing(brief の8本 + 曖昧性解決分8本)、`-W error` で警告ゼロ。

Ruling 12: 実装者が報告した **Netskope の TLS 傍受プロキシ**問題を Task 12/13 に申し送る。この Mac では `docker run python:3.11-slim pip install` が SSL エラーになり、Netskope のルート+テナント CA をコンテナに入れて `update-ca-certificates` + `PIP_CERT` が必要だった。**Task 12 の Dockerfile は CA を焼き込まず、ビルド時に任意の CA を注入できる ARG を持つ形にする** — 読者が再実行するのに企業プロキシ前提の Dockerfile を渡すのは不適切なため。anakin/blade/dgxspark で同じ問題が出るかは Task 13 の疎通時に確認する。コスト: 確認を怠ると4ホストのビルドが一斉に落ちる。

Ruling 13: ご依頼の「worktree を並列で」は、**実装サブエージェントについては実現できない**。この harness ではサブエージェントがセッションのワークツリーに強制的に閉じ込められる(調査7本すべてが指定した共有チェックアウトのパスに書けず、ワークツリー側に書いた)。別ワークツリーを作ってもサブエージェントをそこに向けられないため、同一ワークツリーで実装エージェントを同時に走らせるしかなく、それは git index の競合を招く。したがって **SDD の「実装サブエージェントは同時に1体」を維持**する。
並列化は別の2手段で確保する: (1) 理論調査7本は実際に同時実行し完了済み。(2) **同形の小タスクはバッチ化する**(SDD が明示的に許可)。T4+T5+T6(プローブ+テストを書く同形タスク)を1ディスパッチ、T7+T8+T9 を1ディスパッチ、T15+T16(図版)を1ディスパッチにまとめ、レビューはバッチ単位で行う。これでディスパッチ数が 18 → 13 に減る。
コスト: バッチ内の1タスクに欠陥があるとバッチ全体が fix ループに入る。基盤タスク(T1/T2/T3/T10/T11)は後続が依存するインタフェースを決めるのでバッチ化せず個別に回す。

Task 1: review dispatched (sonnet) — diff 3ded887..7f706d4、313行のレビューパッケージ。ULP のゼロ跨ぎ・符号処理を名指しで重点確認させた(ここが壊れると全実測が静かに腐るため)。

Ruling 12(撤回・改訂): ユーザーが Netskope を無効化。素の `docker run python:3.11-slim pip install numpy==2.2.6` が通ることを確認済み(2026-10-14)。**Task 12 の Dockerfile に CA 注入の ARG は入れない** — 企業プロキシ対策を同梱すると読者の再実行環境で無駄な複雑さになる(YAGNI)。Dockerfile は素のまま書く。anakin/blade/dgxspark でのビルド疎通は Task 13 の Step 2 ドライランで確認する。コスト: 将来 Netskope が再び有効になった環境でビルドが落ちるが、その時は実行側の環境変数で対処すればよく、成果物に残す必要はない。

Task 1: review returned — Spec ✅ compliant、Task quality Approved、Critical 0 / Important 1 / Minor 2。
レビュアーは最高リスク箇所(ULP の符号処理とゼロ跨ぎ)を独立したスクリプトで実測検証し、**実装は数値的に正しい**と確認(-0.0 vs +0.0 → 0、-1.0 と nextafter(-1.0,-2.0) → 1、ulp_diff(-1.0, 1.0) が 2×bits(1.0) に一致)。int64 オーバーフロー対策に object dtype を経由している点も「過剰ではなく必要な修正」と評価。
Important(fix ループへ): **負の数同士・ゼロ跨ぎの ULP 距離を検証するテストがコミット済みスイートに無い。** 現実装は正しいがその保証がレビュアーのスクラッチ検証にしか存在せず、将来 `_ordered_bits` を編集しても pytest が気づかない。T4-T11 が依存する基盤なので塞ぐ。
Task 1: minor (deferred): compare_arrays の max_abs/max_rel は NaN を含む差分で np.max が NaN を伝播する(max_ulp のみ非有限の扱いが文書化されている)。最終レビューで要トリアージ。
Task 1: minor (deferred): list/tuple/dict のエンコードに要素数の外側プレフィックスが冗長に付く。害は無いが brief の明示要求を超える。

Task 1: fix round 1/5 — 実装者が production コードは変えず **テストを5本追加**(commit 5b8a9ea)。-0.0 vs +0.0 = 0、負側の隣接 1 ULP、負側の加法性(2段で 2 ULP)、ゼロ跨ぎ = 2 ULP、そして `ulp_diff(-1.0, 1.0)` が `2 * bits(1.0)` かつ実値 9214364837600034816 に一致することで remap の向きそのものを固定。21 tests passing、`-W error` で警告ゼロ。素の docker コマンド(CA 回避なし)で実行できたことも確認。
Task 1: fix round 1 の scoped re-review を dispatch(sonnet)。新テストが循環的(テスト対象の `_ordered_bits` で期待値を再計算している等)でないかを名指しで確認させた。

Task 1: re-review — All findings addressed, no new breakage。再レビュアーは `_ordered_bits` を in-process で monkey-patch してオフセットの向きを入れ替える退行を注入し、`test_ulp_diff_cross_zero_crossing_is_two` と `test_ulp_diff_cross_sign_symmetry_pins_the_remap` が実際に落ちることを確認 = **テストが remap の向きを本当に固定している**ことを実証。fix diff は test only(36 insertions, 0 deletions, 1 file)。
Task 1: complete (commits 3ded887..5b8a9ea, review clean)
Task 2: dispatched (implementer, sonnet) — BASE 5b8a9ea

Task 2: implementer DONE — commit 3913d48、13 tests(brief の4本 + resolution 分9本)、リポジトリ全体 34/34 passing、`-W error` で警告ゼロ。
実装者が正当な指摘: **brief は `cuda` キーを必須にしているのに中身を規定していない**。実装は `{"available": bool, "device_count": int}`(torch 不在時は None)という最小解釈を採用。

Ruling 14: その最小解釈を**採用する**。追加で driver version を Task 2 に持たせることは**しない** — GPU の素性(name / compute capability / SM 数)は `gpus` に、CUDA ビルド版と cuDNN 版は `torch` に既に入っており、残る driver version は**プロセス単位ではなくホスト単位の属性**だから。代わりに **Task 13 の `run_on_host.sh` に `nvidia-smi --query-gpu=driver_version,name` の取得を追加し、ホスト単位のメタデータとして results に残す**。コスト: driver 差が GPU 実測差の交絡要因だった場合、プロセス内 JSON 単体では説明できずホスト側メタデータと突き合わせる必要がある(突き合わせは Task 14 の比較時に行うので実害なし)。

実装者の自己修正で1点見るべきものがあった: `-W error` が `numpy.core._multiarray_umath` の DeprecationWarning を例外に変え、狭い `except ImportError:` をすり抜けて `numpy_cpu_features` が黙って空になる事故が実装中に発生。新パス(`numpy._core`)を先に試す順序に変えて解消。**記事の中核証拠になるフィールドが静かに欠落する**類の事故で、`-W error` で回していなければ気づかなかった。
Task 2: review dispatched (sonnet) — diff 5b8a9ea..3913d48

Task 2: review returned — Spec ✅(列挙キー・resolution はすべて実装済み)、Task quality **Needs fixes**、Critical 0 / Important 2 / Minor 3 / ⚠️ 1。
Important 1: `_safe()` が全例外を無診断で飲み込む(envinfo.py:18-25)。将来の API 変更由来の AttributeError と「この環境に元々無い」が同じ `None` に潰れる。**この module の存在理由そのものを無効化する**ので stderr へのフィールド名+例外の記録を要求。
Important 2: `torch` / `cuda` / `gpus` の**辞書の形を検証するテストが無い**。テストコンテナに torch が無いため `gpus` が list であることしか見ていない。`multi_processor_count` のキー名を打ち間違えても、anakin/blade で実走するまで気づかず、共用ホストの10分枠を綴り間違いの発見に使うことになる。fake torch(SimpleNamespace)でキー集合を固定させる。

Ruling 15: レビュアーが ⚠️(diff からは検証不能)として挙げた `host` の件を、**私が実ギャップと確認して fix ループに入れた**。`platform.node()` はコンテナ内ではコンテナ自身のホスト名を返すので anakin/blade/dgxspark を識別できない。比較器(Task 11)は `{host}/{image}/{config_label}` でランをラベル付けするため、**全ランが一意ラベルになり横断比較が成立しなくなる**。`image`/`config_label` が既に `REPRO_IMAGE`/`REPRO_CONFIG` を読んでいるのと同じ形で、`host` も `REPRO_HOST` を優先し `platform.node()` にフォールバックさせる。コスト: 放置すると Task 14 の実測を全部やり直すことになっていた(共用ホストの占有が倍になる)。

Task 2: minor (deferred): `_gpus()` が後続デバイスで例外時に取得済みデバイス分を捨てる(全ホスト単一GPUなので実害なし)。
Task 2: minor (deferred): `_import_torch()` が1回の collect_env で3回モジュール探索する。
Task 2: minor (deferred): `_safe` のフィールド独立性を守る回帰テストが無い。
Task 2: fix round 1/5 dispatched — Important 3件(うち1件は ⚠️ からの昇格)。

Task 2: fix round 1/5 — commit f9766fd、3件すべて対応。`_safe(name, fn)` が stderr に `envinfo: field {name!r} failed: {exc!r}` を出すようになり(戻り値は `None` のまま、再 raise なし)、fake torch による形状テスト4本でキー集合を厳密固定、`_host()` が `REPRO_HOST` 優先 + `host_source` キーで経路を記録。19/19(focused)、40/40(全体)、`-W error` で警告ゼロ。
実装者が fix 中に自己注入の退行を1件捕捉: 中間版の `_gpus()` が torch 不在でも `_safe(...)` を無条件に呼び、**通常の CPU 専用ランすべてで偽陽性の診断が出る**状態になっていた。早期 return で解消。
Task 2: fix round 1 の scoped re-review を dispatch(sonnet)。(a) torch 不在時に診断が出ないこと(alpine/CPU イメージが偽陽性で埋まると診断チャネル自体が無価値になる)、(b) 形状テストが `set(d) == {...}` で**厳密一致**を見ているか(`<=` だと打ち間違いも余剰キーも捕まらない)、の2点を名指しで確認させた。

Task 2: re-review — All findings addressed, no new Critical/Important breakage。再レビュアーが2つの重点項目を実地確認: (a) `_torch_info`/`_cuda_info`/`_gpus` はいずれも `_import_torch()` が None を返した直後に早期 return しており、**torch 不在の通常ランで診断は一切出ない**、(b) 形状テストは3本とも `set(d) == {...}` の厳密一致で、`multi_processor_count` の綴りが実装側(envinfo.py:211-213)とテスト側(test_envinfo.py:19,174,179)で一致 = **テストが実装の綴りを写しただけ**ではない。
Task 2: minor (deferred): `_host()` と `_host_source()` が `REPRO_HOST`→`platform.node()` の優先順位を別々に実装しており、片方だけ編集すると `host` と `host_source` が静かに食い違う。現状は一致しており両経路ともテスト済み。
Task 2: complete (commits 5b8a9ea..f9766fd, review clean)
Task 3: dispatched (implementer, sonnet) — BASE f9766fd

=== 2026-10-14 セッション中断 ===
Task 3: 実装エージェントを**自己レビュー/コミット直前で停止**(ユーザー要請によるキリのいい中断)。
停止直前の報告: フルスイート 58 passed(既存40 + 新規18)、`-W error` でクリーン。
未追跡のまま残置: `reproducibility/reprobench/registry.py`, `reproducibility/reprobench/probes/`,
`reproducibility/tests/test_registry.py`。**削除していない** — 動作しているコードなので。
再開時は (a) 新エージェントに「既存ファイルを自己レビューしてコミット」と指示、
または (b) 未追跡ファイルを消して task-3-brief.md からやり直し。

中断時の保全処置:
- `.gitignore` から `research_notes/` を外し、調査ノート7本(368KB)をコミット。理由: 記事の全主張の
  一次情報であり、ワークツリー削除で失われる位置にあったため。あわせて `__pycache__/` を追加。
- spec / plan / 新規 `docs/superpowers/RESUME.md` をコミット。
- RESUME.md に、ホスト構成・調査で確定した事実・**執筆時の禁止事項**(裏が取れなかった主張の一覧)・
  T3 の中断状態・再開手順を記載。
- この台帳(.superpowers/ 配下)は git 管理外のスクラッチのまま。**15個の ruling はここにしかない**。

=== 2026-10-14 セッション再開 ===
状態確認: ワークツリー・ブランチ・台帳(ruling 22件)・brief 18本すべて健在。HEAD c2a1045。
T3 の「58 passed」という停止直前の報告を**コントローラ自身がコンテナで再実行して検証** → 58 passed、`-W error` でクリーン。報告は正しかった。
ファイル: registry.py 157行、probes/__init__.py 12行、tests/test_registry.py 207行。task-3-report.md は未作成(停止が自己レビュー前だったため)。

Ruling 16: T3 は RESUME.md の選択肢 (a) を採る — 既存ファイルを捨てずに、新しい実装エージェントに自己レビュー+レポート作成+コミットだけ行わせる。理由: コードが実在し、テストが実際に通ることをコントローラが独立に検証済みで、作り直しは数分の純粋な手戻りだから。コスト: 前エージェントの設計意図を引き継げないため、自己レビューが「他人のコードの初見レビュー」になる。これは実質的に不利ではなく、むしろ厳しめに見られる方向に働く。
Task 3: re-dispatched (implementer, sonnet) — BASE c2a1045、既存の未追跡ファイルを引き継ぎ

Task 3: implementer(引き継ぎ)DONE — commit 5fc8280、7つの resolution をすべて MET と判定し**コード変更なし**でコミット。58 passed、`-W error` クリーン。
引き継ぎ実装者からの鋭い自己申告: テストコンテナには torch も huggingface_hub も無いため、実物の `check_requirement("cuda")`/`("torch")` は `find_spec` 失敗で元々 False を返す。**したがって brief の monkeypatch テスト2本は、`run_probe` がエイリアスを握っていても偶然 pass する**。vacuousness を排除しているのはテストの合否ではなくコードの構造的読解だけ、と明記。
Task 3: review dispatched (sonnet) — diff c2a1045..5fc8280。レビュアーには (a) この diff は**どの実装者もレビューを受けて直したことがないコード**である点、(b) 上記の偶然 pass 問題を自分で判断し、「brief の逐語テストを変えずに補助テストを足せば塞げるのでは」まで評価すること、を指示。

Task 3: review returned — Spec ❌(resolution 1 は**実装は正しいがテストが検証していない**)、Critical 0 / Important 1 / Minor 4。
レビュアーは7つの resolution を report に頼らず自分でソースを読んで individually 確認し、**コントローラが名指しした偶然 pass 問題に独立に到達**(私は答えを渡していない)。18本のテストを走査し、「本当に不在の依存に対して monkeypatch で True を返させ、プローブ本体が実行されることを assert する」テストが1本も無いことを確認。つまり **`run_probe` をエイリアス握りつぶしに書き換えても全テストが通る**。
レビュアーは引き継ぎ実装者の却下理由(「Python の名前解決をテストすることになる」)を明確に否定 — テスト対象は `run_probe` と `check_requirement` の配線であり、それはこのプロジェクトの契約そのもの。
Task 3: minor (deferred): 未登録名に対する bare KeyError / 否定テストが lightgbm のみ / huggingface_hub 追加で陳腐化するコメント / typing.Callable。
Task 3: fix round 1/5 dispatched — 補助テスト1本の追加。**意図的に壊した `run_probe` で落ちることを確認してから**最終版に戻すよう指示(両方で通るテストこそ排除対象なので)。

Task 3: fix round 1/5 — commit 2c9cf49、テストのみ +22行、`registry.py` は net ゼロ変更。
RED 証拠が要求通り: `run_probe` を**デフォルト引数での def 時キャプチャ**(resolution 1 がまさに警告している反パターン)に意図的に壊してフルスイートを実行 → **1 failed, 58 passed**。新テストだけが `SkippedProbe(reason='missing requirement: torch')` で落ち、**既存58本は壊れたコードに対して全部通った** = 既存テストではこのバグを捕まえられないことの実証。その後 registry.py を前コミットのバイト列に復元して 59 passed。
Task 3: fix round 1 の scoped re-review を dispatch(sonnet)。(a) fix diff 内で `registry.py` が net ゼロ変更か(意図的な破壊実験中に実装がドリフトしていないか)、(b) 新テストが本当に両実装を区別するかをレビュアー自身が論理で再確認すること、の2点を指示。

Task 3: re-review — All findings addressed, no new breakage。レビュアーは報告された RED 出力に頼らず**自分で両実装の挙動を論理で追い**、デフォルト引数キャプチャだと def 時に束縛されるため monkeypatch 前の実物が残り、torch 不在コンテナで `find_spec` が False を返して SkippedProbe になる → テストが落ちる、を独立に確認。`registry.py` が fix diff に一切現れない(net ゼロ変更、破壊実験のドリフト無し)ことも確認。テスト関数数を数えて 21+19+19=59 が報告値と一致することまで検算。
Task 3: complete (commits c2a1045..2c9cf49, review clean)

=== バッチ1: T4+T5+T6 ===
Ruling 17: Ruling 13 の方針どおり T4(S1 PRNG生ビット)・T5(S2/S3/S4 縮約・BLAS・libm)・T6(S0 隠れた入力差)を**1ディスパッチにまとめる**。3つとも「プローブモジュール1本 + テスト1本を registry の契約に対して書く」同形タスクで、ファイルは互いに素。SDD の batching 規定に合致。レビューはバッチ単位で1回。
Ruling 18: プローブモジュールの登録経路を明示する — 各プローブモジュールは `probes/__init__.py` から import させ、`import reprobench.probes` で全登録が完了する形にする。plan はこの配線を規定しておらず、放置すると Task 10 のランナーが `all_probes()` を呼んでも**空の辞書が返る**(モジュールが未 import なので登録が走らない)。コスト: T7-T9 でも同じ配線が要るため、バッチ2の指示にも引き継ぐ。
Ruling 19: `S2_sum_shuffled_float32` のシャッフルは `default_rng(20261014).permutation(n)` による**固定置換**とする。plan は「同じ値を別順で足す」としか書いておらず、置換が非決定だとプローブ自体が環境差と無関係にばらつき、**非結合性の実証という目的を果たさない**。コスト: 置換の生成自体が PCG64 に依存するが、PCG64 のビット列は S1 で別途 L0 として検証されるので循環しない。
Task 4+5+6: dispatched as one batch (implementer, sonnet) — BASE 2c9cf49

Task 4+5+6: implementer DONE — commits f776847 / 35f7acd / 5b22ec2、**22個のプローブを新規登録**、70 passed(59 + 11)、`-W error` クリーン。並列完走順プローブの揺れを見るため3回連続実行して安定を確認。

Ruling 20: 実装者が判断を仰いだ「numpy/stdlib を関数本体ではなくモジュール先頭で import した」件は**実装者の読みを採用する**。私の Resolution 2 が広すぎた。registry の「依存は関数本体で import」規則は**条件的に不在になりうる重い依存**(torch / lightgbm / sklearn / transformers / huggingface_hub)を対象とするもので、全イメージに存在する numpy と stdlib には適用しない。既存の `digest.py` も numpy を先頭で import しており、そちらが前例として正しい。関数本体 import を機械的に強制すると可読性を落として得るものが無い。コスト: なし。ただし**バッチ2(T7-T9)では sklearn / lightgbm / torch / transformers が本当に不在になりうる**ので、そちらは関数本体 import が必須。指示に明記する。

実装者の良い判断が1つ: `S0_parallel_completion_order` のジッタに**グローバル `random` でも本プロジェクトの固定 seed でもない、未シードの `random.Random()` を使った**。グローバルを使うと `test_s1_prng.py` の `random.seed(0)` の状態を拾ってしまい、「実行ごとに変わる」ことを示すはずのプローブが逆に決定的になりかねなかった。
Task 4+5+6: review dispatched (opus) — diff 2c9cf49..5b22ec2

Task 4+5+6: review returned (opus) — Spec ❌、Critical 0 / Important 5 / Minor 6。数値コアは solid 評価(native dtype でダイジェスト、S2 の非結合性実証は空虚でない、3-MT19937 テストは偶然では通らない、S1/S2/S3/S4 に隠れエントロピー無し)。実装者が `list(d.keys())` を digest した判断は「S0 唯一の罠を回避した」と名指しで評価(dict を直接 digest すると digest.py がキーをソートするため定数になり、何も証明しないプローブになっていた)。

**このバッチで捕まった「きれいに動いて別のものを測っている」2件:**
Important 1: `S0_parallel_completion_order` が**自前の乱数 sleep で不安定性を製造**していた。32ジョブ32ワーカーで全ジョブが即座に sleep に入るため、完走順は 32個の一様乱数のソート順そのもの。**構造上どの2ランも一致しえず、ホストについて何の情報も運ばない**。しかも docstring に「遅延が無いと本プローブの主張を実証できなくなる」と書かれており、これは Global Constraint「仮説に合う実測を選びに行かない」の反転。
Important 2: **S1 の正規分布プローブ2本が libm を経由している**のに、モジュールは「L0の証明」と銘打たれていた。`RandomState.standard_normal` の polar 法は全出力に対して `log` を呼び、`Generator.standard_normal` の ziggurat は棄却枝で `exp`/`log` を呼ぶ。つまり aarch64 と x86_64 の不一致が **PRNG 層ではなく glibc の libm 由来**でありうる。

Important 3: legacy(MT19937+polar)vs Generator(**PCG64**+ziggurat)は**変数が2つ動いており、変換アルゴリズムの差を単独で示せない**。brief が第4章 L2 の実例として置いたペアだが、変換が同一でも差が出る。
Important 4: **22プローブ中8本がどのテストからも一度も実行されていない**。かつ `run_probe` は例外を `SkippedProbe` に落とすので、実測ホストで落ちても**そのプローブが静かに消えるだけ**。最悪ケースは `S1_generator_random_float32`(NEP-19 の実証プローブ、価値の全部が native float32 ダイジェストに乗っている)。
Important 5: Resolution 7(float.hex)が S2 にしか適用されていない。S3 の trace/fro_norm と S4 の mean は `sample_b64` から復元不能で、かつ記事が引用する数値。

Ruling 21: Minor 扱いだった **S4 の定義域問題を Important に昇格**。`S4_sin_float64` は uniform(-1e6,1e6) だが NumPy のベクトル化 sin/cos は概ね 1.17e5 未満にしか適用されず**大半の入力が AVX-512 経路を通らない可能性がある** — S4 は記事の冒頭ネタの最有力候補なので、4ホスト測定後に気づくと全部再測になる。`S4_log_float64` も uniform(1e-9,1e9) のため 99.9% が 1e6 超に落ち、log の出力が実質1デケードしか掃けていない。既存プローブは残したまま `S4_sin_small_float64`(|x|≤1e4)と `S4_log_loguniform_float64` を**追加**させる。コスト: プローブが2本増える(実行時間は無視できる)。
Ruling 22: Minor 11(決定性テストが 8本中4本/9本中3本しかカバーしていない)を Important 4 のスモークテストに**統合**。`all_probes()` 全走査で「ProbeResult か missing requirement のスキップであること」と「同一プロセス内2回でダイジェストが一致すること」を同時に見る(並列プローブのみ除外)。1本のテストで2つの穴が塞がる。
Task 4+5+6: minor (deferred): `S0_dict_from_set` は `S0_set_iteration_order` と構造上ダイジェストが常に一致し、2本で1ビットの情報しか運ばない。brief 規定どおりで主張は成立するため据え置くが、**比較器の出力を「独立した2つの確認」と誤読しない**こと。
Task 4+5+6: fix round 1/5 dispatched — Important 5件 + 昇格2件 + 小修正1件。

Task 4+5+6: fix round 1/5 — commits 290f405 / 6911cd9 / 9343c62 / 5791ad5、80 passed、5回連続実行で安定。
1: 乱数 sleep を撤去し、**ジョブ id から決まる決定的で不均一なビジーループ**に置換(`meta["work_sizes"]`)。2: S1 の docstring の L0 主張を uniform プローブに限定、正規分布プローブに `bit_generator`/`method` を記録。3: `S1_numpy_normal_generator_mt19937` を追加し変換と bit generator を独立に切り分け。4: `test_all_probes_smoke.py` を新設し `all_probes()` 全走査 + 同一プロセス内ダイジェスト安定性。5: S3/S4 に float.hex を追加。6: `S4_sin_small_float64` と `S4_log_loguniform_float64` を追加。7: glob を同一ディレクトリで2回呼び `stable_within_process` を記録。8: `permuted`/`perm_seed` を meta に追加。
Task 4+5+6: fix round 1 の scoped re-review を dispatch(opus)。3点を名指しで判断させた: (a) finding 1 が**同じ欠陥の別バージョンに「修正」されていないか** — 置換後の作業量が本当に決定的(経路のどこにも RNG が無い)で、観測される揺れが真にスケジューラ由来か。かつビジーループのコストが 10分枠を食わないか(64スレッド Xeon で可視な揺れを出す長さは 20コア Grace では過大になりうる)。(b) スモークテストが**ハードコードした一覧ではなく本当に全登録プローブを走査**し、除外が並列プローブ1本に限定されているか(広い skip だと T7-T9 の新プローブが静かに免除される)。(c) S4 の新プローブが**本当に定義域が違う**か — log-uniform が名ばかりの uniform になっていないか。

Task 4+5+6: re-review(opus)— **8件すべて ADDRESSED**。レビュアーは `_work_size(i)` がジョブ id の純関数で経路に RNG が無いことを独立に追跡し、`import random`/`import time` がモジュールから消えていることまで確認。ビジーループのコストを約0.8秒/呼び出しと見積もり、GIL 直列化のため 64スレッド Xeon と 20コア Grace で同一と判断(私が懸念した点はクリア)。スモークテストが live `_REGISTRY` を走査しハードコード一覧でないこと、2回の呼び出しが本当に別々であること、`"raised: ..."` 形式のスキップを弾くことも確認。
しかし **fix 自体が新たな Important を2件**生んだ。

Ruling 23(重要): New Breakage #1 — `s1_prng.py:36-45` が「legacy(MT19937+polar)vs Generator-MT19937(ziggurat)は変換のみを分離する」と記述しているが**それは成り立たない**。`RandomState(int)` は NumPy 独自の `mt19937_seed`、`Generator(MT19937(int))` は `SeedSequence` で seed するため、**両者は異なる一様ストリームを消費する**。同じファイルの12行上(`:17-22`)と `test_three_mt19937_paths_give_three_different_streams` が自らそれを証明しており、**ファイルが自己矛盾している**。finding 3 で除こうとした交絡と構造的に同じものが、修正の中で再生産された。
文言修正では足りない。**真の変換分離が実際に構成可能**: `RandomState` は BitGenerator インスタンスを受け取れるので、`RandomState(MT19937(seed))` と `Generator(MT19937(seed))` は**seed 経路が完全に同一で変換だけが違う**。`S1_numpy_normal_randomstate_seedseq` を追加させ、3つの比較それぞれが何を分離するのかを正確に記述させる。コスト: プローブ1本増。放置すると**記事の第4章 L2 の実例が誤った主張として公開される**。

Ruling 24: New Breakage #6 — fix report の貼り付けテスト出力が**自己矛盾**(フルスイート 80 passed が、その部分集合である 4ファイル 21 passed より速い)。レビュアーは S2 が13回・10M要素配列を再生成し、matmul 7回、ビジーループ4回×0.8秒であることから、実時間は 2.9秒の数倍のはずと独立に見積もった(80という件数自体は 21+19+19+7+7+5+2 で検算一致)。**並列プローブが非空虚であるという主張の唯一の根拠が「アドホックスクリプトで検証」「5回再実行」であり、貼られた出力がそれを支えていない**ため、クリーンな再実行と、実際に観測された完走順の記録を要求。順序が安定なら Global Constraint どおりそれが正直な結果。
Ruling 25: Minor #2 を昇格 — `~1.17e5` は NumPy の **float32** SIMD 境界だが、S4 は float64 プローブ。float64 の `sin` は SVML / libm で別の範囲縮小境界を持つ。**未検証の機構を docstring に事実として書く**のは、私が執筆時の禁止事項として管理してきたものそのもの。プローブは残し、理由を「定義域が2桁 vs 18桁で違う」に言い換えさせる。
Minor #3 昇格: スモークテストが `S0_glob_order` のダイジェスト安定性を assert しているが、同プローブは呼び出しごとに別の一時ディレクトリを作るため**構造上変わりうる**。ext4/XFS/tmpfs では通るが保証ではなく、落ちたホストでは安全網が無関係な理由で落ちる。`meta["stable_within_process"]` を見るよう変更。
Minor #5 の一部昇格: `s4_libm.py` の `meta["n"]` がモジュール定数由来 → `len(x)` に。S1 の `mean` に hex を追加(finding 5 の後、S1 だけが復元不能なスカラーを残していた)。
Task 4+5+6: minor (deferred): `**extra_meta` が `seed`/`n`/`dtype` を黙って上書きしうる / S2 が13回 10M配列を再生成 / `S0_dict_from_set` の重複 / `s1_prng.py:4-8` と `:24-26` の文言の緊張。
スモークテストの将来コストは**対応不要と裁定**: T7-T9 のプローブは全て sklearn/lightgbm/torch/cuda/transformers/hf_model を `requires` に宣言し、テストイメージにはどれも無いので実行されずスキップされる。10分/ホストの予算は実測ランに掛かるものでありテストスイートには掛からない。
Task 4+5+6: fix round 2/5 dispatched。

Task 4+5+6: fix round 2/5 — commits d1fc535 / 35174fa / 9ecda8e、81 passed。

Ruling 26(証拠整合性の決着): Ruling 24 で提起したタイミングの疑義を、**コントローラ自身が独立に計測して解決**した。`pytest -q -W error --durations=8` を mandated コンテナで実行 → **81 passed in 2.88s、real 5.484s**。最も重いのはスモークテストの 1.27s(25プローブ×2回)、次が並列プローブの 0.46s。
結論: **実装者の数値は正確で、レビュアーのコスト見積もりの方が誤っていた**。レビュアーは自分のローカル機の測定値(10M要素 standard_normal = 0.355s、ビジーループ = 0.8s)から外挿したが、M5 Pro 上の実測はその数倍速い。round 1 の「部分集合が全体より遅い」も実際に単なるノイズ。**証拠の捏造ではなかった**ことをここに明記する — 疑いを掛けた以上、晴れたことも同じ重さで記録する。
実装者は再貼り付けで済ませず原因を調査し、自分で per-probe コストを測り直した上で、5回の独立プロセス実行における**実際の完走順を verbatim で記録**した(全て相異なり、いずれも投入順と不一致、`stable_within_process=False`)。要求した「読者が検証できる証拠」の形になっている。
A: `S1_numpy_normal_randomstate_seedseq`(`RandomState(MT19937(seed))`)を追加し、状態の直接比較で `Generator(MT19937(seed))` と seed 経路を共有することを確認。3つの比較それぞれが何を分離するかを記述し直し、legacy-vs-default は「3軸すべてが動く比較」と明示ラベル。`meta["seeding"]` を全正規分布プローブに追加。
C/D/E: 未検証の float64 SIMD 閾値の主張を削除、`S0_glob_order` のダイジェスト安定性 assert を `stable_within_process` に変更、`meta["n"]` を `len(x)` に、S1 スカラーに hex を追加。
Task 4+5+6: fix round 2 の scoped re-review を dispatch(opus)。タイミング疑義は**決着済みとして再燃させないよう明示**し、A の新しい分離主張が今度こそ文字通り正しいか(`RandomState(BitGenerator)` が本当に再 seed しないか)に集中させる。

Task 4+5+6: fix round 2 re-review(opus)— **A〜E 全件 ADDRESSED、Critical/Important の新規 breakage なし**。
レビュアーは Finding A を報告に頼らず自分で実行検証: `rs._bit_generator is bg` が True、key 配列が wrap 前後でビット一致、`pos` 未変更(623=fresh)= **再 seed も copy も reset も起きない**。決定打として `RandomState(MT19937(s)).random_sample(8) == Generator(MT19937(s)).random(8)` が True — 両ファサードが**同一の一様ストリームを消費**しており、残る差は polar vs ziggurat だけ。今度は「変換のみ」が文字通り成立。
Finding B の完走順データも妥当性チェック済み: 5試行すべて 0..31 の正当な順列で相異なり、`_work_size` でソートされた形跡が無い(前半/後半の平均仕事量が 6.5/6.5, 7.0/6.0, 5.88/7.12, 6.12/6.88, 6.62/6.38)= **手で作った出力ではなく実スケジューラのノイズと整合**。
Task 4: complete / Task 5: complete / Task 6: complete (commits 2c9cf49..9ecda8e, review clean after 2 fix rounds)

Task 4+5+6: minor (deferred, **T18 で必ず拾うこと**): `s4_libm.py:19-21` の**10進桁数の誤り** — `[-1e6,1e6]` は6桁、`[-1e4,1e4]` は4桁なのに「約2桁強」「約2桁弱」と書かれており、同じ差を強とも弱とも言っている。**Finding C でまさに「唯一確実な事実」として据えた文がこれ**であり、記事はこの docstring から書かれる。検証可能な誤りなので未検証機構より質が悪い。
Task 4+5+6: minor (deferred): Finding A の共有ストリーム性質を**固定するテストが無い**。`RandomState` が将来 bit generator を採用しなくなれば、プローブは緑のまま静かに交絡を測る状態に戻る。1行で塞げる: `RandomState(MT19937(_SEED)).random_sample(k) == Generator(MT19937(_SEED)).random(k)`。
Task 4+5+6: minor (deferred): `s1_prng.py:51` の見出しが「3本の比較はそれぞれ次だけを単離する」と書きつつ第3項が「何かの単離ではない」と自ら述べている / Finding E に覆うテストが無い(S1 の hex も `len(x)` も、修正前でも通るテストしか存在しない)/ `s1_prng.py:94` だけ `_N` のまま(S4 の双子は `len(x)` に変更済み)/ `seeding="bitgen_seedsequence"` ラベルの接頭辞 / `_SELF_REPORTED_STABILITY` が手書き集合。
注記: out-of-scope で挙がった `s4_libm.py:5-6` の「AVX-512 と非 AVX-512 で ~4 ULP 差」は**調査で裏が取れている**(NumPy issue #23523、メンテナ公認)。C と同種の未検証主張ではない。

=== バッチ2: T7+T8+T9 ===
Task 7+8+9: implementer DONE_WITH_CONCERNS — commits 0a6e854 / bf3e2ec / c716989、標準スイート 94 passed / 9 skipped(81 から回帰なし)。
アドホック検証コンテナで**実際に実行したプローブ**: S5/S6(sklearn 1.7.2 + lightgbm 4.6.0)全6本、S7(CPU torch)両方(OMP 1/4 スレッドで5回反復し安定を確認)、**S9 は実物の Qwen2.5-0.5B-Instruct で9本すべて end-to-end 実行**(transformers が temperature=0 + do_sample=True に対して投げる ValueError を実地で確認)。
**S8(GPU)6本はどこでも未実行** — このサンドボックスからコンテナ経由で GPU に到達できないため。CPU-only torch に対して "missing requirement: cuda" で正しくスキップすることだけ確認済み。TF32 トグル、CUBLAS_WORKSPACE_CONFIG の正経路、bf16 学習、**特に `S8_curand_randn_sizes`** は一度も実ハードで動いていない。

Ruling 27: S8 の実地検証を**コントローラ自身が anakin で行う**。anakin に `vllm/vllm-openai:v0.20.1`(torch 同梱、23.3GB)が既にあり新規 pull が要らない。T14 で共用ホストを占有してから API ミスが発覚すると、4ホスト分の段取りをやり直すことになるため、レビューと並行して先に潰す。なお anakin の GPU は現在 59% 使用・2.3GB 占有(他ユーザーが使用中)だが、S8 の最大プローブでも 67M float32 = 約268MB なので 24GB カード上で短時間なら支障なし。

実装者が**隠さずに表に出した発見**(要検証): `S5_kmeans_centers` は `random_state` とスレッド数を固定しても**同一プロセス内で非決定**。sklearn の OpenMP 並列 E-step のスレッドスケジューリング由来の浮動小数ジッタを、`n_init` の best-of-n 選択が別の局所最適の選択へ増幅する、という説明。単スレッド化で隠さず `stable_within_process` で自己申告し `_SELF_REPORTED_STABILITY` に追加した。Global Constraint の精神には合致。
**ただしレビュアーに強く検証させている**: 「`OMP_NUM_THREADS=1` でも不安定が続くなら、スレッドスケジューリングは説明になりえない」(単スレッドの OpenMP リダクションは順序が固定)。env var がライブラリのスレッドプール初期化前に効いていなかったか、あるいは真因が別(未シードの RNG / 周囲のグローバル状態 / dict・set の反復順 / `random_state` が全段に届いていない)かのどちらか。**未診断の理由で非決定なプローブに、自信のある説明を付けて出荷する**のは、このバッチで最も警戒すべき失敗モードそのもの。しかも `_SELF_REPORTED_STABILITY` に入れるとスモークテストから永久に見えなくなる。

=== S8 の実地検証(コントローラ実施、anakin / RTX 4090 / torch 2.11.0+cu130)===
S7/S8 全8本が GPU 上でクリーンに実行、全て in-process で安定。SM 数 128、cc 8.9、TF32 既定値は matmul=False / cudnn=True(調査どおりの非対称)を実機で確認。
`S8_curand_randn_sizes` は4サイズそれぞれ別ダイジェストを記録できている。TF32 on/off は**実際に別ダイジェスト**(730bb75… vs fcda0d2…)= プローブは機能している。

**実行して初めて判明した重大な設計欠陥**: `S8_mlp_train10_cuda` と `S8_mlp_train10_cuda_deterministic` が**同一ダイジェスト**(8644dd3aff0f6319)。deterministic モードを入れても何も変わらない = **このプローブは GPU の非決定性を原理的に捉えていない**。
仮説を立てて実測で検証した(anakin, 5回反復):
| 演算 | 既定 | use_deterministic_algorithms(True) |
|---|---|---|
| MLP(Linear+ReLU)10step 学習 | **5回ともビット一致** | (変わらない) |
| `index_add_`(反復 index、atomicAdd 経路) | **一致せず、max\|diff\| = 8.240e-04** | **5回ともビット一致** |

Ruling 28: S8 に **atomics ベースのプローブを追加する**(batch 2 の fix round に統合)。理由: Linear+ReLU の MLP は CUDA 上で完全に決定的であり、**L4(GPU 非決定性)の実証にならない**。cuBLAS GEMM は単一ストリーム・固定ワークスペースなら run-to-run で決定的。非決定性は atomicAdd を使う演算(scatter_add / index_add / embedding backward / 一部 pooling backward)に局在する。コスト: プローブ2本追加。放置した場合、**記事が「GPU は非決定的」と書きながら、添付した実測データは全部一致している**という自己矛盾した状態で公開されていた。

記事への影響(第6章 G4 を書き換える): 「GPU は非決定的」という大雑把な主張は**誤り**。正しくは (1) GPU 演算の大半は run-to-run でビット一致する、(2) 非決定性は atomics を使う特定可能な演算群に局在する、(3) `use_deterministic_algorithms(True)` はその演算に対して実際に効く(本実測で確認)。**納品時に何を約束できるかを判断する読者にとって、こちらの方がはるかに有用**。しかも全部一次実測。
注記: anakin に `~/reprobench-check/` を残置(検証用)。T14 完了後に削除すること。

Task 7+8+9: review returned (opus) — Spec ❌、**Critical 3 / Important 5 / Minor 8**。23プローブすべて存在、過不足なし。構造面は全て正しいと評価(5モジュールとも関数本体 import、23本すべて `requires=` 正確、`_tf32_flags` の save/restore と `with` 内での `_device_meta` 呼び出し、bf16 の `.float()` 昇格、`CUBLAS_WORKSPACE_CONFIG` を `import torch` 前に確認、`_MODEL_CACHE` が (tokenizer, model) のみ保持し全 S9 が reseed)。S6 が S5 の `_split` 再利用を拒んで sklearn を到達可能性から外した判断は「非自明な good catch」と名指し。

**Critical 3件 + 私の実機発見1件、いずれも「測定後に復元不能」:**
Critical 0(私の発見、Ruling 28): S8 の学習プローブは **Linear+ReLU の MLP が CUDA 上で決定的なため、GPU 非決定性を原理的に実証できない**。anakin 実測で確定。atomics プローブ2本を追加。
Critical 1: `S5_kmeans_centers` のコメント(`s5_sklearn.py:161`「OMP_NUM_THREADS=1 でも再現した」)が**実装者自身の報告(`task-7-9-report.md:116`「1に固定すると 5/5 完全に安定」)と正面から矛盾**。同じ主張がテスト2箇所にも伝播。単スレッドの OpenMP リダクションは順序固定なので、出荷された側が自己矛盾。
Critical 2: `S5_kmeans_centers` が**クラスタの無いデータをクラスタリングしている**。`standard_normal((20000,30))` は等方ガウス1個で、分類プローブ用のデータセット。30次元・構造なしで k=4 は近接同値な局所最適が大量にあり、`n_init=10` の best-of-10 が ULP ノイズで反転する(102反復かかっている)。正直な所見は「構造のないデータでの `n_init>1` KMeans はカオス的」であって「`random_state` を固定しても sklearn KMeans は非決定的」ではない。**後者が今の枠組みが記事に言わせようとしているもの**。→ 真のクラスタ構造を持つ seeded データセットを与えて再測定。安定になればそれも所見。`_SELF_REPORTED_STABILITY` の免除は不安定が残る場合のみ維持。あわせて `cluster_centers_` の行順を lexsort で正規化(KMeans は重心をソートしないため、**同一のクラスタリングを見つけた2ホストが行順違いで別ダイジェストになり、比較器が存在しない不一致を報告する**)。
Critical 3: **S8 の matmul と CUDA 学習プローブが、`S8_curand_randn_sizes` がまさに測ろうとしている RNG で入力を生成している**。2048×2048 = 4,194,304 は `_CURAND_SIZES` の値そのもの。単一ホスト内では清潔(私の anakin 実測が正しく見えたのはこのため)だが、**ホスト跨ぎでは `a` と `b` が別の行列でありうる**。→ 入力を CPU 生成して `.to("cuda")` で移し、演算と RNG を分離。入力のダイジェストも meta に記録。
Critical 4: **S9 の temperature プローブが temperature を単離していない**。Qwen2.5-Instruct の `generation_config.json` が `top_p=0.8, top_k=20, repetition_penalty=1.05` を同梱しており、warper は temperature スケーリングの**後**に適用される。つまり「temperature 1.2」ではなく「temperature 1.2 を上位20トークン・累積80%に切り詰めたもの」。実装者自身の出力(t=1.2 で完全に筋の通った散文)がこれを裏づける。0.5B モデルの無制約 t=1.2 は通常そうならない。→ 全 S9 プローブに `generation_config.to_dict()` を記録(初回実行で経験的に決着)、かつ S1 と同じ形で「単離版(`top_p=1.0, top_k=0, repetition_penalty=1.0`)」と「現実のデフォルト版」に分割。logit margin は単離設定で測る。

Important 昇格2件: `meta["digests"]` の int キー(JSON で文字列化され Task 11 の比較器で破綻)、S5/S6 が**異なる分割なのに同じ `n_test: 4000` を報告**(S5 は層化 train_test_split、S6 は permutation)。
Important: 環境変数由来の意図的スキップ `"missing env var: ..."` がスモークテストの `"missing requirement: "` 判定に落ちる → **GPU ホスト全台でスモークテストが落ちる**(現在はバレていないだけ)。`use_deterministic_algorithms(False)` を finally でハードセットしており**同一ラン内の後続プローブを汚染**。S9 が device を記録せず、かつモデルを CUDA に移していないため **GPU ホストでも黙って CPU 実行**。`s7_torch_cpu.py:70` の dtype が後付けキャスト後の numpy dtype。`torch_dtype=` が非推奨。
Task 7+8+9: fix round 1/5 dispatched — Critical 4件(うち1件は私の実機発見)+ Important 7件 + 小修正3件。

Task 7+8+9: fix round 1/5 — commit 0d8a8c6、**全13件対応**。標準スイート 103 passed / 14 skipped。sklearn+lightgbm コンテナ 14 passed(3回反復、フレーキーでない)、CPU-torch コンテナ 16 passed、**実物 Qwen2.5-0.5B コンテナ 10 passed(max_new_tokens=128 のフル設定)**。
Critical 1 の決着: 新しいトランスクリプトで再現確認 — **`OMP_NUM_THREADS` を固定値(1 でも 4 でも)に固定すれば 5/5 一致、固定しなければ不一致**。コードのコメントが誤りで報告書が正しかった。3箇所すべて修正。
Critical 2 の決着: 4ブロブの専用データセットに変更(sklearn.datasets 不使用)。20回フィットで再測定 → `n_iter_` が常に 2(局所最適カオスではなく**支配的な最適解が1つ**に変わった)、`inertia_` は 20回中 4回 約1 ULP のジッタ。**スレッド数未固定下での不安定性は残る**ため自己申告の免除は維持。lexsort による行順正規化と `effective_thread_count`(threadpoolctl 経由)を追加。
→ **記事にとってこちらの方が良い所見**: 「`random_state` を固定しても KMeans は非決定的」ではなく「**スレッド数を固定しなければ再現しない**」。実務の指針として直接使える。

=== S8 実地再検証(コントローラ実施、anakin / RTX 4090)===
全10本が実行成功。決定的な結果:
- `S8_index_add_atomics`: **stable_in_process=False** ← 新プローブが GPU 非決定性を実際に捕捉
- `S8_index_add_atomics_deterministic`: **stable_in_process=True** ← deterministic モードが実際に効く
- TF32 on/off のダイジェストが修正前から変化(fcda0d2→e3a7029、730bb75→98e7edb)、`S8_mlp_train10_cuda` も 8644dd3→226fb72 = **入力の CPU 生成への切り替えが実際に効いている**ことの確認
- `S7_mlp_forward` の dtype が float64 → **float32** に修正されている(Important 7 の確認)
- `S8_mlp_train10_cuda` と `_deterministic` は依然同一ダイジェスト(226fb72c7a259433)。これは**今や既知の所見**であり沈黙した失敗ではない — MLP は決定的で flag に直すものが無く、L4 の実証は atomics プローブが担う。
→ **GPU 章の筋が実測で完全に通った**: (1) GPU 演算の大半は run-to-run でビット一致、(2) 非決定性は atomics を使う演算に局在、(3) `use_deterministic_algorithms(True)` はそれに実際に効く。全て一次実測。
注記: transformers 5.19.0 は `dtype=` を受け付ける(Task 12 のピン留め制約)。`repetition_penalty` は実測で 1.05 ではなく 1.1(バージョンドリフト、所見は変わらず)。

Task 7+8+9: fix round 1 re-review(opus)— 0/2(機構)/3/4/5/6/7/8/9/10 と小修正3件は ADDRESSED。私が最重点を指示した2点はいずれも成立: finding 4 の単離は `_ISOLATING_KWARGS` が transformers が判定に使う閾値ちょうど(`top_k=0`/`top_p=1.0`/`repetition_penalty=1.0`)で、かつ **`generate()` の明示 kwarg なのでモデル同梱 config を上書きする側**。finding 3 の分離も本物で、デバイス側の draw は測定対象の `S8_curand_randn_sizes` のみ。
**しかし finding 1 は未対応、かつ fix が新たな Critical を生んだ。**

Ruling 29(重要・記事の所見が変わる): レビュアーが実装者のコンテナ構成(python:3.11-slim / numpy 2.2.6 / sklearn 1.7.2 / 8 CPU)で**30回×6条件の再測定**を行い、置き換えた説明も偽であることを示した:
| 条件 | 相異なる重心ダイジェスト / 30 | n_iter_ | inertia_ 相対広がり |
|---|---|---|---|
| OMP=1 | 1 | 2 | 0 |
| OMP=2 | 1 | 2 | 0 |
| OMP=4 | 1 | 2 | 0 |
| **OMP=8(固定、nproc と同値)** | **13** | 2 | 1.9e-16 |
| 未設定 + OMP_DYNAMIC=FALSE | **15** | 2 | 3.9e-16 |
| 未設定(既定) | 19 | 2 | 3.9e-16 |
→ 「固定値にすれば安定」は**偽**(8 は固定しても不安定)。「スレッド数が呼び出し間で変動している」も**偽**(OMP_DYNAMIC=FALSE が効かない)。実装者の5回トランスクリプトはデータと整合していたが、**判別する唯一の値を試していなかっただけ**。
測定が支持する所見(記事にはこれを書く): **スレッド数そのものが変数** — 1/2/4 では安定、8 CPU 機で 8 にすると不安定。そして**不安定性は解の選択ではなく縮約の丸め** — 180フィット全てで `n_iter_ == 2`、inertia の広がりは最大 2 ULP。さらに**各スレッド数が固有の安定したダイジェストを持つ**(1→972e0934…, 2→42e031c0…, 4→034d84cd…)ため、**このプローブのホスト跨ぎ比較は `effective_thread_count` が一致するときのみ有意**。

Ruling 30(以後の規律): このバッチで**3度**、測定していない機構が自信を持って説明として付けられた(当初の「スレッドスケジューリング」、「1スレッドでも再現」、その置き換え)。以後、コメント・docstring・meta に書いてよいのは**実際に測った相関のみ**。変化させなかった因子を原因として名指ししてはならない。「X のとき不安定、Y のとき安定、N 回反復で測定」と書いて止める。**未測定の機構は説明が無いより悪い** — 記事はこれらのファイルから書かれ、実測を添えて公開された誤った機構は確立した事実として読まれるため。
finding 1: 3箇所のうち `tests/test_ml_probes.py:23-29` が**未修正のまま**。反駁済みの「スレッド数を固定しても一致しない」と、自身の再測定が撤回した「n_init=10 のベスト選択が増幅」の両方を含む。
I1(新規 Important): プローブが**自分の診断をホスト上で再現できない** — 2回目のフィットの `n_iter_`/`inertia_` が bool に畳まれて破棄され meta に届かない。「縮約の丸めであって別の最適解ではない」の主張は `n_iter_` 不変性に乗っているので、ホストが `stable_within_process=False` を報告したとき**今回分離したばかりの2つのケースを記録から区別できない**。
Task 7+8+9: fix round 2/5 dispatched — C1 + finding 1 の残り + I1 + 小修正4件 + S9 のスモークテスト未実施カバレッジ。

Task 7+8+9: fix round 2 — commit 2beaf70。**全件 ADDRESSED、新規 Critical/Important なし**。
実装者は散文に触れる前に自分で30回×6条件を再測定し、3つの安定スレッド数のダイジェスト先頭(972e0934/42e031c0/034d84cd)まで一致を確認。さらに**自分の round-1 テキストを新ルールで監査して1件自力発見**(`_effective_thread_count` の docstring が検証コンテナの事実を「4台の実測ホスト全部」に一般化していた)。
カバレッジの穴も解消: 実物モデルコンテナで**スモークテストを初めて実行**(13 passed / 176s)。S9 の sampling_t1_2 と batch8/fp32/bf16/fp16 greedy の in-process 安定性がこれで初めて検証された。**免除が必要なプローブはゼロ**で、それを所見として報告している。
コントローラによる anakin 最終確認: 全10本実行、**変化したダイジェストは `S8_index_add_atomics` のみ**(e2d7a07→fae96c5)= 毎回変わる設計のプローブ。他が全て不変であり、round 2 が測定挙動に一切触れていないことの独立確認。

Ruling 31: レビュアーが「scope ルール上ブロックしない」とした残り2件について、**例外的に3回目の fix round を回す**。理由: 2件とも**記事が書かれるテキストそのもの**であり、standing rule(測った相関のみを書く)が守ろうとしている対象そのものだから。(1) 撤回済みの「n_init=10 のベスト選択が増幅」という枠組みが `s5_sklearn.py:149-154` と `:132-134` に生存している — しかも round 1 が再測定していない**旧データセット**について語る未測定の機構。自己監査が部分的だった。(2) コミットされた測定表が独立実行と 6行中2行で食い違い、**inertia_ 広がりの列が "8" 行と "unset" 行で入れ替わっている**(16/15/17 vs 13/15/19、3.9e-16/1.9e-16 vs 1.9e-16/3.9e-16)。定性的主張と安定ダイジェスト3つは完全一致するので所見は揺るがないが、**distinct-digest 件数はラン毎に変動する量であり定数のように表に置くと記事が「30回中16回」と引用してしまう**。
コスト: 短い1ラウンド。放置した場合、撤回済みの機構と入れ替わった数値が T18 まで残り、文脈が失われた状態で拾う羽目になる。
あわせて standing-rule 系の小修正7件(既定環境への過度な一般化、n_iter_ 単独からの推論→測定済みの inertia 広がりと max_abs_center_delta を引用、2つの OpenMP ランタイムを全順序化しないソートキー、bf16 の「切り捨て」→最近接偶数丸め、許容誤差を「測定値」ではなく「安全側の上界」と記述、S9 meta の全体 JSON 等価性、atomics の機構主張に PyTorch 公式 docs を出典として付す)。
Task 7+8+9: fix round 3/5 dispatched。

Task 7+8+9: fix round 3 — commit 2de13f7。Residual 2 と小修正 1/3/4/5/6/7 は ADDRESSED。書き直された測定表は「1回の30フィット実行の記録」と明示し、**どちらか一方を正とせず2回のランの食い違う数値を並記**、共通点を定性パターン・安定ダイジェスト先頭・`n_iter_`・≤3.9e-16 の上界に正しく絞っており、食い違いを過大にも見せていないと評価。
**しかし Residual 1 が3度目の生き残り**: `tests/test_ml_probes.py:104-106` に「the classification dataset was not built for clustering, **so k=4 on it had a huge family of near-tied local optima**」が残存。前提節だけ `s5_sklearn.py` の修正に合わせて書き換えられ、**因果の帰結節が手付かず**。しかも今回の差分が他の理由で編集したまさにそのハンク内。3ラウンド連続で、狙った行を直すたびに同じ主張の写しが同一ファイルの別の場所に残っている。
小修正2も同じ箇所で部分対応: `s5_sklearn.py` の docstring は正しく「`n_iter_` 不変だけでは証拠にならない」と明記して inertia 広がりと `max_abs_center_delta` を引用する形に直ったが、テスト側は「n_iter_ が変わらない = one dominant optimum の徴候」という置き換えを求めた推論のまま。

Ruling 32: fix round 4 では**行番号指定の編集ではなく、`tests/test_ml_probes.py` と `s5_sklearn.py` を端から端まで読ませ**、撤回済みの語彙(局所最適 / near-tied / amplif / best-of / signature of)を grep して全ヒットを実測と突き合わせさせる。理由: 狙撃的な修正を3回繰り返して3回とも取りこぼしているため、同じ手法の4回目は期待値が低い。
Ruling 33: 実装者の「このラウンドはテキストのみ、測定ロジックには触れていない」という報告は**不正確**だった。`_thread_pool_info()` のソートキーが 2-tuple → 3-tuple になっており、これは `meta["effective_thread_count"]`(モジュールの相関表全体が依存するフィールド)に供給される実行ロジック。CPU 専用・sklearn プローブ限定で、テスト済みコンテナは全て OpenMP ランタイムが1つのためタイが発生せず観測可能な影響なし → **GPU 再検証は不要**と判断。ただし報告の文言は訂正させる。**この種の文を根拠に anakin を再実行するかを決めているため、包括的な「テキストのみ」は再導出なしに行動する類の記述**だから。
Task 7+8+9: fix round 4/5 dispatched(残り2箇所 + 報告の訂正 + docstring 1行)。

Task 7+8+9: fix round 4 — commit 5a541ba。
Ruling 34: round 4 の検証は**サブエージェントを立てずコントローラが直接行った**。理由: 残件が「撤回済みの語彙が残っていないか」という機械的に判定可能な問いで、grep の方がモデルの読解より確実であり、差分も2ハンクのコメントのみで全文を直接読めるため。独立性より確実性と比例性を優先。結果:
- `grep -rniE "near-tied|amplif|best-of|signature of|局所最適" reproducibility/reprobench reproducibility/tests` → **ヒットゼロ**
- 差分全文を確認。帰結節は削除され、前提のみ(「二値分類のために作られたデータセットであり、だから S5_kmeans_centers は自前のデータセットを使う」)。さらに「`n_iter_` 不変だけでは2反復で到達した別の最適解を排除できない」と明記し、直接証拠として inertia 広がり(≤~3.9e-16)と `max_abs_center_delta`(~6.9e-18)を引用。ソートキーの辞書式順序も文書化済み。
実装者は今回 5ファイルを端から端まで grep し、報告の訂正も**履歴を書き換えず round-3 セクションに訂正注記を追記**する形で行った。
Task 7: complete / Task 8: complete / Task 9: complete (commits 9ecda8e..5a541ba, 4 fix rounds)

**プローブ実装フェーズ完了。登録プローブ総数 25+(S0〜S9)。標準スイート 104 passed / 14 skipped。**
T11(比較器)に持ち込む必須の判定ルール(ここまでの実測で確定):
- `S5_kmeans_centers`: **`effective_thread_count` が一致するホスト間でのみ**ダイジェスト比較が有意。不一致でもスレッド数が違えば「差あり」と数えてはいけない。
- `S8_index_add_atomics`: **不一致が正常**(非決定性の実証that自体が目的)。`stable_within_process=False` を自己申告する。
- `S0_parallel_completion_order`, `S0_glob_order`: 同上、`stable_within_process` を見る。
- `S0_dict_from_set` と `S0_set_iteration_order`: **構造上常に一致**するため、2本で1ビットの情報。独立した2つの確認と数えない。
- `S9_logit_margin` と `S9_greedy_batch1`: `repetition_penalty` が違うため**同一の greedy 軌跡ではない**。同じランとして扱わない。
- S5 と S6 は**異なる分割**(層化 vs permutation)。`train_idx` のダイジェストが meta にあるので、指標を直接比較しない。

=== T10: ランナーCLI ===
Task 10: implementer DONE — commit 03d58f6、24テスト(brief の4本 + 全 resolution を覆う20本)、**124 passed / 14 skipped**。
コンテナでの end-to-end 実行確認: **53プローブ登録、26本が実ProbeResult、27本が SkippedProbe**(全て `missing requirement: <torch|sklearn|lightgbm|transformers|hf_model>`)。既定の `results/{host}__{image}__{config}.json` パスと上書き拒否(exit 1、stderr、元ファイル無傷)もコンテナ内で実地確認。自己レビュー中にセレクタの trailing-comma バグを自分で発見・修正。
Resolution 8 の判断: **上書きを拒否**(非ゼロ終了、何も書かない)。理由は「既存ファイルは実際に占有した共用ホスト時間を表しており、それを黙って失うのは空ファイルと同じ失敗クラス」。妥当と判断し採用。
Ruling 35: `main(argv: list[str] | None = None)` という brief の literal(`main() -> None`)からの逸脱を**採用する**。sys.argv を monkeypatch せずにユニットテストできるようにするためで、引数なし呼び出しは brief の仕様と同一挙動。テスト可能性のための純粋な拡張であり、契約を変えていない。
Task 10: review dispatched (sonnet) — diff 5a541ba..03d58f6。重点: (a) JSON 往復テストが**本物のペイロード**か(手組み dict の往復は `run()` が実際に書くものについて何も証明しない)、(b) 比較器が**このファイルだけで** `{host}/{image}/{config_label}` のラベルを再構成できるか、(c) `"raised: "` のスキップ理由が要約・切り詰め・正規化されずに**逐語で**残るか(ここが失われると、ホスト上で壊れたプローブを記事が「この環境では非該当」と報告しうる)、(d) 共用ホスト上での失敗モード(ハング・書込不可・中断時に**完全に見える不完全ファイル**が残らないか)。

Task 10: review returned (sonnet) — Spec ✅(8つの resolution と brief の literal interface すべて)、Critical 0 / Important 3 / Minor 5。
レビュアーは各 resolution を個別に追跡: 登録を `run()` 内で行うため `main()` が忘れられない、空選択の2経路とも実プロセスの非ゼロ終了、スキップ理由は無変換パススルーで prefix 非依存(`registry.py` を named-risk read で確認)、`probe_order` がループが回すのと同じリスト由来なので「計画した順」でなく「実際に回した順」、`host`/`image`/`config_label` が平坦な文字列として `env` に到達し比較器がラベルを再構成可能。JSON 往復テストが**本物の `run(selected=None)` ペイロード**であることも確認。`python:3.11-slim` が UTF-8 既定であることまで実地確認して `ensure_ascii=False` + `write_text()` の安全性を裏取りしている。
Important 1(良い指摘): **プローブを全部実行してから出力ファイルの存在を確認している**。`runner.py:143-144` で測定ワークロード全体を回し、`:151` で初めて `out_path.exists()` を見る。既に埋まっているパスへの誤った再実行は、**共用ホストの占有枠を1つ丸ごと使い切ってから拒否される**。実装者自身の「既存ファイルは実際に占有した共用ホスト時間を表す」という論理が、この順序では2枠目を捨てて1枠目を守る形になっている。`--out` ならパスは即座に判明し、既定パスも `collect_env()` だけで決まるので両方とも前倒し可能。
Important 2: 報告の「24テスト」が diff の 20(literal 4 + 16)と食い違う。報告内の後段の算術は 20 と整合しており見出しだけが誤り。
Important 3: **スキップ理由が切り詰められないことを無条件に証明するテストが無い**。該当テストが `if "skipped" in entry:` で囲まれており、torch がある環境では本体が実行されず素通りする。決定的にスキップを強制するテストの方は `"skipped" in p` しか見ていない。挙動自体は正しい(`_run_one` は無変換パススルー)が保証が固定されていない。

Ruling 36: **ハングしたプローブへの番犬は測定プロセス内に置かない**。レビュアーの指摘(ハング時に完了済みプローブも道連れ、ファイルは最後に1度だけ書かれるため)は正しいが、シグナルベースのウォッチドッグは**第4のスキップ分類**を生み、スモークテストと比較器に波及する。実際の損失は「1ランの再実行」に留まるので割に合わない。**番犬は1層外のシェル** — Task 13 のホストスクリプトでコンテナを外側の `timeout` 配下で走らせる。runner の docstring にその旨を1文書かせ、次の読者が同じ問いを再導出しないようにする。
あわせて atomic write(temp + `os.replace`)、`--repeat <= 0` の暗黙クランプをエラー化、`mkdir`/`write_text` の OSError を他の2経路と同じ clean な非ゼロ終了に統一。
Task 10: fix round 1/5 dispatched。

Task 10: fix round 1 re-review(sonnet)— **全件 ADDRESSED、新規 Critical/Important なし**。
レビュアーは Important 1 の順序テストを**両方の順序で手動トレース**して検証: 旧順序では stub の `AssertionError` が `SystemExit` でも `ValueError` でもないため `pytest.raises(SystemExit)` を素通りして落ちる = **テストは旧順序で確かに失敗する**(空虚でない)。Important 3 のテストも環境非依存であることを確認(`check_requirement` を常に False に monkeypatch し、`S7_mlp_train10` の `requires=("torch",)` が単一要素なので torch の実在に関係なく理由文字列が決定的)。アトミック書き込みの一時ファイルが同一親ディレクトリであること(`os.replace` の原子性の前提)、`.tmp` が宛先を占有して前倒しチェックを破らないことも確認。`collect_env()` の二重呼び出しが意図的かつ無害(host/image/config_label は静的 env var と `platform.node()` 由来でプロセス中に変化しえない)であることも検証。
Task 10: minor (deferred): `write_text`/`os.replace` 自体が OSError を投げた場合に `.tmp` が掃除されない(`mkdir` 失敗経路では tmp は作られないのでテスト済み経路では起きない)。宛先は壊れず、前倒しチェックも破られず、再実行時の無駄な再測定も起きないため純粋に `results/` のゴミ。
Task 10: complete (commits 5a541ba..56f2eb3, review clean after 1 fix round)

=== T11: 比較器 ===
Task 11: dispatched (implementer, sonnet) — BASE 56f2eb3

Task 11: implementer DONE — commit 578c577、21テスト、150 passed / 14 skipped。宣言的 `_POLICY` 表、raised-skip 専用列、グループ化、linked pair の相互注記、cross-probe caveat を実装。3本の実結果ファイルを生成して end-to-end 確認。
Task 11: review returned (opus) — Spec ❌、**Critical 1 / Important 3 / Minor 7**。ポリシー層・S8 の非対称性・スキップ分類・数値経路は正しいが、**レンダリングされた markdown が resolution が守ろうとした区別を2つ落としている**。
レビュアーの評価点: `_POLICY.get(probe, _DEFAULT_POLICY)` が**完全一致ルックアップ**で、`S8_index_add_atomics_deterministic` が独自の `normal` エントリを持ち免除に巻き込まれていない(「最も間違えやすい箇所で、実際に正しい」)。`absent` と4つのスキップ分類を**ポリシー分岐の前に全 kind について計算**しているため、免除されたプローブ上の `raised:` も必ず浮上する = **免除がクラッシュを飲み込めない**。ポリシー注記が機構を主張せず測定を記録したモジュールを引用している点も spot-check 済み。

**Critical: グループ化された行が、データが支持しない一致を公表する。** `compare.py:501-508` がグループ分岐で行レベルの `missing_col`(`:483`)ではなく `agg["missing"]` を描画しているが、sub-agreement の `agg["missing"]` は**構造上常に空**(`sub_results` が `group_labels` に絞られ、`group_labels` は既に非スキップのエントリを持つラベルしか含まない)。したがって sklearn が無いホストや S5 を走らせなかったホストが「欠落なし」と描画される。さらに `_verdict_symbol` が1グループのとき `✓` を返す(`all_equal = len(groups) <= 1`)ため、**観測1個のグループが「比較対象の全ラベルで一致、欠落なし」として出力される**。
**実装者自身の end-to-end 出力(task-11-report.md:94)にこの偽の一致がそのまま写っており、成功として読まれていた。** 記事の証拠表が読み出される出力で、このタスクが名指しする支配的な失敗モードが実際に発生していた。

Important 1: `effective_thread_count` が未決定のランが互いに比較される(`None` が文字列 `"None"` のグループキーになる)。`_effective_thread_count()` は `int | None` 宣言で `threadpool_info()` が空なら `None` を返し、その docstring 自身が「4台の実測ホストでは未確認」と書いている。**スレッド数が違うだけかもしれない2ホストの差を、ホスト跨ぎの不一致として公表しうる**。
Important 2: 観測1個が通常行でも `✓` になる。データ層(`all_equal=True`)は brief 規定なので触らず、**描画のみ**を区別させる。
Important 3: **同一ラベルの結果ファイル2本が黙って潰れ、本物の不一致が `✓` になりうる**(last-write-wins)。1ホストで1設定を再実行するのは**記事が測るホスト内再現性の条件そのもの**なので仮想的な話ではない。
Important 4(私が昇格): `rounds` が黙って無視されており、`--repeat` で集めたデータを読む経路が無い = **記事の9条件のうち1つに分析経路が無い**。比較器に per-label のラウンド間比較を追加させる。本ラウンド唯一のスコープ追加。
昇格した小修正: `raised_col` がグループ行ごとに再描画され1つのクラッシュが N 回出る / `compare_arrays` の `float("inf")` が `json.dumps` で裸の `Infinity` になり下流パーサを壊す(形状不一致はホスト跨ぎで起こりうる、`allow_nan=False` で assert)/ `None` の自己申告が「unstable」と描画される(測っていない値を測定された不安定性として公表)/ 凡例が欠落列を「いずれも想定内で無害」と言うが `unknown_skip:` もそこに入る / linked pair の markdown 描画が未テスト。
レビュアーの偽陰性監査からの採用: **比較器は linked pair の2プローブが互いに一致することを一度も確認していない**。「構造上同一」という前提こそが「2本で1ビット」の根拠なので、前提が壊れたら記事の主張が偽になるのにレポートはそれを示さない。安価な assert で、文書化された仮定を検査された仮定に変える。
Task 11: fix round 1/5 dispatched。

Task 11: fix round 1 — commit 0011718、**全件 ADDRESSED、新規 Critical/Important なし**、165 passed / 14 skipped。
`_verdict_symbol` が観測数を数えて `n=1`(1個)/`–`(0個)/`✓`・`✗`(2個以上)を返すようになり、例の行は実データで `n=1` に変わった。`None` のスレッド数は `ungroupable` に分離。ラベル衝突は専用列に出し、**digest が一致する衝突は conflict としない**(偽陽性を作らない)ことも別テストで確認。`rounds` は `rounds[0] == probes` を踏まえてラウンド1を二重計上しない形で読む。linked pair の前提違反は `⚠ LINKED PREMISE VIOLATED` で描画。
レンダリング状態の総当たり監査: 11状態(一致 / 不一致 / 観測1個 / 自己申告 stable・unstable / 自己申告欠落=unknown / 不在 / クラッシュ / unknown_skip / グループ不成立 / ラベル衝突)がすべて互いに区別可能と確認。
Task 11: minor (deferred): `_fmt_labels` が dead code 化 / グループ継続行の空欄と `–` の区別が凡例を読まないと自明でない(表だけ見る読者が「無し」と誤読しうる)/ `label_for` の `"unknown"` フォールバックにより、host/image/config を報告できない2ホストが同一ラベルに潰れ、新しい衝突検出で偽の衝突を生じうる(Task 10 側の既存挙動)。
Task 11: complete (commits 56f2eb3..0011718, review clean after 1 fix round)

=== バッチ3: T12+T13(インフラ)===
Ruling 37: T12(Dockerイメージ7種 + Makefile)と T13(配布・実行スクリプト)を**1ディスパッチにまとめる**。両方ともインフラで、ファイルは互いに素(`docker/`・`requirements/`・`Makefile`・`README.md` 対 `scripts/`)。かつ **T13 のスクリプトは T12 の Makefile ターゲットを呼ぶので、別エージェントに分けると後者が前者のインタフェースを推測することになる**。同一エージェントなら検証も自然に繋がる。
Task 12+13: dispatched as one batch (implementer, sonnet) — BASE 0011718

Task 12+13: implementer DONE_WITH_CONCERNS — commits 53583e7 / cb09695。
このホスト(aarch64 Mac)でビルド成功: base / py313 / oldpy / oldlibs / alpine。mkl は x86_64 専用のため conda solve の dry-run のみ、gpu は GPU 非到達のため wheel 可用性のみ確認。リモートには一切成果物を残していないことを確認済み。
**実装者が隠さず報告した2件:**
(1) `oldlibs` に torch が入らない — cp310 の torch wheel(1.10.2〜1.13.1)が numpy==1.21.6 と ABI 非互換(`RuntimeError: Numpy is not available`)。
(2) 既存テスト `test_collect_env_survives_torch_module_not_found_error` が torch 不在を前提としており base の内容と矛盾 → `make test` が 178 passed / **1 failed**。実装者はこれを「きれいに見せようとはしなかった」と明記。
途中で実バグ2件を発見・修正: `libgomp1` 欠落(lightgbm の import 失敗)、**ssh を while ループ内で呼ぶことによる stdin 奪取**(anakin に対して実地検証済み)。

Ruling 38: (1) の主張は**テストしたアーキテクチャの artifact である可能性が高い**。実装者は Mac 上でビルドしており Docker は `linux/arm64` を作る。**torch は 2.x 系まで公式の aarch64 Linux wheel が存在しない**ため、測ったものは「ABI 非互換」ではなく「aarch64 wheel が無い」かもしれない。`oldlibs` が実際に走るのは **x86_64 の blade** なので区別は実務上重要。`--platform linux/amd64` ビルドで判別させ、**測った範囲に限定して記述**させる(standing rule 適用)。
Ruling 39: いずれにせよ **`oldlibs` に torch 無しを許容する**。`oldlibs` の目的は NEP 19 の numpy ストリーム互換性条件であり、それを測るのは numpy だけを要する S1〜S4。torch プローブは `missing requirement: torch` として正しくスキップされる。**古い torch を入れることは torch 自身のバージョン互換性を測ることになり、記事はそれについて何も主張していない**。README とイメージのコメントに「ライブラリ版条件は numpy と scikit-learn で測る」と理由付きで記録させる。
Ruling 40: (2) は**このバッチ内で直させる**。`make test` は記事の読者が最初に打つコマンドであり、失敗で迎えるわけにいかない。monkeypatch 版が同じ経路を無条件に覆っているので、実不在版は torch が import 可能なときに skip させる。base では通り、alpine では依然として意味を持つ。Task 2 のテストへのクロスタスク編集として理由付きで報告させる。
Task 12+13: fix round 1/5 dispatched。あわせて**コントローラが gpu イメージを anakin で実ビルドする**(T14 のリスクを先に潰すため。実装者にリモートビルドはさせない)。

=== GPU イメージの実ビルド検証(コントローラ実施、anakin)===
`docker build -f docker/Dockerfile.gpu` が anakin で**成功**。torch 2.14.1+cu130 / transformers 5.19.0 / numpy 2.2.6。
`docker run --gpus all` で torch が CUDA を認識(True、RTX 4090、SM 128、cc 8.9)。
`python -m reprobench.runner --probes S7,S8` が**完走し結果 JSON を出力**。全10プローブ OK、合計約2.2秒。env に torch 版/cuda 版/cudnn 版/TF32 両フラグ/GPU 名/SM 数/`CUBLAS_WORKSPACE_CONFIG=:4096:8` がすべて正しく記録されている。

**重要な副産物(記事に使える一次データ)**: この公式イメージ(**torch 2.14.1+cu130**)のダイジェストが、私が先に ad-hoc の vllm コンテナ(**torch 2.11.0+cu130**)で取ったダイジェストと**全プローブで完全一致**した(毎回変わる設計の `S8_index_add_atomics` を除く)。同一 GPU・同一 CUDA 13 の下で **torch のマイナーバージョンを跨いでビット一致**している。PyTorch 公式は「リリース間の再現性を保証しない」と明記しているが、**保証が無いことと実際に壊れることは別**であり、ここでは壊れなかった。記事の L1(明文の規約)の節で、「保証されていない」と「実際に変わる」の距離を示す実例として使える。

Ruling 41: **`gpu` イメージに pytest が入っていない**ことを発見。`COPY tests/ ./tests/` しているのに `pytest: not found` で実行できず、矛盾している。スモークテストは**プローブ契約(in-process 安定性・クリーンなスキップ)が自動検査される唯一の場所**であり、GPU ホストで測定枠を使う前に走らせられる価値が高い。`requirements/gpu.txt` に pytest を追加させる。コスト: イメージが数MB増えるだけ。
anakin の後片付け: スクラッチディレクトリ(`~/reprobench-gpubuild`、`~/reprobench-check`、`~/show_result.py`)を削除済み。`repro-gpu:check` イメージは T14 のビルドを速くするため残置(389GB 空き、91GB が reclaimable のホストで、ベースイメージの再 pull を避けるため)。T14 完了後に削除すること。

Task 12+13: fix round 1 — commit e813405。実装者が**自分の前回の主張を `--platform linux/amd64` で再検証して取り下げた**: torch 1.13.1(cp310)は numpy 1.21.6 と正常動作し、「ABI 非互換」は **aarch64 固有の artifact をアーキ非依存であるかのように書いていた誤り**だった。消さずに可視の訂正として残している(Ruling 38 の通り)。`make test` on base が 178 passed / **1 skipped** でクリーンに。lightgbm 4.5.0 と sklearn 1.0.2 が numpy 1.21.6 に対して**実際に fit/predict できる**ことを両アーキで確認。root 所有問題は `--user` ではなく chown-after で解決(`--user` だと gpu イメージの HF キャッシュが mode-700 の `/root` 配下で通れなくなるため)。`fetch_model.sh` にも同じ修正を適用。

Task 12+13: review returned (opus) — Spec ❌、**Critical 2 / Important 4 / Minor 9**。15 resolution の大半は充足、シェルの難所は正しいと評価(全 ssh に `< /dev/null`、`declare -f` + `printf %q` での関数輸送、`$HOME`/`id -u` の遠隔解決を quoting より先に行う、bash 3.2 の空配列回避のための `--gpus all` の畳み込み、`threadsmax` の `nproc` 失敗時に何もビルドせず中断)。**私が最重点で確認させた「遠隔側での kill」は正しい** — `run_container` を `declare -f` で輸送しているため `docker kill` はコンテナのある側で実行される。chown がタイムアウト経路でも走ることも確認。実装者が S5/S6 が `REPRO_THREADS` を直接読むことに気づいて6個目の変数を足した点は「real catch」と評価。

**Critical 1: 全リモート実行がコンテナ終了後も 480 秒ハングする。** ウォッチャー `( sleep … && docker kill … ) &` が親シェルの stdout/stderr を継承し、`kill "$watcher"` はサブシェルしか殺さない。`sleep` は別の子プロセスとして孤児化し、継承した fd を開いたまま生き残る。リモート実行ではその stdout は sshd のパイプなので、**孤児 `sleep` が満了するまで `ssh` が返らない**。レビュアーは同じパターンを `cat` にパイプして機構を実測検証(6秒 vs 0秒)。結果、各リモート行が約2秒の測定に対し8分 ssh セッションを保持 = 見かけ上ハングした数時間の campaign になり、**10分/ホストの予算が測定ではなくオーケストレーションで溶ける**。ローカルの end-to-end テストで見えなかったのは、ローカル経路では孤児が端末を保持し誰も待っていないため。
**Critical 2: `rsync --delete` が毎回 HF キャッシュを破壊する。** `hf-cache/` が `~/reprobench-run/` 配下にあるのに除外されていないため、`fetch_model.sh` の後の最初の `run_on_host.sh` が約1GB の Qwen スナップショットを削除する。`fetch_model.sh` 自身も再取得前に自分のキャッシュを消すので、**モデルを既に持つホストで no-op になりえない**。resolution 15 が禁じている multi-GB pull そのもの。

Ruling 42: Important 2(`mkl` イメージが非 root)を**コントローラが実測で確定**。`docker run --rm mambaorg/micromamba:2.0.5 id` → `uid=57439(mambauser)`。名指しリスクではなく事実。blade の枠を使って確認させる必要はないと実装者に伝達済み。
Ruling 43: run matrix(`run_all.sh` が30行、プロジェクトの枠組みは12ラン)は**このラウンドでは触らせない**。実装者は「brief のものではなく自分の計画」と正しく開示しており、**campaign 前に私が決める**。
Task 12+13: fix round 2/5 dispatched。

Task 12+13: fix round 2 — commit 546fb62。実装者が **Critical 1 を standalone A/B テストで先に再現**(旧 6s / 新 1s、私の検証とほぼ一致)してから修正。さらに**修正の副作用として `--rm` が `docker logs` と競合することに気づき**、`docker run -d` + `wait` + `logs` から単一フォアグラウンド実行へ構造ごと作り直した。Important 3 の深い修正(ssh 切断に耐える)は「同期的な collect-after-ssh-returns モデルを壊す」として**意図的に見送り、限定された残存ギャップとして文書化**。中途半端な再設計を避ける正しい判断。
小修正も実測付きで完了: alpine と oldlibs に pytest 追加後、**コミット済みイメージからフルスイートを実行**(alpine 165 passed/14 skipped、oldlibs 174 passed/5 skipped)。

=== リモート経路の実検証(コントローラ実施、anakin)===
`./scripts/run_on_host.sh anakin base threads1` を実行。**実装者が原理的に検証できない4点をすべて確認**:
- **Critical 1 解消**: イメージビルド込みで **37.9秒**(修正前なら480秒ハング)。
- **root 所有問題 解消**: リモートの結果ファイル所有者が `yutamasubuchi`、root ではない。chown-after が実 Linux ホストで機能。
- **残留コンテナ 無し**(`docker ps --filter name=reprobench` が空)。
- **resolution 12 機能**: `thread_env` の5変数すべてが `'1'`。
回収した JSON も妥当(schema 1、995KB、ローカルユーザー所有、host/image/config ラベル正しい、`libc: glibc 2.41` を `platform.libc_ver` 経由で取得、`numpy_cpu_features` 記録、34プローブ実行/19スキップ)。`anakin__nvidia-smi.csv` も resolution 11 どおり生成。

Ruling 44: この pre-flight で得た結果行は**削除した**(ローカル・リモート両方)。有効な測定ではあるが、再レビューがまだ走っており base イメージが変わる可能性がゼロではない。**再現性についての記事の campaign に、他の行と異なるイメージ状態から出た行を混ぜたくない**。コスト: T14 で38秒かけて取り直すだけ。
Ruling 45: `run_all.sh` の30行マトリクスを**承認**。spec §4-6 と T14 brief Step 1-3 まで辿れる形で構成されている(base×4ホスト×5設定=20、blade の nosimd512=1、blade のライブラリ系5本、dgxspark の alpine=1、gpu×3ホスト=3、計30)。spec の「12ラン」はイメージ×ホストの数であって設定の展開を含んでいなかった。T14 はこのマトリクスで実行する。
anakin に `~/reprobench-run/`(rsync 先の作業ツリー)を残置。T14 が再利用するため意図的。campaign 完了後に `repro-gpu:check` / `reprobench:base` / `reprobench:gpu` イメージとあわせて削除すること。

Task 12+13: fix round 2 re-review(sonnet)— Critical 2 / Important 1-4 / 小修正9件はすべて ADDRESSED。だが **Critical 1 の修正が、より重大な Critical を持ち込んだ**。
`run_on_host.sh:233-236`: `sleep` は `run_container` 自身のシェルの子。3行目のサブシェルは別個に fork されたプロセスであり、OS レベルで `sleep` はその子では**ない**。したがってサブシェル内の `wait "$sleep_pid"` は bash の「pid is not a child of this shell」で即座に失敗し、`&&` が短絡して **`docker kill` が一度も実行されない**。レースではなく決定的。レビュアーは2回再現(うち1回は end-to-end: 5秒タイムアウトに対し8秒のフェイクコンテナが kill されず、8秒丸ごと経過)。
**結果は直した bug より悪い**: `docker run` が `-d` 無しのフォアグラウンドになったため、本当にハングしたコンテナは**無制限にブロック**する。以前は少なくとも480秒で打ち切られていた。他人が使っているホスト上で、この層が防ぐために存在するまさにその失敗。

Ruling 46: **この欠陥は実装者のテストでも私の anakin 検証でも捕まらなかった**。実装者の A/B テストも end-to-end ランも、私の 37.9 秒のランも、すべて**タイムアウトが発火する必要のないコンテナ**だった。コードの修正と同時に、**タイムアウトを実際に発火させるテスト**を書かせる。直近2ラウンドはこの機構の2つの必要挙動(早期終了時に即座に返る / 超過時に kill する)を、片方ずつ直しては片方を壊している。両方にテストが要る。
最小修正はレビュアーの提案どおり単一サブシェル形に戻し、必要だった部分(fd リダイレクト)だけ残す — そこでは `sleep` がサブシェル自身の実の子なので孤児問題はリダイレクトだけで解決し、`kill "$watcher"` も正常に効く。`--rm` とフォアグラウンド `docker run` は維持(`--rm` が `docker logs` と競合するという実装者の指摘は正しい)。
Task 12+13: fix round 3/5 dispatched。**次ラウンド後、私が anakin で再検証する際はタイムアウトを強制するランを含める**。
繰り越す文書化済み残存事項(修正しない): Important 1 の既定 umask 644 依存(現状普遍的に真だが実リモートホストで未検証)、Important 4 の skip 判定がローカル収集ファイル基準であること(scp 失敗時に手動のリモート掃除が要る)。

Task 12+13: fix round 3 — commit 4732439。単一サブシェル形に戻し、round 2 から必要だった fd リダイレクトのみ残した。`--rm` とフォアグラウンド `docker run` は維持。
実装者の良い設計判断: **`run_container` を `scripts/lib_run_container.sh` に切り出し**、テストが「リモートに輸送されるのと同じ関数」を検証する形にした(`declare -f run_container` が切り出し前後でバイト一致することを確認)。コピーをテストしても意味が無いため。
新テスト `scripts/test_run_container.sh` が**2つの必要挙動を毎回 assert**: 早期終了(timeout=10s)→ 約0秒で exit 0 / 超過(timeout=3s、sleep 30)→ **exit 137 で約3秒**。後者を round 2 のコードに対して一時的に戻して実行し、実際にハングすることを確認してからテストとして確定させている。
テストスクリプト自身のバグも2件発見・修正: `set -e` 下の裸の `[ ] && [ ]` が FAIL を印字する前にスクリプトを落とす / bash の trap は積み重ならないため `run_container` 内部の trap がテスト側の cleanup trap を上書きする。

=== リモート・タイムアウト経路の実検証(コントローラ実施、anakin)===
`TIMEOUT_SECS` がハードコードのため、`lib_run_container.sh` を ssh で輸送し、`python` が 300 秒 sleep するだけのテストイメージ(`reprobench:timeouttest`)に対して timeout=5 で `run_container` を直接呼んだ = **run_on_host.sh がリモートで行うのと同じ機構**。
結果: **`RESULT elapsed=5s exit_code=137`**。5秒で打ち切られ(300秒ではない)、137 = SIGKILL であり `docker kill` が実際に発火している。残留コンテナ無し。
**これは実装者にも、私の前回の 37.9 秒のランにも不可能だった検証** — どちらもタイムアウトが発火する必要のないコンテナしか走らせていなかった。Ruling 46 で要求した「両方の挙動にテストを」が、ローカル(実装者)とリモート(私)の両面で満たされた。
anakin の後片付け: `reprobench:timeouttest` イメージと `/tmp/tmo` を削除済み。

Task 12+13: fix round 3 re-review(sonnet)— **全件 ADDRESSED、新規 breakage なし**。
レビュアーは実装者の自己申告を信用せず独立検証: コミット済み `lib_run_container.sh` を source して 3秒タイムアウト対 30秒コンテナを実行 → `elapsed=3s exit_code=137`。コミット済みテストスクリプトも実行して両ケース通過。**さらにスクラッチの tmpdir に round 2 の壊れた body を移植して実テストを走らせ**、`elapsed=30s exit_code=0` となりテストが `FAIL: ...` を印字して exit 1 することを確認 = (a) テストが round 2 のバグを実際に弁別する、(b) テスト自身の失敗報告経路が `set -e` 下で機能する、の両方を実証。
抽出の監査: ローカル経路(:193 で無条件 source)、リモート経路(同じ source 済み定義から `declare -f`)、`--dry-run`(`run_container` を呼ばない)、`run_all.sh`(サブプロセス呼び出し)、テスト(同じファイルを source)すべて健全。`scripts/` と Makefile を grep して**インライン定義の残骸が無い**ことも確認。
Task 12+13: minor (deferred): `run_on_host.sh:216` の `--dry-run` プレビュー文字列が `--rm` を欠いており実経路とずれている(round 2 由来の既存ギャップ、本ラウンドでは未接触)。
Task 12: complete / Task 13: complete (commits 0011718..4732439, 3 fix rounds)

=== T14: 本番実測(コントローラ自身が実行)===
Ruling 47: **T14 はサブエージェントに委譲せずコントローラが実行する**。理由: (1) 他人が使用中の共用マシン4台に対する取り返しのつきにくい操作であり、手を置くべきは私である。(2) 実装ではなく「検証済みスクリプトの操作」と「結果が仮説と合うかの裁定」であり、後者は記事の核そのもの。(3) リモート検証はここまで一貫して私が行っており、ホストの状態を把握している。

=== T14 実行前の仮説(結果を見る前に記録)===
Ruling 48: **結果を見る前に仮説を書き出す**。事後に「そう思っていた」と言うのは、このプロジェクトが3回繰り返した失敗(測っていない機構を自信を持って説明する)と同じ構造だから。以下は予測であって主張ではない。外れた項目は**外れたこととして記事に書く**。

G1(時間): 1a/1b/1c とも、`hashseed0` では全プローブ一致、`hashseedfree` では S0 の set 反復順系が割れる、と予測。
G2(ソフトウェア):
- 2a(base 3.11 vs oldpy 3.10、numpy 同一): S1〜S4 は一致すると予測。Python 版だけで numpy の数値が変わる理由が無いため。
- 2b(oldpy vs oldlibs、numpy 2.2.6 vs 1.21.6): **`S1_generator_random_float32` が割れる**と予測(NumPy 1.22.0 の float32 uniform 最下位ビット修正を跨ぐため)。NEP 19 の実証。他の S1 は一致の可能性。
- 2c(mkl vs base、BLAS 実装差): S3(行列積)が割れると予測。
- 2d(alpine vs base、musl vs glibc): **S4(exp/log/sin)が割れる**と予測。libm 実装が違うため。S1〜S3 は一致の可能性。
G3(ハードウェア):
- 3a(スレッド数 1/8/max): S3(BLAS)と S5(KMeans)が割れると予測。S2 は不明。
- 3b(anakin AVX2 vs blade AVX-512、同一イメージ): **S4 が割れる公算が高い**(NumPy issue #23523 で sin/cos/exp/log の約4 ULP 差が文書化済み)。**`S2`(np.sum)は真の未知数** — 同種の文書化報告が存在しない。割れなければ「縮約は意外と頑健」が所見になる。
- 3c(x86_64 vs aarch64): S4 が割れると予測(glibc 2.27 マニュアルで log2 が AArch64 1 ULP / x86_64 2 ULP)。
- 3d(nosimd512、blade 上で AVX-512 無効化): blade-nosimd512 が **anakin(AVX2)と一致**すれば、3b の原因が SIMD だと確定する。これが対照実験。
G4(アクセラレータ):
- 4a(CPU vs GPU): S7 と S8 の MLP が割れると予測。
- 4b/4c: `S8_index_add_atomics` が `stable_within_process=False`、`_deterministic` が True(anakin で実測済み、他2台でも同様と予測)。
- 4e(GPU 世代差 Ada / Blackwell / GB10): **`S8_curand_randn_sizes` がどのサイズから割れるかが最大の関心**。SM 数 128(4090)と他2台で異なるため、閾値超のサイズで乱数値そのものが変わると予測。**公式ドキュメント未記載なので、出れば一次情報**。
S9(LLM): greedy の batch1 vs batch8 が割れると予測。temperature 単離版と shipped-defaults 版で分岐位置が変わると予測。

=== T14 実測完了: 仮説との照合(Ruling 48 に対する答え合わせ)===
campaign 30行すべて成功・失敗ゼロ。全3 GPU ホストで CUDA 有効、S8 8/8・S9 11/11 実行。`make report` が 30ファイルから report.json / matrix.md を生成。

**【外れた仮説 1】`np.sum` は SIMD 世代で割れない。** S2(sum/mean、float32/float64)は anakin(AVX2)・blade(AVX-512)・dgxspark(aarch64)で**全て一致**。調査が「文書化報告なし」と言っていた通りで、予測を「真の未知数」と書いたのは正しかった。**所見: 縮約は頑健**。記事はこれを「割れなかったこと」として書く。
**【外れた仮説 2】アーキテクチャは効かなかった。SIMD 世代が効いた。** 3c(x86_64 anakin vs aarch64 dgxspark)は S2・S4 すべて**一致**。一方 3b(同じ x86_64 同士、AVX2 vs AVX-512)は S4_exp/log が**不一致**。つまり**決め手はアーキテクチャではなく、どの SIMD カーネルが選ばれたか**。予測(3c で割れる)は外れ、より良い所見になった。
**【3d 対照実験 成功】** blade で AVX-512 を無効化すると、**anakin(AVX2)と全プローブでビット一致**。3b の差の原因が SIMD ディスパッチであることが実験的に確定した。これが記事の冒頭ネタ。
**【S4 の内訳が割れた】** 3b では exp/log が不一致・sin は一致。2d(glibc vs musl)では**逆に sin が不一致・exp/log は一致**。2つの実験が相補的に、**exp/log は SIMD 依存・libc 非依存、sin は libc 依存・SIMD 非依存**であることを示している。機構(numpy 内製カーネル vs libm フォールバック)は推測なので、記事では**測った相関として書く**。
**【2b NEP 19 実証 成功】** numpy 2.2.6 vs 1.21.6 で **`S1_generator_random_float32` だけが不一致**、他の S1(stdlib MT19937 / RandomState / PCG64 / Philox)は全一致。NumPy 1.22.0 の float32 uniform 最下位ビット修正を外科的に捉えている。**legacy は動かず Generator だけが動いた** = NEP 19 の条文そのもの。
**【2a 確認】** Python 3.10 vs 3.11(numpy 同一)は S1 全一致。
**【2c 確認】** OpenBLAS vs MKL は S3(行列積)不一致・S4(libm)一致。MKL は BLAS を置き換えるが libm は置き換えない。
**【3a 確認+細部】** スレッド数 1 vs 8 で `S3_matmul_float32` 不一致・**float64 は一致**・S2 一致。
**【G1 確認】** PYTHONHASHSEED=0 vs 未設定で S0 全3本が不一致。`S0_set_iteration_order` と `S0_dict_from_set` のダイジェストが同値であることも確認 = linked pair の前提が実データで成立。

**【4e GPU: 本命が出た。閾値つきで確定】**
| n | RTX 4090 (SM 128) | RTX PRO 6000 (SM 188) | GB10 (SM 48) |
|---|---|---|---|
| 1,024 | c9020ce1 | c9020ce1 | c9020ce1 |
| 65,536 | fb9b6d89 | fb9b6d89 | fb9b6d89 |
| 4,194,304 | 947daf61 | **abf64dd5** | **9e849c24** |
| 67,108,864 | 9638c416 | **383fa914** | **12803be3** |
→ **閾値は 65,536 と 4,194,304 の間**。それ未満では同一 seed から3台とも同じ乱数。それを超えると**3台が3通りの異なる乱数を出す**。丸めの差ではなく**乱数の値そのものが違う**。PyTorch issue #194676 の挙動で、公式 docs への追記 PR は未マージ = **未文書化**。SM 数 48/128/188 の3点での一次データ。記事の GPU 章の目玉。

Task 15: fix round 1 — commit a00d66b。**全件 ADDRESSED、新規 breakage なし**。
Critical 1(cuBLAS の atomicAdd 未使用という未出典主張)削除、機構ボックスは**実測事実と cuBLAS 文書の near-verbatim な言い換えのみ**に。Critical 2 の「固定ワークスペース」の誤帰属も削除し「同一アーキ・SM数 / 単一ストリームの下」に。Critical 3 は ✕ を cx=300→150 に移し、矢印を実際に分断(上側セグメントを削除、矢頭も除去)。
Important 5 の機構記述が**技術的に正しくなった**: 「a[i]とa[i+8]をレーンごとに加算(i=0..7) → P[0..7]」→「P[0..7]を水平加算(log2(8)=3段階)」。実際の two-load / lane-pair / horizontal-reduce の機構。
Important 10 の裁定(L1 の約束だが条件は L3)が **F3 自身の色(#558b2f=L1、#ef6c00=L3)の inline tspan** で表現され、レビュアーが「技術的に存在するだけでなく読者に読める」と評価。
コンバータは base をダイジェスト固定、**47パッケージ全部をピン留め**(前回は2/47)。
レビュアーは PNG を全7枚読み直し、**PNG の高さが現 viewBox の正確に2倍**であることから「古い PNG ではなく現ソースから再レンダリングされている」ことまで確認。実装者が自己申告した F10 のキャプション溢れが最終状態に残っていないことも、報告を信用せず直接 PNG で確認。
Task 15: minor (deferred): レベル配色が F2/F7 で ok/ng の意味も兼ねている(バッジで曖昧性は解消)/ コンバータが自分でレンダリングしていない PNG も数える・孤児 PNG を掃除しない / F5・F10 の脚注の行間。
Task 15: complete (commits 4732439..a00d66b, review clean after 1 fix round)

=== T14 追加分析: 第7章(LLM)の確定事実 ===
ホストの素性(base イメージ基準、記事の表に使う):
  local    aarch64 / glibc 2.41 / NEON,ASIMD
  anakin   x86_64  / glibc 2.41 / **AVX2 のみ** / RTX 4090 SM=128 cc=8.9
  blade    x86_64  / glibc 2.41 / **AVX2+AVX512F+AVX512_SKX** / RTX PRO 6000 SM=188 cc=12.0
  dgxspark aarch64 / glibc 2.41 / NEON,ASIMD,**SVE** / GB10 SM=48 cc=12.1
alpine は両ホストとも `{'name':'musl','version':None,'method':'musl_loader_glob'}` = **2d の条件は本物**。

**【1. temperature=0 の実測。一次引用が取れた】** transformers は ValueError を投げ、メッセージは逐語で:
  「`temperature` (=0.0) has to be a strictly positive float, otherwise your next token scores will be invalid. If you're looking for greedy decoding strategies, set `do_sample=False`.」
**ライブラリ自身が「その慣用句は存在しない、greedy なら do_sample=False を使え」と言っている。** 第7章の核。

**【2. 外れた仮説: batch 1 vs batch 8 の greedy 出力は一致した】** 3 GPU すべてで 68 トークン完全一致、分岐なし。Thinking Machines が Qwen3-235B で報告した batch-invariance の効果は、**この規模(0.5B・batch 8・transformers)では再現しなかった**。記事はこれを「観測されなかった」として書く。機構を推測しない。
**【3. 外れた仮説: dtype は greedy 出力を変えなかった】** 同一 GPU 内で fp32 / bf16 / fp16 がすべて 68 トークン一致。

**【4. 代わりに出た鋭い結果: bf16 では GPU 世代が効く】**
  `S9_greedy_batch1`(fp32): 3 GPU すべて同一ダイジェスト(相異なる数=1)
  `S9_greedy_bf16`: **相異なるダイジェスト数=2** — anakin(Ada cc 8.9)が blade/dgxspark(Blackwell cc 12.0/12.1)と異なり、Blackwell 2台は互いに一致。
→ **精度を bf16 に落とすと GPU アーキテクチャが効き始める**。fp32 では効かない。実務的に直接使える所見。

**【5. 単離の効果が巨大】**
  t=0.7: 同梱デフォルト 87 トークン vs 単離 90 トークン、**最初の分岐はトークン16**
  t=1.2: 同梱デフォルト 119 トークン vs 単離 65 トークン、**最初の分岐はトークン0**
→ t=1.2 では**最初のトークンから違う**。「temperature=1.2」と言っても、同梱の top_p=0.8/top_k=20/repetition_penalty=1.1 を中和しない限り、測っているのは temperature ではない。

**【6. logit margin(脆さの定量)】** 3 GPU すべてで同一: n=84、min=0.0037、median=2.1005、max=13.2557、**0.1 未満が 3.6%**。
→ batch 差は観測されなかったが、**位置の 3.6% は 0.1 未満のマージンしかない**。最小は 0.0037。この大きさの摂動が入れば順位は入れ替わる。**機構を主張せず、脆さを定量で示せる**。
