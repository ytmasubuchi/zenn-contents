# 再開メモ — 乱数の再現性 Zenn 記事

最終更新: 2026-10-15 / 中断時点の HEAD: `c697647`

## これは何

「同じ seed・同じ入力でも結果の再現性をどこまで約束できるか」を、低レイヤの原理と
4ホストでの実測で裏づける Zenn 長編記事1本を、再実行可能なコンテナ実測ハーネスごと作る。
問題意識は「成果物を納品するとき、どこまで再現性に言及していいのか」。

- 設計書: `docs/superpowers/specs/2026-10-14-randomness-reproducibility-design.md`
- 実装計画: `docs/superpowers/plans/2026-10-14-randomness-reproducibility.md`(全18タスク)
- **判断記録: `docs/superpowers/decision-log-2026-10-14.md`(ruling 48件 + 実測の分析すべて)**
- 作業ブランチ: `feature/randomness-reproducibility`
- ワークツリー: `/Users/ljmb20260017/Documents/zenn-contents-wt/randomness-reproducibility`

## 進捗: 18タスク中16完了

| | タスク | 状態 |
|---|---|---|
| 調査 | 理論の一次情報 7本 | **完了** `research_notes/乱数の再現性と保証レベル/`(368KB、全主張に出典URLと取得日) |
| T1〜T13 | ハーネス一式 | **完了**・レビュー clean(53プローブ、165 passed) |
| T14 | **4ホスト30ランの実測** | **完了**・失敗ゼロ `results/` にコミット済み |
| T15 | 原理図 SVG 7枚 | **完了**・レビュー clean |
| T16 | 実測図 F4/F8/F9 | **実装完了・コミット済み(`c697647`)だがタスクレビュー未実施** |
| T17 | 記事執筆 | 未着手 |
| T18 | 事実確認レビューと PR | 未着手 |

## 再開のしかた

1. ワークツリーに入る(`EnterWorktree` で `path` 指定)
2. `docs/superpowers/decision-log-2026-10-14.md` を読む — 48件の ruling と実測分析が全部ここにある
3. **まず T16 のタスクレビューを回す**(`c33ae45..c697647` の diff、PNG を必ず目で見る)
4. その後 T17(執筆)→ T18(事実確認・PR)

テストは必ずコンテナ内で:
```
docker run --rm -e PYTHONPATH=/w \
  -v "/Users/ljmb20260017/Documents/zenn-contents-wt/randomness-reproducibility/reproducibility":/w \
  -w /w python:3.11-slim \
  sh -c "pip install -q numpy==2.2.6 pytest && pytest -q -W error"
```

## 記事に書く確定事実(すべて一次実測)

### 冒頭に置く:対照実験で原因を特定できた
同じ Docker イメージ・同じ seed・どちらも x86_64 の2台(anakin / blade)で `exp` と `log` の
結果がビット不一致。**blade で AVX-512 を無効化すると、anakin と全プローブでビット一致**。
原因が SIMD ディスパッチであることを推測ではなく実験で確定している。

### 外れた仮説(そのまま書く)
- **`np.sum` は割れなかった** — AVX2 / AVX-512 / aarch64 で全一致。縮約は頑健。
- **アーキテクチャは効かなかった。SIMD 世代が効いた** — x86_64 vs aarch64 は全一致、
  同じ x86_64 同士の AVX2 vs AVX-512 で割れる。
- **LLM の batch 1 vs batch 8 は一致した** — 3 GPU すべてで 68 トークン完全一致。
  Thinking Machines が Qwen3-235B で報告した batch-invariance は**この規模では再現せず**。
- **dtype は greedy 出力を変えなかった**(同一 GPU 内で fp32/bf16/fp16 一致)。

### 層の切り分け
`exp`/`log` は **SIMD 依存・libc 非依存**、`sin` は **libc 依存・SIMD 非依存**。
2つの実験(3b と 2d)が相補的に示している。機構は未測定なので**測った相関として書く**。

### NEP 19 の外科的な実証
numpy 2.2.6 → 1.21.6 で **`S1_generator_random_float32` だけ**が不一致。stdlib MT19937 /
RandomState / PCG64 / Philox は全一致。**legacy は動かず Generator だけが動いた**。

### GPU の目玉:閾値つきの未文書化挙動
`S8_curand_randn_sizes`、SM 数 48 / 128 / 188 の3台:

| n | RTX 4090 (SM 128) | RTX PRO 6000 (SM 188) | GB10 (SM 48) |
|---|---|---|---|
| 1,024 | c9020ce1 | c9020ce1 | c9020ce1 |
| 65,536 | fb9b6d89 | fb9b6d89 | fb9b6d89 |
| 4,194,304 | 947daf61 | **abf64dd5** | **9e849c24** |
| 67,108,864 | 9638c416 | **383fa914** | **12803be3** |

閾値は 65,536 と 4,194,304 の間。超えると**同一 seed から3通りの異なる乱数**(丸めではなく値そのもの)。
PyTorch issue #194676、docs 追記 PR は未マージ = **未文書化**。

### GPU の非決定性は atomics に局在する
RTX 4090 実測: MLP(Linear+ReLU)10step は5回ともビット一致。`index_add_` は一致せず
最大差 8.24e-04。`use_deterministic_algorithms(True)` でビット一致に戻る。
→ 「GPU は非決定的」は誤り。**大半は決定的で、非決定性は atomics を使う演算に局在する。**

### bf16 にすると GPU 世代が効き始める
fp32 の greedy は 3 GPU 完全一致。**bf16 では anakin(Ada cc 8.9)が Blackwell 2台と割れる**
(Blackwell 同士は一致)。

### 第7章(temperature)の一次引用
transformers に `temperature=0` を渡すと ValueError:
「`temperature` (=0.0) has to be a strictly positive float, otherwise your next token scores
will be invalid. If you're looking for greedy decoding strategies, set `do_sample=False`.」
**ライブラリ自身が「その慣用句は存在しない」と言っている。**

単離の効果: t=1.2 で同梱デフォルト 119 トークン vs 単離 65 トークン、**最初の分岐はトークン0**。
同梱の `top_p=0.8` / `top_k=20` / `repetition_penalty=1.1` を中和しない限り temperature を測っていない。

margin(脆さの定量、3 GPU 同一): n=84、min=0.0037、median=2.0748(`np.median`)、**0.1 未満が 3.6%**。
ULP差(float32): anakin-blade 完全一致17/84・差の中央値18 ULP・最大49,152 ULP、anakin-dgxspark 完全一致5/84・中央値32 ULP・最大40,960 ULP、blade-dgxspark 完全一致10/84・中央値24 ULP・最大8,192 ULP。

### その他の確定事実
- Python 3.10 vs 3.11(numpy 同一)は S1 全一致
- OpenBLAS vs MKL は行列積が不一致、libm は一致
- スレッド数 1 vs 8 で float32 GEMM が不一致、**float64 は一致**、`np.sum` も一致
- `PYTHONHASHSEED` 未設定 vs 0 で S0 全3本が不一致
- KMeans はスレッド数を固定すれば 1/2/4 で安定、8 CPU 機で 8 にすると不安定(縮約の丸め、
  `n_iter_` は180フィット全てで 2、inertia の広がりは最大 2 ULP)
- 同一 GPU・同一 CUDA 13 で **torch 2.11 と 2.14 のダイジェストが全一致**
  (公式は「リリース間の再現性を保証しない」— 保証が無いことと実際に壊れることは別)

## 計測ホスト(base イメージ基準)

| ホスト | arch | libc | SIMD | GPU |
|---|---|---|---|---|
| local (Mac) | aarch64 | glibc 2.41 | NEON, ASIMD | コンテナからは不可 |
| anakin | x86_64 | glibc 2.41 | **AVX2 のみ** | RTX 4090 SM=128 cc=8.9 |
| blade | x86_64 | glibc 2.41 | **AVX2+AVX512F+AVX512_SKX** | RTX PRO 6000 SM=188 cc=12.0 |
| dgxspark | aarch64 | glibc 2.41 | NEON, ASIMD, **SVE** | GB10 SM=48 cc=12.1 |

alpine イメージは両ホストとも musl(`musl_loader_glob` で検出)= 2d の条件は本物。

## 執筆時の規律(最重要)

**コメント・docstring・図のラベル・本文に書いてよいのは、実際に測った相関か出典が言っていることだけ。
変化させなかった因子を原因として名指ししてはならない。**

このプロジェクトは測っていない機構を自信を持って説明して**4回**修正している
(当初の「スレッドスケジューリング」、「1スレッドでも再現」、その置き換え、
そして図版の「cuBLAS は atomicAdd を使わない(NVIDIA文書)」という未出典の帰属)。

### 書いてはいけないこと(調査で裏が取れなかった)
- IEEE 754 の subclause 番号(primary PDF から抽出できず二次情報由来)
- Apple Accelerate/vecLib の ULP(公式文書が存在しない)
- 「コンテナはホストのカーネル CSPRNG を共有する」(一次情報なし。かつ Mac では
  コンテナは Linux VM 配下でホストのカーネルは macOS ではない)
- LLM の分岐率の定量値を査読済みとして扱うこと(すべて 2026 年のプレプリント)
- scikit-learn に「バージョン間で保証しない」という文が存在するかのように書くこと(存在しない)
- Stream-K への言及(一次情報が見つからなかった)
- `np.sum` が SIMD 世代で割れるという主張(文書化報告が無く、**本実測でも割れなかった**)

## 残務と注意

- **T16 のタスクレビューが未実施。** `c33ae45..c697647`。PNG(F4/F8/F9)を必ず目で見ること。
- `results/` は 31MB(30ラン分の生データ)。記事の全主張の証拠なので意図的にコミットしている。
- リモートホストに残置: anakin に `~/reprobench-run/`、各ホストに `reprobench:*` イメージと
  `repro-gpu:check`、GPU ホストに HF キャッシュ(約1GB)。**記事完成後に削除すること。**
- 繰り越し済みの軽微な指摘は decision-log の "minor (deferred)" 行を参照。T18 でトリアージする。
