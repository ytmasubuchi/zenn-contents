# 乱数の再現性 — Zenn長編記事 設計書

- 作成日: 2026-10-14
- スラッグ(暫定): `reproducibility-of-randomness`
- 形式: Zenn 長編記事 1本 + 図版 + 再実行可能なコンテナ実測ハーネス

## 1. 目的とブリーフ

### 目的
「インプットがすべて同じでも、結果の再現性をどこまで約束できるか」の線引きを、
低レイヤの原理と4ホストでの実測の両方で裏付ける。読者が納品の場面で判断を誤らないようにする。

### 想定読者
Python は書けるが、IEEE 754・SIMD・BLAS・libm といった層を意識したことがない人。

### 成功条件
1. 読者が手元の処理を見て「これは保証レベル L0〜L4 のどれか」を判定できる
2. 納品文書にそのまま書ける条件文テンプレートを持ち帰れる
3. 記事中の全実測が、公開されたコンテナ定義で第三者に再実行できる

### 制約(依頼者指定)
- 計測は**必ずコンテナ内**で行う。ホストマシン上での直接実行は不可。
- 理論は一次情報(国際規格・論文・公式ドキュメント)で裏取りする。deep-research スキルを使う。
- オーケストレーターは調査・実装・執筆を直接行わず、Sonnet サブエージェント + git worktree に振って並列実行する。
- 図解を積極的に用いる。SVG を生成してよい。

## 2. 記事の骨格(承認済み:実測ショック型 + レベル軸ハイブリッド)

| 章 | 内容 |
|---|---|
| 1 | はじめに — 納品の場面設定、TL;DR 3点 |
| 2 | **冒頭の実測** — 同じ seed・同じ Docker イメージ・どちらも x86_64 の2台で、NumPy の結果がビット不一致 |
| 3 | seed を固定すると何が起きているのか — 階層降下(アプリ → ライブラリPRNG → Python ランタイム → OS CSPRNG → ハードウェア TRNG) |
| 4 | 再現性の保証レベル L0〜L4 |
| 5 | 「インプットが全部同じ」は案外成立していない(glob順・set反復順・並列完走順) |
| 6 | 実測マトリクス G1〜G4 |
| 7 | LLM と temperature |
| 8 | 判定フローと納品文言テンプレート |
| 9 | まとめ |

冒頭で語彙を2つ定義する。
- **再実行性 / 移植再現性** — ACM Artifact Review バッジの Repeatability(同チーム同環境)/ Reproducibility(別チーム同環境)/ Replicability(別チーム別環境)。納品で揉めるのはほぼ3つ目。
- **ビット一致 / 数値的に近い** — 「再現します」が `==` なのか `allclose(rtol=1e-9)` なのかで約束の重さが二桁違う。

## 3. 保証レベルの定義(記事の背骨)

| | レベル | 保証の出所 | 代表例 | 壊れる条件 | 納品で言えること |
|---|---|---|---|---|---|
| L0 | 規格 | 国際規格・公開アルゴリズム仕様 | IEEE 754 の `+ - * / sqrt fma`(正しく丸められる)、SHA-256、MT19937 / PCG64 / Philox の生成ビット列 | 言語・OS・CPU を跨いでも壊れない | バージョン固定でビット一致を約束できる |
| L1 | 明文の規約 | ライブラリが文書で約束 | NumPy NEP 19(`RandomState` はストリーム互換を保証、`Generator` は非保証)、PyTorch「リリース間の再現性は保証しない」 | バージョンを上げた瞬間 | バージョン固定を条件に約束できる |
| L2 | 事実上の実装 | 文書化なし、実装がたまたま決めている | 正規乱数の変換法(ziggurat / Box-Muller)、`np.sum` の pairwise summation 閾値、sklearn 内部の乱数消費順 | バージョン差 | 実際には一致するが**根拠が無いので書かない方がいい** |
| L3 | 実行時環境 | 実行時に選ばれる | SIMD ディスパッチ(AVX2 / AVX-512 / SVE2)、BLAS スレッド数、OpenMP の分割、libm(glibc / musl / Apple)、FMA 契約 | 同じコード・同じ版でも CPU・スレッド数・ベースイメージが変われば | 「同一イメージ + 同一CPU世代 + スレッド数固定」まで条件を明記すれば約束できる |
| L4 | 保証なし | 原理的に非決定 | `PYTHONHASHSEED`、GPU の `atomicAdd` 順序、cuDNN benchmark のアルゴリズム自動選択、TF32 の自動適用、並列 reduce の完了順 | 同じプロセスを2回走らせても | 約束できない。許容誤差でしか語れない |

依頼者の当初3段階との対応: 1 → L0 / 2 → L1 / 3 → **L2・L3・L4 に分割**。
分割の根拠は「壊れる条件が違えば、納品文書に書ける条件文も違う」こと。

### 本文に必ず入れる但し書き
- IEEE 754 が保証するのは**1演算**であって**演算列**ではない。浮動小数の加算に結合則がないため、順序が変われば結果が変わる。これが L3 の根本原因。
- `exp` / `log` / `sin` などの超越関数は IEEE 754-2019 では "recommended operations" どまりで、正しい丸めが**要求されていない**。libm 実装差が出る。

## 4. 実測設計

### 4-1. 計測ホスト

| ホスト | CPU | Arch | SIMD | GPU | 備考 |
|---|---|---|---|---|---|
| MacBook | Apple M5 Pro (18c) | aarch64 | NEON | 不可 | Docker Desktop(VM 経由) |
| anakin | i9-13900K (32t) | x86_64 | **AVX2 まで** | RTX 4090 (Ada) | 共用 |
| blade | Xeon w7-3565X (64t) | x86_64 | **AVX-512 あり** | RTX PRO 6000 (Blackwell) | 共用 |
| dgxspark | GB10 Grace (20c) | aarch64 | **SVE2 あり** | GB10 (Blackwell) | 共用 |

anakin と blade はどちらも x86_64 だが SIMD 世代が違う。これが記事冒頭のネタになる。

### 4-2. 条件軸(G1〜G4)

| グループ | ID | 条件 | 狙うレベル |
|---|---|---|---|
| G1 時間 | 1a | 同一プロセス内で2回 | L4 のみ |
| | 1b | 同一コンテナを再起動 | L4(`PYTHONHASHSEED`) |
| | 1c | 同一イメージから別コンテナ | L4 |
| G2 ソフトウェア | 2a | Python 3.11 / 3.13 | L1・L2 |
| | 2b | ライブラリ版差(numpy 1.26 / 2.x, sklearn 1.3 / 最新, torch) | **L1(NEP 19 の実例)** |
| | 2c | BLAS 実装差(OpenBLAS / MKL) | L3 |
| | 2d | ベースイメージ差(debian/glibc vs alpine/musl) | L3(libm) |
| G3 ハードウェア | 3a | スレッド数 1 / 8 / 32 | L3 |
| | 3b | 同一アーキ別CPU(anakin AVX2 vs blade AVX-512) | **L3・記事冒頭** |
| | 3c | 別アーキ(x86_64 vs aarch64) | L3 |
| | 3d | SIMD 明示無効化 `NPY_DISABLE_CPU_FEATURES` | 3b の原因同定用の対照実験 |
| G4 アクセラレータ | 4a | CPU vs GPU | L3・L4 |
| | 4b | 同一GPU 反復 | L4 |
| | 4c | `torch.use_deterministic_algorithms` on/off | L4 |
| | 4d | TF32 / dtype(fp32, bf16, fp16) | L3・L4 |
| | 4e | 別GPU世代(Ada / Blackwell / GB10) | L3 |

「別OS」は、コンテナ内計測の制約のもとで **2d(glibc vs musl)= libc/libm が変わる**と解釈し直す。
加えて Mac(Docker Desktop = VM 経由)vs ネイティブ Linux も G3 に自然に含まれる。

### 4-3. プローブ(S1〜S9)

| ID | プローブ | 狙い |
|---|---|---|
| S1 | `random` / `RandomState` / `default_rng`(PCG64) / Philox の生ビット | **L0 の証明**。全条件で一致するはず |
| S2 | `np.sum`(1e7 要素, float32/64)・`np.mean` | pairwise summation × SIMD 幅 → 3b の主犯 |
| S3 | `A @ B`(2048², float32/64) | BLAS。スレッド数・実装差 |
| S4 | `np.exp` / `np.log` 大配列 | libm 差(2d)。ULP で測る |
| S5 | sklearn: LogisticRegression(lbfgs) / RandomForest(n_jobs) / KMeans(k-means++) | 「乱数は合うのにモデルが違う」実例 |
| S6 | LightGBM(num_threads 可変) | 勾配ブースティングの並列非決定性 |
| S7 | PyTorch CPU: MLP 10 step 学習 | 誤差の積み上がり |
| S8 | PyTorch GPU: 同上 + deterministic / TF32 on-off | **L4 の証明** |
| S9 | LLM 推論(下記) | AI 文脈の本丸 |

加えて **S0: 隠れた入力差**(`glob` 順 / `set` 反復順 / 並列ワーカー完走順)を独立プローブとして持つ。第5章の材料。

### 4-4. S9: LLM と temperature

モデル: **Qwen2.5-0.5B-Instruct**(Apache-2.0、全ホストの GPU と CPU で動く)。推論は transformers。

1. **`temperature=0` なら決定的、は成り立つか**
   同一 GPU・同一プロンプト・greedy で、**バッチサイズだけ 1 と 8 に変える**。
   バッチ形状が変われば行列積のタイル分割が変わり logits が最下位ビットで動く。
   top-1 と top-2 の差が小さいトークンで順位が入れ替わると、そこから出力が発散する。
2. **temperature を上げると何が変わるか**
   `temperature>0` で初めて乱数が登場する。seed 固定なら同一環境では一致する。
   温度は「乱数の強さ」ではなく**ロジット差の拡大率**であり、再現性の観点では**非決定性の増幅器**。
   top-1/top-2 margin の分布を実測し、温度ごとに「何トークン目で分岐するか」を出す。
3. **dtype** — fp32 / bf16 / fp16 で margin がどれだけ縮むか。量子化せずとも dtype だけで再現性が変わる。
4. **別GPU世代**(4090 / RTX PRO 6000 / GB10)で同一 seed・同一 temperature が一致するか。

### 4-5. 比較方法

全プローブ共通:
- 出力を正規化してバイト列化 → **SHA-256** で一致/不一致を判定
- 数値配列は float64 に揃えて **max 絶対差 / 相対誤差 / ULP 差** を記録(「どのくらい違うか」が結論の許容誤差論に繋がる)
- 大きい配列は、全体の縮約値 + 決定的サブサンプル(固定インデックス 4096 点)を保存
- LLM は**トークン列の一致**と**最初に分岐した位置**を記録

結果 JSON のスキーマ:
```
{
  "env": { host, arch, libc, python, numpy, numpy_cpu_features, blas, torch, cuda, gpu, image, config },
  "probes": [ { "probe": str, "digest": str, "scalars": {...}, "sample_b64": str|null, "meta": {...} } ]
}
```

### 4-6. イメージ割り当て(コンテナ 12 ラン)

| イメージ | 流すホスト | カバー |
|---|---|---|
| `base`(python:3.11-slim / glibc) | 4 ホスト全部 | 1a・1b・1c・3a・3b・3c・3d |
| `py313`(python:3.13-slim) | blade | 2a |
| `oldlibs`(numpy 1.26 / sklearn 1.3 / 旧 torch) | blade | 2b |
| `alpine`(musl) | blade + dgxspark | 2d |
| `mkl`(conda-forge MKL 版 numpy) | blade | 2c |
| `gpu`(CUDA 13 + torch cu130 + transformers) | anakin・blade・dgxspark | 4a〜4e・S9 |

既知の制約(記事に注記する):
- alpine には PyTorch の musl wheel が無い → alpine では S1〜S6 のみ
- MKL は x86_64 専用 → 2c は blade のみ

## 5. 図版

SVG を正としてリポジトリに置き、記事には PNG 変換版を貼る(Zenn での SVG 表示が不確実なため)。

| ID | 図 | 依存 |
|---|---|---|
| F1 | 乱数生成の階層(HW TRNG → OS CSPRNG → ライブラリ PRNG → アプリ)と seed 固定で切れる線 | 原理のみ |
| F2 | 「乱数は一致しても結果は一致しない」分岐図 | 原理のみ |
| F3 | 保証レベル L0〜L4 の地図 | 原理のみ |
| F4 | L0〜L4 × 条件 の ○× マトリクス | **実測依存** |
| F5 | 浮動小数の非結合性 | 原理のみ |
| F6 | SIMD 縮約木(AVX2 8並列 vs AVX-512 16並列) | 原理のみ |
| F7 | GPU 並列縮約 / atomic の非決定 | 原理のみ |
| F8 | LLM greedy の margin 分布と分岐位置 | **実測依存** |
| F9 | 判定フローチャート + 納品文言テンプレ | 原理のみ |
| F10 | BLAS タイル分割とスレッド数 | 原理のみ |

## 6. 成果物レイアウト

```
articles/reproducibility-of-randomness.md
images/reproducibility-of-randomness/*.png
reproducibility/
  images_src/*.svg
  docker/Dockerfile.{base,py313,oldlibs,alpine,mkl,gpu}
  probes/            S0〜S9
  runner/            run_probes.py / collect.py / compare.py / make_figs.py
  results/<host>__<image>__<config>.json
  README.md
reports/             deep-research 最終レポート(コミットする)
research_notes/      調査の生ノート(.gitignore)
```

## 7. フェーズ計画

| | 内容 | 並列化 |
|---|---|---|
| P1 | ① 理論の一次情報調査(deep-research)/ ② 実測ハーネス実装 / ③ 原理図 SVG(F1・F2・F3・F5・F6・F7・F10) | ①②③を同時実行。②③は Sonnet サブエージェント + 独立 worktree |
| P2 | 4 ホストへ配布 → ビルド → 12 ラン実行 → 結果回収 → 比較マトリクス生成 | ホスト単位で並列 |
| P3 | 実測依存の図(F4・F8・F9)+ 記事執筆 | 図と執筆を分離 |
| P4 | 事実確認レビュー(調査結果と本文の突き合わせ)→ ブランチ作成・PR | レビューは独立サブエージェント |

## 8. リスクと対応

| リスク | 対応 |
|---|---|
| ~~dgxspark(GB10 / sm_121)で PyTorch が動かない~~ **解消(2026-10-14 実測)** | GB10 上で torch 2.13/2.14 + cu130 が動作、compute capability (12,1)、ドライバ 580.126。**ただし GPU イメージは CUDA 13 系が必須** — cu12 ビルドでは sm_121 を叩けない。3ホストともドライバ 580.x で CUDA 13 対応済み |
| 共用マシンの占有 | 各ホスト実測 10 分以内・GPU メモリ 8GB 以下。計測対象が速度でなくビット一致なので他ユーザーの負荷と干渉しない |
| HF モデル取得(各ホスト約 1GB) | 事前取得。失敗したホストは S9 をスキップして記録 |
| Zenn で SVG が表示されない | SVG を正、記事には PNG |
| 冒頭の実測(3b)で差が出ない可能性 | その場合は 3a(スレッド数)または 2d(libm)を冒頭に差し替える。**実測が仮説と違ったら記事の筋を実測に合わせる**(仮説に合わせて実測を選ばない) |

## 9. 非スコープ

- 速度ベンチマーク(本記事は値の一致のみを扱う)
- 暗号論的乱数の安全性評価
- 分散学習(マルチノード)の再現性
- llama.cpp / vLLM での追加検証(余力があれば付録)
