# 再開メモ — 乱数の再現性 Zenn 記事

最終更新: 2026-10-14 / 中断時点の HEAD: `f9766fd`

## これは何

「同じ seed・同じ入力でも結果の再現性をどこまで約束できるか」を、低レイヤの原理と
4ホストでの実測で裏づける Zenn 長編記事1本を、再実行可能なコンテナ実測ハーネスごと作る。
問題意識は「成果物を納品するとき、どこまで再現性に言及していいのか」。

- 設計書: `docs/superpowers/specs/2026-10-14-randomness-reproducibility-design.md`
- 実装計画: `docs/superpowers/plans/2026-10-14-randomness-reproducibility.md`(全18タスク)
- 作業ブランチ: `feature/randomness-reproducibility`
- ワークツリー: `/Users/ljmb20260017/Documents/zenn-contents-wt/randomness-reproducibility`
  (メインは `/Users/ljmb20260017/Documents/zenn-contents`、main ブランチ)

## 再開のしかた

1. ワークツリーに入る(セッションは `EnterWorktree` で `path` 指定、上記パス)
2. 台帳 `.superpowers/sdd/2026-10-14-randomness-reproducibility/progress.md` を読む
   — **15個の ruling(私が代理で下した判断)が全部ここにある**。git では追跡していない
   スクラッチなので、ワークツリーを消すと失われる点に注意
3. `superpowers:subagent-driven-development` スキルで Task 3 から再開
4. 各タスクの brief は `.superpowers/sdd/.../task-N-brief.md` に18本すべて生成済み

テストは必ずコンテナ内で回す(ホスト直実行は禁止)。現時点のコマンド:

```
docker run --rm -e PYTHONPATH=/w \
  -v "/Users/ljmb20260017/Documents/zenn-contents-wt/randomness-reproducibility/reproducibility":/w \
  -w /w python:3.11-slim \
  sh -c "pip install -q numpy==2.2.6 pytest && pytest -q -W error"
```

## 進捗

| | タスク | 状態 |
|---|---|---|
| 調査 | 理論の一次情報調査 7本 | **完了**。`research_notes/乱数の再現性と保証レベル/` に7ファイル(368KB)。全主張に出典URLと取得日、未検証項目は Gaps に明示 |
| T1 | ダイジェストと数値比較の基盤 | **完了** `3ded887..5b8a9ea` レビュー clean |
| T2 | 環境情報の採取 | **完了** `5b8a9ea..f9766fd` レビュー clean |
| T3 | プローブ登録と契約 | **中断** — 下記参照 |
| T4〜T18 | | 未着手 |

## T3 の中断状態(重要)

実装エージェントを**自己レビューとコミットの直前で停止**させた。作業は捨てていない。

- 未追跡のまま残っているファイル: `reproducibility/reprobench/registry.py`,
  `reproducibility/reprobench/probes/`, `reproducibility/tests/test_registry.py`
- 停止直前の報告: **フルスイート 58 passed**(既存40 + 新規18)、`-W error` で出力クリーン
- 未実施: 実装者の自己レビュー、コミット、タスクレビュー

再開の選択肢は2つ。
- (a) 新しい実装エージェントに「ファイルは既にある、58テスト通過済み、自己レビューして
  コミットせよ」と指示する(速いが、前エージェントの意図を引き継げない)
- (b) 未追跡ファイルを消して `task-3-brief.md` から作り直す(確実だが数分の手戻り)

どちらでも、その後は通常どおり タスクレビュー → fix ループ → 完了。

## 残りのフェーズ

- T4〜T11: プローブ群とランナーと比較器(**T4+T5+T6 と T7+T8+T9 はバッチ化して1ディスパッチにする方針**。
  Ruling 13 参照)
- T12〜T13: Docker イメージ7種と4ホストへの配布スクリプト
- T14: 実測の実行(4ホスト・12ラン)← ここが本番
- T15〜T16: 図版(原理図7枚 + 実測図3枚)
- T17: 記事執筆
- T18: 事実確認レビューと PR

## 計測ホスト(全台 docker + ssh 疎通確認済み)

| ホスト | CPU | Arch | SIMD | GPU |
|---|---|---|---|---|
| このMac | Apple M5 Pro (18c) | aarch64 | NEON | コンテナからは不可 |
| anakin | i9-13900K (32t) | x86_64 | **AVX2 まで** | RTX 4090 (Ada) |
| blade | Xeon w7-3565X (64t) | x86_64 | **AVX-512 あり** | RTX PRO 6000 (Blackwell) |
| dgxspark | GB10 Grace (20c) | aarch64 | **SVE2 あり** | GB10, CC (12,1) |

全ホスト共用マシン。占有は各10分以内・GPUメモリ8GB以下。
GPU イメージは **CUDA 13 + torch cu130 必須**(GB10 は sm_121 で cu12 ビルドでは動かない)。

## 調査で確定した、記事に効く事実(詳細は台帳と research_notes)

- IEEE 754 WG 自身が「**規格は準拠システム間で同一結果を保証しない**」と明記。第4章の冒頭引用に使う
- 超越関数が「推奨」止まりな理由は **Table Maker's Dilemma**
- glibc 2.27 マニュアルで `log2`(double)が **AArch64 1 ULP / x86_64 2 ULP** — クロスアーキ差の文書化された実例
- NumPy の AVX-512 と非 AVX-512 で **sin/cos/exp/log が最大 ~4 ULP 違う**(issue #23523、メンテナ公認)。
  一方 `np.sum` 自体の SIMD 差は文書化された報告が見つからない → **冒頭ネタは S4 が本命、S2 は要実測**
- **MT19937 の仕様は state→出力のみを固定し、seed→state は固定しない**。CPython / RandomState /
  Generator(MT19937) の3経路が同一 seed で3つの異なるストリームを出す
- NEP 19 原文: RandomState は「SHALL NOT be made」、Generator は X.Y で破壊を許容。**"frozen" とは書いていない**
- PyTorch 公式「identical seed でも CPU と GPU の間で再現しないことがある」
- cuBLAS のビット再現性は「単一 CUDA ストリーム時のみ」かつ「同一アーキ**かつ同一 SM 数**」
- **PyTorch の CUDA RNG は SM 数依存** — 要素数が閾値を超えると同じ seed でも GPU モデルが違えば
  乱数の値そのものが変わる(issue #194676)。**公式 docs への追記 PR #196733 は未マージ = 未文書化**
- transformers に `temperature=0` という決定的モードは**存在しない**(`do_sample=False` が正)。
  vLLM は `temperature<1e-5` を greedy に落とし `0<t<0.01` を 0.01 にクランプ
- ACM の Repeatability/Reproducibility/Replicability は **v1.0(2016)と v1.1(2020)で意味が入れ替わっている**
- ACM 原文に「完全一致は要求しない、許容誤差内で一致し主要な主張を変えなければよい」
  → **第8章の納品文言テンプレの一次的な根拠**

## 執筆時の禁止事項(調査で裏が取れなかったもの)

- IEEE 754 の細かい subclause 番号を書かない(primary PDF から抽出できず二次情報由来)
- Apple Accelerate/vecLib の ULP を断定しない(公式文書が存在しない)
- 「コンテナはホストのカーネル CSPRNG を共有する」を断定しない(一次情報なし)
- LLM の分岐率の定量値は**未査読プレプリント**由来であることを明記する
- scikit-learn に「バージョン間で保証しない」という文は**存在しない**。要約で創作しない
- Stream-K には言及しない(一次情報が見つからなかった)
