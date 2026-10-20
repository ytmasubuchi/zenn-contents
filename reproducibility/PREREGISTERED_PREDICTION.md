# 事前登録された予測 (追加実測 round 2)

記録日時: 2026-10-20
記録者: 本ファイルは **実測を走らせる前に** コミットされる。
目的: 第6章G2の機構(「AVX-512専用カーネルの有無が、libmを経由するかどうかを決める」)を
      実測で検証する。既存の30ランは全て記事執筆前に取得済みであり、記事中の「仮説」は
      事後の文章化である。本ファイルの予測のみが事前登録である。

## 機構の仮説

NumPy 2.2.6 のソースから導かれる:

- `loops_exponent_log.dispatch.c.src`: DOUBLE_exp/DOUBLE_log は
  AVX512F があれば NumPy 内製カーネル(AVX512F_exp_DOUBLE / AVX512F_log_DOUBLE、libc非依存)、
  無ければ npy_exp/npy_log のスカラー経路 → プラットフォームの libm。
- `loops_trigonometric.dispatch.cpp`: 「Disable SIMD code sin/cos f64 and revert to libm」
  → float64 の sin/cos は常に libm。

したがって exp/log が libc 実装差の影響を受けるかどうかは、
**AVX-512F が使えるかどうか**で決まる。

## 既存データとの整合(これは事後確認)

| label | S4_exp | S4_log | S4_sin |
|---|---|---|---|
| blade/base (AVX-512あり, glibc) | 5119b269 | a6b6b7fa | ddf398b7 |
| blade/alpine (AVX-512あり, musl) | 5119b269 | a6b6b7fa | 601eb0fa |
| dgxspark/base (aarch64, glibc) | cba8bfa7 | 09cef322 | ddf398b7 |
| dgxspark/alpine (aarch64, musl) | ab300a28 | 84b2ce10 | 601eb0fa |
| anakin/base (x86_64 AVX2, glibc) | cba8bfa7 | 09cef322 | ddf398b7 |
| local/base (aarch64, glibc) | cba8bfa7 | 09cef322 | ddf398b7 |

## 予測 (実測前に記録)

### 実験1: `blade alpine nosimd512` — 同一ハードウェアでの決定実験
blade (AVX-512搭載機) で、muslイメージのまま AVX-512 のみを無効化する。
ハードウェア・libc・numpy版・Python版を全て固定し、AVX-512 の有効/無効だけを変える。

予測:
- `S4_exp_float64` は `5119b269` から変化し、**`ab300a28`** になる
  (= dgxspark/alpine と一致。musl の libm 値)
- `S4_log_float64` は `a6b6b7fa` から変化し、**`84b2ce10`** になる
- `S4_sin_float64` は `601eb0fa` のまま変化しない (sin は元々 libm なので AVX-512 は無関係)
- `S2_sum_float64` は `f9458b5c` のまま変化しない

外れる場合: exp/log が `5119b269`/`a6b6b7fa` のまま変わらなければ、
「AVX-512の有無が libm 経由かを決める」という機構は誤り。

### 実験2: `anakin alpine threadsmax` — AVX2機でのlibc比較(未実施区間)
anakin は AVX2 どまり。glibc では exp/log が `cba8bfa7`/`09cef322` (= dgxspark/local と一致)。

予測:
- `S4_exp_float64` は `cba8bfa7` から変化し、**`ab300a28`** になる
- `S4_log_float64` は `09cef322` から変化し、**`84b2ce10`** になる
- `S4_sin_float64` は `ddf398b7` から **`601eb0fa`** になる
- `S2_sum_float64` は `f9458b5c` のまま

外れる場合: exp/log が glibc 値のまま変わらなければ、
「AVX2機では exp/log が libm 依存」という読みが誤り。

## 副次的に確認すること(予測なし)

- `S3_matmul_float64` が base→alpine で変わることが blade/dgxspark で確認されている。
  anakin でも変わるか。これは予測を立てる根拠が無いので実測のみ。
