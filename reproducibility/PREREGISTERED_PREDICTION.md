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

---

# 実測結果 (2026-10-20 18:06 実行、予測コミット 63d9e38 の後)

| label | S4_exp | S4_log | S4_sin | S2_sum | S3_matmul_f64 |
|---|---|---|---|---|---|
| blade/base/threadsmax | 5119b269 | a6b6b7fa | ddf398b7 | f9458b5c | 76042a4d |
| blade/alpine/threadsmax | 5119b269 | a6b6b7fa | 601eb0fa | f9458b5c | 8875959c |
| **blade/alpine/nosimd512 (新規)** | **ab300a28** | **28ff757a** | 601eb0fa | f9458b5c | 8875959c |
| anakin/base/threadsmax | cba8bfa7 | 09cef322 | ddf398b7 | f9458b5c | f86a1f40 |
| **anakin/alpine/threadsmax (新規)** | **ab300a28** | **28ff757a** | **601eb0fa** | f9458b5c | **c924d2fa** |
| dgxspark/base/threadsmax | cba8bfa7 | 09cef322 | ddf398b7 | f9458b5c | 0cb8db34 |
| dgxspark/alpine/threadsmax | ab300a28 | 84b2ce10 | 601eb0fa | f9458b5c | 86d16106 |

## 予測の採点

### 実験1: blade alpine nosimd512 (同一ハードウェアでの決定実験)
- S4_exp: 予測 ab300a28 → 実測 **ab300a28** ... 的中
- S4_log: 予測 84b2ce10 → 実測 **28ff757a** ... **外れ**(変化はしたが値が違う)
- S4_sin: 予測 601eb0fa のまま → 実測 601eb0fa ... 的中
- S2_sum: 予測 f9458b5c のまま → 実測 f9458b5c ... 的中

### 実験2: anakin alpine threadsmax
- S4_exp: 予測 ab300a28 → 実測 **ab300a28** ... 的中
- S4_log: 予測 84b2ce10 → 実測 **28ff757a** ... **外れ**
- S4_sin: 予測 601eb0fa → 実測 601eb0fa ... 的中
- S2_sum: 予測 f9458b5c のまま → 実測 f9458b5c ... 的中

## 確定したこと

**機構は実証された。** blade という1台の中で、同一の musl イメージ・同一の numpy 2.2.6・
同一の Python 3.11.17 のまま、AVX-512 の numpy ディスパッチ対象を無効化しただけで
exp/log の値が動いた(5119b269→ab300a28、a6b6b7fa→28ff757a)。
かつ無効化後の exp は、AVX2 しか持たない anakin の musl 値と完全一致した。

→ exp/log は、AVX-512F があれば NumPy 内製カーネル(libc非依存)、
  無ければプラットフォームの libm にフォールバックする。
  「exp/log は libc 非依存」という一般化は誤りで、
  正しくは「AVX-512 専用カーネルの有無が、libm を経由するかどうかを決める」。

sin は AVX-512 の有無に一切反応しなかった(601eb0fa のまま)。
NumPy 2.2.6 が float64 の sin/cos の SIMD 経路を撤回している
(loops_trigonometric.dispatch.cpp の「Disable SIMD code sin/cos f64 and revert to libm」)
ことと整合する。

## 外れから分かった新しいこと

musl の log は ISA によって違うビットを返す:
- x86_64 の musl: 28ff757a (blade/alpine/nosimd512 と anakin/alpine で一致)
- aarch64 の musl: 84b2ce10 (dgxspark/alpine)

一方、musl の exp と sin は両 ISA で同一(ab300a28 / 601eb0fa)。
glibc は log も両 ISA で同一(09cef322)。

つまり「同じ libc 実装なら ISA を跨いでも同じ」という素朴な読みは、musl の log では成立しない。
**この原因は未検証である。** FMA の有無(musllinux の x86-64 ベースラインは FMA3 を含まず、
aarch64 ベースラインは FMA を含む)が候補だが、本プロジェクトは検証していない。

## 副次的な観測(予測を立てていなかった項目)

- S3_matmul_float64 は base→alpine で全ホストで変わる
  (anakin f86a1f40→c924d2fa、blade 76042a4d→8875959c、dgxspark 0cb8db34→86d16106)。
  glibc/musl の比較軸は BLAS も同時に動かしており、libm 単独の軸ではない。
