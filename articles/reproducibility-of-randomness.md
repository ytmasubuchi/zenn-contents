---
title: "乱数のseedを固定しても、結果が一致するとは限らない"
emoji: "🎲"
type: "tech"
topics: ["python", "numpy", "pytorch", "reproducibility", "docker"]
published: false
publication_name: "jdsc_sol_dev"
---

## 1. はじめに

> 同じseed・同じDocker・同じコードでも、2台のマシンで結果がビット単位で食い違うことがある。本記事は、どこまでなら「再現します」と言い切れるのかを、4台のホストでの実測で確かめた記録である。

納品物に「同じ入力なら同じ結果が出ます」と書きたくなる場面は多い。
しかし乱数を使う計算では、この一文がどこまで正しいのかを厳密に言うのは意外と難しい。
seedを固定しても、CPUの世代やスレッド数、ライブラリのバージョンが変われば、最終結果が変わることがある。

本記事は、この問題に実測で答える。

TL;DRは次の3点である。

- 乱数生成のビット列そのものが一致する保証と、それを使った計算の最終結果が一致する保証は別物である。
- 保証はL0(規格)からL4(保証なし)まで5段階に分かれる。段階によって壊れる条件も、納品文書に書ける文言も違う。
- MacBook・anakin・blade・dgxsparkの4台、7種類のDockerイメージ、30回の実測で境界を確かめた。割れなかった仮説もそのまま書く。

---

## 2. 冒頭の実測(同じseed・同じコンテナでビットが割れた)

> 同じDockerイメージ、同じseed、どちらもx86_64の2台で、NumPyの`exp`/`log`の結果がビット単位で食い違った。原因は推測ではなく、対照実験で確定している。

### 理論

NumPyは起動時にCPUが対応する命令セットを検出し、演算ごとに別のコードパスを選ぶ「CPUディスパッチ」を行う。
AVX2やAVX-512は、Intel系CPUが持つSIMD命令セットの世代名であり、どちらが選ばれるかでコードパスが変わる。
NumPy issue #23523では、メンテナのmattip氏が「使用する実装によっては、golden resultsとの差を最大4 ULPまで許容している」と述べている。
`sin`/`cos`/`exp`/`log`でAVX-512と非AVX-512のコードパスが異なる結果を返すことは、同issueで公式に確認されている。
ULPはunits in the last placeの略で、浮動小数点数の最下位ビット1つ分の差を指す。

:::details SIMDとCPUディスパッチとは

SIMD(Single Instruction, Multiple Data)とは、CPUが1回の命令で複数の数値をまとめて計算する仕組みである。
NumPyで配列同士の`+`や`np.exp`のような演算を書くと、内部ではこのSIMD命令が使われていることが多い。
AVX2やAVX-512は、Intel系CPUが持つSIMD命令セットの世代名であり、AVX-512の方が1回に処理できる数値の幅が広い。

同じCPUでも、実行時にどのSIMD命令セットを使うかを自動的に選ぶ処理を「CPUディスパッチ」と呼ぶ。
NumPyは起動時にCPUが対応する命令セットを検出し、AVX-512が使えるCPUではAVX-512向けのコードパスを選ぶ。
命令セットごとに別のコードパスが選ばれるため、同じ入力でも最終ビットが変わることがある。
つまりSIMDとCPUディスパッチは、Pythonのコードを1行も変えなくても、実行先のCPU次第で結果に影響しうる層にあたる。

:::

### 仮説

issue #23523の4 ULP許容と、sin/cos/exp/logの差が公式に確認済みという記述から、anakinとbladeで超越関数(`exp`/`log`/`sin`)が割れると予想した。
一方`np.sum`のような縮約については、同種の文書化された報告が見当たらず、理論から割れる・割れないのどちらかを導けなかった。
そのため`np.sum`は「真の未知数」と位置づけ、実測して確かめることにした。

### 実験設定

anakin(Intel、AVX2まで)とblade(Intel、AVX-512あり)の2台で、同じbaseイメージ・同じnumpy 2.2.6・同じseedを使った。
複数のNumPy計算(`S4_exp_float64`・`S4_log_float64`・`S4_sin_float64`・`S2_sum_float64`)を実行した。
原因がSIMD命令セットの違いなのか、CPUアーキテクチャそのものの違いなのかを切り分けるため、bladeでAVX-512を無効化する対照実験(`NPY_DISABLE_CPU_FEATURES`)も行った。
この対照実験では、ベースイメージ・numpyのバージョン・seedは実験前と同一のまま、AVX-512の有効/無効だけを変えた。

### 結果

`np.exp`と`np.log`の結果がビット単位で一致しなかった。
一方`np.sin`と`np.sum`は一致した。
AVX-512を無効化したbladeは、anakinと全プローブでビット一致した。
原因がSIMDディスパッチであることは、この対照実験によって確定している。

| プローブ         | anakin(AVX2) | blade(AVX-512) | blade(AVX-512無効化) |
| ---------------- | ------------ | -------------- | -------------------- |
| `S4_exp_float64` | cba8bfa7     | 5119b269       | cba8bfa7             |
| `S4_log_float64` | 09cef322     | a6b6b7fa       | 09cef322             |
| `S4_sin_float64` | ddf398b7     | ddf398b7       | ddf398b7             |
| `S2_sum_float64` | f9458b5c     | f9458b5c       | f9458b5c             |

### 仮説との照らし合わせ

`exp`/`log`が割れたことは仮説通りだった。
一方`sin`は、本章の実験単体ではanakinとbladeの間で割れなかった。
内訳の違い(`exp`/`log`はSIMD依存・libc非依存、`sin`はlibc依存・SIMD非依存)は、第6章G2で別の対照実験とあわせて扱う。

> 外れた仮説: `np.sum`はAVX2とAVX-512で割れなかった。縮約(総和)は、超越関数よりも頑健だった。

:::details 縮約とpairwise summationとは

縮約(reduction)とは、複数の数値を1つにまとめる計算全般を指し、合計(sum)や平均はその代表例である。
浮動小数点の足し算には結合則が無いため、同じ数値の集合でも足す順序が変われば、合計の最下位ビットが変わることがある。
`np.sum`のような縮約は、要素数が多い配列では先頭から順に足すのではなく、途中で部分和に分けてから合体させる方式で計算されることが多い。

この方式はpairwise summation(部分和を作ってから足し合わせる方法)と呼ばれ、単純な逐次合計より誤差が小さくなる利点がある。
部分和をどの大きさで区切るかという閾値は、ライブラリの実装が内部で決めており、公開された規約として文書化されているわけではない。
そのため縮約の結果がバージョン間で変わっても、それはバグではなく実装都合の変更でありうる。

:::

詳しい内訳は第6章の実測マトリクスでまとめて扱う。

---

## 3. seedを固定すると何が起きているのか

> seedを固定して止まるのは、ライブラリがOSに乱数を問い合わせる、その1回の呼び出しだけである。OSのCSPRNGもCPU内蔵の乱数源もPYTHONHASHSEEDのハッシュ鍵も、seedとは無関係に動き続ける。

乱数生成は階層構造になっている。
階層は上からアプリケーション、ライブラリ/ランタイムのPRNG、OSのCSPRNG、ハードウェア乱数源の順に並ぶ。
下に行くほどソフトウェアから遠くなる。

![F1. 乱数生成の階層。seed固定が切るのは1本の呼び出しだけ。](/images/reproducibility-of-randomness/f1_layers.png)
_F1. 乱数生成の階層(原理図)。_

`random.seed(42)`のようにseedを指定すると、ライブラリは本来OSに尋ねるはずだった`os.urandom()`の呼び出しをスキップする。
止まるのはこの1箇所だけで、それより下の層は無関係に動き続ける。

OS側では、2022年のLinuxカーネルの書き直し(5.17/5.18)で、入力プールの処理がSHA-1からBLAKE2sに、出力段がChaCha20に変わった。
カーネル5.18以降はCPUごとに別々の生成器を持つ。
「`/dev/random`はエントロピー切れでブロックする」という説明は、現行カーネルでは正しくない。
man7の`random(4)`は、「Linux 5.6以降、`/dev/random`は起動直後を除いてブロックしなくなった」と明記している。

:::details カーネルの意味とCSPRNG・エントロピーとは

本記事の「カーネル」は、文脈によって2つの別のものを指している。
1つはOSカーネルで、Linuxなどのオペレーティングシステムの中核部分であり、乱数生成や`/dev/random`もここで動いている。
もう1つは計算カーネル(compute kernel)で、GPU上で1回にまとめて実行される計算処理の単位を指し、これをGPU上で動かすときに「GPUカーネル」と呼ぶ。
NumPyの内部実装がSIMD命令を使って配列演算を行う処理も、同じ意味でのカーネル(内製カーネル)と呼ばれることがある。
OSカーネルの話をしている箇所と、GPUカーネルの話をしている箇所は別であり、同じ単語でも指しているものは違う。

CSPRNG(Cryptographically Secure Pseudo-Random Number Generator)とは、暗号用途にも耐える強度で乱数列を生成する仕組みである。
OSはこのCSPRNGを内部に持っており、`os.urandom()`や`/dev/random`はここから乱数を取り出す窓口にあたる。
エントロピーとは、CSPRNGの種になる予測不可能性の量を指し、キーボード入力やハードウェア乱数などから集められる。
現行のLinuxカーネルのCSPRNGは、起動直後の初期化が済んだ後は、エントロピーの蓄積を待ってブロックすることなく出力を返す。

:::

ハードウェア側では、Intelの`RDRAND`はSP 800-90A準拠のDRBGから読み、`RDSEED`はSP 800-90B/C準拠の生に近いエントロピーを直接読む。
RDRANDは1つのシードから最大1022個の連続した乱数しか生成せず、超えると自動的に再シードされる。
ARMの`RNDR`/`RNDRRS`も同じ構図で、`RNDRRS`は呼び出すたびに強制的に再シードする。

コンテナがホストのカーネルCSPRNGを共有しているかどうかは、今回の調査では一次情報を確認できなかった。
Mac上のDocker DesktopはそもそもLinuxの仮想マシンを経由しており、ホストのmacOSカーネルを直接使っているわけではない。
この記事では、この点について断定しない。

PYTHONHASHSEEDも、seedとは別の軸にある設定である。
目的はハッシュフラッディング攻撃(CVE-2012-1150)への対策で、PEP 456によりSipHash24が採用された。
`dict`はPython 3.7以降、挿入順を保持する言語仕様になっており、PYTHONHASHSEEDの影響を受けない。
一方`set`はハッシュテーブルのスロット位置に依存するため、PYTHONHASHSEEDを変えると反復順が変わりうる。

この違いは第5章で実測する。

---

## 4. 再現性の保証レベル L0〜L4

> 「乱数の再現性」は一つの保証ではなく、保証の出所も壊れる条件も違う5段階に分かれる。段階を区別しないまま「再現します」と書くと、守れない約束をしていることがある。

まず語彙を2つ定義する。
ACM Artifact Review and Badging(v1.1、2020年8月)は、次の3つの観点を区別している。

- Repeatability: 同チーム・同環境
- Reproducibility: 別チーム・同じ作成物を使った同環境
- Replicability: 別チーム・独自に作った別環境

注意点として、ACMはv1.0(2016年承認)からv1.1(2020年)で、ReproducibilityとReplicabilityの意味を入れ替えている。
NISOの勧告に従った変更だとACM自身が明記しており、古い定義のまま読むと話が噛み合わなくなる。

もう1つの軸は、「ビット一致」と「数値的に近い」の違いである。
ACMの同じ文書は、「完全な再現や複製は要求されておらず、期待もされていない。結果は、その実験の種類にとって妥当とされる許容誤差の範囲で一致していればよい」と明記している。
「再現します」が`==`なのか`np.allclose(rtol=1e-9)`なのかで、約束の重さは大きく変わる。

![F2. 乱数が一致しても、最終結果が一致するとは限らない。](/images/reproducibility-of-randomness/f2_divergence.png)
_F2. PRNGのビット列の一致と、それを使った計算結果の一致は別の保証(原理図)。_

保証のレベルをまとめると次の表になる。

| レベル          | 保証の出所                           | 代表例                                                                                                                       | 壊れる条件                                                          | 納品で言えること                                                      |
| --------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------- |
| L0 規格         | 国際規格・公開アルゴリズム仕様       | IEEE754の`+ - × ÷ √`・FMA(Fused Multiply-Add、積和を1回の丸めで計算する演算。正しく丸め)、MT19937/PCG64/Philoxの生成ビット列 | 言語・OS・CPUを跨いでも壊れない                                     | バージョン固定でビット一致を約束できる                                |
| L1 明文の規約   | ライブラリが文書で約束               | NumPy NEP19(RandomStateは保証、Generatorは非保証)、PyTorch「リリース間の再現性は保証しない」                                 | バージョンを上げた瞬間                                              | バージョン固定を条件に約束できる                                      |
| L2 事実上の実装 | 文書化なし。実装がたまたま決めている | 正規乱数の変換法(ziggurat/polar法)、`np.sum`のpairwise summationの閾値                                                       | バージョン差(根拠なく変わりうる)                                    | 実際には一致するが、根拠が無いので書かない方がよい                    |
| L3 実行時環境   | 実行時に環境が選ぶ                   | SIMDディスパッチ、BLASスレッド数、libm(glibc/musl)                                                                           | 同じコード・同じ版でもCPU世代・スレッド数・ベースイメージが変われば | 「同一イメージ+同一CPU世代+スレッド数固定」まで条件を書けば約束できる |
| L4 保証なし     | 原理的に非決定                       | PYTHONHASHSEED、GPUの`atomicAdd`順序、cuDNN benchmarkの自動選択                                                              | 同じプロセスを2回走らせても                                         | 約束できない。許容誤差でしか語れない                                  |

![F3. 再現性の保証レベルL0〜L4。](/images/reproducibility-of-randomness/f3_levels.png)
_F3. 各レベルで何が壊すかを並べた原理図。_

本文に必ず入れておくべき但し書きが2つある。

- IEEE 754が保証するのは演算1つであって、演算の並びではない。IEEEのサイト(grouper.ieee.org)で公開されている背景資料は、「the IEEE standard does not guarantee that the same program will deliver identical results on all conforming systems(同じプログラムが全ての準拠システムで同一の結果を返すことを規格は保証しない)」と明記している。この資料は規格の規範的な本文ではなく背景説明用の文書であり、ワーキンググループ内での正式な位置づけまでは確認できていないが、信頼できる二次資料として扱う。同じ資料は「移植した際に途中結果が食い違うなら必ずソフトウェアのバグだ」という通説も否定している。
- `exp`/`log`/`sin`のような超越関数は、IEEE 754では正しく丸めることが要求されておらず、推奨にとどまる。任意の点で正しく丸めるのに何ビット余分に必要かを一般に抑える手法が無いため、主要なlibm実装はどれも全ての超越関数を正しく丸めてはいない。これが第6章でlibm実装差が出る根本原因になる。

:::details libcとlibmとは

libcとは、Cで書かれたプログラムが使う標準的な基盤ライブラリで、ファイル操作や文字列処理など基本機能を提供する。
libmはlibcの一部、または別ライブラリとして提供される数学関数集であり、`sin`・`cos`・`exp`・`log`のような関数の実装はここに入っている。
PythonからNumPyで`np.sin`や`np.log`を呼んだとき、最終的に計算しているのはPython自身ではなく、このlibmの実装であることが多い。

glibcはLinuxディストリビューションで広く使われるlibcの実装であり、muslはAlpine Linuxなどの軽量イメージで使われる別の実装である。
同じ`sin`という関数名でも、glibc版とmusl版では内部のアルゴリズムが異なるため、同じ入力に対して返すビット列が違うことがある。
どちらがより正しいかという話ではなく、どちらも規格の許容範囲内で別の近似をしている、という理解が実務上は役に立つ。

:::

L2(事実上の実装)は、記事中では「今回の実測範囲では一致した」としか書かない。
文書化された保証ではないため、約束の根拠にできない。
たとえばlegacyの`RandomState.standard_normal`はpolar法、`Generator.standard_normal`は256段のziggurat法を使っており、変換方式そのものが違う。
NumPy公式ドキュメントは「it is not possible to reproduce the exact random values using Generator for these distributions」と明記している。
日本語にすると「Generatorでこれらの分布の正確な乱数値を再現することはできない」であり、同じseedでも正規分布の値は一致しない。

---

## 5. 「インプットが全部同じ」は案外成立していない

> seedと環境変数をそろえても、globの列挙順・setの反復順・並列ワーカーの完了順は、気づかないうちに違う入力になっていることがある。

納品時に「同じ入力」と言うとき、通常は「同じデータ」「同じコード」「同じseed」を指す。
しかしコードの中には、seedで止められない隠れた入力がいくつかある。

### 理論

第3章で確認した通り、`dict`はPython 3.7以降挿入順を保持する言語仕様であり、PYTHONHASHSEEDの影響を受けない。
一方`set`はハッシュテーブルのスロット位置に依存するため、PYTHONHASHSEEDを変えると反復順が変わりうる。
PYTHONHASHSEEDの目的はハッシュフラッディング攻撃(CVE-2012-1150)への対策であり、PEP 456でSipHash24が採用された、という経緯も第3章の通りである。

`glob.glob`のファイル列挙順について、順序の安定性を明文で保証する規格は、本記事が参照した一次情報の範囲では確認できていない。
一般にファイルシステムの実装に依存すると理解されているにとどまる。
並列ワーカーの完了順はOSのスケジューラが決めており、第4章のL4(保証なし)の定義が示す通り、原理的に非決定な領域に属する。

### 仮説

PYTHONHASHSEEDを0に固定すれば、4ホストで`set`由来の値が一致すると予想した。
固定しない既定状態では、ホスト間はもちろん、同じホストの再起動間でも値が割れると予想した。
並列ワーカーの完了順は、L4の定義からして、同一プロセス内の2回の計測でも一致しないと予想した。
`glob`の列挙順については、ファイルシステム実装に依存するという理論から「環境を跨げば変わりうる」とは言えても、同一プロセス内の2回で実際に安定するかどうかまでは理論から導けず、実測で確かめることにした。

### 実験設定

`glob.glob`でのファイル一覧取得、`set`の反復順(`hash('reproducibility')`を例に使用)、32ジョブを使った並列ワーカーの完了順の3つを対象にした。
4ホストで、PYTHONHASHSEEDを0に固定した場合と、設定しない既定の場合を比較した。
globはext4・XFS・tmpfsの3種類のファイルシステム上で、同一プロセス内で2回呼び出した。
並列ワーカーは同一プロセス内で2回計測し、完了順を比較した。

### 結果

- `glob.glob`でファイル一覧を取ると、順序はファイルシステムに依存する。今回の実測では、同一プロセス内で2回`glob`を呼んだ場合、30ラン全てで安定していた(`stable_within_process`として自己申告)。ただしこれはext4・XFS・tmpfsでの結果であり、全てのファイルシステムで保証されているわけではない。
- `set`の反復順は、ハッシュテーブルのスロット位置に依存する。PYTHONHASHSEEDを0に固定すると、4ホストとも`hash('reproducibility')`が一致した。PYTHONHASHSEEDを設定しない既定状態では、同じコンテナを再起動するだけで4ホスト全てで値が食い違い、同じイメージを別コンテナで4回立てただけで4通りの値が出た。
- 並列ワーカーの完了順は、seedとは無関係にOSのスケジューラが決める。同一プロセス内で2回計測しても、32ジョブの完了順は毎回異なった(`stable_within_process=False`として自己申告)。これは測定上の不備ではなく、並列処理そのものの性質であり、L4(保証なし)に相当する。

### 仮説との照らし合わせ

`set`は仮説通りで、PYTHONHASHSEED=0で一致し、既定状態では不一致になった。
並列ワーカーの完了順も仮説通り、同一プロセス内でも毎回異なった。
`glob`は理論からは導けなかったが、実測では30ラン全てで安定していた(ext4・XFS・tmpfsの範囲)。
理論が予測していなかった分、この安定性は実測で初めて分かった事実として扱う。

この3つは、seedを固定した「正しい」コードの中にも紛れ込む。
納品前のチェックリストに「`glob`の結果をソートしているか」「`PYTHONHASHSEED`を固定しているか」を加える価値がある。

---

## 6. 実測マトリクス(何が割れて、何が割れなかったか)

> 4ホスト・7イメージ・30ランの実測を、保証レベル×条件のマトリクスにまとめた。Python版・BLAS実装・SIMD世代・スレッド数・GPU世代ごとに、割れた条件と割れなかった条件をそのまま並べる。

以下は、全ての実測結果をレベル×条件で並べたマトリクスである。
緑は一致、赤は不一致、オレンジは複数プローブの結果が混在していることを示す(凡例は画像内に記載)。

![F4. 保証レベル×環境条件。](/images/reproducibility-of-randomness/f4_matrix.png)
_F4. 30ランの実測結果を自動生成したマトリクス。_

ここから先の差は、ほぼ全てが同じ原因に行き着く。
浮動小数点の加算には結合則が無い。

![F5. 浮動小数点の加算に結合則はない。](/images/reproducibility-of-randomness/f5_assoc.png)
_F5. `(x+y)+z`と`x+(y+z)`が別の値になる例(Goldberg 1991)。_

IEEE 754は個々の演算の丸めは規定するが、複数の演算をどの順序・どのグループで評価するかは規定しない。
順序はコンパイラ・SIMD幅・スレッド分割など、実行環境が決める。

### G1 時間・コンテナ

#### 理論

第5章で確認した通り、L0(規格)に属する計算はPYTHONHASHSEEDのような実行時条件と無関係であり、L4(保証なし)に属するPYTHONHASHSEED依存の挙動は、プロセスを再起動するだけで変わりうる。
この整理は第4章のレベル定義そのものである。

#### 仮説

同一プロセス内の2回実行・同一コンテナの再起動・同一イメージからの別コンテナ、のいずれでも、L0の計算は常に一致すると予想した。
一方PYTHONHASHSEEDに依存する値は、この時間軸で崩れると予想した。

#### 実験設定

比較対象は、同一プロセス内の2回実行・同一コンテナの再起動・同一イメージからの別コンテナ、という3つの条件である。
具体的な実測条件と結果は第5章にまとめている。

#### 結果

同一プロセス内の2回実行・同一コンテナの再起動・同一イメージからの別コンテナについては、第5章で扱った。
L0(規格レベル)の計算はこの軸でも常に一致し、L4(PYTHONHASHSEEDなど)はこの軸で崩れた。

#### 仮説との照らし合わせ

結果は仮説通りだった。
L0がこの時間軸で崩れないこと、L4がこの軸で崩れることは、ともに第4章のレベル定義と整合する。

### G2 ソフトウェア

#### 理論

Python自体のバージョンがNumPyの乱数の値に影響する経路は、NumPyの乱数生成器の実装構造上見当たらない。
この点を明言した一次文書は確認できておらず、以下はアーキテクチャからの消去法による予想である。

NEP19は、「Bug fixes that change the stream SHALL NOT be made to RandomState(ストリームを変えるバグ修正はRandomStateに対して行わない)」と明記する。
一方Generatorについては、「breaking stream-compatibility ... will be allowed with caution ... on X.Y releases, never X.Y.Z」としている。
機能追加版(X.Yリリース)に限り、注意しながらストリーム互換性を破ることを許容する、という意味である。
NumPy 1.22.0のリリースノートには、float32型の一様乱数で最下位ビットが常に0になっていたバグの修正が記載されている。

第4章で確認した通り、IEEE754は個々の演算の丸めを規定するが、複数の演算をどの順序・どのグループで評価するかは規定しない。
BLAS実装(OpenBLAS・MKL)は、この規定されていない部分を独自に実装しており、libm(超越関数)とは別のライブラリである。
同じく第4章の通り、`exp`/`log`/`sin`のような超越関数はIEEE754で正しく丸めることが要求されておらず、glibcとmuslのような別々のlibm実装は異なる近似をしうる。

#### 仮説

Python 3.10と3.11(numpy版は同一)では、S1(PRNGの生ビット列を見るプローブ群)が全て一致すると予想した。
numpy 2.2.6と1.21.6(NEP19が実際に境界を跨ぐ組み合わせ)では、NumPy 1.22.0のバグ修正を踏まえ、`S1_generator_random_float32`だけが不一致になると予想した。
legacyのRandomStateは一致すると予想した。
OpenBLASとMKLでは、`S3_matmul_float32`/`float64`が不一致になり、libm(`S4_exp_float64`)は一致すると予想した。
glibcベースのイメージとmusl(alpine)ベースのイメージでは、libm実装差により`S4`(`exp`/`log`/`sin`)のいずれかが不一致になると予想したが、どの関数が割れるかまでは理論から特定できなかった。

#### 実験設定

Python 3.10と3.11(numpyバージョンは同一)を比較した。
numpy 2.2.6と1.21.6(NEP19が境界を跨ぐ組み合わせ)を比較した。
OpenBLASとMKL(ともにスレッド数固定)を比較した。
glibcベースのイメージとmusl(alpine)ベースのイメージを比較した。

#### 結果

Python 3.10と3.11(numpy版は同一)を比較すると、S1は全て一致した。
Python自体のバージョン差は、NumPyの乱数の値を変える理由にならなかった。

numpy 2.2.6と1.21.6を比較すると、`S1_generator_random_float32`だけが不一致になった。
stdlibのMT19937、legacyのRandomState、Generator経由のPCG64・Philoxは全て一致した。
これはNumPy 1.22.0のリリースノートに記載された「float32型の一様乱数で最下位ビットが常に0になっていたバグ修正」を、ピンポイントで捉えている。

OpenBLASとMKL(ともにスレッド数固定)を比較すると、`S3_matmul_float32`/`float64`はどちらも不一致になった。
一方`S4_exp_float64`と`S2_sum_float64`は一致した。
BLASの実装差は行列積(線形代数)を変えるが、libm(超越関数)には影響しなかった。

:::details BLASとGEMM(OpenBLAS・MKL)とは

BLAS(Basic Linear Algebra Subprograms)とは、行列やベクトルの計算を高速に行うための標準的なインタフェースの仕様である。
NumPyで`A @ B`のように行列積を書いたとき、実際にその計算をしているのはNumPy自身ではなく、BLASという別のライブラリであることが多い。
GEMM(General Matrix Multiply)は、このBLASが定める代表的な演算の1つで、行列同士の積を指す。

OpenBLASとMKL(Math Kernel Library)は、どちらもBLASの実装の一種であり、オープンソースかIntel製かという違いがある。
どちらも同じBLASの仕様を満たすが、内部でどう計算を分割し、どの順序で足し合わせるかという実装の詳細は実装ごとに異なる。
そのため、同じ行列積でも使うBLAS実装が変われば、結果のビット列が変わることがある。

:::

glibcベースのイメージとmusl(alpine)ベースのイメージを比較すると、`S4_sin_float64`が不一致になった。
`S4_exp_float64`・`S4_log_float64`・`S2_sum_float64`は一致した。
これは第2章のAVX2/AVX-512の結果と対になっている。

#### 仮説との照らし合わせ

Python版差は仮説通りだった。
NEP19の実証も仮説通りで、legacyは動かず、Generatorだけが動いた、という条文通りの結果になった。
BLAS実装差も仮説通りで、行列積は割れ、libmは影響を受けなかった。

glibc/muslの比較は、「S4のどれかが割れる」という予想自体は当たったが、割れた関数は予想と逆だった。
第2章ではexp/logが割れてsinは一致したのに対し、本章ではsinが割れてexp/logは一致した。

> `exp`/`log`はSIMD依存・libc非依存、`sin`はlibc依存・SIMD非依存。2つの独立した対照実験(SIMD世代の違いとlibc実装の違い)が、互いに補い合う形でこれを示している。機構(NumPy内製カーネルかlibmフォールバックか)までは実測していないため、測った相関として書くにとどめる。

### G3 ハードウェア

#### 理論

第4章で確認した通り、IEEE754は演算の順序までは規定しない。
BLAS実装はスレッド数に応じて行列を分割するタイル数を変えるため、タイルの足し合わせ順が変わりうる。
KMeansの重心計算も、各クラスタに属する点の合計という縮約を内部で使っている。

一般に、異なるCPUアーキテクチャ(x86_64とaarch64)は、使えるSIMD命令セットやコンパイラの最適化が異なり、結果が変わりうると広く理解されている。
本記事が参照した一次資料の中に、x86_64とaarch64の違いそのものを論じたものは無く、以下はこの一般的な理解に基づく予想である。

#### 仮説

スレッド数を1・8・最大に変えると、`S3_matmul_float32`は複数の値に割れると予想した。
`np.sum`(`S2_sum_float64`)への影響は理論から断定できなかった。
KMeans(`S5_kmeans_centers`)は、スレッド数を固定しない設定や8スレッドの設定で、重心が複数パターンに分かれると予想した。
x86_64とaarch64の違いで、`S4`(exp/log)が割れると予想した。

#### 実験設定

スレッド数を1・8・最大(ホストの論理コア数)に変えて`S3_matmul_float32`/`float64`と`S2_sum_float64`を比較した。
KMeans(`S5_kmeans_centers`)は、`effective_thread_count`を1・8で固定した場合と、固定しない既定設定を比較した。
別の1回の計測では、8CPU構成のコンテナでスレッド数を1・2・4・8・既定の各設定にし、30回ずつフィットを繰り返した。
第2章のanakin(x86_64、AVX2)とblade(x86_64、AVX-512)の対照実験に加え、x86_64(anakin)とaarch64(dgxspark)を比較した。

#### 結果

`S3_matmul_float32`はanakinで3通りに割れた(threads1/threads8/threadsmaxがそれぞれ別の値)。
`S3_matmul_float64`は4ホスト・全スレッド数設定で一致した。
`np.sum`(`S2_sum_float64`)も全スレッド数設定で一致した。

![F10. BLASの行列積(スレッド数が変わるとタイルの足し合わせ順が変わる)。](/images/reproducibility-of-randomness/f10_blas_tiles.png)
_F10. スレッド数ごとのタイル割り当ての違い(原理図)。_

KMeansは、`effective_thread_count`が1のとき4ホスト全てで同じ重心に収束した。
8に固定すると、6通りの設定のうち4通りの異なる重心に分かれた。
30回×6条件の繰り返し実行では、スレッド数を1・2・4に固定すると30回とも同じ重心になり、8に固定すると13回、既定設定では19回、異なる重心になった。
この回数は実行のたびに多少変動する値であり、固定した定数としては読まないでほしい。
この生データは`results/`には無く、プロジェクトの判断記録にのみ残っている。

一方で`n_iter_`(反復回数)は180回のフィット全てで2、重心の分散(inertia)の広がりは最大でも3.9×10^-16程度に収まっていた。

x86_64(anakin)とaarch64(dgxspark)を比較すると、`S2`/`S4`は全て一致した。

#### 仮説との照らし合わせ

スレッド数の影響は仮説通りで、float32のBLASは割れ、float64とnp.sumは一致した。
KMeansの不安定も仮説通りだった。
つまり不安定の正体は、別の局所最適を見つけているのではなく、縮約(合計)の丸め方の違いである。

第2章の実測を振り返ると、`S4_exp_float64`/`S4_log_float64`は同じx86_64同士(anakinのAVX2とbladeのAVX-512)で不一致になった。
一方、本章のx86_64とaarch64の比較では全て一致した。

> 外れた仮説: アーキテクチャ(x86_64かaarch64か)は決め手ではなかった。決め手はSIMD世代だった。当初はx86_64とaarch64の違いで結果が割れると予測していたが、実際に割れたのは同じx86_64同士のAVX2とAVX-512の間だった。

![F6. SIMD幅が変わると縮約の足す順序が変わる。](/images/reproducibility-of-randomness/f6_simd.png)
_F6. 8並列と16並列でレーンの組み方が変わる例(原理図)。_

### G4 アクセラレータ

#### 理論

NVIDIAのcuBLAS公式ドキュメントは、「同一ツールキット版のcuBLAS APIルーチンは、同一アーキテクチャ・同一SM数のGPU上で実行する限り、毎回ビット単位で同じ結果を生成する」と明記している。
この保証は複数のCUDAストリームが同時に有効な場合には成り立たない。
cuDNN公式ドキュメントは「異なるアーキテクチャ間では、いかなるcuDNNルーチンもビット単位の再現性を保証しない」とVolta/Turingを例に明記している。

atomicAdd自体は1回ずつの書き込みを保証するが、どのスレッドがどの順番で書き込むかまでは保証しない。
浮動小数点の足し算には結合則が無いため、足す順序が実行のたびに変わると、合計の最下位ビットも実行のたびに変わりうる。

:::details atomic演算(atomicAdd)とは

atomic演算とは、複数の処理が同時に同じメモリ上の値を書き換えようとしても、1回ずつ確実に反映されることを保証する演算である。
GPUでは数千のスレッドが同時に動くため、複数のスレッドが同じ場所に値を足し込みたいとき、`atomicAdd`のようなatomic演算を使う。
atomicAdd自体は1回ずつの書き込みを保証するが、どのスレッドがどの順番で書き込むかまでは保証しない。

浮動小数点の足し算には結合則が無いため、足す順序が実行のたびに変わると、合計の最下位ビットも実行のたびに変わりうる。
これが、同じGPU・同じコードでも`index_add_`のような処理の結果が毎回一致しない理由である。
`torch.use_deterministic_algorithms(True)`のような設定は、足し込む順序を固定することで、この種の非決定性を抑える。

:::

PyTorchのCUDA RNGは`curand_init`をスレッドのインデックスでシードしており、スレッドと要素の対応はSM数に依存して決まる(PyTorch issue #194676)。
この挙動を公式ドキュメントに追記するPR(#196733)は、2026年10月14日時点でレビュー承認済みだが未マージであり、現時点では公式に文書化されていない。

:::details fp16・bf16・TF32・dtypeとは

dtypeとは、NumPyやPyTorchの配列・テンソルが持つ、要素の型を表す属性であり、`float32`や`float16`もdtypeの一種である。
fp32(float32)は32ビット、fp16(float16)は16ビットで1つの数値を表現し、ビット数が少ないほど表現できる精度や範囲は狭くなる。
bf16(bfloat16)もfp16と同じ16ビットだが、指数部にfp32と同じ桁数を割り当て、仮数部(精度)を削る設計になっている。

TF32(TensorFloat-32)はNVIDIA GPU上の内部計算形式で、fp32の入出力を保ちながら、内部の掛け算だけをfp32より粗い精度で行う。
どの形式を使うかによって丸め誤差の大きさが変わり、わずかな誤差がGPU世代ごとの結果の分かれ方に影響することがある。
dtypeを変えるという操作は、単に数値の型を変えるだけでなく、計算の精度そのものを変える操作である。

:::

#### 仮説

当初の実測設計は、CPU実行とGPU実行の結果を比較して、GPUの非決定性を示すことを狙っていた。
cuBLASの保証から、単一ストリーム・同一アーキテクチャ・同一SM(Streaming Multiprocessor、GPU内の演算ユニット群)数の下でのGEMMは決定的になると予想した。
一方atomicAddを使う`index_add_`は、同一GPU上でも5回の実行で一致しないと予想し、`torch.use_deterministic_algorithms(True)`を指定すればビット一致に戻ると予想した。
決定的モードにしても、cuDNNの文書の通り、異なるGPUアーキテクチャ間では一致しない可能性があると予想した。
PyTorch issue #194676の記述から、要素数が一定を超えるとSM数の異なるGPU間で`curand_randn`の値が変わると予想した。

#### 実験設定

anakin(RTX 4090)で、Linear層+ReLUのMLPを10ステップ学習させる処理と、重複したインデックスへの`index_add_`を、それぞれ5回ずつ実行した。
`index_add_`は`torch.use_deterministic_algorithms(True)`の有無で比較した。
決定的モードの`index_add_`は、3GPU(RTX 4090・RTX PRO 6000・GB10)で比較した。

TF32は無効化と有効化の両方で、3GPUを比較した。
greedy復号のトークン列は、fp32・fp16・bf16の3dtypeで、3GPUを比較した。
`torch.randn`を同じseedで4サイズ(1,024 / 65,536 / 4,194,304 / 67,108,864)生成し、サイズごとにダイジェストを取った(`S8_curand_randn_sizes`、3GPU比較)。

#### 結果

anakin(RTX 4090)で確かめると、Linear層+ReLUのMLPを10ステップ学習させる処理は、5回ともビット一致した。
GPUで動かすだけでは、非決定性は現れなかった。

一方、重複したインデックスへの`index_add_`(内部で`atomicAdd`を使う処理)は、5回とも一致しなかった(最大絶対差 8.24×10^-4)。
`torch.use_deterministic_algorithms(True)`を指定すると、同じ`index_add_`が5回ともビット一致した。
MLPと`index_add_`それぞれの5回反復分の生データは`results/`には無く、プロジェクトの判断記録にのみ残っている。

![F7. GPU上の非決定性(atomic演算を使う処理は毎回変わりうる)。](/images/reproducibility-of-randomness/f7_gpu_reduce.png)
_F7. anakinでの5回反復実測(MLP学習は一致、`index_add_`は不一致)。_

今回の実測は単一ストリームで行っている。
`index_add_`を決定的モードで3GPU(RTX 4090・RTX PRO 6000・GB10)で比較すると、anakinとbladeは一致し、dgxsparkだけ別の値になった。

TF32を無効化すると、anakinとbladeは一致し、dgxsparkだけ別の値になった。
有効化すると、3GPUとも別々の値になった。
greedy復号のトークン列は、fp32とfp16では3GPUとも常に一致した。

bf16にすると、anakin(Adaアーキテクチャ)はfp32・fp16と同じ68トークンのまま変わらなかった。
一方blade・dgxspark(ともにBlackwellアーキテクチャ)は、bf16にすると63トークンの別の列になった(2台は互いに一致)。

`torch.randn`を同じseedで4サイズ生成した結果は次の通りである。

| n          | RTX 4090(SM128) | RTX PRO 6000(SM188) | GB10(SM48) |
| ---------- | --------------- | ------------------- | ---------- |
| 1,024      | c9020ce1        | c9020ce1            | c9020ce1   |
| 65,536     | fb9b6d89        | fb9b6d89            | fb9b6d89   |
| 4,194,304  | 947daf61        | abf64dd5            | 9e849c24   |
| 67,108,864 | 9638c416        | 383fa914            | 12803be3   |

65,536要素までは3GPUとも同じ乱数が出た。
4,194,304要素を超えると、3GPUが3通りの異なる乱数を返した。
これは丸めの差ではなく、乱数の値そのものが違う。

#### 仮説との照らし合わせ

しかし実際には、CPU実行とGPU実行の比較という当初の設計では、GPUの非決定性を示せなかった。

> 外れた仮説: 「GPUは非決定的」という大雑把な理解は誤りだった。cuBLASのGEMM(行列積)は、単一ストリーム・同一アーキテクチャ・同一SM数という条件下では決定的である。非決定性は`atomicAdd`を使う一部の演算に局在している。

`index_add_`が割れ、決定的モードで直ることは仮説通りだった。
今回の実測は単一ストリームで行っており、cuBLASの保証の前提と一致する。
決定的モードでもdgxsparkだけ別の値になったことは、cuDNN文書の「異なるアーキテクチャ間では再現性を保証しない」という枠組みで理解できる。

> 精度を落とすと、GPU世代が効き始める。fp32とfp16では3GPUとも違いが見えなかったが、bf16にすると、Ada世代とBlackwell世代の間で出力そのものが変わった。

`S8_curand_randn_sizes`の閾値(65,536と4,194,304の間)が実在したことは仮説通りだった。
ただし具体的な閾値の位置や、3GPUが3通りの異なる値に分かれる点は、PyTorch issue #194676からは読み取れる。
一方、公式ドキュメントは未記載(PR #196733は未マージ)であり、理論からの厳密な予測ではなく推測の確認にとどまる。

---

## 7. LLMとtemperature

> temperature=0は「決定的」を意味する公式なモードではない。batchサイズ1と8の比較は検出力不足で現象を捉えられる設計になっていなかったが、同梱のデフォルト設定を外すと、生成はもっと早いトークンで分岐した。

Qwen2.5-0.5B-Instructを使い、transformersで生成した。

### temperature=0は存在しない

#### 理論

temperatureは、生成時にモデルの出力するロジットを割って候補のスコアを広げたり狭めたりする値である。
数学的には、temperatureで割る処理そのものが、0で割るという未定義の操作になる。
vLLM公式ドキュメントは、内部で`temperature<1e-5`をgreedy扱いにする仕様を明記しており、同じ「temperature=0」という入力でも、実装によって扱いが異なりうることを示している。
温度0での割り算が未定義になるという理論だけでは、transformersが`temperature=0`をどう扱うかは決まらない。vLLMのように内部でgreedy扱いに読み替える可能性と、エラーとして拒否する可能性の両方がありうるため、実際の挙動は実測で確認する必要がある。

#### 実験設定

transformersで、Qwen2.5-0.5B-Instructに対して`temperature=0`を指定して生成を呼び出した。

#### 結果

transformersに`temperature=0`を渡すと、`ValueError`が出た。
メッセージは次の通りである。

> `temperature` (=0.0) has to be a strictly positive float, otherwise your next token scores will be invalid. If you're looking for greedy decoding strategies, set `do_sample=False`.

#### 理論との照合

transformersはvLLMのような内部での読み替えをせず、エラーとして拒否する側を選んでいた。
ライブラリ自身が、「temperature=0」という慣用句は存在しないと言っている。
決定的な生成がしたいなら`do_sample=False`(greedy復号)を使う。
vLLMなど他の推論基盤は内部で`temperature<1e-5`をgreedy扱いにするなど、実装ごとに「temperature=0」の意味が違う。

:::details greedyデコーディングとlogits・top_p・top_kとは

LLMは次の1トークンを生成するとき、語彙全体に対するスコア(logits)を出力し、そこから次のトークンを選ぶ。
greedyデコーディングとは、そのスコアが最大のトークンを毎回そのまま選ぶ、最もシンプルな選び方である。
transformersでは`do_sample=False`、つまりサンプリングをしない設定が、このgreedyデコーディングに相当する。

top_kは、スコア上位k個のトークンだけを候補に絞ってからランダムに選ぶ方式で、top_pは上位から確率の合計がp(例えば0.8)に達するまでの候補に絞る方式である。
repetition_penaltyは、既に出たトークンのスコアを下げ、同じ語の繰り返しを抑える設定である。
いずれもgreedyより候補を広げたり絞ったりする調整であり、temperatureと組み合わさることで生成結果は大きく変わりうる。

:::

### batch1とbatch8

#### 理論

Thinking Machines Labは2025年9月のブログで、batch-invariance(バッチの構成によってGPUカーネルの縮約順が変わり出力が変わる現象)を報告している。
このブログは査読論文ではなくエンジニアリングブログであり、Qwen3-235B-A22B-Instruct-2507で1000回のgreedy生成を行い、出力の種類が80通りから1通りに減ったと報告している。
このブログが主張する中心メカニズムは、「ほぼ全てのLLM推論エンドポイントが非決定的である一番の理由は、負荷(したがってbatchサイズ)が非決定的に変動することにある("the primary reason nearly all LLM inference endpoints are nondeterministic is that the load (and thus batch-size) nondeterministically varies!")」というものである。
つまり、本番の推論サーバでは他ユーザの同時リクエストによって継続的バッチング(continuous batching)の構成が実行のたびに非決定的に変わり、同じリクエストでもバッチに同居する他系列が毎回変わることで出力が変わる、という本番推論サーバ固有の現象を指している。
この理論からは、バッチの構成を変えればgreedy出力が分岐しうることが導かれる。

#### 実験設定

Qwen2.5-0.5B-Instructを使い、3GPU(anakin・blade・dgxspark)で、batchサイズ1と8のgreedy出力をそれぞれ1回ずつ比較した。
batch8は、同一プロンプトをbatch方向に8本複製したものであり(`tokenizer([text] * batch, ...)`)、長さの異なる実プロンプトを混在させてはいない。

#### 結果

batchサイズ1と8のgreedy出力は3GPUとも一致した。

#### 理論との照合

> 理論で説明できなかった結果: batchサイズ1と8のgreedy出力は3GPUとも一致した。

ただしこれは「再現しなかった」と言うより、「検出できる設計ではなかった」と言うべきである。理由は次の3点である。

1. batch8の中身は同一プロンプトの8本複製であり、長さの異なる実プロンプトが混ざっていない。そのため、パディングとattention mask由来の経路分岐が原理的に発生しない構成になっている。
2. 比較は3GPU×各条件1回、つまりn=1の比較にすぎない。TMLが報告した分岐率は80/1000 = 8%であり、数%程度の分岐率をn=1の比較で「一致した」と結論することは統計的に意味を持たない。TMLの1000回試行に対し、本実測は統計的検出力が大きく劣る。
3. TMLが主張する中心メカニズムは、本番推論サーバにおける負荷変動由来の継続的バッチング構成の非決定性である。本実測は単一プロセス内でバッチの中身を固定して行った静的な比較であり、TMLが実証した現象と同じ土俵に立っていない。

したがって、今回の実測から「batch-invarianceは起きない」と結論することはできない。長さの異なる実プロンプトを混在させた現実的なバッチでの挙動は、未検証のまま残っている。

なお、TMLもQwen3-235B-A22B-Instruct-2507で1000回生成した、紛れもない一次データである。試行回数で見れば本実測の統計的検出力はTMLより大きく劣るため、「一次データとして測定している点ではこの報告より強い」という優位の主張はできない。

### dtypeとgreedy出力

#### 理論

dtypeを変えると、仮数部のビット数が変わり、計算の丸め誤差の大きさが変わる(第6章G4の説明の通り)。
greedyデコーディングは、ロジットが最大のトークンをそのまま選ぶ操作であり、順位が入れ替わるのは候補間のスコア差(margin)が丸め誤差より小さい場合に限られる。
dtypeがgreedy出力そのものを変えるかどうかを明言した一次資料は、本記事が参照した範囲では見当たらなかった。
通常はtop-1とtop-2のスコア差がdtypeによる丸め誤差より大きいため、この理論からは、dtypeを変えてもgreedy出力は変わらないという帰結が導かれる。

#### 実験設定

fp32・fp16・bf16の3dtypeで、Qwen2.5-0.5B-Instructのgreedy出力を3GPU(anakin・blade・dgxspark)で比較した(第6章G4と同一の実測)。

#### 結果

fp32とfp16のgreedy出力は、3GPUとも常に一致した(第6章参照)。
bf16にすると、anakin(RTX 4090)は元のfp32・fp16と同じ68トークンのまま変わらなかったが、blade(RTX PRO 6000 Blackwell)とdgxspark(GB10)は63トークンの別の列になった(この2台は互いに一致)。
greedyの範囲では、dtypeがgreedy出力を変えるかどうかはGPUモデル次第だった。
anakin・blade・dgxsparkのCUDA compute capabilityはそれぞれ8.9・12.0・12.1であり、3台とも異なる(第6章参照)。bladeとdgxsparkは「Blackwell」という製品世代名こそ共通するが、compute capabilityとしては別物であり、この2台が一致した理由が製品世代の一致によるものかどうかは、本実測だけでは特定できない。

#### 理論との照合

> 理論と部分的に食い違った結果: fp32とfp16についてはGPUモデルによらず一致し、理論の帰結通りだった。bf16では、anakinと、blade・dgxsparkとで出力が分かれた。

bf16は仮数部が7ビットしかなく、fp16(10ビット)より丸め誤差が大きい。「top-1とtop-2のスコア差は通常dtypeの丸め誤差より大きい」という理論の前提が、bf16のように丸め誤差自体が大きいdtypeでは成立しにくくなる方向とは整合する。
ただし、どのGPUモデルで出力が変わるかという切り分けの理由(アーキテクチャ世代・SM数・ホストCPUアーキテクチャに起因するビルド差のいずれが効いているか)は、本実測の範囲では特定できていない(第6章参照)。

### temperatureを上げると何が変わるか

ここでは2種類の生成を比較する。

- 同梱デフォルト: `repetition_penalty=1.1`・`top_p=0.8`・`top_k=20`を使った生成
- temperature単離: `top_p=1.0`・`top_k=0`・`repetition_penalty=1.0`に揃えた生成

#### 理論

temperatureは、生成時にロジットを割ってスコアの差を縮めたり広げたりする値であり、値が大きいほど低スコアのトークンが選ばれる確率が上がる。
同梱デフォルトの`top_p`/`top_k`/`repetition_penalty`は、temperatureとは別に候補を絞ったり偏らせたりする設定であり、これらを変えずに温度だけ上げても、実際には複数の設定が同時に変わっている。
この理論からは、temperatureを上げるほどサンプリングのランダム性が増し、同梱デフォルト設定と単離設定とで生成されるトークン列の長さや内容が異なること、また複数GPU間でも、temperatureを上げるほど早いトークン位置から出力が分岐することが導かれる。

#### 実験設定

temperature 0.7と1.2のそれぞれで、同梱デフォルトと単離の2設定を比較し、3GPU(anakin・blade・dgxspark)間の分岐位置も記録した。

#### 結果

| temperature | 同梱デフォルト | 単離       | 最初に食い違うトークン位置 |
| ----------- | -------------- | ---------- | -------------------------- |
| 0.7         | 87トークン     | 90トークン | 16トークン目               |
| 1.2         | 119トークン    | 65トークン | 0トークン目(最初から)      |

temperature 1.2では、最初のトークンから既に違う生成になった。

![F8b. 温度を上げるほど、GPUモデル間の分岐は早いトークンで起きる。](/images/reproducibility-of-randomness/f8_llm_divergence.png)
_F8b. anakinとdgxsparkが最初に食い違うトークン位置(温度別)。_

3GPU間の分岐位置も、temperatureを上げるほど早くなった。
greedyでは3GPUとも生成末尾までトークン列が一致した。
temperature 0.7では、同梱デフォルト設定で49トークン目、単離設定で85トークン目からanakinとdgxsparkの間で食い違いが始まった。
temperature 1.2では、同梱デフォルト設定で31トークン目、単離設定で4トークン目から食い違い始めた。

#### 理論との照合

「temperature 1.2で生成した」と言っても、同梱の`top_p`/`top_k`/`repetition_penalty`を中和しない限り、実際に測っているのはtemperatureだけではない。
3GPU間の分岐がtemperatureを上げるほど早くなった点は理論の帰結通りだった。
一方、単離設定とデフォルト設定で分岐の早さの順序が逆転している点は、理論だけでは説明がつかない組み合わせであり、測定結果としてそのまま書く。

### margin(脆さの定量)

#### 理論

greedyデコーディングでは、top-1のロジットがわずかでも大きければ、丸め誤差があっても選ばれるトークンは変わらない。
逆にtop-1とtop-2のロジット差(margin)が小さい位置では、GPU間の丸め誤差だけでも順位が入れ替わりうる。
この関係は、第6章・本章で確認した「GPU間でビットパターンがULP単位でずれる」という実測結果と、浮動小数点演算の丸め誤差の理論から導かれる。
この理論からは、ほとんどの生成ステップではtop-1とtop-2のmarginが十分大きく、GPU間の丸め誤差では順位が入れ替わらないこと、一方でmarginが極端に小さい一部のステップは、わずかな摂動でも順位が入れ替わりうる「脆い」位置になることが導かれる。
ただし、順位が実際に入れ替わるかどうかを左右するのは、top-1・top-2それぞれのロジットの絶対誤差とmarginの絶対誤差の大小関係であって、margin自体の値が持つ相対誤差(ULP)の大小ではない。marginは近い2つの値の差であるため、marginが小さいほど桁落ち(catastrophic cancellation)によってmargin自身の相対誤差(ULP)が拡大するのは、浮動小数点演算の性質として必然であり、これと「順位が入れ替わりやすいこと」は別の命題である。この区別は、後の照合で使う。

#### 実験設定

greedy生成の84ステップ全てで、top-1とtop-2のロジット差(margin)を記録した。個々のtop-1・top-2のロジット値そのものはmetaに記録しておらず、環境間で比較できるのはmarginの値のみである。
同じ84ステップについて、各ステップのmargin値のfloat32ビットパターンを、ULP単位で3GPU間(anakin・blade・dgxspark)を総当たりで比較した。すなわちここでのULP比較の対象はmarginの値であり、ロジットそのものではない。

#### 結果

最小値は0.0037、中央値は約2.07、最大値は13.26だった。
0.1未満のステップは84個中3個(約3.6%)だった。
3GPUともmarginの大きさの分布(ヒストグラムの形)は一致したが、84個の値そのものは1つも揃わなかった。

anakinとbladeで完全一致したのは84個中17個、差がある要素の中央値は18 ULP、最大49,152 ULPだった。
anakinとdgxsparkでは完全一致5個・中央値32 ULP・最大40,960 ULP、bladeとdgxsparkでは完全一致10個・中央値24 ULP・最大8,192 ULPだった。
「最下位ビットがずれる」という表現が示す1 ULPの桁とは、ここまで大きく違う。

![F8a. LLM greedy生成のlogit margin。](/images/reproducibility-of-randomness/f8_llm_margin.png)
_F8a. margin分布(左)とホスト間差分(右)。_

#### 理論との照合

全84要素でmarginの大きさとULP差の関係を確認すると、marginが小さい要素ほどULP差が大きくなる傾向があった。
ただし、これは「marginが小さい位置は脆い」という理論の裏付けにはならない。marginは近い2つのロジットの差であるため、margin自体が小さければ桁落ちでmarginの相対誤差(ULP)が大きくなるのは、浮動小数点演算として自明である。「marginが小さい要素ほどULP差が大きい」という観測は、この算術的な必然を確認したにすぎず、「順位が入れ替わりやすい」こととは別命題である。
最大のULP差(49,152 ULP)が出た要素が、3GPUいずれでもmarginが最小だった要素(anakin 0.0037326813 / blade 0.0037212372)と一致しているのも、同じ理由(margin自体が小さいほど、marginのULP誤差が大きくなる)で説明がつき、「脆さ」の追加の裏付けにはならない。順位の入れ替わりやすさを左右するのは、top-1・top-2それぞれのロジットの絶対誤差とmarginの絶対誤差の大小関係であり、本実測は個々のロジット値を記録していないため、これを直接測ってはいない。

一方、GPU間で実際に観測されたmarginの絶対差は、84ステップ・3ペア(anakin-blade/anakin-dgxspark/blade-dgxspark)を通して最大でも2.10×10^-5程度である(先述の49,152 ULPは、marginが0.0037程度まで小さい要素でULP間隔が極端に狭くなることで生じた数字であり、同じ要素を絶対量に直すと1.14×10^-5程度にとどまる)。84ステップ中もっとも小さいmargin(0.0037326813)でも、今回観測された最大の絶対差(2.10×10^-5)の約178倍の大きさがある。つまり、今回のGPU間の丸め誤差の実測スケールでは、84ステップのどこにおいても実際に順位が入れ替わる水準には届いていない。実際、この84ステップの生成トークン列は3GPUとも完全に一致している。

「0.1程度の摂動が入れば順位が入れ替わりうる位置が3.6%ある」という解釈は撤回する。0.1という閾値の根拠は本文になく、実測されたGPU間の摂動スケール(10^-5のオーダー)よりはるかに大きい。0.1未満のステップが84個中3個(約3.6%)あるという数値自体は事実だが、それが実際の脆弱性を意味するかどうかは、今回観測された摂動スケールでは確認されていない。
batchサイズの変化でも、margin自体のULP差という尺度でも、今回の実測範囲で実際の順位入れ替わりは確認されなかった。脆い位置が理論上存在しうること自体は、margin最小値(0.0037)が中央値(2.07)より3桁小さいという分布の形から読み取れるが、それ以上の定量的な主張は今回のデータでは裏付けられない。

---

## 8. 判定フローと納品文言テンプレート

> 手元の計算を上から順に当てはめるだけで、L0〜L4のどこに属するかが決まる。各レベルの末尾に、そのまま納品文書に貼れる一文を用意した。

ここまでの実測を判定フローにまとめた。
判定は「計算」だけに対して行うのではなく、「計算+設定」のペアに対して行う。たとえば`index_add_`は、`torch.use_deterministic_algorithms(True)`を指定しない設定ではL4(保証なし)だが、指定した設定ではビット一致する(第6章参照)。使う設定を固定せずに同じコードをこのフローに通すと、一度はYES、一度はNOという答えが両方出てしまう。
手元の計算と、それを動かす設定(deterministicアルゴリズムの指定・スレッド数の固定・SIMD/BLAS実装の指定等)を1組選び、上から順に問いに答えていく。

1. 同一プロセス・同一設定を2回実行しただけで結果が変わりうるか。YESならL4(保証なし)。NOなら次へ。
2. 今、結果を決めているのはIEEE754の基本演算(`+ - × ÷ √`・FMA)**単独**、または公開されたPRNG仕様の生ビット列(MT19937/PCG64/Philox)だけか。IEEE754が規格として保証するのは演算1つの結果であって、演算の並び(結合順序)ではない(第4章参照)。複数の演算を組み合わせている場合はここでNOに倒す。YESならL0(規格)。NOなら次へ。
3. 使っているAPIについて、ライブラリが版間のビット互換性を文書で明言しているか。YESならL1(明文の規約)。NO(非保証と明記、または文書なし)なら次へ。
4. 現在の設定で、SIMDディスパッチ・BLASスレッド数・libm実装・GPUモデル・deterministicアルゴリズムの指定の有無といった既知の変動源を、実測または文献で既に固定・検証したか。未検証なら「分からない」に倒す。検証済みで、それらを変えると結果が変わることを確認している場合はL3(実行時環境)。検証済みで、今回の実測範囲では変えても一致した場合はL2(事実上の実装)。

![F9. 保証レベルの判定フロー+納品文言テンプレート。](/images/reproducibility-of-randomness/f9_flowchart.png)
_F9. 判定フローと、レベルごとの文言の雛形(図は代表的な計算のみを示す簡略版であり、実際の判定では使用する設定も合わせて固定する必要がある)。_

各レベルの納品文言テンプレートは次の通りである。
角括弧の中は、自分の計算・バージョン・環境に置き換えて使う。

```text
L0 規格:
本計算はIEEE754の基本演算(演算1つ単位)、またはMT19937/PCG64/Philoxが
公開仕様として定める生ビット列のみに依存する。この生成アルゴリズム自体は
言語・OS・CPUを跨いでも結果は変わらない。
ただし、seedから内部状態への変換(例: NumPyのSeedSequence)や乱数の
消費順序は、規格そのものではなくライブラリ側の実装であり、版が変わると
挙動が変わった実績がある(例: NumPy 1.19.0でのSeedSequence spawn挙動の
変更、NumPy 1.21.0でのPCG64DXSM追加)。そのためライブラリのバージョンを
[X.Y.Z]に固定することで、アルゴリズム単体の規格としての安定性に加えて、
周辺実装を含めた全体のビット一致を保証する。

L1 明文の規約:
本計算は[ライブラリ名]が文書(例: NEP19)で明言するAPIのみを使う。
[ライブラリ名]==[X.Y.Z]に固定する限り、実行環境を問わずビット単位で
一致することを約束する。バージョンを上げた場合は無効。

L2 事実上の実装:
本計算の結果は、現在のバージョン[X.Y.Z]・現在の環境では一致することを
実測で確認した。ただしこれは文書化された保証ではなく、将来のバージョン
アップで予告なく変わりうるため、一致は約束しない。

L3 実行時環境:
本計算の一致は、コンテナイメージ[digest]・CPU世代[例: AVX-512非対応機に
統一]・スレッド数[N]・GPUモデル[型番]を固定する環境でのみ実測により
確認した。これらのいずれかが変われば一致しない。

L4 保証なし:
本処理(GPU上のatomic演算・ハッシュ順序等)は原理的に非決定であり、同一
入力・同一コードでも結果が一致することは保証されない。一致は許容誤差
(atol/rtol等)の範囲でのみ論じる。
```

これらは本プロジェクトの実測(4ホスト・7イメージ・30ラン、2026年10月14日。加えて2026年10月20日の追加実測2ラン)に基づく雛形である。
自分の計算について、まず上のフローで判定してから使う。

---

## 9. まとめ

> seedを固定するだけでは、ビット単位の再現性は保証されない。何が保証の出所で、何が壊れる条件かをレベルごとに言葉にできれば、納品文書に書いてよいことと書いてはいけないことの境界がはっきりする。

本記事の理論整理(各章の「理論」節)は、実測の後に行っている。最初の30ラン(4ホスト・7イメージ、2026年10月14日)の結果を見てから記事の執筆を始めており、各章の「理論」は結果を踏まえた事後の整理である。本記事中で唯一、実行前に予測を文書として固定してから走らせたのは、次段落で触れる追加実測(`PREREGISTERED_PREDICTION.md`、2026年10月20日)だけである。

今回の実測は、理論で説明できたことと、理論だけでは説明がつかなかったことの両方を含む。
AVX2とAVX-512の違いで`exp`/`log`が割れることは理論と整合したが、`np.sum`は割れなかった。
決め手はx86_64とaarch64というCPUアーキテクチャの違いではなく、AVX-512専用カーネルの有無だった。AVX-512専用カーネルがあるかどうかで、exp/logの計算がlibmを経由するかどうかが変わる。ISAが異なっていてもlibmの実装が同じなら一致し、libmが異なれば(glibcとmuslのように)割れる。この機構は、bladeのハードウェアを固定したままAVX-512のnumpyディスパッチ対象だけを無効化する追加実測で確認した。実行前に記録した4項目の予測のうち3項目(exp・sin・np.sumの挙動)は的中し、logの挙動だけ予測(aarch64のmuslと同じ値になる)が外れた。外れた結果から、muslのlogはISA(x86_64かaarch64か)によって異なるビットを返すという、実測前には分からなかった事実が新たに分かった。
LLMのbatch1とbatch8の比較は、「少なくとも今回の規模では一致した」と単純には言えない。batch8が同一プロンプトの複製で長さの異なる実プロンプトを混在させていないこと、3GPU×各条件1回というn=1の比較でありThinking Machines Labの1000回試行・8%の分岐率に対して統計的検出力が大きく劣ることから、現象を検出できる設計ではなかった、というのが正確な言い方である。
GPUは全体として非決定的ではなく、非決定性は`atomicAdd`を使う一部の処理に局在していた。

納品物に再現性を書くときは、「再現します」を一言で済ませず、どのレベルの保証なのかを明記するとよい。
第8章の判定フローと文言テンプレートは、そのための最小限の道具である。

本記事の実測ハーネスと生データは、そのままリポジトリに残す予定である。
<!-- TODO: 公開時にリポジトリURLを差し替える -->
実測ハーネスと生データ: [reprobench(公開準備中のリポジトリ)](https://example.com/TODO-repository-url)
同じ手順で、読者自身の環境でも再実行できる。

---

## 参考文献

> 本文が依拠した外部の出典を、テーマ別に種別を区別して列挙する。規格・公式文書・Issue/PR・査読論文・エンジニアリングブログを同格には扱わず、種別を明記した。URLと取得日はすべてresearch_notesの調査記録に基づく。

### 規格

- IEEE 754-2019 背景資料「Differences Among IEEE 754 Implementations」(grouper.ieee.org上で公開。ページ本文に「The following remarks are extracted from Sun's Numerical Computation Guide Appendix D, an expansion upon David Goldberg's 'What Every Computer Scientist Should Know About Floating-Point Arithmetic.'」「This material was not written by David Goldberg, but it appears here with his permission.」とあり、フッターに「Copyright 1997 Sun Microsystems, Inc.」とある。1997年のSun Microsystems社Numerical Computation Guide付録Dの転載で、David Goldberg自身の執筆ではなく許諾を得て掲載されている二次資料)。https://grouper.ieee.org/groups/msc/ANSI_IEEE-Std-754-2019/background/addendum.html (取得日: 2026-10-14)

### PRNGの仕様と方針

- Matsumoto, M. and Nishimura, T. (1998) "Mersenne Twister: A 623-Dimensionally Equidistributed Uniform Pseudo-Random Number Generator." ACM Transactions on Modeling and Computer Simulation, Vol.8, No.1(査読論文)。https://dl.acm.org/doi/10.1145/272991.272995 (取得日: 2026-10-14)。ACMは有料のため、著者による公開ページも挙げる。https://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/emt.html (取得日: 2026-10-15)
- Salmon, J. K., Moraes, M. A., Dror, R. O. and Shaw, D. E. (2011) "Parallel Random Numbers: As Easy as 1, 2, 3." Proceedings of SC11(ACM Best Paper Award、査読論文)。https://dl.acm.org/doi/10.1145/2063384.2063405 (取得日: 2026-10-14)。ACMは有料のため、著者による公開PDFも挙げる。https://www.thesalmons.org/john/random123/papers/random123sc11.pdf (取得日: 2026-10-15)
- NEP 19「Random Number Generator Policy」(NumPy公式文書、著者Robert Kern、Status: Final)。https://numpy.org/neps/nep-0019-rng-policy.html (取得日: 2026-10-14)
- NumPy公式ドキュメント「What's New or Different」(Generatorでは正規分布等の正確な乱数値を再現できないと明記)。https://numpy.org/doc/stable/reference/random/new-or-different.html (取得日: 2026-10-14)
- NumPy 1.22.0 リリースノート(公式文書。float32一様乱数の最下位ビットが常に0になるバグの修正を記載)。https://numpy.org/doc/stable/release/1.22.0-notes.html (取得日: 2026-10-14)

### SIMD・BLAS・libm

- Goldberg, D. (1991) "What Every Computer Scientist Should Know About Floating-Point Arithmetic." ACM Computing Surveys, Vol.23, No.1(査読論文。浮動小数点加算の非結合則の例)。https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html (取得日: 2026-10-14)
- NumPy Issue #23523「BUG: AVX-512 results differ from non-AVX-512 for sin, cos, exp, log, etc.」(GitHub Issue。メンテナmattip氏による4 ULP許容の発言)。https://github.com/numpy/numpy/issues/23523 (取得日: 2026-10-14)
- NumPy公式ドキュメント「CPU build options」(NPY_DISABLE_CPU_FEATURES / NPY_ENABLE_CPU_FEATURESの仕様)。https://numpy.org/doc/stable/reference/simd/build-options.html (取得日: 2026-10-14)

### GPUとCUDA

- NVIDIA cuBLAS Documentation「Results Reproducibility」(公式ドキュメント。ただし`cublas<t>symv()`/`cublas<t>hemv()`は`cublasSetAtomicsMode()`でatomics使用を選ぶと高速ルーチンに切り替わり、ビット再現性は保証されないという例外が明記されている。本記事の実測ではこれらのルーチンは使用していない)。https://docs.nvidia.com/cuda/cublas/ (取得日: 2026-10-14)
- NVIDIA cuDNN Backend Documentation「Odds and Ends」(公式ドキュメント。異なるアーキテクチャ間のビット再現性は非保証と明記)。https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/misc.html (取得日: 2026-10-14)
- PyTorch GitHub Issue #194676「CUDA RNG is not reproducible across GPU models above a launch-geometry threshold」。https://github.com/pytorch/pytorch/issues/194676 (取得日: 2026-10-14)
- PyTorch GitHub PR #196733「[docs] CUDA generator output depends on the device's SM count」(レビュー承認済み、2026年10月20日時点でも未マージ)。https://github.com/pytorch/pytorch/pull/196733 (取得日: 2026-10-14、再確認日: 2026-10-20)

### LLM推論

- He, H. et al.「Defeating Nondeterminism in LLM Inference」Thinking Machines Lab, Connectionism Blog, 2025年9月10日(査読を経ていないエンジニアリングブログ)。https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ (取得日: 2026-10-14)
- vLLM公式ドキュメント「SamplingParams」v0.11.0(temperature<1e-5をgreedy扱いにする仕様を記載)。https://docs.vllm.ai/en/v0.11.0/api/vllm/sampling_params.html (取得日: 2026-10-14)

### 再現性の用語

- ACM「Artifact Review and Badging」Version 1.1、2020年8月24日(ACM公式サイトが自動取得を拒否するため、Wayback Machineのアーカイブを参照した)。http://web.archive.org/web/20240102044447/https://www.acm.org/publications/policies/artifact-review-and-badging-current (取得日: 2026-10-14)

### OS・ハードウェア乱数

- Intel「Digital Random Number Generator (DRNG) Software Implementation Guide」(公式ドキュメント。RDRAND/RDSEEDの仕様とSP 800-90A/B/C準拠、再シード間隔)。https://www.intel.com/content/www/us/en/developer/articles/guide/intel-digital-random-number-generator-drng-software-implementation-guide.html (取得日: 2026-10-14)
- NIST SP 800-90A Rev.1「Recommendation for Random Number Generation Using Deterministic Random Bit Generators」(規格文書)。https://nvlpubs.nist.gov/nistpubs/specialpublications/nist.sp.800-90ar1.pdf (取得日: 2026-10-14)
- NIST SP 800-90B「Recommendation for the Entropy Sources Used for Random Bit Generation」(規格文書)。https://nvlpubs.nist.gov/nistpubs/SpecialPublications/nist.sp.800-90b.pdf (取得日: 2026-10-14)
- NIST SP 800-90C「Recommendation for Random Bit Generator (RBG) Constructions」(規格文書)。https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-90C.pdf (取得日: 2026-10-14)
- ARM Architecture Reference Manual、AArch64システムレジスタRNDR/RNDRRSの記述(第三者による転記サイトで確認。ARM公式サイトでは未確認)。https://arm.jonpalmisc.com/latest_sysreg/AArch64-rndrrs (取得日: 2026-10-14)
- man7.org「random(4)」マニュアルページ(Linux 5.6以降、/dev/randomが起動直後を除き非ブロック化したと明記)。https://man7.org/linux/man-pages/man4/random.4.html (取得日: 2026-10-14)
- kernel-internals.org「Random Number Generation」(コミュニティによる技術解説記事。kernel.org公式ページではない)。https://kernel-internals.org/crypto/rng/ (取得日: 2026-10-14)
- The Register「New Linux kernel has improved random-number generation」2022年3月21日(ニュース記事)。https://www.theregister.com/2022/03/21/new_linux_kernel_has_improved/ (取得日: 2026-10-14)
- PEP 456「Secure and interchangeable hash algorithm」(Python公式文書。SipHash24採用とハッシュフラッディング対策を記載)。https://peps.python.org/pep-0456/ (取得日: 2026-10-14)
- CVE-2012-1150(共通脆弱性識別子。ハッシュ衝突によるDoS)。https://app.opencve.io/cve/CVE-2012-1150 (取得日: 2026-10-14)
