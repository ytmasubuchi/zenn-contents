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
NumPy issue #23523では、2023年4月(NumPy 1.24.2)時点の報告として、メンテナのmattip氏が「使用する実装によっては、golden resultsとの差を最大4 ULPまで許容している」と述べ、`sin`/`cos`/`exp`/`log`でAVX-512と非AVX-512のコードパスが異なる結果を返すことが同issue内で確認されている。
ULPはunits in the last placeの略で、浮動小数点数の最下位ビット1つ分の差を指す。
このissueは本記事の執筆時点でもOpenのままだが、同issue内でメンテナのseberg氏は後に「we removed the biggest precision offenders again (not sure if in NumPy 1.25 or 1.26)」と述べており、その後のNumPyでは状況が変わっている。本記事が実測に使ったNumPy 2.2.6では、float64の`sin`/`cos`のSIMD経路は撤回済みで常にlibmを経由する(機構の詳細は第6章G2で扱う)。

:::details SIMDとCPUディスパッチとは

SIMD(Single Instruction, Multiple Data)とは、CPUが1回の命令で複数の数値をまとめて計算する仕組みである。
NumPyで配列同士の`+`や`np.exp`のような演算を書くと、内部ではこのSIMD命令が使われていることが多い。
AVX2やAVX-512は、Intel系CPUが持つSIMD命令セットの世代名であり、AVX-512の方が1回に処理できる数値の幅が広い。

同じCPUでも、実行時にどのSIMD命令セットを使うかを自動的に選ぶ処理を「CPUディスパッチ」と呼ぶ。
NumPyは起動時にCPUが対応する命令セットを検出し、AVX-512が使えるCPUではAVX-512向けのコードパスを選ぶ。
命令セットごとに別のコードパスが選ばれるため、同じ入力でも最終ビットが変わることがある。
つまりSIMDとCPUディスパッチは、Pythonのコードを1行も変えなくても、実行先のCPU次第で結果に影響しうる層にあたる。

:::

issue #23523の4 ULP許容と、sin/cos/exp/logの差が確認済みという記述からは、anakinとbladeで超越関数(`exp`/`log`/`sin`)が割れうることが読み取れる。
一方`np.sum`のような縮約については、同種の文書化された報告が見当たらず、理論から割れる・割れないのどちらかを導けない。
そのため`np.sum`は実測して初めて方向が分かる「真の未知数」として扱った。

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

### 理論との照合

`exp`/`log`/`sin`のうち、`exp`/`log`は割れた。
一方`sin`は、本章の実験単体ではanakinとbladeの間で割れなかった。
`exp`/`log`と`sin`とで割れ方が違う内訳(AVX-512専用カーネルの有無がlibmを経由するかどうかを決めており、sinはAVX-512の有無によらず常にlibmを経由する)は、第6章G2で別の対照実験とあわせて扱う。

> 理論で説明できなかった結果: `np.sum`はAVX2とAVX-512で割れなかった。理論は縮約についてどちらに転ぶか予測材料を持たなかったため、これは実測して初めて分かったことである。縮約(総和)は、この実測では超越関数よりも頑健だった。

:::details 縮約とpairwise summationとは

縮約(reduction)とは、複数の数値を1つにまとめる計算全般を指し、合計(sum)や平均はその代表例である。
浮動小数点の足し算には結合則が無いため、同じ数値の集合でも足す順序が変われば、合計の最下位ビットが変わることがある。
`np.sum`のような縮約は、要素数が多い配列では先頭から順に足すのではなく、途中で部分和に分けてから合体させる方式で計算されることが多い。

この方式はpairwise summation(部分和を作ってから足し合わせる方法)と呼ばれ、単純な逐次合計より誤差が小さくなる利点がある。
部分和をどの大きさで区切るかという閾値は、NumPy 1.9.0のリリースノートに「only along the fast axis and for groups of the values <= 8192 in length」と明記された歴史があり、ソース中には`PW_BLOCKSIZE 128`という定数もある。ただし現行の公式APIドキュメントが、この数値を安定した規約として保証しているわけではない。
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

本章の各節は「理論」「実験設定」「結果」「理論との照合」の4部構成を取る。
断っておくと、本記事の理論整理は実測結果を見た後に行っている。測定結果31ファイルのコミットから約2.5時間後に初稿を書いており、理論節の記述は事後的な整理である。本記事中で実測前に予測を記録した唯一の例は、本章G2で後述する追加実測(`PREREGISTERED_PREDICTION.md`)であり、そこでは明示的に「事前登録」と呼ぶ。それ以外の節の理論は、一次資料(ソースコード・公式ドキュメント・リリースノート)が何を言っているかの整理であって、実測前の予測ではない。

また、本章が尽くしているのは実測したプローブが対象とした軸に限られる。以下の軸は本章のマトリクスには含まれていない。

- 乱数消費順序に対するリファクタ耐性(実務で最も頻繁に問題になる軸だが、それを測るプローブ自体を作っていない)
- コンパイラ/ビルドフラグ(`-ffast-math`、最適化レベルなど)
- メモリアライメント/非連続ストライドの配列
- OS・カーネルバージョンの差(`envinfo.py`でカーネルのリリース文字列は収集しているが、軸として比較には使っていない)
- `MKL_CBWR`(MKLの再現性制御環境変数。`envinfo.py`では収集しているが本文では扱っていない)

### G1 時間・コンテナ

時間・コンテナという軸(同一プロセス内の2回実行・同一コンテナの再起動・同一イメージからの別コンテナ)は、具体的な実測条件と結果を第5章にまとめている。
結論だけ述べると、L0(規格レベル)の計算はこの軸で常に一致し、L4(PYTHONHASHSEEDなど)はこの軸で崩れた。これは第4章のレベル定義とそのまま整合する。

### G2 ソフトウェア

#### 理論

Python自体のバージョンがNumPyの乱数の値に影響する経路は、NumPyの乱数生成器の実装構造上見当たらない。
この点を明言した一次文書は確認できておらず、以下はアーキテクチャからの消去法による整理である。

NEP19は、「Bug fixes that change the stream SHALL NOT be made to RandomState(ストリームを変えるバグ修正はRandomStateに対して行わない)」と明記する。
一方Generatorについては、「breaking stream-compatibility ... will be allowed with caution ... on X.Y releases, never X.Y.Z」としている。
機能追加版(X.Yリリース)に限り、注意しながらストリーム互換性を破ることを許容する、という意味である。
NumPy 1.22.0のリリースノートには、float32型の一様乱数で最下位ビットが常に0になっていたバグの修正が記載されている。

第4章で確認した通り、IEEE754は個々の演算の丸めを規定するが、複数の演算をどの順序・どのグループで評価するかは規定しない。
BLAS実装(OpenBLAS・MKL)は、この規定されていない部分を独自に実装しており、libm(超越関数)とは別のライブラリである。
同じく第4章の通り、`exp`/`log`/`sin`のような超越関数はIEEE754で正しく丸めることが要求されておらず、glibcとmuslのような別々のlibm実装は異なる近似をしうる。

第2章では、`exp`/`log`が割れて`sin`が割れないという結果を、AVX2/AVX-512のSIMDディスパッチで説明した。
しかし「SIMDに依存するのでlibcには依存しない」という単純な二分法は、NumPy 2.2.6のソースと照らすと正確ではない。
`loops_exponent_log.dispatch.c.src`のDOUBLE_exp/DOUBLE_logは、(1) SVMLが使えるAVX512_SKX環境ではSVMLカーネル、(2) AVX512Fが使える環境ではNumPy内製のAVX512F_exp_DOUBLE/AVX512F_log_DOUBLE(libc非依存)、(3) どちらも使えなければnpy_exp/npy_logのスカラー経路、つまりプラットフォームのlibmを呼ぶ、という3段構えのディスパッチになっている。
一方`loops_trigonometric.dispatch.cpp`には「Disable SIMD code sin/cos f64 and revert to libm」というコメントがあり、float64の`sin`/`cos`はNumPy 2.2.6では常にlibmを経由する(`npy_math_internal.h.src`も「We assume the above are always available in their double versions」としてlibc依存を前提にしている)。
したがって、`exp`/`log`がlibc実装の違いに反応するかどうかは「AVX-512専用カーネルが使えるかどうか」で決まり、無条件に「libc非依存」とは言えない。`sin`/`cos`は常にlibc依存である。

#### 実験設定

Python 3.10・3.11・3.13(numpyバージョンは同一)を比較した。
なお「同じbaseイメージ」と呼んでいるものも、実際にはPythonのパッチ版がホスト間で完全には揃っていない。dgxsparkだけPython 3.11.15で、他3ホストは3.11.17である。Dockerfileが`python:3.11-slim`という浮動タグを使っておりdigest固定をしていないためで、「同じイメージと言っても同じとは限らない」という本記事のテーマそのものの実例でもある。
numpy 2.2.6と1.21.6(NEP19が境界を跨ぐ組み合わせ)を比較した。この比較はPython 3.10に固定しており(oldpy=Python 3.10・numpy 2.2.6、oldlibs=Python 3.10・numpy 1.21.6)、Python版自体の影響とnumpy版の影響を分離できる設計になっている。
OpenBLASとMKL(ともにスレッド数固定)を比較した。
glibcベースのイメージとmusl(alpine)ベースのイメージを比較した。この比較は当初blade(x86_64、AVX-512あり)とdgxspark(aarch64)でのみ実施しており、AVX2どまりの機体(anakin)でのlibc比較が欠けていた。

この欠けていた区間と、AVX-512の有無が本当に決め手かどうかを確かめるため、blade1台の中でmuslイメージを固定したままAVX-512のnumpyディスパッチ対象だけを無効化する決定実験と、anakin(AVX2どまり)でのglibc/musl比較を追加で実施した。これらは実行前に結果を予測し、`PREREGISTERED_PREDICTION.md`としてコミット63d9e38で記録してから実行している。本記事中で実測前に予測を登録したのはこの実験だけであり、他の節の「理論」は一次資料の事後整理である。
`NPY_DISABLE_CPU_FEATURES`で無効化したのは、NumPy 2.2.6の`__cpu_dispatch__`が持つAVX512系8項目(AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL)であり、ハードウェアの検出結果自体(AVX512VL/BW/DQなど)は変えていない。

#### 結果

Python 3.10・3.11・3.13(numpy版は同一)を比較すると、S1は全て一致した。
Python自体のバージョン差は、NumPyの乱数の値を変える理由にならなかった。

numpy 2.2.6と1.21.6を比較すると、S1(PRNGの生ビット列)では`S1_generator_random_float32`だけが不一致になった。
stdlibのMT19937、legacyのRandomState、Generator経由のPCG64・Philoxは全て一致した。
これはNumPy 1.22.0のリリースノートに記載された「float32型の一様乱数で最下位ビットが常に0になっていたバグ修正」を、ピンポイントで捉えている。

ただし、numpy版の違いはS1だけにとどまらない。同じoldpy(Python 3.10・numpy 2.2.6)とoldlibs(Python 3.10・numpy 1.21.6)の比較で、`S3_matmul_float32`/`float64`、`S4_exp_float64`/`log_float64`/`log_loguniform_float64`、`S5_kmeans_centers`、`S5_logreg_coef`も不一致になった(`S4_sin_float64`・`S2_sum_float64`・`S5_random_forest_pred`は一致)。
原因は、numpy 1.21.6が旧いOpenBLASバンドル方式を内包していること(2.2.6はscipy-openblas64パッケージ方式)で、BLASの実装自体と超越関数のカーネル実装自体が版ごと入れ替わる。
つまり本章が「numpy版」「BLAS実装」「glibc/musl」を独立した3つの軸として並べている構成は、実際には独立していない。numpy版を変えると、BLASの実装もlibmへのフォールバック経路の実装自体も同時に動く。

OpenBLASとMKL(ともにスレッド数固定)を比較すると、`S3_matmul_float32`/`float64`はどちらも不一致になった。
一方`S4_exp_float64`と`S2_sum_float64`は一致した。
BLASの実装差は行列積(線形代数)を変えるが、libm(超越関数)には影響しなかった。
なお、mklイメージの`np.__config__`には"mkl"という文字列が一度も現れず(`Build Dependencies.blas.name`は"blas"、"openblas configuration"は"unknown")、MKLが実際にリンクされていることをこの実測単体では確認できていない。

:::details BLASとGEMM(OpenBLAS・MKL)とは

BLAS(Basic Linear Algebra Subprograms)とは、行列やベクトルの計算を高速に行うための標準的なインタフェースの仕様である。
NumPyで`A @ B`のように行列積を書いたとき、実際にその計算をしているのはNumPy自身ではなく、BLASという別のライブラリであることが多い。
GEMM(General Matrix Multiply)は、このBLASが定める代表的な演算の1つで、行列同士の積を指す。

OpenBLASとMKL(Math Kernel Library)は、どちらもBLASの実装の一種であり、オープンソースかIntel製かという違いがある。
どちらも同じBLASの仕様を満たすが、内部でどう計算を分割し、どの順序で足し合わせるかという実装の詳細は実装ごとに異なる。
そのため、同じ行列積でも使うBLAS実装が変われば、結果のビット列が変わることがある。
なお、本記事のS3プローブは2048×2048の単一サイズのみで計測しており、サイズを変えた場合の挙動は未検証である。

:::

glibcベースのイメージとmusl(alpine)ベースのイメージを比較すると、blade(x86_64、AVX-512あり)では`S4_sin_float64`のみが不一致になり、`S4_exp_float64`・`S4_log_float64`・`S2_sum_float64`は一致した。
ところが同じ比較をdgxspark(aarch64、AVX-512なし)で行うと、`S4_exp_float64`・`S4_log_float64`も`S4_sin_float64`と同様に不一致になった。

| 条件 | `S4_exp_float64` | `S4_log_float64` | `S4_sin_float64` |
| --- | --- | --- | --- |
| blade/base(x86_64、AVX-512あり、glibc) | 5119b269 | a6b6b7fa | ddf398b7 |
| blade/alpine(x86_64、AVX-512あり、musl) | 5119b269 | a6b6b7fa | 601eb0fa |
| dgxspark/base(aarch64、glibc) | cba8bfa7 | 09cef322 | ddf398b7 |
| dgxspark/alpine(aarch64、musl) | ab300a28 | 84b2ce10 | 601eb0fa |

bladeだけが`exp`/`log`でlibc実装差に反応しなかった。これはAVX-512の有無で説明できる(詳しくは本節末尾)。
`S4_sin_small_float64`(入力レンジを変えた`sin`)も同じ傾向で、glibc系の値とmusl系の値に分かれて割れた。`S4_log_loguniform_float64`(入力分布を変えた`log`)も、AVX-512の有無に沿って分かれるパターンが同じく確認できた。振幅やサンプリング分布を変えても、`S4_exp_float64`/`S4_log_float64`/`S4_sin_float64`で見たのと同じ方向の結果だった。

なお、glibc/muslの比較は`S4`だけでなく`S3_matmul_float64`も動かす(blade 76042a4d→8875959c、dgxspark 0cb8db34→86d16106、anakin f86a1f40→c924d2fa)。ベースイメージを変えるとBLASの実装も同時に変わるためであり、libm単独の軸ではない。

#### 決定実験(事前登録)

ここまでの結果は、「bladeだけexp/logが割れない」という観測事実と、「NumPyのソースはAVX-512専用カーネルの存在を示している」という理論を、別々の対照実験から集めたものにすぎない。両者が同じ機構を指しているかどうかを確かめるため、1台のハードウェア(blade)の中でAVX-512の有無だけを切り替える決定実験を行い、実行前に予測を記録した。

予測は次の4点だった。(1) `S4_exp_float64`は`5119b269`から`ab300a28`(musl libmの値)に変化する。(2) `S4_log_float64`は`a6b6b7fa`から`84b2ce10`に変化する。(3) `S4_sin_float64`は`601eb0fa`のまま変化しない。(4) `S2_sum_float64`は`f9458b5c`のまま変化しない。あわせて、anakin(AVX2どまり)でalpineイメージを計測する実験も同時に予測登録した。

実測結果は次の通りである。

| 条件 | `S4_exp_float64` | `S4_log_float64` | `S4_sin_float64` | `S2_sum_float64` |
| --- | --- | --- | --- | --- |
| blade/alpine(AVX-512あり、musl) | 5119b269 | a6b6b7fa | 601eb0fa | f9458b5c |
| blade/alpine、AVX-512ディスパッチ無効化(musl) | ab300a28 | 28ff757a | 601eb0fa | f9458b5c |
| anakin/alpine(AVX2どまり、musl、新規) | ab300a28 | 28ff757a | 601eb0fa | f9458b5c |

4項目の予測のうち3項目が的中した。(1)の`exp`は的中、(3)の`sin`は的中、(4)の`np.sum`は的中。(2)の`log`は外れた——変化すること自体は当たったが、予測した`84b2ce10`ではなく`28ff757a`になった。

#### 理論との照合

Python版差・NEP19の実証・BLAS実装差については、理論の整理と結果は一致する。legacyは動かず、Generatorだけが動いた、という条文通りの結果になった。BLAS実装差では行列積が割れ、libmは影響を受けなかった。

glibc/muslの比較は、第2章時点では「bladeという1台のAVX-512搭載機で、exp/logが一致してsinだけ割れた」という結果だけを見ていた。本章でdgxsparkを加えると、同じglibc/musl比較でexp/logも割れる。これは「exp/logはlibc非依存」という一般化がそのままでは誤りであることを、プロジェクト自身のデータが示している。

決定実験は、この矛盾を解消する機構を裏付けた。bladeという1台の中で、同一のmuslイメージ・同一のnumpy 2.2.6・同一のPython 3.11.17のまま、AVX-512のnumpyディスパッチ対象だけを無効化すると、`exp`は`ab300a28`に、`log`は`28ff757a`に変化し、無効化後の`exp`はAVX2しか持たないanakinのmusl値と完全一致した。`sin`は一切反応しなかった。

> `exp`/`log`は、AVX-512Fがあればlibc非依存(NumPy内製カーネル)、無ければlibc依存(libmへのフォールバック)。`sin`/`cos`(float64)はAVX-512の有無によらず常にlibc依存である。「SIMD依存・libc非依存」という二分法ではなく、「AVX-512専用カーネルの有無が、libmを経由するかどうかを決める」という1つの機構として理解する必要がある。この機構は、同一ハードウェア上でAVX-512ディスパッチのみを無効化した決定実験で確認した。

異なるISA(x86_64のAVX2とaarch64)のglibc環境同士が`exp`/`log`でビット一致したのも、この機構で説明できる。どちらもAVX-512専用カーネルを持たずlibmにフォールバックするため、あとは双方のglibc(ともにバージョン2.41)が`exp`/`log`で同じ結果を返すかどうかだけで決まる。今回はそれが一致した、という観測である。

一方、予測が外れた`log`は新しい事実を明らかにした。muslの`log`はISAによって異なるビットを返す(x86_64のmusl: `28ff757a`、aarch64のmusl: `84b2ce10`)。musl版の`exp`と`sin`は両ISAで同一(`ab300a28`/`601eb0fa`)であり、glibc版の`log`も両ISAで同一(`09cef322`)である。つまり「同じlibc実装ならISAを跨いでも同じ」という読みは、muslの`log`では成立しない。**この原因は未検証である。** FMA(積和演算)の有無が候補として考えられる(muslのx86-64ベースラインはFMA3を含まず、aarch64ベースラインはFMAを含む)が、本プロジェクトでは検証していない。

### G3 ハードウェア

#### 理論

第4章で確認した通り、IEEE754は演算の順序までは規定しない。
BLAS実装は、スレッド数に応じて各スレッドが担当するサブレンジの割り当てが変わり、部分和の結合順が変わりうる。
KMeansの重心計算も、各クラスタに属する点の合計という縮約を内部で使っている。

一般に、異なるCPUアーキテクチャ(x86_64とaarch64)は、使えるSIMD命令セットやコンパイラの最適化が異なり、結果が変わりうると広く理解されている。
本記事が参照した一次資料の中に、x86_64とaarch64の違いそのものを論じたものは無く、以下はこの一般的な理解に基づく整理である。
ただし第2章・本章G2で見た通り、`exp`/`log`の割れ方を実際に決めているのはCPUアーキテクチャの種別そのものではなく、AVX-512専用カーネルが使えるかどうかである。

#### 実験設定

スレッド数を1・8・最大(ホストの論理コア数)に変えて`S3_matmul_float32`/`float64`と`S2_sum_float64`を比較した。S3は2048×2048の単一サイズ、S2は1,000万要素・連続配置1次元配列のみで計測しており、サイズや配置を変えた場合の挙動は未検証である。
KMeans(`S5_kmeans_centers`)は、`effective_thread_count`を1・8で固定した場合と、固定しない既定設定を比較した。
別の1回の計測では、8CPU構成のコンテナでスレッド数を1・2・4・8・既定の各設定にし、30回ずつフィットを繰り返した。
第2章のanakin(x86_64、AVX2)とblade(x86_64、AVX-512)の対照実験に加え、x86_64(anakin)とaarch64(dgxspark)を比較した。
PyTorch(CPU)のMLP順伝播(`S7_mlp_forward`)・学習10ステップ(`S7_mlp_train10`)を、スレッド数1・8・既定で比較した。
LightGBM(`S6_lgbm_pred`/`S6_lgbm_tree_structure`)を、スレッド数1/8/最大・MKL/OpenBLAS・Python 3.11/3.13・oldlibsを跨いで比較した。

#### 結果

`S3_matmul_float32`はanakinで3通りに割れた(threads1/threads8/threadsmaxがそれぞれ別の値)。
`S3_matmul_float64`は、各ホスト内ではスレッド数を変えても不変だった(threads1/threads8/threadsmaxが同一digest)が、ホスト間では4通りに分かれた(anakin f86a1f40、blade 76042a4d、dgxspark 0cb8db34、local baba98ed)。
一方`np.sum`(`S2_sum_float64`)は、4ホスト×3スレッド数設定の全12セルが真に同一digest(`f9458b5c`)で一致した。「ホスト内では安定、ホスト間では別」というS3_matmul_float64の結果と、「全セルが真に一致」というS2_sum_float64の結果は別の現象であり、混同しないようにしたい。

![F10. BLASの行列積(スレッド数が変わると各スレッドの担当サブレンジが変わる)。](/images/reproducibility-of-randomness/f10_blas_tiles.png)
_F10. スレッド数ごとの担当範囲割り当ての違い(原理図)。_

KMeansは、`effective_thread_count`が1のとき4ホスト全て(anakin/blade/dgxspark/local)で同じ重心に収束した。
`effective_thread_count`が8になった6本のラン(anakin/threads8・blade/threads8・dgxspark/threads8・local/threads8・local/hashseed0・local/hashseedfree。localは8コア構成のため、スレッド数を明示的に固定していないhashseed0・hashseedfreeでも`effective_thread_count`が8になる)では、4通りの異なる重心に分かれた。
30回×6条件の繰り返し実行(別の1回の計測、8CPU構成のコンテナ)では、スレッド数を1・2・4に固定すると30回とも同じ重心になり、8に固定すると13回、既定設定では19回、異なる重心になった。
この回数は実行のたびに多少変動する値であり、固定した定数としては読まないでほしい。
この生データは`results/`には無く、プロジェクトの判断記録(`reprobench/probes/s5_sklearn.py`のdocstring)にのみ残っている。

一方で`n_iter_`(反復回数)は180回のフィット全てで2、重心の分散(inertia)の広がりは最大でも3.9×10^-16程度に収まっていた。

x86_64(anakin)とaarch64(dgxspark)を比較すると、`S2`/`S4`は全て一致した。

PyTorch(CPU)のMLP順伝播・学習10ステップは、スレッド数1・8・既定でそれぞれ別のdigestになった(最大絶対差は`S7_mlp_forward`で約1.79×10^-7、`S7_mlp_train10`で約7.6×10^-8)。ホスト間でも別の値になった。BLAS・KMeans・PyTorch(CPU)が同じ構造(非結合則の演算をスレッド分割で計算する)で揃って割れており、「CPU実行は安全で、GPUだけが危ない」という理解はこの3つを並べると成立しない。

一方LightGBM(`S6_lgbm_pred`/`S6_lgbm_tree_structure`)は、スレッド数1/8/最大・MKL/OpenBLAS・Python 3.11/3.13・oldlibsを跨いだ25ラベル全てでビット一致した。`reprobench/probes/s6_lightgbm.py`には、決定的モードを意図的に設定していない(`deterministic: True`を指定しない)ことが明記されている。np.sumと並んで、縮約を使う処理が必ず割れるわけではないことを示す例である。勾配ブースティングはマルチスレッドで不安定という実務の通念とは逆の結果になった。

#### 理論との照合

スレッド数の影響は理論の整理と一致し、float32のBLASは割れた。float64のBLASはホスト内では安定し、ホスト間では割れた。np.sumはスレッド数・ホストのいずれでも割れなかった。
KMeansの不安定も理論の整理と一致した。つまり不安定の正体は、別の局所最適を見つけているのではなく、縮約(合計)の丸め方の違いである。

PyTorch(CPU)とLightGBMの結果は対照的だった。PyTorchのMLPはBLAS・KMeansと同じ理由(非結合則×スレッド分割)で割れたのに対し、LightGBMは同じ軸(スレッド数・BLAS実装・Python版)を跨いでも割れなかった。

第2章の実測を振り返ると、`S4_exp_float64`/`S4_log_float64`は同じx86_64同士(anakinのAVX2とbladeのAVX-512)で不一致になった。
一方、本章のx86_64とaarch64の比較では全て一致した。

> 理論で説明できなかった結果: アーキテクチャ(x86_64かaarch64か)は決め手ではなかった。決め手はAVX-512専用カーネルの有無である。本節の理論は「CPUアーキテクチャの違いが結果を変える」という一般的な理解に基づいていたが、実際に割れたのは同じx86_64同士のAVX2とAVX-512の間であり、x86_64とaarch64という異なるISA同士は(どちらもAVX-512を持たない組み合わせでは)一致した。機構はG2で確認した通り、AVX-512専用カーネルの有無がlibmを経由するかどうかを決めるというものである。

![F6. SIMD幅が変わると縮約の足す順序が変わる。](/images/reproducibility-of-randomness/f6_simd.png)
_F6. 8並列と16並列でレーンの組み方が変わる例(原理図)。_

### G4 アクセラレータ

#### 理論

NVIDIAのcuBLAS公式ドキュメントは、「同一ツールキット版のcuBLAS APIルーチンは、同一アーキテクチャ・同一SM数のGPU上で実行する限り、毎回ビット単位で同じ結果を生成する」と明記している。
この保証は複数のCUDAストリームが同時に有効な場合には成り立たない(また`cublasSetAtomicsMode()`でatomics使用を有効にした`symv`/`hemv`も原文では対象外とされているが、本実測ではこれらのルーチンも`cublasSetAtomicsMode()`も使っていない)。
cuDNN公式ドキュメントは「異なるアーキテクチャ間では、いかなるcuDNNルーチンもビット単位の再現性を保証しない」とVolta/Turingを例に明記している。
ただしこれはcuDNNのルーチンについての記述である。後述する`index_add_`はcuDNNのルーチンではなく、PyTorchのATen/CUDAカーネル(`atomicAdd`ベース)であり、この保証・非保証は直接は適用されない。

atomicAdd自体は1回ずつの書き込みを保証するが、どのスレッドがどの順番で書き込むかまでは保証しない。
浮動小数点の足し算には結合則が無いため、足す順序が実行のたびに変わると、合計の最下位ビットも実行のたびに変わりうる。

:::details atomic演算(atomicAdd)とは

atomic演算とは、複数の処理が同時に同じメモリ上の値を書き換えようとしても、1回ずつ確実に反映されることを保証する演算である。
GPUでは数千のスレッドが同時に動くため、複数のスレッドが同じ場所に値を足し込みたいとき、`atomicAdd`のようなatomic演算を使う。
atomicAdd自体は1回ずつの書き込みを保証するが、どのスレッドがどの順番で書き込むかまでは保証しない。

浮動小数点の足し算には結合則が無いため、足す順序が実行のたびに変わると、合計の最下位ビットも実行のたびに変わりうる。
これが、同じGPU・同じコードでも`index_add_`のような処理の結果が毎回一致しない理由である。
PyTorch公式の定義では、`torch.use_deterministic_algorithms(True)`は「決定的な実装があればそれに差し替え、無ければRuntimeErrorを送出する」という設定であり、一般に「足し込む順序を固定する」という意味ではない。`index_add_`についてはこの設定で決定的な実装に切り替わり、結果として足し込む順序が固定される。

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

3GPUの比較に入る前に、条件に残る交絡を明記しておく。3台のGPUのcompute capabilityは、anakin(RTX 4090)が8.9、blade(RTX PRO 6000 Blackwell)が12.0、dgxspark(GB10)が12.1であり、3台とも別々である。SM数も128/188/48とそれぞれ異なる。
また、torch/CUDA/cuDNNのバージョン文字列(`torch==2.14.1+cu130`・CUDA 13.0・cuDNN 92400)は3台で一致しているが、dgxsparkはaarch64ホストであるため、同一の文字列でも実際には別ビルド(別バイナリ)である。GPUドライバのパッチ版も3台で異なる(anakin 580.178.04 / blade 580.173.02 / dgxspark 580.126.09)。
以降の比較結果を「GPUアーキテクチャの差」だけに帰属させることはできない。

#### 実験設定

当初の実測設計は、CPU実行とGPU実行の結果を比較して、GPUの非決定性を示すことを狙っていた。

anakin(RTX 4090)で、Linear層+ReLUのMLPを10ステップ学習させる処理と、重複したインデックスへの`index_add_`を、それぞれ5回ずつ実行した。
`index_add_`は`torch.use_deterministic_algorithms(True)`の有無で比較した。
決定的モードの`index_add_`は、3GPU(RTX 4090・RTX PRO 6000・GB10)で比較した。

TF32は無効化と有効化の両方で、3GPUを比較した。
greedy復号のトークン列は、fp32・fp16・bf16の3dtypeで、3GPUを比較した。
`torch.randn`を同じseedで4サイズ(1,024 / 65,536 / 4,194,304 / 67,108,864)生成し、サイズごとにダイジェストを取った(`S8_curand_randn_sizes`、3GPU比較)。

#### 結果

anakin(RTX 4090)で確かめると、Linear層+ReLUのMLPを10ステップ学習させる処理は、5回ともビット一致した。
CPU実行とGPU実行を比較するという当初の設計のままでは、非決定性は現れなかった。

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

bf16にすると、anakinはfp32・fp16と同じ68トークンのまま変わらなかった。
一方blade・dgxsparkは、bf16にすると63トークンの別の列になった(2台は互いに一致)。blade・dgxsparkはどちらも製品世代としては「Blackwell」と呼ばれるが、前述の通りcompute capabilityは12.0と12.1で異なる。

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

#### 理論との照合

CPU実行とGPU実行の比較という当初の実験設計では、GPUの非決定性を示すことができなかった。非決定性は`atomicAdd`を使う一部の演算に局在していた。

> 理論で説明できなかった結果: 「GPUは非決定的」という大雑把な理解は誤りだった。cuBLASのGEMM(行列積)は、単一ストリーム・同一アーキテクチャ・同一SM数という条件下では決定的である。非決定性は`atomicAdd`を使う一部の演算に局在している。

`index_add_`が割れ、決定的モードで直ることは理論の整理と一致した。
今回の実測は単一ストリームで行っており、cuBLASの保証の前提と一致する。

決定的モードでもdgxsparkだけ別の値になった点、TF32無効化でもdgxsparkだけ別の値になった点について、当初はcuDNN文書の「異なるアーキテクチャ間では再現性を保証しない」という枠組みで説明していたが、この説明は撤回する。理由は2つある。
第一に、`index_add_`はcuDNNのルーチンではなくPyTorchのATen/CUDAカーネル(`atomicAdd`ベース)であり、cuDNN文書の記述はそのままでは適用できない。
第二に、「anakinとbladeが一致し、dgxsparkだけ別」というグルーピングを「異なるアーキテクチャ」で説明しようとしても、3台のcompute capabilityはanakin 8.9・blade 12.0・dgxspark 12.1であり、3台とも互いに異なる。cuDNNの文書が述べているのは「異なるアーキテクチャ間では保証しない」であって「異なるアーキテクチャなら必ず結果が変わる」ではなく、非保証を「同じ結果になる・ならない」の予測モデルとして使うことはできない。3台とも互いに異なるアーキテクチャである以上、anakinとbladeが一致した理由を「アーキテクチャ」だけでは説明できない。SM数の近さ(anakin 128・blade 188)や、前述のdgxsparkがaarch64ホストであることによるビルド差・ドライバのパッチ版の違いが候補として考えられるが、どれが効いているかは本実測だけでは切り分けられない。

> 精度を落とすと、GPU世代が効き始める。fp32とfp16では3GPUとも違いが見えなかったが、bf16にすると、anakinと、blade・dgxsparkの間で出力そのものが変わった。

ただしblade・dgxsparkは製品の世代名としては共に「Blackwell」と呼ばれるものの、compute capabilityは12.0と12.1であり、同一アーキテクチャとは言えない。bf16での挙動を分けている要因が世代レベルの機能差なのか、個別のcompute capability差なのか、あるいは前述のビルド差・ドライバ差なのかは、本実測だけでは確定できない。

`S8_curand_randn_sizes`の閾値(65,536と4,194,304の間)が実在したことは、PyTorch issue #194676の記述と整合する。
ただし具体的な閾値の位置や、3GPUが3通りの異なる値に分かれる点は、公式ドキュメントには未記載(PR #196733は未マージ)であり、一次資料からの厳密な予測ではなく、issueの記述を実測で確認したものにとどまる。

---

## 7. LLMとtemperature

> temperature=0は「決定的」を意味する公式なモードではない。batchサイズを1から8に変えても出力は変わらなかったが、同梱のデフォルト設定を外すと、生成はもっと早いトークンで分岐した。

Qwen2.5-0.5B-Instructを使い、transformersで生成した。

### temperature=0は存在しない

#### 理論

temperatureは、生成時にモデルの出力するロジットを割って候補のスコアを広げたり狭めたりする値である。
数学的には、temperatureで割る処理そのものが、0で割るという未定義の操作になる。
vLLM公式ドキュメントは、内部で`temperature<1e-5`をgreedy扱いにする仕様を明記しており、同じ「temperature=0」という入力でも、実装によって扱いが異なりうることを示している。

#### 仮説

温度0での割り算が未定義になるという理論から、transformersが`temperature=0`をどう扱うかは理論だけでは決まらないと考えた。
vLLMのように内部でgreedy扱いに読み替える可能性と、エラーとして拒否する可能性の両方がありうるため、実測で確かめることにした。

#### 実験設定

transformersで、Qwen2.5-0.5B-Instructに対して`temperature=0`を指定して生成を呼び出した。

#### 結果

transformersに`temperature=0`を渡すと、`ValueError`が出た。
メッセージは次の通りである。

> `temperature` (=0.0) has to be a strictly positive float, otherwise your next token scores will be invalid. If you're looking for greedy decoding strategies, set `do_sample=False`.

#### 仮説との照らし合わせ

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

Thinking Machines Labは2025年9月のブログで、batch-invariance(batchサイズによってGPUカーネルの縮約順が変わり出力が変わる現象)を報告している。
このブログは査読論文ではなくエンジニアリングブログであり、Qwen3-235Bで1000回のgreedy生成を行い、出力の種類が80通りから1通りに減ったと報告している。

#### 仮説

この報告から、batchサイズ1と8でgreedy出力が分岐すると予想した。

#### 実験設定

Qwen2.5-0.5B-Instructを使い、3GPU(anakin・blade・dgxspark)で、batchサイズ1と8のgreedy出力を比較した。

#### 結果

batchサイズ1と8のgreedy出力は3GPUとも一致した。

#### 仮説との照らし合わせ

> 外れた仮説: batchサイズ1と8のgreedy出力は3GPUとも一致した。Thinking Machines Labが2025年9月のブログで報告したbatch-invariance(batchサイズによってGPUカーネルの縮約順が変わり出力が変わる現象)は、少なくとも今回の規模(0.5Bモデル・batch8・3GPU)では再現しなかった。

我々の実測ははるかに小さいモデル・小さいbatchだが、一次データとして測定している点ではこの報告より強い。
batchサイズだけでは分岐しない場合もある、という事実として書いておく。

### dtypeとgreedy出力

#### 理論

dtypeを変えると、仮数部のビット数が変わり、計算の丸め誤差の大きさが変わる(第6章G4の説明の通り)。
greedyデコーディングは、ロジットが最大のトークンをそのまま選ぶ操作であり、順位が入れ替わるのは候補間のスコア差(margin)が丸め誤差より小さい場合に限られる。
dtypeがgreedy出力そのものを変えるかどうかを明言した一次資料は、本記事が参照した範囲では見当たらなかった。

#### 仮説

通常はtop-1とtop-2のスコア差がdtypeによる丸め誤差より大きいと想定し、dtypeを変えてもgreedy出力は変わらないと予測した。

#### 実験設定

fp32・fp16・bf16の3dtypeで、Qwen2.5-0.5B-Instructのgreedy出力を3GPU(anakin・blade・dgxspark)で比較した(第6章G4と同一の実測)。

#### 結果

fp32とfp16のgreedy出力は、3GPUとも常に一致した(第6章参照)。
bf16はAdaでは変わらず、Blackwell2台では63トークンの別の列になった。
greedyの範囲では、dtypeがgreedy出力を変えるかどうかはGPU世代次第だった。

#### 仮説との照らし合わせ

> 外れた仮説(部分的中): 当初は「dtypeを変えてもgreedy出力は変わらない」と予測していた。fp32とfp16については3GPUとも予測通りだったが、bf16はAda(anakin)でのみ予測通りで、Blackwell(blade・dgxspark)では出力そのものが変わった。

### temperatureを上げると何が変わるか

ここでは2種類の生成を比較する。

- 同梱デフォルト: `repetition_penalty=1.1`・`top_p=0.8`・`top_k=20`を使った生成
- temperature単離: `top_p=1.0`・`top_k=0`・`repetition_penalty=1.0`に揃えた生成

#### 理論

temperatureは、生成時にロジットを割ってスコアの差を縮めたり広げたりする値であり、値が大きいほど低スコアのトークンが選ばれる確率が上がる。
同梱デフォルトの`top_p`/`top_k`/`repetition_penalty`は、temperatureとは別に候補を絞ったり偏らせたりする設定であり、これらを変えずに温度だけ上げても、実際には複数の設定が同時に変わっている。

#### 仮説

temperatureを上げるほど、サンプリングのランダム性が増し、同梱デフォルト設定と単離設定とで生成されるトークン列の長さや内容が異なると予想した。
また複数GPU間でも、temperatureを上げるほど、早いトークン位置から出力が分岐すると予想した。

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

#### 仮説との照らし合わせ

「temperature 1.2で生成した」と言っても、同梱の`top_p`/`top_k`/`repetition_penalty`を中和しない限り、実際に測っているのはtemperatureだけではない。
3GPU間の分岐がtemperatureを上げるほど早くなった点は仮説通りだった。
一方、単離設定とデフォルト設定で分岐の早さの順序が逆転している点は、予想していなかった組み合わせであり、測定結果としてそのまま書く。

### margin(脆さの定量)

#### 理論

greedyデコーディングでは、top-1のロジットがわずかでも大きければ、丸め誤差があっても選ばれるトークンは変わらない。
逆にtop-1とtop-2のロジット差(margin)が小さい位置では、GPU間のULP単位の丸め誤差だけでも順位が入れ替わりうる。
この関係は、第6章・本章で確認した「GPU間でビットパターンがULP単位でずれる」という実測結果と、浮動小数点演算の丸め誤差の理論から導かれる。

#### 仮説

ほとんどの生成ステップではtop-1とtop-2のmarginが十分大きく、GPU間のULP差では順位が入れ替わらないと予想した。
一方で、marginが極端に小さい一部のステップは、わずかな摂動でも順位が入れ替わりうる「脆い」位置になると予想した。

#### 実験設定

greedy生成の84ステップ全てで、top-1とtop-2のロジット差(margin)を記録した。
同じ84ステップについて、float32のビットパターンをULP単位で3GPU間(anakin・blade・dgxspark)を総当たりで比較した。

#### 結果

最小値は0.0037、中央値は約2.07、最大値は13.26だった。
0.1未満のステップは84個中3個(約3.6%)だった。
3GPUともmarginの大きさの分布(ヒストグラムの形)は一致したが、84個の値そのものは1つも揃わなかった。

anakinとbladeで完全一致したのは84個中17個、差がある要素の中央値は18 ULP、最大49,152 ULPだった。
anakinとdgxsparkでは完全一致5個・中央値32 ULP・最大40,960 ULP、bladeとdgxsparkでは完全一致10個・中央値24 ULP・最大8,192 ULPだった。
「最下位ビットがずれる」という表現が示す1 ULPの桁とは、ここまで大きく違う。

![F8a. LLM greedy生成のlogit margin。](/images/reproducibility-of-randomness/f8_llm_margin.png)
_F8a. margin分布(左)とホスト間差分(右)。_

#### 仮説との照らし合わせ

全84要素でmarginの大きさとULP差の関係を確認すると、marginが小さい要素ほどULP差が大きくなる傾向があり、仮説と整合していた。
最大のULP差(49,152 ULP)が出た要素は、3GPUいずれでもmarginが最小だった要素(anakin 0.0037326813 / blade 0.0037212372)と一致している。
この3.6%という数字が意味するのは、0.1程度の摂動が入れば順位が入れ替わりうる位置が、全体の3.6%ある、ということである。
batchサイズの変化では今回分岐は見られなかったが、脆い位置自体は実在する。

---

## 8. 判定フローと納品文言テンプレート

> 手元の計算を上から順に当てはめるだけで、L0〜L4のどこに属するかが決まる。各レベルの末尾に、そのまま納品文書に貼れる一文を用意した。

ここまでの実測を判定フローにまとめた。
手元の計算を1つ選び、上から順に問いに答えていく。

1. 同一プロセスを2回実行しただけで結果が変わりうるか。YESならL4(保証なし)。NOなら次へ。
2. IEEE754の基本演算(`+ - × ÷ √`・FMA)、または公開されたPRNG仕様の生ビット列(MT19937/PCG64/Philox)だけに依存しているか。YESならL0(規格)。NOなら次へ。
3. 使っているAPIについて、ライブラリが版間のビット互換性を文書で明言しているか。YESならL1(明文の規約)。NO(非保証と明記、または文書なし)なら次へ。
4. SIMDディスパッチ・BLASスレッド数・libm実装・GPUモデルを変えると結果が変わることを、実測または文献で確認したか。未検証なら「分からない」に倒す。YESならL3(実行時環境)。NO(今回の実測範囲では環境を変えても一致)ならL2(事実上の実装)。

![F9. 保証レベルの判定フロー+納品文言テンプレート。](/images/reproducibility-of-randomness/f9_flowchart.png)
_F9. 判定フローと、レベルごとの文言の雛形。_

各レベルの納品文言テンプレートは次の通りである。
角括弧の中は、自分の計算・バージョン・環境に置き換えて使う。

```text
L0 規格:
本計算はIEEE754の基本演算、またはMT19937/PCG64/Philoxの公開仕様が定める
生ビット列のみに依存する。ライブラリのバージョンを[X.Y.Z]に固定すれば、
実行環境を問わずビット単位で一致する。

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

これらは本プロジェクトの実測(4ホスト・7イメージ・30ラン、2026年10月14日)に基づく雛形である。
自分の計算について、まず上のフローで判定してから使う。

---

## 9. まとめ

> seedを固定するだけでは、ビット単位の再現性は保証されない。何が保証の出所で、何が壊れる条件かをレベルごとに言葉にできれば、納品文書に書いてよいことと書いてはいけないことの境界がはっきりする。

今回の実測では、仮説通りだったことと、外れたことの両方があった。
AVX2とAVX-512の違いで`exp`/`log`が割れることは予測通りだったが、`np.sum`は割れなかった。
x86_64とaarch64の違いよりも、同じx86_64同士のSIMD世代の違いの方が効いた。
LLMのbatch1とbatch8は、少なくとも今回の規模では一致した。
GPUは全体として非決定的ではなく、非決定性は`atomicAdd`を使う一部の処理に局在していた。

納品物に再現性を書くときは、「再現します」を一言で済ませず、どのレベルの保証なのかを明記するとよい。
第8章の判定フローと文言テンプレートは、そのための最小限の道具である。

本記事の実測ハーネスと生データは、そのままリポジトリに残している。
同じ手順で、読者自身の環境でも再実行できる。

---

## 参考文献

> 本文が依拠した外部の出典を、テーマ別に種別を区別して列挙する。規格・公式文書・Issue/PR・査読論文・エンジニアリングブログを同格には扱わず、種別を明記した。URLと取得日はすべてresearch_notesの調査記録に基づく。

### 規格

- IEEE 754-2019 背景資料「Differences Among IEEE 754 Implementations」(IEEE Microprocessor Standards Committee名義。背景説明用の二次資料で、ワーキンググループ内での正式な位置づけは未確認)。https://grouper.ieee.org/groups/msc/ANSI_IEEE-Std-754-2019/background/addendum.html (取得日: 2026-10-14)

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

- NVIDIA cuBLAS Documentation「Results Reproducibility」(公式ドキュメント)。https://docs.nvidia.com/cuda/cublas/ (取得日: 2026-10-14)
- NVIDIA cuDNN Backend Documentation「Odds and Ends」(公式ドキュメント。異なるアーキテクチャ間のビット再現性は非保証と明記)。https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/misc.html (取得日: 2026-10-14)
- PyTorch GitHub Issue #194676「CUDA RNG is not reproducible across GPU models above a launch-geometry threshold」。https://github.com/pytorch/pytorch/issues/194676 (取得日: 2026-10-14)
- PyTorch GitHub PR #196733「[docs] CUDA generator output depends on the device's SM count」(レビュー承認済み、2026年10月14日時点で未マージ)。https://github.com/pytorch/pytorch/pull/196733 (取得日: 2026-10-14)

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
