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
- local(MacBook)・anakin・blade・dgxsparkの4台、7種類のDockerイメージ、35回の実測で境界を確かめた。理論で説明がつかなかった結果もそのまま書く。

なお、各章の「理論」節は実測結果を見た後に書いている。本記事中で実測前に予測を記録したのは2か所だけである。第6章の決定実験(`PREREGISTERED_PREDICTION.md`)と、第7章のbatch構成実験(`BATCH_INVARIANCE_REDESIGN.md`)であり、それ以外の「理論」は一次資料の事後整理である(詳細は第10章)。

---

## 2. 冒頭の実測(同じseed・同じコンテナでビットが割れた)

> 同じDockerイメージ、同じseed、どちらもx86_64の2台で、NumPyの`exp`/`log`の結果がビット単位で食い違った。原因は推測ではなく、対照実験で確定している。

### 理論

NumPyは起動時にCPUが対応する命令セットを検出し、演算ごとに別のコードパスを選ぶ「CPUディスパッチ」を行う。AVX2・AVX-512はこの切り替えに関わるSIMD命令セットの世代名である(詳細は後述のボックスを参照)。
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
AVX-512を無効化したbladeは、anakinとここで実行した4プローブ全てでビット一致した。
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

OS側では、出力段のChaCha20ベースCRNG自体は2016年のカーネル4.8(Theodore Ts'o)で既に導入済みである。
2022年のLinuxカーネルの書き直し(5.17/5.18、Jason Donenfeld)が変えたのはその先で、5.17では入力プールの処理がSHA-1からBLAKE2sに変わり、5.18では出力段の鍵の持ち方がper-NUMAからper-CPUに再設計された。
ChaCha20そのものは2022年の変更対象ではない。
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

ハードウェア側では、Intelの`RDRAND`はSP 800-90A準拠のDRBGから読み、`RDSEED`はSP 800-90B/C準拠のconditioner(AES-CBC-MACベース)を経たエントロピーサンプルを読む。RDRANDのようなDRBGによる長い擬似乱数展開を経ない分、生エントロピーに近い。
RDRANDは1つのシードから最大1022個の連続した乱数しか生成せず、超えると自動的に再シードされる。
ARMの`RNDR`/`RNDRRS`も同じ構図で、`RNDRRS`は呼び出すたびに強制的に再シードする。

通常のLinuxコンテナは、ホストのカーネルCSPRNGを共有する。
コンテナの分離はnamespace機構で実現されているが、man7の`namespaces(7)`が列挙するnamespaceの種類はcgroup・IPC・network・mount・PID・time・user・UTSの8種類のみで、RNGやエントロピー専用のnamespaceは存在しない。
RNG専用のnamespaceが無い以上、通常のLinuxコンテナはホストと同一のカーネルCSPRNGインスタンスを使わざるを得ない。
ただし、この点を一文で明記したDocker公式等の文書は見つけられていない。
Mac上のDocker DesktopはそもそもLinuxの仮想マシンを経由しており、ホストのmacOSカーネルを直接使っているわけではない。

PYTHONHASHSEEDも、seedとは別の軸にある設定である。
目的はハッシュフラッディング攻撃(CVE-2012-1150)への対策で、PEP 456によりSipHash24が採用された(この採用はPEP 456時点の経緯であり、現行のCPythonのデフォルトはPython 3.11以降SipHash13に変わっている)。
CPython公式ドキュメントによれば、ハッシュランダム化の対象は`str`と`bytes`(およびそれらに依存する型)に限られ、`int`のハッシュはPYTHONHASHSEEDの影響を受けない。
`dict`はPython 3.7以降、挿入順を保持する言語仕様になっており、PYTHONHASHSEEDの影響を受けない。
一方`set`は、要素が`str`/`bytes`(やそれらに依存する型)を含む場合、ハッシュテーブルのスロット位置がそれらの値に依存するため、PYTHONHASHSEEDを変えると反復順が変わりうる。要素が`int`のみのsetは、この影響を受けない。

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
| L0 規格         | IEEE754は国際規格。MT19937/PCG64/Philoxは査読論文・公開仕様(国際規格ではない) | IEEE754の`+ - × ÷ √`・明示的に呼んだ`fma()`(Fused Multiply-Add、積和を1回の丸めで計算する演算。正しく丸め)、MT19937/PCG64/Philoxの状態遷移・出力計算(同一の内部状態から次状態・出力を計算する部分に限る) | 言語・OS・CPUを跨いでも壊れない。ただし`a*b+c`のようなFMAへの自動contractや、seedから内部状態への変換はこの「壊れない」範囲に含まれない(下記参照) | コア部分(同一内部状態からの次状態・出力計算)に限ってビット一致を約束できる                                |
| L1 明文の規約   | ライブラリが文書で約束               | NumPy NEP19(RandomStateは保証、Generatorは非保証)、PyTorch「リリース間の再現性は保証しない」                                 | バージョンを上げた瞬間                                              | バージョン固定を条件に約束できる                                      |
| L2 事実上の実装 | 文書化なし。実装がたまたま決めている | 正規乱数の変換法(ziggurat/polar法)、`np.sum`のpairwise summationの閾値                                                       | バージョン差(根拠なく変わりうる)                                    | 実際には一致するが、根拠が無いため無条件の一致保証としては書かない方がよい(条件付きの言い回しは第9章L2テンプレート参照)                    |
| L3 実行時環境   | 実行時に環境が選ぶ                   | SIMDディスパッチ、BLASスレッド数、libm(glibc/musl)、cuBLAS GEMM(同一アーキテクチャ・同一SM数なら決定的。第6章G4参照)         | 同じコード・同じ版でもCPU世代・スレッド数・ベースイメージ・GPUアーキテクチャが変われば | 「同一イメージ+同一CPU世代+スレッド数固定」(GPU計算では同一GPUアーキテクチャ・同一SM数も)まで条件を書けば約束できる |
| L4 保証なし     | 原理的に非決定                       | PYTHONHASHSEED、GPUの`atomicAdd`順序(いずれも本記事で実測)。一般に知られた他の例としてcuDNN benchmarkの自動選択があるが、本記事にはこれを測るプローブが無く未実測                                                              | 性質が2種類ある。`atomicAdd`等は同一プロセス内の2回実行でも変わる。PYTHONHASHSEED等は同一プロセス内では変わらないが、プロセス/コンテナを起動し直すと変わる(第5章の実測参照) | 約束できない。許容誤差でしか語れない                                  |

![F3. 再現性の保証レベルL0〜L4。](/images/reproducibility-of-randomness/f3_levels.png)
_F3. 各レベルで何が壊すかを並べた原理図。_

L0の範囲は、「同一の内部状態から次状態・出力を計算する」というコア部分に限られる。
MT19937(Matsumoto and Nishimura, 1998, ACM Transactions on Modeling and Computer Simulation掲載の査読論文)やPCG64(O'Neillの論文自体が"A Family of..."を名乗っており、PCGは特定の1アルゴリズムではなくアルゴリズムの族を指す。NumPyはその中から「PCG XSL RR 128/64」という特定のバリアントを選んでいる)は、IEEE754のような国際規格ではなく、査読論文または公開された実装仕様である。
とはいえ、同じ内部状態から同じ手順で次状態と出力を計算する部分(コア部分)は、版を跨いで仕様として固定されているため、ここをL0として扱うこと自体は妥当である。

一方、seedという整数値を内部状態に変換する部分は、このコア部分の外側にある。
NumPyの`SeedSequence`はバージョン1.19.0でspawn時の処理を変更しており、公式リリースノートは「Spawned SeedSequences will produce different results than in the previous release」(spawnしたSeedSequenceは前リリースと異なる結果を生成する)と明記している。
また`default_rng()`が返すビット生成器自体も将来変わりうる設計になっている。NumPyは1.21.0でPCG64DXSMを追加しており、公式の互換性方針は「In particular, default_rng is allowed to change the default BitGenerator that it uses (again, with caution and plenty of advance warning)」(default_rngは、既定で使うBitGeneratorを変更してよい。ここでも慎重に、十分な事前告知を伴って)と述べている。
つまり、seedから内部状態への変換はL1(明文の規約)寄りの領域にあり、ライブラリのバージョンを固定しない限り約束できない。
第9章の納品文言テンプレートが「バージョンを[X.Y.Z]に固定すれば」という条件を付けているのは、このseed変換を含めた生成パイプライン全体を指しているためであり、コア部分だけを指す上表の「壊れる条件」の記述と矛盾するものではない。

本文に必ず入れておくべき但し書きが3つある。

- IEEE 754が保証するのは演算1つであって、演算の並びではない。IEEEのサイト(grouper.ieee.org)で公開されている背景資料は、「the IEEE standard does not guarantee that the same program will deliver identical results on all conforming systems(同じプログラムが全ての準拠システムで同一の結果を返すことを規格は保証しない)」と明記している。この資料は規格の規範的な本文ではなく背景説明用の文書であり、ワーキンググループ内での正式な位置づけまでは確認できていないが、信頼できる二次資料として扱う。同じ資料は「移植した際に途中結果が食い違うなら必ずソフトウェアのバグだ」という通説も否定している。
- `exp`/`log`/`sin`のような超越関数は、IEEE 754では正しく丸めることが要求されておらず、推奨にとどまる。任意の点で正しく丸めるのに何ビット余分に必要かを一般に抑える手法が無いため、主要なlibm実装はどれも全ての超越関数を正しく丸めてはいない。これが第6章でlibm実装差が出る根本原因になる。
- FMAがL0なのは、`fma()`を明示的に呼んだ場合に限られる。ソースコード中の`a*b+c`という書き方がFMA命令1つにcontract(融合)されるかどうかはIEEE 754の範囲外で、言語標準・コンパイラ・`-ffp-contract`のようなフラグ・対象CPUがFMA命令を持つかに依存する。C標準は`#pragma STDC FP_CONTRACT OFF`でcontractを抑制できることになっているが、GCCはこのpragmaを無視してFMAに融合することが知られている。したがって自動contractによる結果はL3寄りの挙動になりうる。なお、同じlibm実装でもISA(命令セットアーキテクチャ)によって結果が変わる例が本記事の追加実測で実際に観測されている(第6章G2)。muslの`log`はソースコード中に`__FP_FAST_FMA`という明示的な`#if`分岐を持ち、FMA命令が使えるISAかどうかで数式そのものを切り替えている。ここで述べた「`a*b+c`の自動contractはL3寄り」という一般論よりもさらに直接的に、ソースに書かれた分岐そのものがISA依存の原因になっている例であり、本章で条件付きでしかL0でないと述べたFMAの依存性が、実測で現れた具体例にあたる。

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
NumPy公式ドキュメントは「it is not possible to reproduce the exact random values using Generator for these distributions or any distribution method that relies on them」と明記している。
日本語にすると「Generatorでこれらの分布、およびそれに依存する分布生成メソッドの正確な乱数値を再現することはできない」であり、「these distributions」が指すのは正規分布だけではない。
原文の直前の文で挙げられているのは正規分布・指数分布・ガンマ分布の3つであり、さらにガンマ分布に依存して実装されている分布(ベータ分布・カイ二乗分布など)もこの射程に含まれる。
同じseedでも、これらの分布の値はRandomStateとGeneratorの間で一致しない。

---

## 5. 「インプットが全部同じ」は案外成立していない

> seedと環境変数をそろえても、globの列挙順・setの反復順・並列ワーカーの完了順は、気づかないうちに違う入力になっていることがある。

納品時に「同じ入力」と言うとき、通常は「同じデータ」「同じコード」「同じseed」を指す。
しかしコードの中には、seedで止められない隠れた入力がいくつかある。

### 理論

第3章で確認した通り、`dict`は挿入順を保持しPYTHONHASHSEEDの影響を受けない一方、`str`/`bytes`型の要素を含む`set`は反復順がPYTHONHASHSEEDに依存する。
この理論からは、PYTHONHASHSEEDを0に固定すれば`str`型要素を含む`set`の反復順はホスト間で揃い、固定しない既定状態では揃わないはずだ、という方向が導ける。

`glob.glob`のファイル列挙順について、Python公式ドキュメントは「The pathnames are returned in no particular order」(列挙順は特定の順序を保証しない)と明記しており、「If you need a specific order, sort the results」(特定の順序が必要ならソートせよ)と推奨している。
順序を実際に左右する要因はファイルシステムの実装に依存する。
ただしこの規約が明言しているのは「順序を保証しない」ことであって、「同一プロセス内で2回呼んだときに毎回結果が変わる」ことまでは述べていない。同一プロセス内の2回呼び出しが安定するかどうかは、この理論だけからは導けない。

並列ワーカーの完了順はOSのスケジューラが決めており、第4章のL4(保証なし)の定義が示す通り、原理的に非決定な領域に属する。L4の定義からすれば、同一プロセス内の2回の計測であっても一致しない方向が自然に予想される。

### 実験設定

`glob.glob`でのファイル一覧取得、`set`の反復順(`hash('reproducibility')`を例に使用)、32ジョブを使った並列ワーカーの完了順の3つを対象にした。
4ホストで、PYTHONHASHSEEDを0に固定した場合と、設定しない既定の場合を比較した。
globはext4・XFS・tmpfsの3種類のファイルシステム上で、同一プロセス内で2回呼び出した。
並列ワーカーは同一プロセス内で2回計測し、完了順を比較した。

### 結果

- `glob.glob`でファイル一覧を取ると、順序はファイルシステムに依存する。今回の実測では、同一プロセス内で2回`glob`を呼んだ場合、32ラン全てで安定していた(`stable_within_process`として自己申告)。ただしこれはext4・XFS・tmpfsでの結果であり、全てのファイルシステムで保証されているわけではない。
- `set`の反復順は、`str`/`bytes`型の要素を含む場合、ハッシュテーブルのスロット位置に依存する。PYTHONHASHSEEDを0に固定すると、4ホストとも`hash('reproducibility')`(str型)が一致した。PYTHONHASHSEEDを設定しない既定状態では、同じコンテナを再起動するだけで4ホスト全てで値が食い違い、同じイメージを別コンテナで4回立てただけで4通りの値が出た。
- 並列ワーカーの完了順は、seedとは無関係にOSのスケジューラが決める。同一プロセス内で2回計測しても、32ジョブの完了順は毎回異なった(`stable_within_process=False`として自己申告)。これは測定上の不備ではなく、並列処理そのものの性質であり、L4(保証なし)に相当する。

### 理論との照合

`set`(`str`型要素を含む場合)は理論の通りで、PYTHONHASHSEED=0で一致し、既定状態では不一致になった。
並列ワーカーの完了順も理論の通り、同一プロセス内でも毎回異なった。
`glob`は、公式ドキュメントが明言する「順序不定」という理論から、同一プロセス内の2回呼び出しが安定するかどうかまでは導けなかった。実測では32ラン全てで安定していた(ext4・XFS・tmpfsの範囲)。
ただし、同一プロセス内で同じ`readdir`の結果がOS側のキャッシュ機構によって再度返ってくること自体は驚くに値しない経路であり、かつ確認できたのは同一プロセス内の2回に限られる。環境(ホスト・ファイルシステム・プロセスの再起動)を跨いだ場合の順序安定性については、本実測は何も示していない。ソートせずに依存してよいという結論にはならない。

この3つは、seedを固定した「正しい」コードの中にも紛れ込む。
納品前のチェックリストに「`glob`の結果をソートしているか」「`PYTHONHASHSEED`を固定しているか」を加える価値がある。

---

## 6. 実測マトリクス(何が割れて、何が割れなかったか)

> 4ホスト・7イメージ・35ランの実測を、保証レベル×条件のマトリクスにまとめた。Python版・BLAS実装・SIMD世代・スレッド数・GPU世代ごとに、割れた条件と割れなかった条件をそのまま並べる。

以下は、全ての実測結果をレベル×条件で並べたマトリクスである。
緑は一致、赤は不一致、オレンジは複数プローブの結果が混在していることを示す(凡例は画像内に記載)。

![F4. 保証レベル×環境条件。](/images/reproducibility-of-randomness/f4_matrix.png)
_F4. 35ランの実測結果を自動生成したマトリクス。_

ここから先の差は、ほぼ全てが同じ原因に行き着く。
浮動小数点の加算には結合則が無い。

![F5. 浮動小数点の加算に結合則はない。](/images/reproducibility-of-randomness/f5_assoc.png)
_F5. `(x+y)+z`と`x+(y+z)`が別の値になる例(Goldberg 1991)。_

IEEE 754は個々の演算の丸めは規定するが、複数の演算をどの順序・どのグループで評価するかは規定しない。
順序はコンパイラ・SIMD幅・スレッド分割など、実行環境が決める。

本章の各節は「理論」「実験設定」「結果」「理論との照合」の4部構成を基本とする。ただしG2だけは、この4部に決定実験(事前登録)の節を加えた5部構成とする。
断っておくと、本記事の理論整理は実測結果を見た後に行っている。実測結果30ランのコミットから約2.5時間後に初稿を書いており、理論節の記述は事後的な整理である。本記事中で実測前に予測を記録したのは、本章G2で後述する追加実測(`PREREGISTERED_PREDICTION.md`)と、第7章で後述するbatch構成実験(`BATCH_INVARIANCE_REDESIGN.md`)の2か所だけであり、そこでは明示的に「事前登録」と呼ぶ。それ以外の節の理論は、一次資料(ソースコード・公式ドキュメント・リリースノート)が何を言っているかの整理であって、実測前の予測ではない。

また、本章が尽くしているのは実測したプローブが対象とした軸に限られる。以下の軸は本章のマトリクスには含まれていない。

- 乱数消費順序に対するリファクタ耐性(実務で最も頻繁に問題になる軸だが、それを測るプローブ自体を作っていない)
- コンパイラ/ビルドフラグ(`-ffast-math`、最適化レベルなど)
- メモリアライメント/非連続ストライドの配列
- OS・カーネルバージョンの差(`envinfo.py`でカーネルのリリース文字列は収集しているが、軸として比較には使っていない)
- `MKL_CBWR`(MKLの再現性制御環境変数。`envinfo.py`では収集しているが本文では扱っていない)
- batch構成によるLLMロジットの変化(S10、第7章で扱う)。S10の3ランはresults/に含まれており、F4のラン数(35ラン)にも数えられているが、「バッチ構成」はG1(時間・コンテナ)/G2(ソフトウェア)/G3(ハードウェア)/G4(アクセラレータ)のどの軸にも属さない――同一の計算を環境を変えて比較する軸ではなく、同一の環境で入力(バッチの組み方)を変える軸であり、本章のマトリクスが前提とする条件軸の構造そのものと噛み合わない。そのためF4には列を設けていない。

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
しかし「SIMDに依存するのでlibmには依存しない」という単純な二分法は、NumPy 2.2.6のソースと照らすと正確ではない。
`loops_exponent_log.dispatch.c.src`のDOUBLE_exp/DOUBLE_logは、(1) SVML(Intel Short Vector Math Library。Intelコンパイラでリンクした場合にのみ有効になるベクトル化数学関数ライブラリで、通常のPyPIホイールでは有効でなく、本記事の実測環境でもこの経路は使われていない)が使えるAVX512_SKX環境ではSVMLカーネル、(2) AVX512Fが使える環境ではNumPy内製のAVX512F_exp_DOUBLE/AVX512F_log_DOUBLE、(3) どちらも使えなければnpy_exp/npy_logのスカラー経路、つまりプラットフォームのlibmを呼ぶ、という3段構えのディスパッチになっている。
ただし(2)のAVX512F_log_DOUBLEには、それ自身の内部にも小さなlibmフォールバックがある。同ソースのコメントは887行目で「if x on (1.0 - 0x1p-4, 1.0 + 0x1.09p-4), calling npy_log()」と明記しており、953〜954行目でそのウィンドウを定数`[0.9375, 1.0647]`として定義し、988行目のif分岐でこの範囲に入る要素だけ、1088行目で`npy_log(ip_fback[ii])`としてnpy_log(プラットフォームのlibm)を個別に呼んでいる。AVX512F_exp_DOUBLEにはこのフォールバックが無い。
一方`loops_trigonometric.dispatch.cpp`には「Disable SIMD code sin/cos f64 and revert to libm」というコメントがあり、float64の`sin`/`cos`はNumPy 2.2.6では常にlibmを経由する(`npy_math_internal.h.src`も「We assume the above are always available in their double versions」としてlibm依存を前提にしている)。
したがって、`exp`がlibm実装の違いに反応するかどうかは「AVX-512専用カーネルが使えるかどうか」だけで決まるのに対し、`log`はAVX-512専用カーネルが使える環境でもなお、入力のうち1.0近傍の窓`[0.9375, 1.0647]`に落ちる要素があるかどうかで部分的にlibm実装差の影響を受けうる。`sin`/`cos`は常にlibm依存である。

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

bladeだけが`exp`/`log`でlibm実装差に反応しなかった。これはAVX-512の有無で説明できる(詳しくは本節末尾)。
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

glibc/muslの比較は、第2章時点では「bladeという1台のAVX-512搭載機で、exp/logが一致してsinだけ割れた」という結果だけを見ていた。本章でdgxsparkを加えると、同じglibc/musl比較でexp/logも割れる。これは「exp/logはlibm非依存」という一般化がそのままでは誤りであることを、プロジェクト自身のデータが示している。

決定実験は、この矛盾を解消する機構を裏付けた。bladeという1台の中で、同一のmuslイメージ・同一のnumpy 2.2.6・同一のPython 3.11.17のまま、AVX-512のnumpyディスパッチ対象だけを無効化すると、`exp`は`ab300a28`に、`log`は`28ff757a`に変化し、無効化後の`exp`はAVX2しか持たないanakinのmusl値と完全一致した。`sin`は一切反応しなかった。

> `exp`は、AVX-512Fがあれば完全にlibm非依存(NumPy内製カーネル)、無ければlibm依存(libmへのフォールバック)である。`log`はAVX-512Fがあっても、入力`x`が1.0近傍の窓`[0.9375, 1.0647]`に落ちる要素だけ、内部でlibm(`npy_log`)を個別に呼ぶため、「libm非依存」とは言い切れない。`sin`/`cos`(float64)はAVX-512の有無によらず常にlibm依存である。「SIMD依存・libm非依存」という単純な二分法ではなく、「AVX-512専用カーネルの有無と、入力がフォールバック窓に落ちるかどうかが、libmを経由するかどうかを決める」という機構として理解する必要がある。`exp`/`sin`については、この機構を同一ハードウェア上でAVX-512ディスパッチのみを無効化した決定実験で確認した。

この`log`のフォールバック窓は、本プロジェクトが用意した2本の`log`プローブの挙動差をそのまま説明する。`S4_log_float64`は`[1e-9, 1e9]`を線形一様サンプリングしており、桁の大きい側に標本が偏るため、同じseed(20261014)・同じN(1,000,000)で計算すると、フォールバック窓`[0.9375, 1.0647]`に入る要素は0個である。一方`S4_log_loguniform_float64`は指数を一様に敷いてサンプリングしており、同じ条件で3,027個の要素が窓に入る(いずれも本プロジェクトで直接計算して確認した値)。blade(AVX-512あり)でglibcとmuslを比較すると、`S4_log_float64`(フォールバック0件)は両者一致し(`a6b6b7fa`)、`S4_log_loguniform_float64`(フォールバック3,027件)も両者一致した(`002faa54`)。フォールバックが一度も発火しない`S4_log_float64`が一致するのは、AVX-512専用カーネルがlibmを経由しないためそのまま説明がつく。一方`S4_log_loguniform_float64`はフォールバックが発火しているにもかかわらずglibcとmuslが一致しており、これは1.0近傍ではblade上のglibcとmuslの`log`がたまたま同じビットを返しているためだと読める。ただし、この「1.0近傍でglibcとmuslが一致する理由」自体は未検証であり、断定はできない。

この一致は、「入力範囲が結論を決めている可能性がある」という方法論上の懸念が、本プロジェクトで具体的に実在した例でもある。もし`S4_log_loguniform_float64`を追加していなければ、`S4_log_float64`の一致だけを見て「AVX-512専用カーネルがあれば`log`は常にlibm非依存」と誤って一般化していた可能性がある。線形一様サンプリングだけでは1.0近傍の挙動を検出できないことを踏まえて入力分布の異なるプローブを追加した判断は、結果的にこの死角を塞ぐ役割を果たした。

異なるISA(x86_64のAVX2とaarch64)のglibc環境同士が`exp`/`log`でビット一致したのも、AVX-512専用カーネルの有無という機構で説明できる。どちらもAVX-512専用カーネルを持たずlibmにフォールバックするため、あとは双方のglibc(ともにバージョン2.41)が`exp`/`log`で同じ結果を返すかどうかだけで決まる。今回はそれが一致した、という観測である。

一方、予測が外れた`log`は新しい事実を明らかにした。muslの`log`はISAによって異なるビットを返す(x86_64のmusl: `28ff757a`、aarch64のmusl: `84b2ce10`)。musl版の`exp`と`sin`は両ISAで同一(`ab300a28`/`601eb0fa`)であり、glibc版の`log`も両ISAで同一(`09cef322`)である。つまり「同じlibc実装ならISAを跨いでも同じ」という読みは、muslの`log`では成立しない。

この原因は、予測が外れた後の追加調査で特定できた。muslのソース`src/math/log.c`には、次のような明示的な`#if`分岐が実在する(`log2.c`にも同じ分岐がある)。

```c
	/* log(x) = log1p(z/c-1) + log(c) + k*Ln2.  */
	/* r ~= z/c - 1, |r| < 1/(2*N).  */
#if __FP_FAST_FMA
	/* rounding error: 0x1p-55/N.  */
	r = __builtin_fma(z, invc, -1.0);
#else
	/* rounding error: 0x1p-55/N + 0x1p-66.  */
	r = (z - T2[i].chi - T2[i].clo) * invc;
#endif
```

`__FP_FAST_FMA`が真のときはFMA命令による積和を1回の丸めで計算し、偽のときは`T2`という別の補正テーブルを使って複数回丸める経路に切り替わる。どちらを通るかで数式そのものが別物になる。`exp.c`や`__sin.c`/`__cos.c`/`__rem_pio2.c`には、この`__FP_FAST_FMA`分岐も`__builtin_fma`呼び出しも一切存在しない。これが`exp`/`sin`はISA依存にならず、`log`だけがISA依存になる理由である。また`src/math/x86_64/`にはdouble精度の`log`/`exp`/`sin`のISA専用アセンブリは存在せず(long double版の`logl.s`/`expl.s`や`fma.c`/`sqrt.c`等があるのみ)、`log.c`/`log_data.c`にも`long double`は登場しない。したがって「ISA専用アセンブリ」や「long double中間値」が原因という代替仮説は成り立たない。

`__FP_FAST_FMA`はmusl自身が定義する値ではなく、GCCがターゲットのISA能力を見て自動的に定義するマクロである。GCCの`gcc/c-family/c-cppbuiltin.cc`にある`mode_has_fma()`が、そのターゲット向けにdouble精度の`fma`命令パターンがバックエンドに存在するか(`HAVE_fmadf4`)を見て決めており、musl側の`arch/aarch64/fp_arch.h`・`arch/generic/fp_arch.h`はこのマクロに一切関与しない。
Alpineのビルド設定(`aports/main/gcc/APKBUILD`)の`build()`は次のようになっている。

```sh
case "$CTARGET" in
aarch64-*-*-*)		_arch_configure="--with-arch=armv8-a --with-abi=lp64";;
esac
```

aarch64には`--with-arch=armv8-a`が明示指定されており(ARMv8-Aはスカラ浮動小数点のFMAを必須で含む)、x86_64に対応するcaseエントリは存在しない。その場合GCCのデフォルトである`x86-64`(SSE2ベースライン、FMA命令なし)が使われる。`aports/main/musl/APKBUILD`にも`-march`や`-mfma`の上書きは無い。この非対称なビルド設定が、aarch64側でだけFMA経路を有効にし、x86_64側では複数回丸めの別経路を使わせている。

muslの数学ライブラリの作者Szabolcs Nagy氏自身も、2019年2月2日のmuslメーリングリストで次のように述べている。

> my new implementations are generic c code, and it's possible to get portable results with the fp semantics of c across targets which implement annex F with FLT_EVAL_METHOD==0: you have to turn fp contraction off, disable builtins and other compiler smartness (musl already does these) and remove any target specific ifdefs or asm that may introduce non-portable results (in the new code only the __FP_FAST_FMA checks).

作者自身が、移植性を崩しうる要因として`__FP_FAST_FMA`の分岐を名指ししている。

ここで確認できていない点も残しておく。Alpineが配布する実バイナリ(`libc.musl-x86_64.so`/`libc.musl-aarch64.so`)を逆アセンブルして`__FP_FAST_FMA`の実値を直接確認したわけではなく、上記はビルド設定からの論理的な推論である。また、`exp`/`sin`でコンパイラによる自動FMA融合(contraction)が実際に起きていないことも、Alpineが使っている実際のCFLAGSに`-ffp-contract=off`相当の指定があるかまでは確認できておらず、作者の一般的な発言と観測の整合からの推測にとどまる。さらに、この現象自体が既知のissueやバグとして報告された形跡は見つからなかった。

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

![F6. BLASの行列積(スレッド数が変わると各スレッドの担当サブレンジが変わる)。](/images/reproducibility-of-randomness/f6_blas_tiles.png)
_F6. スレッド数ごとの担当範囲割り当ての違い(原理図)。_

KMeansは、`effective_thread_count`が1のとき4ホスト全て(anakin/blade/dgxspark/local)で同じ重心に収束した。
`effective_thread_count`が8になった6本のラン(anakin/threads8・blade/threads8・dgxspark/threads8・local/threads8・local/hashseed0・local/hashseedfree。localは8コア構成のため、スレッド数を明示的に固定していないhashseed0・hashseedfreeでも`effective_thread_count`が8になる)では、4通りの異なる重心に分かれた。
30回×6条件の繰り返し実行(別の1回の計測、8CPU構成のコンテナ)では、スレッド数を1・2・4に固定すると30回とも同じ重心になり、8に固定すると13回、既定設定では19回、異なる重心になった。
この回数は実行のたびに多少変動する値であり、固定した定数としては読まないでほしい。
この生データは`results/`には無く、プロジェクトの判断記録(`reprobench/probes/s5_sklearn.py`のdocstring)にのみ残っている。

一方で`n_iter_`(反復回数)は180回のフィット全てで2、重心の分散(inertia)の広がりは最大でも3.9×10^-16程度に収まっていた。

x86_64(anakin)とaarch64(dgxspark)を比較すると、`S2`/`S4`は全て一致した。

PyTorch(CPU)のMLP順伝播・学習10ステップは、スレッド数ごとの割れ方が一様ではなかった。anakinでは`S7_mlp_forward`・`S7_mlp_train10`の両方が、bladeでは`S7_mlp_train10`のみが、スレッド数1・8・既定でそれぞれ別のdigestになった(最大絶対差は`S7_mlp_forward`で約1.79×10^-7、`S7_mlp_train10`で約7.6×10^-8)。一方dgxsparkとlocal(いずれもaarch64)は、両プローブとも今回の実測範囲ではスレッド数を変えても一致した。x86_64の2台(anakin・blade)では割れ、aarch64の2台(dgxspark・local)では割れなかったという非対称性自体が観察として価値があるが、原因は未検証である。BLAS・KMeans・PyTorch(CPU)は同じ構造(非結合則の演算をスレッド分割で計算する)を持ち、実際に割れるケースが見つかっており、「CPU実行は安全で、GPUだけが危ない」という理解はこの3つを並べると成立しない。

一方LightGBM(`S6_lgbm_pred`/`S6_lgbm_tree_structure`)は、スレッド数1/8/最大・MKL/OpenBLAS・Python 3.11/3.13・oldlibsを跨いだ25ラベル全てでビット一致した。`reprobench/probes/s6_lightgbm.py`には、決定的モードを意図的に設定していない(`deterministic: True`を指定しない)ことが明記されている。np.sumと並んで、縮約を使う処理が必ず割れるわけではないことを示す例である。勾配ブースティングはマルチスレッドで不安定という実務の通念とは逆の結果になった。

#### 理論との照合

スレッド数の影響は理論の整理と一致し、float32のBLASは割れた。float64のBLASはホスト内では安定し、ホスト間では割れた。np.sumはスレッド数・ホストのいずれでも割れなかった。
KMeansの不安定も理論の整理と一致した。つまり不安定の正体は、別の局所最適を見つけているのではなく、縮約(合計)の丸め方の違いである。

PyTorch(CPU)とLightGBMの結果は対照的だった。PyTorchのMLPはBLAS・KMeansと同じ理由(非結合則×スレッド分割)で割れたのに対し、LightGBMは同じ軸(スレッド数・BLAS実装・Python版)を跨いでも割れなかった。

第2章の実測を振り返ると、`S4_exp_float64`/`S4_log_float64`は同じx86_64同士(anakinのAVX2とbladeのAVX-512)で不一致になった。
一方、本章のx86_64とaarch64の比較では全て一致した。

> 理論で説明できなかった結果: アーキテクチャ(x86_64かaarch64か)は決め手ではなかった。決め手はAVX-512専用カーネルの有無である。本節の理論は「CPUアーキテクチャの違いが結果を変える」という一般的な理解に基づいていたが、実際に割れたのは同じx86_64同士のAVX2とAVX-512の間であり、x86_64とaarch64という異なるISA同士は(どちらもAVX-512を持たない組み合わせでは)一致した。機構はG2で確認した通り、AVX-512専用カーネルの有無がlibmを経由するかどうかを決めるというものである。

![F7. AVX-512専用カーネルの有無が、exp/log/sinのlibm経由を決める。](/images/reproducibility-of-randomness/f7_simd.png)
_F7. `np.exp`/`np.log`/`np.sin`のディスパッチ機構と、blade機内でAVX-512を無効化した決定実験の実測値(原理図+実測)。_

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

本文で述べた通り、atomicAdd自体は書き込む順序までは保証しないため、これが同じGPU・同じコードでも`index_add_`のような処理の結果が毎回一致しない理由になる。
PyTorch公式の定義では、`torch.use_deterministic_algorithms(True)`は「決定的な実装があればそれに差し替え、無ければRuntimeErrorを送出する」という設定であり、一般に「足し込む順序を固定する」という意味ではない。`index_add_`についてはこの設定で決定的な実装に切り替わり、結果として足し込む順序が固定される。

:::

PyTorchのCUDA RNGは`curand_init`をスレッドのインデックスでシードしており、スレッドと要素の対応はSM数に依存して決まる(PyTorch issue #194676)。
この挙動を公式ドキュメントに追記するPR(#196733)は、2026年10月20日時点でレビュー承認済みだが未マージであり、現時点では公式に文書化されていない。

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
`torch.randn`を同じseedで4サイズ(1,024 / 65,536 / 4,194,304 / 67,108,864)生成し、サイズごとにdigestを取った(`S8_curand_randn_sizes`、3GPU比較)。

#### 結果

anakin(RTX 4090)で確かめると、Linear層+ReLUのMLPを10ステップ学習させる処理は、5回ともビット一致した。
CPU実行とGPU実行を比較するという当初の設計のままでは、非決定性は現れなかった。

一方、重複したインデックスへの`index_add_`(内部で`atomicAdd`を使う処理)は、5回とも一致しなかった(最大絶対差 8.24×10^-4)。
`torch.use_deterministic_algorithms(True)`を指定すると、同じ`index_add_`が5回ともビット一致した。
MLPと`index_add_`それぞれの5回反復分の生データは`results/`には無く、プロジェクトの判断記録にのみ残っている。

![F8. GPU上の非決定性(atomic演算を使う処理は毎回変わりうる)。](/images/reproducibility-of-randomness/f8_gpu_reduce.png)
_F8. anakinでの5回反復実測(MLP学習は一致、`index_add_`は不一致)。_

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

> temperature=0は「決定的」を意味する公式なモードではない。batch構成を変えるとロジットは数値水準で動く(GPU間差と同じ10^-5オーダー)が、この規模のgreedy生成ではトークン列は分岐しなかった。同梱のデフォルト設定を外すと、生成はもっと早いトークンで分岐した。

Qwen2.5-0.5B-Instructを使い、transformersで生成した。

### temperature=0は存在しない

#### 理論

temperatureは、生成時にモデルの出力するロジットを割って候補のスコアを広げたり狭めたりする値である。
数学的には、temperatureで割る処理そのものが、0で割るという未定義の操作になる。
vLLMの公式の実装(ソースコード)は、内部で`temperature<1e-5`をgreedy扱いにする仕様(`_SAMPLING_EPS = 1e-05`)を持っており、同じ「temperature=0」という入力でも、実装によって扱いが異なりうることを示している。
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

### batch構成とbatch-invariance(事前登録)

#### 理論

Thinking Machines Labは2025年9月のブログで、batch-invariance(バッチの構成によってGPUカーネルの縮約順が変わり出力が変わる現象)を報告している。
このブログは査読論文ではなくエンジニアリングブログであり、Qwen3-235B-A22B-Instruct-2507で1000回のgreedy生成を行い、出力の種類が80通りから1通りに減ったと報告している。
このブログが主張する中心メカニズムは、「ほぼ全てのLLM推論エンドポイントが非決定的である一番の理由は、負荷(したがってbatchサイズ)が非決定的に変動することにある("the primary reason nearly all LLM inference endpoints are nondeterministic is that the load (and thus batch-size) nondeterministically varies!")」というものである。
つまり、本番の推論サーバでは他ユーザの同時リクエストによって継続的バッチング(continuous batching)の構成が実行のたびに非決定的に変わり、同じリクエストでもバッチに同居する他系列が毎回変わることで出力が変わる、という本番推論サーバ固有の現象を指している。
この理論からは、バッチの構成を変えればロジットが変わりうることが導かれる。
ただしgreedyデコーディングはargmaxであるため、出力トークンが変わるかどうかは別の話である。top-1とtop-2のロジット差(margin)を跨ぐ摂動がない限り、ロジットが動いても選ばれるトークンは変わらない(margin自体の分布は本章末尾で実測する。中央値は約2.07、最小値は0.0037である)。

#### 実験設定(事前登録)

本節の実験は、設計そのものを一度作り直している。最初に実装した`S9_greedy_batch1`/`S9_greedy_batch8`は、Qwen2.5-0.5B-Instructを使い、3GPU(anakin・blade・dgxspark)でbatchサイズ1と8のgreedy出力をそれぞれ1回ずつ比較するものだったが、以下の3つの欠陥を抱えていた。

1. **バッチ構成が均質**: batch8は`tokenizer([text] * batch, ...)`で同一プロンプトを8本複製したものであり、長さの異なる実プロンプトを混在させていなかった。全行が同じ長さなので、パディングもattention mask由来の経路分岐も原理的に発生しない。
2. **観測量が鈍すぎる**: digestの対象が生成トークンID列だった。greedyはargmaxなので、marginを跨ぐ摂動がなければ出力は一切変わらない。marginの中央値(約2.07)に対し、本節で後述する通り実際に観測された摂動は10^-5オーダーであり、2桁違う量を閾値判定(トークン列が変わるかどうか)で見ようとしていたことになる。
3. **各条件1回のみ**: 比較は3GPU×各条件1回、つまりn=1だった。TMLが報告した分岐率は80/1000 = 8%であり、数%程度の事象をn=1の比較で「一致した」と結論するのは統計的に意味を持たない。

これら3つの欠陥を修正するため、観測量を生成トークン列からロジットそのものに変え、バッチ構成を実際に動かす設計に作り直した。新しい設計と5項目の予測は、実装・実測の前に`BATCH_INVARIANCE_REDESIGN.md`としてコミット60294f2(2026-10-20 18:54:51)で記録してから実行している。本記事中で実測前に予測を記録したのは、第6章G2の決定実験とこの実験の2か所だけである。

新設計は、固定した被験プロンプトP(実トークン長43)に対し、同じバッチに何が同居するかを変えながら、Pの最終実トークン位置のロジット行をビット単位で比較する。

| config | 内容 |
| --- | --- |
| `b1` | [P] 単体(基準) |
| `b8_same` | [P]×8、Pは位置0(旧設計と同じ構成。対照として残す) |
| `b8_mixed_first_right` | [P] + 長さの異なるfiller 7本、Pは先頭(位置0)、右詰めパディング |
| `b8_mixed_first_left` | 同上、左詰めパディング |
| `b8_mixed_last` | filler 7本 + [P]、Pは末尾(位置7) |
| `b2_long` | [P, 非常に長いfiller] |
| `b16_mixed` | [P] + 長さの異なるfiller 15本 |
| `b4_padheavy` | [P] + Pより大幅に長いfiller 3本 |

各configにつき、`b1`を基準とした最大絶対差(max_abs)・argmaxトークンID・top-1とtop-2のmarginを記録し、同一configを同一プロセス内で2回実行してプロセス内決定性も確認した。副次実験として、同じconfig群でgreedy生成(最大128トークン)も行い、`b1`を基準とした最初の分岐位置を記録した。

事前登録した予測は次の5点である。

1. `b1` vs `b8_same`: ロジットは不一致になる。
2. 混成バッチ(`b8_mixed_*`)の`b1`との差は、均質バッチ(`b8_same`)の差より大きい。
3. `b8_mixed_first_right` vs `b8_mixed_last`(バッチ内の位置だけが違う): ロジットは不一致になる。
4. トークン列は全configでほぼ一致する。分岐が起きるとしても生成の後半。
5. 同一configのプロセス内2回実行は完全に一致する。

なお、実測JSONには`max_ulp_vs_b1`というULP差も記録されているが、本節ではこの値を引用しない。`reprobench.digest.ulp_diff`はfloat32の値をfloat64にキャストしてからビット距離を取るため、値が10^14〜10^15オーダーになり、本章末尾のmargin節で使うfloat32ビット空間のULP(18 ULP、49,152 ULPなど)とは規約が異なる。同じ記事に2つのULP規約を混在させると読者が混乱するため、本節では`max_abs`(絶対差)のみを使う。

#### 結果

3GPU(anakin・blade・dgxspark)で実測した。

ロジット(Pの最終実トークン位置のロジット行をビット単位で比較、`b1`を基準とした最大絶対差):

| config | anakin | blade | dgxspark | `b1`とビット一致 |
| --- | --- | --- | --- | --- |
| `b1`(基準) | 0 | 0 | 0 | — |
| `b8_same`(旧設計と同じ構成) | 2.85e-5 | 2.53e-5 | 2.77e-5 | 3ホストともFalse |
| `b8_mixed_first_right` | 2.92e-5 | 3.39e-5 | 3.48e-5 | False |
| `b8_mixed_first_left` | 3.15e-5 | 3.15e-5 | 2.67e-5 | False |
| `b8_mixed_last` | 2.92e-5 | 3.39e-5 | 3.48e-5 | False |
| `b2_long` | 3.48e-5 | 2.77e-5 | 4.24e-5 | False |
| `b16_mixed` | 3.58e-5 | 3.15e-5 | 4.77e-5 | False |
| `b4_padheavy` | 4.29e-5 | 4.20e-5 | 4.24e-5 | False |

全config・全ホストで`stable_within_process=True`(プロセス内2回実行は完全一致)であり、top-1とtop-2のmarginは全config・全ホストを通じて1.3991〜1.3992の範囲に収まっていた。

`b8_mixed_first_right`と`b8_mixed_last`は、3ホストともビット単位で完全一致した(digest・max_absともに同一)。Pが同じバッチ内で先頭(位置0)にあるか末尾(位置7)にあるかは、ロジットに影響しなかった。

greedy生成は、7 config × 3ホスト = 21通りすべてが同一digest(`2d84d0ac...`、68トークン)になった。`first_divergence_vs_b1`は全条件でnullであり、`n_tokens == n_tokens_raw`でパディングの混入も確認できなかった。

想定外の観察として、dgxsparkでのみ`b2_long`(バッチサイズ2、パディング後系列長819)と`b4_padheavy`(バッチサイズ4、パディング後系列長429)のロジットdigestが完全に一致した。バッチサイズもパディング後系列長もどちらも異なる2つのconfigが一致しており、anakinとbladeでは別の値になっている。この一致の機構は未調査である。また、max_absが最大になるconfigはホストによって異なり(anakin・bladeは`b4_padheavy`、dgxsparkは`b16_mixed`)、`b1`そのもののロジットdigestも3ホスト間で一致しなかった。これは第6章・本章で既に確認しているGPU間の浮動小数点非決定性と整合する既知の現象であり、バッチ構成を変える以前に、ベースラインの`b1`自体が既にGPU依存である。

#### 理論との照合

事前登録した5つの予測の採点は次の通りである。

- 予測1(`b1` ≠ `b8_same`): **3ホストとも支持**された。均質なバッチでも、バッチサイズが1から8に変わるだけでロジットは変化する。
- 予測2(混成バッチの差 > 均質バッチの差): 概ね支持された。ただし**dgxsparkの`b8_mixed_first_left`だけ逆転している**(2.67e-5 < `b8_same`の2.77e-5)。パディング側(left/right)という別の軸が同時に変わっているための交絡がありうるが、予測通りにならなかった事実としてそのまま記録する。
- 予測3(バッチ内の位置の影響): **3ホストとも反証された。** `b8_mixed_first_right`と`b8_mixed_last`はビット単位で完全一致しており、Ada(anakin)・Blackwell workstation(blade)・GB10(dgxspark)という3つの異なるアーキテクチャ全てで同じ結果だった。
- 予測4(トークン列はほぼ一致、分岐しても後半): 支持された。予測より強く、分岐そのものが21通り全てでゼロだった。
- 予測5(プロセス内2回実行の一致): 3ホストとも支持された。

> batch-invarianceは数値水準では破れている。greedy復号は、この規模(Qwen2.5-0.5B-Instruct、最大128トークン)ではそれを隠しているだけである。

TMLの報告との関係も改めて整理しておく。旧実験は「batchサイズ1と8のgreedy出力は3GPUとも一致した」ことを「理論で説明できなかった結果」として扱っていたが、これは不当な評価だった。今回の結果はTMLの報告と矛盾せず、むしろ補強する。TMLが指摘するのはカーネルのbatch依存性であり、今回の実測はそれをロジットの数値水準で確認した。
ただし両者は同じ実験ではない。TMLが指摘する中心的な経路は本番サーバの継続的バッチング(負荷によってバッチ構成が非決定的に変わる)であり、今回の実測は単一プロセス内でバッチ構成を明示的に変える静的な実験である。カーネルのbatch依存性という同じ機構を突いてはいるが、同じ土俵には立っていない。
また、TMLが観測した出力の分岐は、サンプリング(temperature > 0)を伴う設定で起きている。サンプリングであれば10^-5オーダーの摂動でも選択が変わりうるため、今回のgreedyで分岐がゼロだったことと矛盾しない。ただしこの説明は本実測が直接確かめたものではなく、断定はできない。

本実測が一次データとして測定している点で先行報告より強いという主張はしない。TMLはQwen3-235B-A22B-Instruct-2507で1000回生成しており、本実測(Qwen2.5-0.5B-Instruct、7 config×3ホスト、各config同一プロセス内2回)より規模も試行回数も大きい。

本節で観測した摂動のオーダー(2.5×10^-5〜4.8×10^-5)が、GPU間で実際に観測される丸め誤差とどう関係するかは、本章末尾のmargin節で改めて数値を揃えて示す。

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

![F9a. 温度を上げるほど、GPUモデル間の分岐は早いトークンで起きる。](/images/reproducibility-of-randomness/f9a_llm_divergence.png)
_F9a. anakinとdgxsparkが最初に食い違うトークン位置(温度別)。_

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
3ペアの数値をまとめると次の表になる(84ステップ中、完全一致した個数/一致しなかった要素のULP差の中央値/最大値)。

| ペア               | 完全一致数(/84) | ULP差の中央値 | ULP差の最大値 |
| ------------------ | ---------------- | -------------- | -------------- |
| anakin - blade      | 17                | 18             | 49,152          |
| anakin - dgxspark    | 5                 | 32             | 40,960          |
| blade - dgxspark     | 10                | 24             | 8,192           |

「最下位ビットがずれる」という表現が示す1 ULPの桁とは、ここまで大きく違う。

![F9b. LLM greedy生成のlogit margin。](/images/reproducibility-of-randomness/f9b_llm_margin.png)
_F9b. margin分布(左)とホスト間差分(右)。右パネルは3ペアのうち2ペア(anakin-blade・anakin-dgxspark)を表示(3ペア目のblade-dgxsparkを含む数値は本文参照)。_

#### 理論との照合

全84要素でmarginの大きさとULP差の関係を確認すると、marginが小さい要素ほどULP差が大きくなる傾向があった。
ただし、これは「marginが小さい位置は脆い」という理論の裏付けにはならない。marginは近い2つのロジットの差であるため、margin自体が小さければ桁落ちでmarginの相対誤差(ULP)が大きくなるのは、浮動小数点演算として自明である。「marginが小さい要素ほどULP差が大きい」という観測は、この算術的な必然を確認したにすぎず、「順位が入れ替わりやすい」こととは別命題である。
最大のULP差(49,152 ULP)が出た要素が、3GPUいずれでもmarginが最小だった要素(anakin 0.0037326813 / blade 0.0037212372)と一致しているのも、同じ理由(margin自体が小さいほど、marginのULP誤差が大きくなる)で説明がつき、「脆さ」の追加の裏付けにはならない。順位の入れ替わりやすさを左右するのは、top-1・top-2それぞれのロジットの絶対誤差とmarginの絶対誤差の大小関係であり、本実測は個々のロジット値を記録していないため、これを直接測ってはいない。

一方、GPU間で実際に観測されたmarginの絶対差は、84ステップ・3ペア(anakin-blade/anakin-dgxspark/blade-dgxspark)を通して最大でも2.10×10^-5程度である(先述の49,152 ULPは、marginが0.0037程度まで小さい要素でULP間隔が極端に狭くなることで生じた数字であり、同じ要素を絶対量に直すと1.14×10^-5程度にとどまる)。84ステップ中もっとも小さいmargin(0.0037326813)でも、今回観測された最大の絶対差(2.10×10^-5)の約178倍の大きさがある。つまり、今回のGPU間の丸め誤差の実測スケールでは、84ステップのどこにおいても実際に順位が入れ替わる水準には届いていない。実際、この84ステップの生成トークン列は3GPUとも完全に一致している。

「0.1程度の摂動が入れば順位が入れ替わりうる位置が3.6%ある」という解釈は撤回する。0.1という閾値の根拠は本文になく、実測されたGPU間の摂動スケール(10^-5のオーダー)よりはるかに大きい。0.1未満のステップが84個中3個(約3.6%)あるという数値自体は事実だが、それが実際の脆弱性を意味するかどうかは、今回観測された摂動スケールでは確認されていない。
batchサイズの変化でも、margin自体のULP差という尺度でも、今回の実測範囲で実際の順位入れ替わりは確認されなかった。脆い位置が理論上存在しうること自体は、margin最小値(0.0037)が中央値(2.07)より3桁小さいという分布の形から読み取れるが、それ以上の定量的な主張は今回のデータでは裏付けられない。

ここでbatch構成の実験(本章前節)と数値を並べると、同じ1本の線が見える。GPU間で実際に観測されたmarginの絶対差は最大2.10×10^-5であり、バッチ構成を変えたときにロジットに生じた摂動(2.5×10^-5〜4.8×10^-5)はこれと同じオーダーである。GPUを変えても、バッチ構成を変えても、観測される摂動は10^-5オーダーにとどまり、今回の実測範囲で記録された最小margin(0.0037)の数十分の一から数百分の一の大きさにしかならない。どちらの摂動源も、今回の実測範囲ではgreedy復号の出力そのものを変える水準には届いていない。

---

## 9. 判定フローと納品文言テンプレート

> 手元の計算を上から順に当てはめるだけで、L0〜L4のどこに属するかが決まる。各レベルの末尾に、そのまま納品文書に貼れる一文を用意した。

ここまでの実測を判定フローにまとめた。
判定は「計算」だけに対して行うのではなく、「計算+設定」のペアに対して行う。たとえば`index_add_`は、`torch.use_deterministic_algorithms(True)`を指定しない設定ではL4(保証なし)だが、指定した設定ではビット一致する(第6章参照)。使う設定を固定せずに同じコードをこのフローに通すと、一度はYES、一度はNOという答えが両方出てしまう。
手元の計算と、それを動かす設定(deterministicアルゴリズムの指定・スレッド数の固定・SIMD/BLAS実装の指定等)を1組選び、上から順に問いに答えていく。

1. 同一プロセス・同一設定を2回実行して結果が変わる、またはプロセス/コンテナを起動し直すだけで結果が変わりうるか。YESならL4(保証なし)。NOなら次へ。
2. 今、結果を決めているのはIEEE754の基本演算(`+ - × ÷ √`)、明示的に呼んだ`fma()`(自動contractされた`a*b+c`は含まない)**単独**、または公開されたPRNG仕様の生ビット列(MT19937/PCG64/Philox)だけか。IEEE754が規格として保証するのは演算1つの結果であって、演算の並び(結合順序)ではない(第4章参照)。複数の演算を組み合わせている場合はここでNOに倒す。YESならL0(規格)。NOなら次へ。
3. 使っているAPIについて、ライブラリが版間のビット互換性を文書で明言しているか。YESならL1(明文の規約)。NO(非保証と明記、または文書なし)なら次へ。
4. 現在の設定で、SIMDディスパッチ・BLASスレッド数・libm実装・GPUモデル・deterministicアルゴリズムの指定の有無といった既知の変動源を、実測または文献で既に固定・検証したか。未検証なら「分からない」に倒す。検証済みで、それらを変えると結果が変わることを確認している場合はL3(実行時環境)。検証済みで、今回の実測範囲では変えても一致した場合はL2(事実上の実装)。

![F10. 保証レベルの判定フロー+納品文言テンプレート。](/images/reproducibility-of-randomness/f10_flowchart.png)
_F10. 判定フローと、レベルごとの文言の雛形(図は代表的な計算のみを示す簡略版であり、実際の判定では使用する設定も合わせて固定する必要がある)。_

各レベルの納品文言テンプレートは次の通りである。
角括弧の中は、自分の計算・バージョン・環境に置き換えて使う。

```text
L0 規格:
本計算はIEEE754の基本演算(演算1つ単位。明示的に呼んだfma()を含む)、
またはMT19937/PCG64/Philoxが公開仕様として定める生ビット列のみに依存
する。この生成アルゴリズム自体は言語・OS・CPUを跨いでも結果は変わらない。
なお、ソース中の`a*b+c`のような自動contractによるFMA融合はこのL0には
含まれない(コンパイラ・CPU依存のL3寄りの挙動になりうる)。
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

これらは本プロジェクトの実測(4ホスト・7イメージ・35ラン。2026年10月14日の30ランに、2026年10月20日の追加実測5ラン——G2の決定実験2ランと、第7章のbatch構成事前登録実験3ラン——を加えたもの)に基づく雛形である。
自分の計算について、まず上のフローで判定してから使う。

---

## 10. まとめ

> seedを固定するだけでは、ビット単位の再現性は保証されない。何が保証の出所で、何が壊れる条件かをレベルごとに言葉にできれば、納品文書に書いてよいことと書いてはいけないことの境界がはっきりする。

本記事の理論整理(各章の「理論」節)は、実測の後に行っている。最初の30ラン(4ホスト・7イメージ、2026年10月14日)の結果を見てから記事の執筆を始めており、各章の「理論」は結果を踏まえた事後の整理である。本記事中で実行前に予測を文書として固定してから走らせたのは2か所だけである。次段落で触れる追加実測(`PREREGISTERED_PREDICTION.md`、2026年10月20日)と、その後に触れるLLMのbatch構成実験(`BATCH_INVARIANCE_REDESIGN.md`、同じく2026年10月20日)である。

今回の実測は、理論で説明できたことと、理論だけでは説明がつかなかったことの両方を含む。
AVX2とAVX-512の違いで`exp`/`log`が割れることは理論と整合したが、`np.sum`は割れなかった。
決め手はx86_64とaarch64というCPUアーキテクチャの違いではなく、AVX-512専用カーネルの有無だった。AVX-512専用カーネルがあれば`exp`は完全にlibmを経由せず、`log`は入力が1.0近傍の窓に落ちる場合だけ部分的にlibmを経由する。ISAが異なっていてもlibmの実装が同じなら一致し、libmが異なれば(glibcとmuslのように)割れる。この機構は、bladeのハードウェアを固定したままAVX-512のnumpyディスパッチ対象だけを無効化する追加実測で確認した。実行前に記録した4項目の予測のうち3項目(exp・sin・np.sumの挙動)は的中し、logの挙動だけ予測(aarch64のmuslと同じ値になる)が外れた。外れた結果から、muslのlogはISA(x86_64かaarch64か)によって異なるビットを返すという、実測前には分からなかった事実が新たに分かった。原因はmuslのソース`src/math/log.c`にある`__FP_FAST_FMA`という明示的な`#if`分岐で、FMA命令が使えるかどうかで数式そのものが切り替わっている(第6章G2参照)。
LLMのbatch構成実験は、観測量を生成トークン列からロジットそのものに変え、バッチの中身を実際に動かす設計に作り直した上で、予測を実装・実測の前に事前登録して実測した。結果は、batch-invarianceがロジットの数値水準では破れていることを3GPU(anakin・blade・dgxspark)すべてで確認した——均質なバッチでもバッチサイズを1から8に変えるだけでロジットは変化し(2.5〜4.8×10^-5)、事前登録した5つの予測のうち3つは支持されたが、「バッチ内の位置がロジットに影響する」という予測は3ホストとも反証された(先頭と末尾でビット単位の完全一致)。一方でgreedy復号のトークン列は、7 config×3ホスト=21通りすべてで一致し、分岐は一度も起きなかった。この摂動の大きさ(10^-5オーダー)は、GPU間で実際に観測されたmarginの絶対差(最大2.10×10^-5)と同じオーダーであり、今回実測した最小margin(0.0037)にはどちらも遠く届いていない。batch-invarianceは数値水準で確認できたが、この規模のgreedy復号はそれを隠している、というのが結論である。
GPUは全体として非決定的ではなく、非決定性は`atomicAdd`を使う一部の処理に局在していた。CPU実行も一様に安全なわけではない。BLAS・KMeansと同じ理由(非結合則の演算をスレッド分割で計算する)でPyTorch(CPU)のMLPもスレッド数ごとに割れた一方、LightGBMは同じ軸(スレッド数・BLAS実装・Python版)を跨いでも割れなかった(第6章G3)。縮約を使うかどうかではなく、個々の実装がスレッド分割をどう行うかで結果が変わる。

納品物に再現性を書くときは、「再現します」を一言で済ませず、どのレベルの保証なのかを明記するとよい。
第9章の判定フローと文言テンプレートは、そのための最小限の道具である。

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
- O'Neill, M. E. (2014) "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation." Harvey Mudd College Technical Report HMC-CS-2014-0905(大学テクニカルレポート、査読論文ではない)。https://www.pcg-random.org/paper.html (取得日: 2026-10-20)
- Salmon, J. K., Moraes, M. A., Dror, R. O. and Shaw, D. E. (2011) "Parallel Random Numbers: As Easy as 1, 2, 3." Proceedings of SC11(ACM Best Paper Award、査読論文)。https://dl.acm.org/doi/10.1145/2063384.2063405 (取得日: 2026-10-14)。ACMは有料のため、著者による公開PDFも挙げる。https://www.thesalmons.org/john/random123/papers/random123sc11.pdf (取得日: 2026-10-15)
- NEP 19「Random Number Generator Policy」(NumPy公式文書、著者Robert Kern、Status: Final)。https://numpy.org/neps/nep-0019-rng-policy.html (取得日: 2026-10-14)
- NumPy公式ドキュメント「What's New or Different」(Generatorでは正規分布等の正確な乱数値を再現できないと明記)。https://numpy.org/doc/stable/reference/random/new-or-different.html (取得日: 2026-10-14)
- NumPy 1.22.0 リリースノート(公式文書。float32一様乱数の最下位ビットが常に0になるバグの修正を記載)。https://numpy.org/doc/stable/release/1.22.0-notes.html (取得日: 2026-10-14)
- NumPy 1.19.0 リリースノート(公式文書。SeedSequenceのspawn挙動変更を「Spawned SeedSequences will produce different results than in the previous release」と明記)。https://numpy.org/doc/stable/release/1.19.0-notes.html (取得日: 2026-10-20)
- NumPy公式ドキュメント「Compatibility Policy」(公式文書。「In particular, default_rng is allowed to change the default BitGenerator that it uses (again, with caution and plenty of advance warning)」と明記)。https://numpy.org/doc/stable/reference/random/compatibility.html (取得日: 2026-10-20)

### SIMD・BLAS・libm

- Goldberg, D. (1991) "What Every Computer Scientist Should Know About Floating-Point Arithmetic." ACM Computing Surveys, Vol.23, No.1(査読論文。浮動小数点加算の非結合則の例)。https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html (取得日: 2026-10-14)
- NumPy Issue #23523「BUG: AVX-512 results differ from non-AVX-512 for sin, cos, exp, log, etc.」(GitHub Issue。メンテナmattip氏による4 ULP許容の発言)。https://github.com/numpy/numpy/issues/23523 (取得日: 2026-10-14)
- NumPy公式ドキュメント「CPU build options」(NPY_DISABLE_CPU_FEATURES / NPY_ENABLE_CPU_FEATURESの仕様)。https://numpy.org/doc/stable/reference/simd/build-options.html (取得日: 2026-10-14)
- NumPy 1.9.0 リリースノート(公式文書。pairwise summationを「only along the fast axis and for groups of the values <= 8192 in length」と明記)。https://numpy.org/doc/stable/release/1.9.0-notes.html (取得日: 2026-10-20)
- NumPy 2.2.6 ソース `loops_exponent_log.dispatch.c.src`(GitHub。DOUBLE_exp/DOUBLE_logのAVX512F_exp_DOUBLE/AVX512F_log_DOUBLE経路とnpy_exp/npy_logスカラー経路への3段ディスパッチ)。https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/umath/loops_exponent_log.dispatch.c.src (取得日: 2026-10-20)
- NumPy 2.2.6 ソース `loops_trigonometric.dispatch.cpp`(GitHub。「Disable SIMD code sin/cos f64 and revert to libm」のコメント)。https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/umath/loops_trigonometric.dispatch.cpp (取得日: 2026-10-20)
- musl libc ソース `src/math/log.c`(git.musl-libc.org。`__FP_FAST_FMA`によるFMA/補正テーブル分岐)。https://git.musl-libc.org/cgit/musl/tree/src/math/log.c (取得日: 2026-10-20)
- musl メーリングリスト、Szabolcs Nagy氏の投稿(2019年2月2日、Message-ID: `<20190202033929.GK21289@port70.net>`。musl数学ライブラリの移植性に関する`__FP_FAST_FMA`チェックへの言及。同じ文面は2019年4月3日のDamian McGuckin氏の投稿内に`>`付きの引用として再掲されている)。https://www.openwall.com/lists/musl/2019/02/02/1 (取得日: 2026-10-20)
- GCC ソース `gcc/c-family/c-cppbuiltin.cc`(GitHubミラー。`__FP_FAST_FMA`を定義する`mode_has_fma()`の実装)。https://github.com/gcc-mirror/gcc/blob/master/gcc/c-family/c-cppbuiltin.cc (取得日: 2026-10-20)
- Alpine Linux aports `main/gcc/APKBUILD`(GitLab。aarch64にのみ`--with-arch=armv8-a`を指定するビルド設定)。https://gitlab.alpinelinux.org/alpine/aports/-/blob/master/main/gcc/APKBUILD (取得日: 2026-10-20)

### GPUとCUDA

- NVIDIA cuBLAS Documentation「Results Reproducibility」(公式ドキュメント。ただし`cublas<t>symv()`/`cublas<t>hemv()`は`cublasSetAtomicsMode()`でatomics使用を選ぶと高速ルーチンに切り替わり、ビット再現性は保証されないという例外が明記されている。本記事の実測ではこれらのルーチンは使用していない)。https://docs.nvidia.com/cuda/cublas/ (取得日: 2026-10-14)
- NVIDIA cuDNN Backend Documentation「Odds and Ends」(公式ドキュメント。異なるアーキテクチャ間のビット再現性は非保証と明記)。https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/misc.html (取得日: 2026-10-14)
- PyTorch GitHub Issue #194676「CUDA RNG is not reproducible across GPU models above a launch-geometry threshold」。https://github.com/pytorch/pytorch/issues/194676 (取得日: 2026-10-14)
- PyTorch GitHub PR #196733「[docs] CUDA generator output depends on the device's SM count」(レビュー承認済み、2026年10月20日時点でも未マージ)。https://github.com/pytorch/pytorch/pull/196733 (取得日: 2026-10-14、再確認日: 2026-10-20)

### LLM推論

- He, H. et al.「Defeating Nondeterminism in LLM Inference」Thinking Machines Lab, Connectionism Blog, 2025年9月10日(査読を経ていないエンジニアリングブログ)。https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ (取得日: 2026-10-14)
- vLLMの公式の実装(ソースコード)、`SamplingParams`クラス定義。v0.11.0のAPIリファレンスページ(ソースから自動生成)に`_SAMPLING_EPS = 1e-05`としてtemperature<1e-5をgreedy扱いにする定数が記載されている。利用者向けのプローズ解説本文には具体的な数値は出てこない。https://docs.vllm.ai/en/v0.11.0/api/vllm/sampling_params.html (取得日: 2026-10-14)

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
- Donenfeld, J. A.「Random number generator enhancements for Linux 5.17 and 5.18」zx2c4.com(カーネル乱数生成器の作者本人による解説。5.17でのBLAKE2s化、5.18でのper-CPU化を記載)。https://www.zx2c4.com/projects/linux-rng-5.17-5.18/ (取得日: 2026-10-20)
- PEP 456「Secure and interchangeable hash algorithm」(Python公式文書。SipHash24採用とハッシュフラッディング対策を記載)。https://peps.python.org/pep-0456/ (取得日: 2026-10-14)
- CPython公式ドキュメント「Configure Python」(configure docs。siphash13が3.11以降のデフォルトであることを記載)。https://docs.python.org/3/using/configure.html (取得日: 2026-10-20)
- CVE-2012-1150(共通脆弱性識別子。ハッシュ衝突によるDoS)。https://app.opencve.io/cve/CVE-2012-1150 (取得日: 2026-10-14)
- man7.org「namespaces(7)」マニュアルページ(cgroup・IPC・network・mount・PID・time・user・UTSの8種類を列挙。RNG/エントロピー専用のnamespaceは存在しない)。https://man7.org/linux/man-pages/man7/namespaces.7.html (取得日: 2026-10-20)
- Python公式ドキュメント「glob — Unix style pathname pattern expansion」(「The pathnames are returned in no particular order.」と明記)。https://docs.python.org/3/library/glob.html (取得日: 2026-10-20)
