# uvを解剖する — 実測結果サマリー

計測日時: 2026-07-28 (UTC)。すべてDockerコンテナ内(ホストは非汚染)で実行。
コンテナは同一の `uv-bench` イメージから起動した1つの常駐コンテナ(`sleep infinity`)に対し、
各実験を `docker exec` で個別に実行した(キャッシュ/venvの状態は各条件ごとに明示的に作り直している)。

### 📝 バージョン訂正(2026-07-28追記)

初回計測ではPoetryが **1.8.3** で入っていた。原因は `Dockerfile` で公式インストーラに `--version 1.8.3` を明示的に指定していたため(インストーラ本来のデフォルト動作なら最新安定版が入る)。技術記事のベンチマークとして古いバージョンでは信頼性を損なうため、`--version` 指定を削除してイメージを再ビルドし、**Poetry 2.4.1**(2026-07-28時点の最新安定版)で実験1・3・4のPoetry関連部分を再実施した。本サマリーの数値は**Poetry 2.4.1を主数値**とし、1.8.3の値は参考値として注記・別ファイルに残す。結論(HTTP/1.1のみ、コピー方式でハードリンクなし)に変化はないが、**インストール速度は2.4.1の方がやや遅い**(後述)という新知見が得られた。

## 0. 環境・ツールバージョン

| 項目 | 値 |
|---|---|
| ベースイメージ | `python:3.12-slim` (Debian GNU/Linux 13 "trixie") |
| Python | 3.12.13 |
| pip | 25.0.1 |
| Poetry | **2.4.1**(公式インストーラのデフォルト最新安定版。参考値のみ1.8.3、下記注記参照) |
| uv | 0.11.32 (公式インストーラで導入) |
| Dockerクライアント(ホスト) | 29.2.1 |
| ネットワーク | コンテナからPyPI/files.pythonhosted.orgへ直接アクセス可能(プロキシなし) |

生データ: `results/env_info.json`(訂正内容は`poetry_version_note`フィールドに記録)、参考値: `results/exp1_poetry_1.8.3_reference.jsonl`

---

## 実験1: インストール速度ベンチマーク(streamlit)

### 手法
- pip: `python -m venv <venv> && pip install streamlit`(単一コマンドで解決+取得+展開)
- poetry / uv: `streamlit` を唯一の依存に持つ最小 `pyproject.toml` を用い、`poetry lock`/`uv lock`(依存解決)と `poetry install`/`uv sync`(取得・展開)を分離して計測
- コールド: `pip cache purge` / `poetry cache clear --all -n <cache>` / `uv cache clean` を実行し、さらにキャッシュディレクトリ(`PIP_CACHE_DIR`/`POETRY_CACHE_DIR`/`UV_CACHE_DIR`、いずれもコンテナ内専用パス)を`rm -rf`。venv/プロジェクトも毎回新規作成
- ウォーム: 直前にキャッシュを温める1回のウォームアップ実行(非計測)を行った上で、venv/プロジェクトのみ毎回新規作成してキャッシュは保持
- 各条件3回実行、中央値(median)を採用

### 結果(秒、中央値。括弧内は3回の全実測値) — Poetryは2.4.1(主数値)

| ツール | 条件 | lock/resolve | install/sync | 合計(中央値) |
|---|---|---|---|---|
| pip | cold | (pip installに内包) | (pip installに内包) | **25.48** (24.60 / 25.48 / 35.23) |
| pip | warm | (pip installに内包) | (pip installに内包) | **21.31** (21.17 / 21.31 / 23.82) |
| poetry 2.4.1 | cold | 5.73 (5.53/5.73/6.28) | 7.56 (7.30/7.56/7.67) | **13.29** (12.83 / 13.29 / 13.94) |
| poetry 2.4.1 | warm | 4.18 (4.18/4.18/4.26) | 5.72 (5.57/5.72/5.99) | **9.97** (9.75 / 9.97 / 10.17) |
| uv | cold | 0.64 (0.46/0.64/0.72) | 2.15 (2.07/2.15/2.37) | **2.82** (2.70 / 2.82 / 2.87) |
| uv | warm | 0.07 (0.065/0.069/0.071) | 0.14 (0.116/0.139/0.144) | **0.20** (0.18 / 0.20 / 0.21) |

参考値(旧計測、Poetry 1.8.3、`results/exp1_poetry_1.8.3_reference.jsonl`):

| ツール | 条件 | lock/resolve | install/sync | 合計(中央値) |
|---|---|---|---|---|
| poetry 1.8.3 | cold | 4.91 (4.82/4.91/6.93) | 6.42 (6.28/6.42/6.47) | 11.29 (11.19 / 11.29 / 13.35) |
| poetry 1.8.3 | warm | 3.65 (3.64/3.65/3.74) | 4.30 (4.15/4.30/4.30) | 7.94 (7.80 / 7.94 / 8.04) |

生データ: `results/exp1_pip_raw.jsonl`, `results/exp1_poetry_raw.jsonl`(2.4.1), `results/exp1_poetry_1.8.3_reference.jsonl`(参考), `results/exp1_uv_raw.jsonl`, `results/exp1_summary.json`

### 倍率(Poetry 2.4.1基準)
- uv cold は pip cold の **約9.0倍速**、poetry cold の **約4.7倍速**
- uv warm は pip warm の **約104倍速**、poetry warm の **約49倍速**(ハードリンクキャッシュの効果。実験4参照)
- poetry も pip よりかなり速い(cold同士で pip の約1.9倍速)。これは poetry も PEP 658 メタデータ取得を使っていること・resolverの実装差によるもの(下記実験2参照)

**⚠️ 新知見**: Poetry 2.4.1はこの環境において1.8.3より**明確に遅い**(cold: 13.29s vs 11.29s、約+18%。warm: 9.97s vs 7.94s、約+26%)。バージョンアップで速くなったわけではない。したがって「uvが速いのはRust実装だからではない」という記事の主張は揺らがない(むしろ、同じPythonツールであるPoetryも新しいバージョンで速くなっていない点は、実装言語よりI/O設計の差が本質という主張を補強する材料になる)。

---

## 実験2: メタデータ取得の実証(PEP658 / レンジリクエスト)

### 対象
`streamlit-1.60.0-py3-none-any.whl`
(PyPI Simple API JSON, `Accept: application/vnd.pypi.simple.v1+json` で取得した `data-dist-info-metadata`/`core-metadata` フィールドあり = PEP658/714対応)

### サイズ対比

| 対象 | サイズ | 備考 |
|---|---|---|
| wheel本体 | 10,419,153 bytes (約9.9MB) | `curl -sI` の `content-length` |
| `.metadata` サイドカー | 10,369 bytes (約10.1KB) | `curl -sI <wheel_url>.metadata` の `content-length` |
| 比率 | **約0.0995%**(wheelの約1005分の1) | メタデータだけならこの量で済む |

生データ: `results/exp2_metadata_vs_wheel_headers.txt`

### レンジリクエストによる自前METADATA抽出デモ
`.metadata` サイドカーを一切使わず、wheel(zip)の末尾からRangeリクエストでEnd Of Central Directory→Central Directory→METADATAエントリのローカルヘッダ位置を辿り、必要な範囲だけをHTTP Range Requestで取得するPythonスクリプト(`bench/exp2_range_metadata.py`)を作成し実行。

| 項目 | 値 |
|---|---|
| wheel全体サイズ | 10,419,153 bytes |
| 転送したバイト数(Rangeリクエスト合計、6回) | 92,655 bytes |
| 転送量/wheel全体比 | **0.89%**(約112分の1) |
| 取得できたMETADATA本体サイズ | 10,369 bytes |

6回のRangeリクエストの内訳(末尾22B→末尾20B→中央ディレクトリ約87KB→ローカルヘッダ2回→METADATA本体3.7KB)は `results/exp2_range_metadata_log.txt` に記録。
生データ: `results/exp2_range_metadata_result.json`

**含意**: PEP658の `.metadata` サイドカー(1万バイト強)は、Rangeリクエストで自前解析する場合(9万バイト強、主にzip中央ディレクトリの読み込み分)よりもさらに約9倍小さい。サイドカー方式はzip構造の解析コストとリクエスト往復回数を追加で削減している。

### ⚠️ 資料との差異: pipのPEP658対応はデフォルトである

スライドには「pipは`.metadata`取得にオプションが必要」との記述があるとのことだが、**pip 25.0.1(python:3.12-slim同梱)で実測したところ、オプション無しの `pip install --dry-run -v streamlit` だけで、41個中すべての依存パッケージについて `.whl.metadata` へのHTTPリクエストで依存情報を取得していることを確認した**(ログ中 `Obtaining dependency information for ... from https://.../*.whl.metadata` および `Using cached ...whl.metadata` が41件)。

pipのPEP658サポートは **pip 22.3(2022年10月リリース)導入時から追加オプション不要**であり(pip公式NEWS.rstで確認: `Use the data-dist-info-metadata attribute from PEP 658 to resolve distribution metadata without downloading the dist yet.`)、現行の pip 25.0.1 でも追加オプションなしに `.metadata` を優先的に取得する。

**訂正(本タスク内で発覚)**: 当初このサマリーでは「pipのPEP658サポート(旧`--use-feature=fast-deps`)」と記載していたが、これは誤り。`--use-feature=fast-deps` はPEP658とは別系統の、HTTPレンジリクエストでwheelを遅延ダウンロードして依存情報を得る実験的機能で、pip 20.2(2020年7月)に導入されたもの。PEP658(pip 22.3、2022年10月)より前から存在し、PEP658対応の旧名でもない。pip本家リポジトリのソース(`src/pip/_internal/cli/req_command.py`)を確認したところ、fast-depsは2026年7月時点の最新版でも実験的フラグとして残っており(「本番環境では未推奨」という警告付き)、廃止されていない。スライドの記述は古いpipバージョン(PEP658対応が普及する前の時代)を前提にしていた可能性が高く、**現在のpipには当てはまらない**という結論自体は変わらないが、fast-depsとの混同は記事化する際に修正が必要。

生データ: `results/exp2_pip_dry_run_verbose.log`(`grep -c 'Obtaining dependency information'` → 41)

### 補足: poetryも同様にPEP658 `.metadata` を取得している(1.8.3・2.4.1両方で確認)
`poetry lock -vvv` の詳細ログで `GET .../*.whl.metadata HTTP/1.1` が多数観測された。1.8.3・2.4.1いずれのバージョンでも同様のパターンで、metadataらしき文字列がログ中それぞれ42件(2.4.1)観測された。つまり「メタデータだけ先に取る」という設計はpip/poetry/uvの3ツールいずれも(少なくとも現行バージョンでは)採用しており、**uvだけの専売特許ではない**。uvとpip/poetryの実測差は主に後述の並列度・HTTP/2・実装効率(Rust vs Python)の複合要因と考えられる。
生データ: `results/exp2_poetry_verbose_lock.log`(旧1.8.3、参考)、`results/exp2_poetry2x_verbose_lock.log`(2.4.1、主データ)

### uvの詳細ログでの`.metadata`取得確認
`RUST_LOG=uv=debug uv lock` のログで、`No cache entry for: .../*.whl.metadata` → `Sending fresh GET request for: .../*.whl.metadata` というパターンが streamlit の依存関係一つ一つに対して観測された(ログ中 `metadata` 一致87件)。
生データ: `results/exp3_uv_debug_lock.log`

---

## 実験3: 並列通信の効果とHTTP/2

### UV_CONCURRENT_DOWNLOADS=1 vs デフォルト(50)
- lockファイルは事前に1回だけ生成(計測対象外)し、`uv sync`(ダウンロード+展開フェーズ)のみをコールドキャッシュで3回計測
- 対象: streamlit(約30個の依存wheel)

| 設定 | 実測値(秒) | 中央値 |
|---|---|---|
| `UV_CONCURRENT_DOWNLOADS=1`(直列) | 2.595 / 3.131 / 2.850 | **2.850** |
| デフォルト(50、並列) | 2.103 / 2.356 / 2.213 | **2.213** |

**並列化の寄与: 約1.29倍(約22%短縮)**。

⚠️ **注記(ネットワーク依存性)**: この検証環境はPyPI CDN(files.pythonhosted.org)への往復レイテンシが非常に小さく帯域も十分なクラウド環境と見られ、直列化のペナルティが理論的な最大値より小さく出ている可能性が高い。レイテンシの大きい回線(自宅回線・地理的に遠いリージョンなど)では、リクエスト往復回数がボトルネックになるため並列化の効果はより大きく出ると推測される。本実験の数値は「この実行環境における実測値」であり、並列化の効果を過小評価している可能性がある点を明記する。

生データ: `results/exp3_concurrency_raw.jsonl`

### HTTP/2対応の確認

`curl -v --http2 -sI` で files.pythonhosted.org への接続を確認:
```
* ALPN: curl offers h2,http/1.1
* SSL connection using TLSv1.3 / TLS_AES_128_GCM_SHA256 / X25519MLKEM768 / RSASSA-PSS
* ALPN: server accepted h2
* using HTTP/2
```
pypi.org / files.pythonhosted.org いずれも `HTTP/2 200` を返す(PyPI CDNはHTTP/2対応)。
生データ: `results/exp3_curl_http2_check.txt`

### uv vs poetry: 実際に使われているHTTPバージョンの実測比較(Poetry 2.4.1で再確認)

| ツール | 確認方法 | 結果 |
|---|---|---|
| uv | `RUST_LOG=trace uv lock` | `ALPN negotiated h2, updating pool` / `http2 handshake complete, spawning background dispatcher task` が観測(2回) → **HTTP/2 (h2) を実際に使用** |
| poetry 2.4.1 | `poetry lock -vvv` | ログ中 `HTTP/1.1` が **107件**、`HTTP/2` は **0件**。全通信が `urllib3.connectionpool` 経由の `HTTP/1.1` リクエストとして記録されている → **HTTP/1.1のみ**(1.8.3時点の計測と同数・同じ結論) |

**2.x系での変化を懸念して依存関係を精査したところ、Poetry 2.4.1はPoetry本体の専用venv(`/root/.local/share/pypoetry/venv`)に `httpx 0.28.1` と `httpcore 1.0.9` を新たに同梱していることが判明した**(1.8.3にはなかった)。ただし:
- `poetry lock -vvv` の実測ログを見ると、PyPIとの通信は依然 `requests`(`urllib3.connectionpool`)経由の `HTTP/1.1` のみで行われており、httpx/httpcoreは(少なくともpackage indexとの通信では)実際には使われていない。
- httpxがHTTP/2を使うために必要な `h2` パッケージは同梱されていない(`h2 not installed` を確認)。そのためhttpxが将来的に使われるようになったとしても、`h2`が入らない限りHTTP/2は使えない。

以上より、**Poetry 2.4.1でもHTTP/1.1のみという結論は変わらない**。ただしhttpx/httpcoreが同梱され始めた点は、将来のPoetryバージョンでHTTP/2対応が入る可能性を示す兆候として記事中で触れる価値がある。

生データ: `results/exp3_uv_trace_lock.log`, `results/exp2_poetry2x_verbose_lock.log`(2.4.1、主データ)、`results/exp2_poetry_verbose_lock.log`(1.8.3、参考)

---

## 実験4: キャッシュ機構(ハードリンク vs コピー)

### 手法
`numpy`(コンパイル済みバイナリを含むwheel)を2つの独立プロジェクトに `uv sync` / `poetry install` でインストールし、venv内ファイルとキャッシュ内ファイルの `stat` (inode, nlink) を比較。

### uv: ハードリンクを実証

| 場所 | inode | nlink | サイズ |
|---|---|---|---|
| project A の venv内 `libscipy_openblas64_*.so` | 20725570 | 3 | 25,128,625 bytes |
| project B の venv内 `libscipy_openblas64_*.so` | 20725570 | 3 | 25,128,625 bytes |
| uvグローバルキャッシュ内の同ファイル | 20725570 | 3 | 25,128,625 bytes |

**3か所すべてinode一致・nlink=3** → uvはグローバルキャッシュの展開済みファイルを両プロジェクトへハードリンクしていることを実証。

生データ: `results/exp4_hardlink_result.json`

### poetry(2.4.1): コピーを実証

| 場所 | inode | nlink |
|---|---|---|
| project A の venv内 `libscipy_openblas64_*.so` | 20610565 | 1 |
| project B の venv内 `libscipy_openblas64_*.so` | 20612179 | 1 |

**inode不一致・nlink=1** → Poetry 2.4.1でも各venvへ個別に展開(コピー)している(1.8.3時点の計測: inode 20727098 vs 20728725、nlink=1 と同じ結論)。
poetryのキャッシュ(`~/.cache/pypoetry/artifacts/.../numpy-*.whl`)は**圧縮された.whl本体のみ**をキャッシュしており(サイズ16,672,469 bytes、inode=20603463, nlink=1)、展開後のファイル単位でのハードリンクは行わない。2つ目のプロジェクトでも「ダウンロードは省略できる(cold: 2.22s → 2回目: 1.61s)が、展開(unzip)は毎回フルコピーで発生する」という挙動(2.4.1でも同様)。

### ディスク使用量(du)比較

`du -sb <projA>/.venv <projB>/.venv --total` を**単一プロセス**で実行し、OSレベルでのinode重複検出(ハードリンクの実ディスク節約)を反映させた値:

| ツール | 2プロジェクト分の実ディスク使用量(結合du) | 素朴な合計(2×単独du) | 差 |
|---|---|---|---|
| uv | **57,197,631 bytes (約54.5MB)** | 114,187,146 bytes (約108.9MB) | 2件目の増分はわずか118,298 bytes(約115KB) |
| poetry 2.4.1 | **125,414,182 bytes (約119.6MB)** | 125,428,498 bytes (約119.6MB) | 差なし(ハードリンクしていないため当然) |
| poetry 1.8.3(参考) | 125,165,528 bytes (約119.4MB) | 125,179,766 bytes (約119.4MB) | 差なし |

**uvはpoetry(2.4.1)の約45.6%のディスク使用量**(57.2MB vs 125.4MB、約2.19倍の差)で同じ2プロジェクトを構築できる。特に「2件目以降のプロジェクトの追加コストがほぼゼロ(115KB)」という点が、複数プロジェクトを持つ開発者にとってのuvの実利用上のメリットとして定量的に示された。Poetryはバージョンが上がっても(1.8.3→2.4.1)キャッシュ機構そのものは変わっておらず、コピー方式のまま。

生データ: `results/exp4_hardlink_result.json`(`combined_du_check`フィールド。`reference_poetry_1_8_3`に旧版の値を保持)

---

## 総括

| 検証項目 | スライドの主張 | 実測結果(Poetryは2.4.1基準) |
|---|---|---|
| インストール速度 | uvが最速 | ✅ 支持。streamlit installでpipの約9〜104倍、poetry(2.4.1)の約4.7〜49倍速(条件により差、warmキャッシュ時に差が最大化) |
| PEP658メタデータ直接取得 | uv高速化の要因の一つ | ✅ 支持。ただし**pip/poetryも同じ仕組みを使っている**ことが判明(uv専用の仕組みではない。poetryは1.8.3・2.4.1両方で確認) |
| 非同期大量並列 | uv高速化の要因の一つ | ✅ 部分的に支持。この実行環境(低レイテンシ)では並列化の寄与は約22%短縮に留まった。高レイテンシ環境ではより大きい寄与が予想される |
| HTTP/2 | uv高速化の要因の一つ | ✅ 支持。uvはALPNでh2を実際に使用、poetry(2.4.1でもrequests/urllib3経由でHTTP/1.1のみ)と実測で明確に対比できた |
| グローバルキャッシュ+ハードリンク | uv高速化・省ディスクの要因 | ✅ 支持。inode一致で実証、2プロジェクトで poetry(2.4.1)の約45.6%のディスク使用量、2件目以降の追加コストはほぼゼロ |

### ⚠️ 資料との差異(まとめ)
1. **「pipは`.metadata`取得にオプションが必要」という記述は現行pip(25.0.1)には当てはまらない。** PEP658対応は2022年末〜2023年頃のバージョンからデフォルト化されている。オプション無しの `pip install --dry-run -v` で全依存パッケージについて `.metadata` 経由の依存情報取得を確認した。スライド/記事はこの点の記述をpipのバージョンに応じた注記に修正するか、「pipも(今は)metadata-onlyで解決するが、それでもuvの方が速い理由は並列度・HTTP/2・実装言語(Rust)の複合効果である」という論旨に更新することを推奨する。
2. (参考、資料との直接矛盾ではないが記事の論旨強化に有用) poetryも実は仕組みの多くをuvと共有している(PEP658メタデータ取得)。したがって「uvが速いのはRustだからではない」という主張を補強する一方で、「メタデータ取得の仕組みだけがuvの差別化要因」という単純化も避けるべきで、**並列度・HTTP/2・キャッシュのハードリンク化の複合効果**として説明するのが実測に忠実。
3. **(本タスク内で発覚・訂正した点)** ベンチマーク構築時、`Dockerfile`で公式インストーラに誤って`--version 1.8.3`を指定していたため、当初の計測は古いPoetry 1.8.3で行われていた。`--version`指定を削除し最新安定版**Poetry 2.4.1**で再計測した。**HTTP/1.1のみ・コピー方式(ハードリンクなし)という質的な結論はバージョンを問わず変わらない**が、**インストール速度は2.4.1の方が1.8.3より遅い**(cold: 13.29s vs 11.29s / warm: 9.97s vs 7.94s)。これは「新しいPoetryバージョンで性能改善された」という期待とは逆方向の結果であり、記事の主張(uvの速さはRust実装そのものではなく通信I/O設計に起因する)を弱めるものではなく、むしろ「Pythonで書かれた同種のツールはバージョンを重ねても同等のI/O設計上のボトルネックを抱え続けている」という補強材料になる。

---

## 再現方法

```bash
cd dissecting-uv
docker build -t uv-bench .
docker run -d --name uvbench -v "$(pwd)/results:/bench/results" -v "$(pwd)/bench:/bench/bench" uv-bench sleep infinity

# 実験1
docker exec uvbench python3 bench/exp1_pip.py --condition cold --runs 3
docker exec uvbench python3 bench/exp1_pip.py --condition warm --runs 3
docker exec uvbench python3 bench/exp1_poetry.py --condition cold --runs 3
docker exec uvbench python3 bench/exp1_poetry.py --condition warm --runs 3
docker exec uvbench python3 bench/exp1_uv.py --condition cold --runs 3
docker exec uvbench python3 bench/exp1_uv.py --condition warm --runs 3
docker exec uvbench python3 bench/exp1_aggregate.py

# 実験2
docker exec uvbench python3 bench/exp2_range_metadata.py

# 実験3
docker exec uvbench python3 bench/exp3_concurrency.py --concurrency 1 --runs 3
docker exec uvbench python3 bench/exp3_concurrency.py --runs 3

# 実験4
docker exec uvbench python3 bench/exp4_hardlink.py both

docker rm -f uvbench
```
