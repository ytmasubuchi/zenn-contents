# 乱数再現性記事 改訂 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 実務で頻出する4軸(乱数消費順序・cudnn設定・CUBLAS_WORKSPACE_CONFIG・DataLoader worker seeding)を実測し、その結果と外部レビュー由来の正確性修正を記事に反映する。

**Architecture:** 既存の `reprobench` プローブ機構に S11〜S15 を追加し、新しい config ラベル `practice` で既存の結果ファイルに触れずに実測する。記事は章を1つ増やし(新第8章)、判定フローに「コード同一性」という前提条件を加える。

**Tech Stack:** Python 3.11 / numpy / torch / scikit-learn / pytest / Docker / bash(bash 3.2互換)

**Spec:** `docs/superpowers/specs/2026-10-21-reproducibility-article-revision-design.md`

## Global Constraints

- **既存の `reproducibility/results/*__*__*.json` を上書き・編集・削除しない。** 実測の前後で全ファイルのSHA-256を比較し、1バイトも変化していないことを確認する。
- **コミットは必ずパスを明示する**: `git add <path>` の後に `git commit -- <path>`。`git commit` 単独は使わない(同一ワークツリーでの並行編集によるレースが過去に3回発生している)。
- **既存6config(`threads1` `threads8` `threadsmax` `nosimd512` `hashseed0` `hashseedfree`)と `s10batch` の挙動を一切変更しない。**
- **重い依存(torch/sklearn/lightgbm/transformers)のimportは関数本体の内部で行う。** モジュールトップレベルに置くと、その依存が無いイメージでモジュール全体がimport不能になり同モジュールの他プローブが道連れになる(`reprobench/probes/__init__.py` のdocstring参照)。
- **事前登録のコミットは実測の前に行う。** 順序が逆になると事前登録として成立しない。
- プローブ名の接頭辞: `S11_` `S12_` `S13_` `S14_` `S15_`。
- docstring・コメントは日本語。既存プローブの文体に倣う。
- テストコマンド: `make test`(= `docker build -f docker/Dockerfile.base -t reprobench:base . && docker run --rm reprobench:base pytest -q -W error`)。`-W error` で警告が例外になる点に注意。
- 作業ディレクトリは `/Users/ljmb20260017/Documents/zenn-contents`。ハーネスは `reproducibility/` 配下。
- 現在のテスト件数は 212 passed, 2 skipped。タスク完了ごとに件数が増えることはあっても減ってはならない。

## Review Focus

- **S11の頑健版が実は壊れるパターンがありうる。** テストは「頑健版は壊れない」をassertしてはならない(予測を実装で固定すると実験の意味が消える)。テストは「基準版との比較が記録されていること」だけを検証する。→ Task 2 で担保。
- **S12/S13/S15はGPUが無い環境でskipされる。** `requires=("torch","cuda")` によるskip経路が実際に `SkippedProbe` を返すことを検証する。→ Task 4・Task 5 で担保。
- **S13は例外が出ることが期待挙動である。** `use_deterministic_algorithms(True)` かつ `CUBLAS_WORKSPACE_CONFIG` 未設定で PyTorch が送出する例外を握りつぶさず、メッセージごと記録する。例外が出ない場合もそのまま記録する。→ Task 4 で担保。
- **`num_workers>0` はコンテナの共有メモリ不足で失敗しうる。** 失敗した場合に測定全体が止まらないよう、worker起動の失敗を記録して続行する。→ Task 3 で担保。
- **`REPRO_PROBES` 未設定時に全プローブが走ること。** `practice` config の追加で既存configのデフォルトが壊れていないことを検証する。→ Task 1 で担保。

---

## Task 1: ハーネスに `practice` config と CUBLAS opt-out を追加

**Files:**
- Modify: `reproducibility/scripts/run_on_host.sh`(CONFIG の case 文、ヘッダコメントの config 一覧、unknown CONFIG のエラーメッセージ、IMAGE=gpu の CUBLAS ブロック)
- Modify: `reproducibility/Makefile`(config 一覧のコメント)
- Modify: `reproducibility/README.md`(config 一覧)
- Test: `reproducibility/tests/test_runner.py`

**Interfaces:**
- Consumes: `reprobench.runner.main` が `--probes` 省略時に環境変数 `REPRO_PROBES` を参照する既存の仕組み(`s10batch` が既に利用)。
- Produces:
  - `reprobench.runner.all_probe_names() -> list[str]` — 登録済み全プローブ名
  - `reprobench.runner.selected_probe_names(argv_probes: list[str] | None) -> list[str]` — `argv_probes` が `None` のとき `REPRO_PROBES`(カンマ区切りの接頭辞)を参照して絞り込む
  - config ラベル `practice`。環境変数は `threadsmax` と同じスレッド数固定を踏襲し、`REPRO_PROBES=S11,S12,S13,S14,S15` を設定する。IMAGE=gpu のとき `CUBLAS_WORKSPACE_CONFIG` を**設定しない**。出力は `results/<host>__<image>__practice.json`。

- [ ] **Step 1: `REPRO_PROBES` が未設定なら全プローブが走ることを確認する失敗テストを書く**

```python
def test_repro_probes_unset_runs_all_probes(monkeypatch):
    monkeypatch.delenv("REPRO_PROBES", raising=False)
    assert runner.selected_probe_names(argv_probes=None) == runner.all_probe_names()


def test_repro_probes_accepts_comma_separated_prefixes(monkeypatch):
    monkeypatch.setenv("REPRO_PROBES", "S11,S12")
    names = runner.selected_probe_names(argv_probes=None)
    assert all(n.startswith(("S11", "S12")) for n in names)
    assert any(n.startswith("S11") for n in names)
```

`selected_probe_names` / `all_probe_names` が現在の `runner.py` に無ければ、既存の絞り込みロジックをこの2関数に切り出してから使う。切り出しによって既存の挙動を変えないこと。

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `cd reproducibility && docker run --rm -v "$PWD":/w -w /w reprobench:base pytest tests/test_runner.py -q -W error`
Expected: FAIL(`selected_probe_names` が無い、またはカンマ区切りが未対応)

- [ ] **Step 3: `reprobench/runner.py` に `selected_probe_names(argv_probes: list[str] | None) -> list[str]` と `all_probe_names() -> list[str]` を実装する**

`REPRO_PROBES` はカンマ区切りの接頭辞リストとして解釈する(現在の `s10batch` は `"S10"` 単体を渡しており、カンマ区切り対応はその上位互換になる)。`--probes` が明示された場合は環境変数を無視する既存の優先順位を保つ。

- [ ] **Step 4: テストが通ることを確認する**

Run: Step 2 と同じコマンド
Expected: PASS

- [ ] **Step 5: `run_on_host.sh` に `practice` の case を追加する**

`s10batch` の case(140-166行付近)を雛形にする。スレッド数は `threadsmax` と同じ解決ロジックを使い、`add_env "REPRO_PROBES" "S11,S12,S13,S14,S15"` を設定する。ヘッダコメントの config 一覧(13-20行付近)と unknown CONFIG のエラーメッセージ(167行付近)にも `practice` を追加する。

- [ ] **Step 6: IMAGE=gpu の CUBLAS ブロックを config 条件付きにする**

現在188行付近は `if [ "$IMAGE" = "gpu" ]; then add_env "CUBLAS_WORKSPACE_CONFIG" ...` と無条件である。`practice` のときだけ `CUBLAS_WORKSPACE_CONFIG` を設定しないようにする(`--gpus all` は `practice` でも必要なので残す)。**既存6configと `s10batch` の経路は変更しないこと。**

この opt-out は S13 の測定対象そのものである。設定しない状態で `use_deterministic_algorithms(True)` を使うと PyTorch が例外を送出することが期待される。

- [ ] **Step 7: dry-run で6パターンを確認する**

Run:
```bash
cd reproducibility
for c in threadsmax s10batch practice; do
  for i in base gpu; do
    bash scripts/run_on_host.sh anakin $i $c --dry-run 2>&1 | grep -E "REPRO_PROBES|CUBLAS_WORKSPACE_CONFIG|would run: docker run" | head -3
  done
done
```
Expected:
- `threadsmax`: `REPRO_PROBES` が出ない。gpu のとき `CUBLAS_WORKSPACE_CONFIG` が出る
- `s10batch`: `REPRO_PROBES=S10`。gpu のとき `CUBLAS_WORKSPACE_CONFIG` が出る
- `practice`: `REPRO_PROBES=S11,S12,S13,S14,S15`。gpu でも `CUBLAS_WORKSPACE_CONFIG` が**出ない**

- [ ] **Step 8: `make test` が通ることを確認する**

Run: `cd reproducibility && make test`
Expected: PASS(件数は 212 passed 以上、2 skipped)

- [ ] **Step 9: コミット**

```bash
git add reproducibility/scripts/run_on_host.sh reproducibility/reprobench/runner.py \
        reproducibility/tests/test_runner.py reproducibility/Makefile reproducibility/README.md
git commit -- reproducibility/scripts/run_on_host.sh reproducibility/reprobench/runner.py \
        reproducibility/tests/test_runner.py reproducibility/Makefile reproducibility/README.md
```
メッセージ: `feat(repro): practice configとCUBLAS_WORKSPACE_CONFIGのopt-out経路を追加`

---

## Task 2: S11 乱数消費順序のリファクタ耐性

**Files:**
- Create: `reproducibility/reprobench/probes/s11_rng_order.py`
- Modify: `reproducibility/reprobench/probes/__init__.py`(1行追加)
- Test: `reproducibility/tests/test_s11_rng_order.py`

**Interfaces:**
- Consumes: `reprobench.registry.register` / `ProbeResult`、`reprobench.digest.digest`
- Produces:
  - `REFACTOR_PATTERNS: tuple[str, ...]` = `("baseline", "loop_split", "swap_order", "drop_dead_call", "early_return")`
  - `IMPL_NAMES: tuple[str, ...]` = `("naive", "spawn")`
  - `generate(impl: str, pattern: str, seed: int) -> tuple[list[float], int]` — 生成した値の列と消費した乱数の個数を返す純粋関数。torch等に依存しない。
  - プローブ `S11_rng_consumption_order`(`requires=()`)

- [ ] **Step 1: 純粋関数の失敗テストを書く**

```python
def test_generate_returns_values_and_consumption_count():
    values, n_consumed = generate("naive", "baseline", seed=20261014)
    assert len(values) > 0
    assert n_consumed > 0

def test_naive_baseline_is_reproducible_for_same_seed():
    assert generate("naive", "baseline", 20261014) == generate("naive", "baseline", 20261014)

def test_all_impl_pattern_combinations_are_generatable():
    for impl in IMPL_NAMES:
        for pattern in REFACTOR_PATTERNS:
            values, n = generate(impl, pattern, 20261014)
            assert len(values) > 0

def test_spawn_impl_uses_independent_streams_per_component():
    # 頑健版は成分ごとに別ストリームを使う。成分数ぶんの子SeedSequenceが
    # 生成されることを、消費した乱数の個数が素朴版と異なることで間接的に見る。
    _, n_naive = generate("naive", "baseline", 20261014)
    _, n_spawn = generate("spawn", "baseline", 20261014)
    assert n_naive > 0 and n_spawn > 0
```

**このテストは「頑健版が壊れない」をassertしてはならない。** 壊れるかどうかは実験の結果であり、テストで固定すると実験の意味が消える(Review Focus 1)。

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `cd reproducibility && docker run --rm -v "$PWD":/w -w /w reprobench:base pytest tests/test_s11_rng_order.py -q -W error`
Expected: FAIL(`s11_rng_order` が無い)

- [ ] **Step 3: `generate` と4リファクタパターンを実装する**

`naive` は単一の `numpy.random.Generator` を使い回す。`spawn` は `numpy.random.SeedSequence(seed).spawn(n)` で成分ごとに独立した `Generator` を作る。

4パターンは仕様書の表の通り。`baseline` に対して、`loop_split` はループを2つに分ける、`swap_order` は2つの呼び出しの順序を入れ替える、`drop_dead_call` は使っていない呼び出しを消す、`early_return` は条件分岐で消費数を変える。

**各パターンが「同じ計算結果を意図したリファクタ」であること。** 値そのものの意味を変える変更(分布を変える、個数を変える)はリファクタではないので避ける。

- [ ] **Step 4: テストが通ることを確認する**

Run: Step 2 と同じコマンド
Expected: PASS

- [ ] **Step 5: プローブ `S11_rng_consumption_order` を実装し、`probes/__init__.py` に登録する**

`meta` に記録する内容: 実装×パターンの各組み合わせについて、値列のdigest、**同じ実装のbaselineとの一致/不一致**、消費した乱数の個数。プローブ全体の `digest` は (impl, pattern) → digest の順序付きマッピングから取る。

比較は常に同じ実装のbaselineとの間で行う。`naive` と `spawn` の値そのものは比較しない(別ストリームなので一致する理由がない)。

- [ ] **Step 6: プローブのテストを追加して通す**

```python
def test_probe_records_all_combinations_against_own_baseline():
    r = run_probe(registry.get("S11_rng_consumption_order"))
    assert isinstance(r, ProbeResult)
    for impl in IMPL_NAMES:
        for pattern in REFACTOR_PATTERNS:
            assert r.meta["results"][impl][pattern]["matches_baseline"] in (True, False)
```

Run: Step 2 と同じコマンド
Expected: PASS

- [ ] **Step 7: コミット**

```bash
git add reproducibility/reprobench/probes/s11_rng_order.py \
        reproducibility/reprobench/probes/__init__.py \
        reproducibility/tests/test_s11_rng_order.py
git commit -- reproducibility/reprobench/probes/s11_rng_order.py \
        reproducibility/reprobench/probes/__init__.py \
        reproducibility/tests/test_s11_rng_order.py
```
メッセージ: `feat(repro): S11 乱数消費順序のリファクタ耐性プローブを追加`

---

## Task 3: S14 DataLoader の worker seeding

**Files:**
- Create: `reproducibility/reprobench/probes/s14_dataloader_seed.py`
- Modify: `reproducibility/reprobench/probes/__init__.py`
- Test: `reproducibility/tests/test_s14_dataloader_seed.py`

**Interfaces:**
- Produces:
  - `RNG_SOURCES: tuple[str, ...]` = `("torch", "numpy")`
  - `WORKER_COUNTS: tuple[int, ...]` = `(0, 2, 4)`
  - プローブ `S14_dataloader_worker_seed`(`requires=("torch",)`)
  - `meta["results"]` の構造は `{rng_source: {str(num_workers): cell}}`。**worker数はJSONキーとして文字列化する**(JSONは整数キーを持てないため)。

- [ ] **Step 1: 失敗テストを書く**

```python
def test_probe_records_each_rng_source_and_worker_count():
    r = run_probe(registry.get("S14_dataloader_worker_seed"))
    if isinstance(r, SkippedProbe):
        assert "missing requirement" in r.reason
        return
    for src in RNG_SOURCES:
        for n in WORKER_COUNTS:
            cell = r.meta["results"][src][str(n)]
            assert "digest" in cell
            assert "duplicate_sample_count" in cell or "worker_start_failed" in cell
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `cd reproducibility && docker run --rm -v "$PWD":/w -w /w reprobench:base pytest tests/test_s14_dataloader_seed.py -q -W error`
Expected: FAIL

- [ ] **Step 3: プローブを実装する**

`torch.utils.data.DataLoader` で `num_workers` を 0/2/4 と変え、各サンプルに乱数由来の拡張を適用した結果列を集める。拡張の乱数源を `torch` 版と `numpy` 版の2通り用意する。`worker_init_fn` は**指定しない**(指定しないことが測定対象のため)。

PyTorchはworkerごとに torch の RNG を再シードするが numpy の RNG は再シードしない。`numpy` 版では複数workerが同じ乱数列を使い、拡張結果が重複することが期待される。`duplicate_sample_count` にその個数を記録する。

**worker起動に失敗した場合(共有メモリ不足など)は例外を握りつぶさず、`worker_start_failed` に例外のメッセージを記録して次の条件へ進む**(Review Focus 4)。測定全体を止めない。

- [ ] **Step 4: テストが通ることを確認する**

Run: Step 2 と同じコマンド
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add reproducibility/reprobench/probes/s14_dataloader_seed.py \
        reproducibility/reprobench/probes/__init__.py \
        reproducibility/tests/test_s14_dataloader_seed.py
git commit -- reproducibility/reprobench/probes/s14_dataloader_seed.py \
        reproducibility/reprobench/probes/__init__.py \
        reproducibility/tests/test_s14_dataloader_seed.py
```
メッセージ: `feat(repro): S14 DataLoaderのworker seedingプローブを追加`

---

## Task 4: S12 cudnn設定 と S13 CUBLAS_WORKSPACE_CONFIG

**Files:**
- Create: `reproducibility/reprobench/probes/s12_cudnn.py`
- Create: `reproducibility/reprobench/probes/s13_cublas_workspace.py`
- Modify: `reproducibility/reprobench/probes/__init__.py`
- Test: `reproducibility/tests/test_s12_cudnn.py`, `reproducibility/tests/test_s13_cublas_workspace.py`

**Interfaces:**
- Produces:
  - `CUDNN_SETTINGS: tuple[tuple[bool, bool], ...]` — `(benchmark, deterministic)` の2×2
  - プローブ `S12_cudnn_benchmark_deterministic`(`requires=("torch","cuda")`)
  - プローブ `S13_cublas_workspace_config`(`requires=("torch","cuda")`)

- [ ] **Step 1: skip経路と記録内容の失敗テストを書く**

```python
def test_s12_skips_without_cuda_and_records_within_process_stability():
    r = run_probe(registry.get("S12_cudnn_benchmark_deterministic"))
    if isinstance(r, SkippedProbe):
        assert "missing requirement" in r.reason
        return
    for bench, det in CUDNN_SETTINGS:
        cell = r.meta["results"][f"benchmark={bench},deterministic={det}"]
        assert "digest" in cell
        assert cell["stable_within_process"] in (True, False)

def test_s13_records_exception_or_result_without_swallowing():
    r = run_probe(registry.get("S13_cublas_workspace_config"))
    if isinstance(r, SkippedProbe):
        assert "missing requirement" in r.reason
        return
    assert r.meta["cublas_workspace_config"] in (None, "") or isinstance(
        r.meta["cublas_workspace_config"], str)
    for op in r.meta["ops"]:
        cell = r.meta["ops"][op]
        assert ("digest" in cell) or ("raised" in cell)
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `cd reproducibility && docker run --rm -v "$PWD":/w -w /w reprobench:base pytest tests/test_s12_cudnn.py tests/test_s13_cublas_workspace.py -q -W error`
Expected: FAIL

- [ ] **Step 3: S12 を実装する**

畳み込みを含む小さなモデルの順伝播と逆伝播を、`torch.backends.cudnn.benchmark` × `torch.backends.cudnn.deterministic` の2×2で実行する。

**各設定でプロセス内反復を行い `stable_within_process` を記録する。** `benchmark=True` は実行時のタイミング計測でアルゴリズムを選ぶため、同一プロセス内の反復でも結果が変わりうる。これが本プローブの要点である。

- [ ] **Step 4: S13 を実装する**

`torch.use_deterministic_algorithms(True)` を有効にした状態で、決定的実装が CUBLAS workspace を要求する演算(行列積を含む複数)を実行する。`CUBLAS_WORKSPACE_CONFIG` の現在値を `os.environ.get` で読んで `meta` に記録する(このプローブは環境変数を設定しない。設定の有無は config 側が決める)。

**例外が出ることが期待挙動である。** 演算ごとに try/except で囲み、例外が出たら `raised` に例外クラス名とメッセージを記録し、出なければ `digest` を記録する。**例外を握りつぶして正常終了したように見せてはならない**(Review Focus 3)。

- [ ] **Step 5: テストが通ることを確認する**

Run: Step 2 と同じコマンド
Expected: PASS(CUDA の無い base イメージでは両プローブとも `SkippedProbe` を返し、skip経路の分岐が通る)

- [ ] **Step 6: コミット**

```bash
git add reproducibility/reprobench/probes/s12_cudnn.py \
        reproducibility/reprobench/probes/s13_cublas_workspace.py \
        reproducibility/reprobench/probes/__init__.py \
        reproducibility/tests/test_s12_cudnn.py reproducibility/tests/test_s13_cublas_workspace.py
git commit -- reproducibility/reprobench/probes/s12_cudnn.py \
        reproducibility/reprobench/probes/s13_cublas_workspace.py \
        reproducibility/reprobench/probes/__init__.py \
        reproducibility/tests/test_s12_cudnn.py reproducibility/tests/test_s13_cublas_workspace.py
```
メッセージ: `feat(repro): S12 cudnn設定とS13 CUBLAS_WORKSPACE_CONFIGのプローブを追加`

---

## Task 5: S15 反復の生データ記録

**Files:**
- Create: `reproducibility/reprobench/probes/s15_repeat_raw.py`
- Modify: `reproducibility/reprobench/probes/__init__.py`
- Test: `reproducibility/tests/test_s15_repeat_raw.py`

**Interfaces:**
- Produces:
  - プローブ `S15_index_add_repeats`(`requires=("torch","cuda")`)
  - プローブ `S15_kmeans_repeats`(`requires=("sklearn",)`)
  - 両者とも `meta["round_digests"]` に各回のdigestを**列として**記録する。

記事が2箇所で「生データは results/ には無く、プロジェクトの判断記録にのみ残っている」と認めている箇所を塞ぐためのプローブである。

- [ ] **Step 1: 失敗テストを書く**

```python
def test_index_add_repeats_records_per_round_digests():
    r = run_probe(registry.get("S15_index_add_repeats"))
    if isinstance(r, SkippedProbe):
        assert "missing requirement" in r.reason
        return
    assert len(r.meta["round_digests"]) == r.meta["n_rounds"]
    assert r.meta["n_rounds"] >= 5

def test_kmeans_repeats_records_per_round_digests():
    r = run_probe(registry.get("S15_kmeans_repeats"))
    if isinstance(r, SkippedProbe):
        assert "missing requirement" in r.reason
        return
    assert len(r.meta["round_digests"]) == r.meta["n_rounds"]
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `cd reproducibility && docker run --rm -v "$PWD":/w -w /w reprobench:base pytest tests/test_s15_repeat_raw.py -q -W error`
Expected: FAIL

- [ ] **Step 3: 2つのプローブを実装する**

`S15_index_add_repeats` は重複インデックスへの `index_add_` を5回実行し、各回のdigestと、1回目を基準とした最大絶対差を記録する。既存の `s8_torch_gpu.py` の `S8_index_add_atomics` と同じ計算内容にすること(同じ現象の生データを取るのが目的のため)。

`S15_kmeans_repeats` は KMeans のフィットを繰り返し、各回の重心のdigestを記録する。既存の `s5_sklearn.py` の `S5_kmeans_centers` と同じデータ・同じパラメータを使う。

`index_add_` は意図的に非決定であるため、ここで得られる値は記事が現在引用している「最大絶対差 8.24×10⁻⁴」とは異なる。これは期待された挙動であり、記事では「別の機会に測った別の5回」として併記する。

- [ ] **Step 4: テストが通ることを確認する**

Run: Step 2 と同じコマンド
Expected: PASS

- [ ] **Step 5: `make test` 全体が通ることを確認する**

Run: `cd reproducibility && make test`
Expected: PASS(212 passed 以上)

- [ ] **Step 6: コミット**

```bash
git add reproducibility/reprobench/probes/s15_repeat_raw.py \
        reproducibility/reprobench/probes/__init__.py \
        reproducibility/tests/test_s15_repeat_raw.py
git commit -- reproducibility/reprobench/probes/s15_repeat_raw.py \
        reproducibility/reprobench/probes/__init__.py \
        reproducibility/tests/test_s15_repeat_raw.py
```
メッセージ: `feat(repro): S15 反復の生データ記録プローブを追加`

---

## Task 6: 予測の事前登録

**Files:**
- Create: `reproducibility/PRACTICE_AXES_PREDICTION.md`

**このタスクは Task 7(実測)より前に完了していなければならない。** 順序が逆になると事前登録として成立しない。

- [ ] **Step 1: 予測ファイルを書く**

仕様書の「事前登録の方針」に従い、**不確実な3点のみ**を予測として記録する。

1. `SeedSequence.spawn()` を使った頑健版が、リファクタ4パターン**全て**を生き延びるか
2. `cudnn.benchmark=True` が**同一プロセス内の反復**で実際に揺れるか
3. `CUBLAS_WORKSPACE_CONFIG` 未設定時に例外が出る演算の範囲

**自明として登録しないものも明記する**: 素朴版がリファクタで壊れること、`worker_init_fn` 未指定時に numpy の乱数が重複すること。理由(文書から予想がつく)も書く。

各予測には「外れた場合に何が言えるか」も書く。既存の `PREREGISTERED_PREDICTION.md` と `BATCH_INVARIANCE_REDESIGN.md` の書式に倣う。

- [ ] **Step 2: 実測がまだ行われていないことを確認する**

Run: `cd reproducibility && ls results/*__*__practice.json 2>/dev/null || echo "未実測(OK)"`
Expected: `未実測(OK)`

- [ ] **Step 3: コミット**

```bash
git add reproducibility/PRACTICE_AXES_PREDICTION.md
git commit -- reproducibility/PRACTICE_AXES_PREDICTION.md
```
メッセージ: `docs(repro): 実務4軸の予測を実測前に事前登録`

- [ ] **Step 4: コミットのタイムスタンプを記録する**

Run: `git log -1 --format='%H %ad' --date=iso -- reproducibility/PRACTICE_AXES_PREDICTION.md`
記事で引用するため、出力をメモしておく。

---

## Task 7: 実測(7ラン)

**Files:**
- Create: `reproducibility/results/{anakin,blade,dgxspark,local}__base__practice.json`
- Create: `reproducibility/results/{anakin,blade,dgxspark}__gpu__practice.json`
- Modify: `reproducibility/results/report.json`, `reproducibility/results/matrix.md`(再集計)

- [ ] **Step 1: 実測前に既存ファイルのSHA-256を記録する**

Run:
```bash
cd reproducibility && find results -type f | sort | xargs shasum -a 256 > /tmp/results-before.sha256
wc -l < /tmp/results-before.sha256
```
Expected: 35ラン分 + report.json + matrix.md + nvidia-smi.csv 3件 の行数

- [ ] **Step 2: dry-run で7ラン分のコマンドを確認する**

Run:
```bash
cd reproducibility
for h in anakin blade dgxspark local; do bash scripts/run_on_host.sh $h base practice --dry-run | tail -2; done
for h in anakin blade dgxspark;       do bash scripts/run_on_host.sh $h gpu  practice --dry-run | tail -2; done
```
Expected: 7件とも `results/<host>__<image>__practice.json` を出力する内容になっている

- [ ] **Step 3: 実測を実行する**

Run:
```bash
cd reproducibility
for h in anakin blade dgxspark local; do bash scripts/run_on_host.sh $h base practice; done
for h in anakin blade dgxspark;       do bash scripts/run_on_host.sh $h gpu  practice; done
```
Expected: 7ファイルが `results/` に増える。失敗したホストがあればエラーをそのまま記録し、握りつぶさない。

- [ ] **Step 4: 既存ファイルが1バイトも変わっていないことを確認する**

Run:
```bash
cd reproducibility && shasum -a 256 -c /tmp/results-before.sha256 2>&1 | grep -v ': OK$' || echo "既存ファイルは全て不変(OK)"
git status --short reproducibility/results/
```
Expected: 既存ファイルに `M` が付いていない。新規7件のみ。**変化していたら事故なので直ちに `git checkout` で戻して報告する。**

- [ ] **Step 5: 再集計する**

Run: `cd reproducibility && make report`
Expected: `report.json` のラベル数が 35 → 42 になる

- [ ] **Step 6: 結果を読んで予測を採点する**

`results/*__*__practice.json` を jq か python で読み、Task 6 の予測3点それぞれについて支持/反証を判定する。**予測に合わせて結果を解釈しない。外れた予測は外れたまま記録する。**

判定結果を `PRACTICE_AXES_PREDICTION.md` に「実測結果」節として追記する。

- [ ] **Step 7: コミット**

```bash
git add reproducibility/results reproducibility/PRACTICE_AXES_PREDICTION.md
git commit -- reproducibility/results reproducibility/PRACTICE_AXES_PREDICTION.md
```
メッセージ: `feat(repro): 実務4軸を7ラン実測(42ラベル)、予測を採点`

---

## Task 8: 記事の章番号を繰り下げる

**Files:**
- Modify: `articles/reproducibility-of-randomness.md`

内容の追加より先に、機械的な番号変更だけを独立したコミットにする。混ぜると差分が読めなくなる。

- [ ] **Step 1: 変更前の参照数を記録する**

Run: `grep -o "第[0-9]*章" articles/reproducibility-of-randomness.md | sort | uniq -c | sort -rn`
Expected(現状): 第6章18 / 第4章6 / 第2章5 / 第7章4 / 第8章3 / 第5章3 / 第9章1 / 第3章1 = 計41件

- [ ] **Step 2: 見出しと参照を繰り下げる**

`## 8. 判定フローと納品文言テンプレート` → `## 9.`、`## 9. まとめ` → `## 10.`。
本文中の「第8章」3件を「第9章」へ、「第9章」1件を「第10章」へ。

**第1〜7章の番号と参照は変更しない。** 新章(第8章)はこのタスクでは作らない。

- [ ] **Step 3: 変更後の参照数を確認する**

Run: `grep -o "第[0-9]*章" articles/reproducibility-of-randomness.md | sort | uniq -c | sort -rn`
Expected: 総数41件のまま。第8章0件、第9章3件、第10章1件。第1〜7章への参照は変更前と同数。

- [ ] **Step 4: 存在しない章への参照が無いことを確認する**

Run:
```bash
cd /Users/ljmb20260017/Documents/zenn-contents
python3 -c "
import re
t=open('articles/reproducibility-of-randomness.md').read()
heads={int(m) for m in re.findall(r'^## ([0-9]+)\.', t, re.M)}
refs={int(m) for m in re.findall(r'第([0-9]+)章', t)}
print('見出し:', sorted(heads)); print('参照:', sorted(refs))
print('存在しない章への参照:', sorted(refs - heads) or 'なし(OK)')
"
```
Expected: `存在しない章への参照: なし(OK)`(この時点では第8章の見出しがまだ無いので、第8章への参照も0件のはず)

- [ ] **Step 5: コミット**

```bash
git add articles/reproducibility-of-randomness.md
git commit -- articles/reproducibility-of-randomness.md
```
メッセージ: `refactor(article): 新章挿入に備えて第8章以降を繰り下げ`

---

## Task 9: 新章の追加と既存章への実測結果の組み込み

**Files:**
- Modify: `articles/reproducibility-of-randomness.md`

**Interfaces:**
- Consumes: Task 7 が生成した `results/*__*__practice.json` の実測値、Task 6 の事前登録コミットのハッシュとタイムスタンプ

- [ ] **Step 1: 新第8章「コードを変えると何が壊れるか」を追加する**

第7章(LLM)の直後、第9章(判定フロー)の直前に挿入する。

既存の実験章と同じ4部構成(理論 / 実験設定 / 結果 / 理論との照合)。冒頭で軸の転換を明示する:

> ここまでの章は一貫して「同じコード・違う環境」を扱ってきた。本章は「違うコード・同じ環境」である。

S11 の結果を、素朴版と頑健版のコードブロックを並べる形で示す。**図は作らない**(コードで示す方が明快であり、かつ図を足すと直前に出現順へ揃えた図番号が再び崩れるため)。

事前登録した予測1(頑健版が4パターン全てを生き延びるか)の採点を書く。外れていれば外れたまま書く。

- [ ] **Step 2: 第5章に S14(DataLoader worker seeding)を追加する**

第5章は「同じ入力のつもりが違う」章であり、既に並列ワーカーの完了順を扱っている。workerごとに numpy の RNG が再シードされず拡張が重複する現象はその発展形である。

`torch` の RNG を使う実装と `numpy` の RNG を使う実装で**片方だけが壊れる**ことを示す。

- [ ] **Step 3: 第6章G4に S12・S13 を追加する**

S12 は `benchmark=True` がプロセス内反復でも揺れるかどうかを中心に書く(予測2の採点)。S13 は例外が出た演算の範囲を書く(予測3の採点)。例外が出なかった場合もそのまま書く。

- [ ] **Step 4: S15 の結果を該当箇所に反映する**

記事が「生データは results/ には無く、プロジェクトの判断記録にのみ残っている」と書いている2箇所を、S15 の実測への参照に置き換える。`index_add_` については、現在引用している値と今回の値が**別の機会に測った別の5回**であることを明記する。非決定性の実例が2組になる。

- [ ] **Step 5: ラン数の記述を42に更新する**

Run: `grep -n "35ラン\|35ラベル" articles/reproducibility-of-randomness.md`
該当箇所を確認し、プロジェクト総ラン数を指すものだけを42に更新する。

**注意**: 第5章の「32ラン」は `S0_glob_order` の観測数であり総ラン数ではない(`practice` config は S11〜S15 のみを走らせるため S0 の観測は増えない)。**変更しないこと。** 同様に第10章の「最初の30ラン」は執筆当時の時系列記述なので変更しない。

- [ ] **Step 6: 章と参照の整合を確認する**

Run: Task 8 Step 4 と同じスクリプト
Expected: 見出しに 8 が現れ、`存在しない章への参照: なし(OK)`

- [ ] **Step 7: コミット**

```bash
git add articles/reproducibility-of-randomness.md
git commit -- articles/reproducibility-of-randomness.md
```
メッセージ: `feat(article): 新第8章とS12〜S15の実測結果を追加`

---

## Task 10: 判定フロー・テンプレート・正確性の修正

**Files:**
- Modify: `articles/reproducibility-of-randomness.md`

- [ ] **Step 1: 判定フローにコード同一性の前提を追加する**

フローの**手前**に前提条件として置く(判定項目としてフローの中に入れない。レベルと並列な軸ではないため)。

> 判定は、同一のコードに対して行う。乱数の消費順序を変えるリファクタは、L0 の計算であっても結果を変える。コードを変更したら、レベル判定は無効になり、やり直しになる。

納品文言テンプレートの全レベルに共通の但し書きとして「本保証はコードを変更しない限りにおいて有効である」を入れる。

- [ ] **Step 2: L2 / L3 の定義を書き換える**

第4章の表と判定フローの両方に、以下の趣旨を入れる。

> L0・L1・L4 は計算の性質で決まる。L2 と L3 は証拠の状態を表す。L2 は「文書化された保証が無く、かつ試した範囲では割れなかった」という主張であり、反証可能である。新しい軸で割れた時点で、その計算は L3 に移る。L2 判定には、自分が何を試したかの記録が伴わなければ意味がない。

- [ ] **Step 3: L1 と L3 のテンプレートを修正する**

**L1**: 現在の「[ライブラリ名]==[X.Y.Z]に固定する限り、実行環境を問わずビット単位で一致することを約束する」は F2(乱数列の一致と計算結果の一致は別の保証)と矛盾する。範囲を**乱数列**に限定し、「その乱数を使った後段の計算は本保証の範囲外であり、L3 の条件が別途かかる」を加える。

**L3**: 「これらのいずれかが変われば一致しない」→「これらのいずれかが変われば、**一致は保証しない**」。

- [ ] **Step 4: 撤回の痕跡2箇所を書き換える**

Run: `grep -n "当初は\|解釈は撤回する" articles/reproducibility-of-randomness.md`

該当2箇所(旧615行・旧875行付近)を、**現在の立場だけを述べる形**に書き換える。読者は撤回対象の主張を見ていない。

**610行付近の「『GPUは非決定的』という大雑把な理解は誤りだった」は変更しない。** これは記事自身の旧稿ではなく世間の通念を正す文である。

- [ ] **Step 5: temperature=0 の見出しを変更する**

`### temperature=0は存在しない` → `### temperature=0の意味は実装ごとに違う`

本文の内容(transformers はエラー、vLLM は内部で greedy に読み替え)は既に正確なので、見出しのみ変更する。

- [ ] **Step 6: 第1章に「なぜビット一致が要るのか / いつ allclose で足りるのか」を追加する**

**両方書く。** ML の仕事の大半はビット一致を必要としない。それを認めたうえで必要になる場面を挙げる方が、記事は押し付けがましくなくなり、読者が自分の状況を判定できる。

ビット一致が要る場面として挙げるもの: 納品契約に「同じ入力なら同じ結果」と書いた場合、allclose の閾値を正当化できない回帰テスト、規制のある領域の監査、原因の二分探索によるデバッグ、計算結果のキャッシュの正当性。

- [ ] **Step 7: 事前登録という語の定義を追加する**

公開リポジトリのコミット履歴で前後関係が検証できて初めて成立する語である。定義を添え、リポジトリURLのプレースホルダを示す。

現時点で事前登録は3件になる(第6章G2の決定実験、第7章のbatch実験、新第8章ほかの実務4軸)。件数の記述がある箇所を更新する。

- [ ] **Step 8: Zenn記法の健全性を確認する**

Run:
```bash
cd /Users/ljmb20260017/Documents/zenn-contents
python3 -c "
t=open('articles/reproducibility-of-randomness.md').read().split('\n')
op=sum(1 for l in t if l.startswith(':::details')); cl=sum(1 for l in t if l.strip()==':::')
print(f'details 開{op}/閉{cl}', 'OK' if op==cl else '不一致!')
"
grep -c '^```' articles/reproducibility-of-randomness.md
```
Expected: details の開閉が一致。コードフェンスの数が偶数。

- [ ] **Step 9: コミット**

```bash
git add articles/reproducibility-of-randomness.md
git commit -- articles/reproducibility-of-randomness.md
```
メッセージ: `fix(article): 判定フローの前提・L2/L3の定義・テンプレート文言・撤回痕跡を修正`

---

## Task 11: 最終検証とレビュー

**Files:**
- 変更なし(検証のみ。レビューで見つかった問題の修正は必要に応じて追加コミット)

- [ ] **Step 1: 全テストを実行する**

Run: `cd reproducibility && make test`
Expected: PASS(212 passed 以上、2 skipped)

- [ ] **Step 2: 既存の実測データが不変であることを再確認する**

Run: `cd reproducibility && shasum -a 256 -c /tmp/results-before.sha256 2>&1 | grep -v ': OK$' || echo "不変(OK)"`
Expected: `不変(OK)`

`/tmp/results-before.sha256` が残っていない場合(別セッション・別エージェントで実行したなど)は、git で代替する:

```bash
cd /Users/ljmb20260017/Documents/zenn-contents
BASE=$(git log --format=%H -1 --before="Task 7の実測コミット" -- reproducibility/results) # 実測直前のコミット
git diff --stat $BASE -- reproducibility/results | grep -v "practice.json\|report.json\|matrix.md" || echo "既存ファイルの変更なし(OK)"
```
Expected: `practice.json` 7件と `report.json` / `matrix.md` 以外に変更が出ない

- [ ] **Step 3: 章・図・参照の整合を確認する**

Run:
```bash
cd /Users/ljmb20260017/Documents/zenn-contents
python3 -c "
import re, os
t=open('articles/reproducibility-of-randomness.md').read()
heads={int(m) for m in re.findall(r'^## ([0-9]+)\.', t, re.M)}
refs={int(m) for m in re.findall(r'第([0-9]+)章', t)}
print('存在しない章への参照:', sorted(refs-heads) or 'なし(OK)')
figs=re.findall(r'\(/images/reproducibility-of-randomness/([^)]+)\)', t)
missing=[p for p in figs if not os.path.exists('images/reproducibility-of-randomness/'+p)]
print('欠損画像:', missing or 'なし(OK)')
nums=re.findall(r'!\[(F[0-9ab]+)\.', t)
print('図の出現順:', nums)
"
```
Expected: 存在しない章への参照なし、欠損画像なし、図の出現順が F1→F2→F3→F4→F5→F6→F7→F8→F9a→F9b→F10

- [ ] **Step 4: 記事を見ていない新しいエージェントにレビューを依頼する**

章番号が動き、新章が入り、判定フローの前提が変わったため、整合が崩れる余地が大きい。**Round 1 の指摘リストは渡さない**(先入観なしに読ませる)。

レビュー観点: 章をまたぐ整合、新章と既存章の重複、事前登録3件の扱いの一貫性、L2/L3 の新定義が判定フローと矛盾していないか、新規実測の数値が `results/*.json` と一致するか、図と本文の対応。

- [ ] **Step 5: レビューの指摘を検証してから修正する**

**指摘をそのまま信用しない。** 過去のレビューで、実在しない誤記の指摘や、記事が正しく監査が誤っていた例が実際にあった。各指摘について実データまたは一次ソースで裏を取ってから修正する。裏が取れないものは「疑わしい」として扱い、誤りと断定しない。

- [ ] **Step 6: 重大な指摘がゼロになるまで Step 4〜5 を繰り返す**

収束条件: 致命的・重大の指摘がゼロ。

- [ ] **Step 7: 最終コミット**

```bash
git add articles/reproducibility-of-randomness.md
git commit -- articles/reproducibility-of-randomness.md
```
メッセージ: `fix(article): 最終レビューの指摘を反映`
