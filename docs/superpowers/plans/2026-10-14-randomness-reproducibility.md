# 乱数の再現性 記事 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 「同じ seed・同じ入力でも結果の再現性をどこまで約束できるか」を、4ホスト×12コンテナの実測と一次情報調査で裏づけた Zenn 長編記事1本を、再実行可能な実測ハーネスごと公開できる状態にする。

**Architecture:** `reprobench` という単一 Python パッケージに、環境非依存の基盤(ダイジェスト計算・環境採取・プローブ登録)と、S0〜S9 のプローブを置く。プローブは「引数なしで呼ぶと `ProbeResult` を返す関数」に統一し、`requires` 宣言でその環境に無い依存(torch / CUDA / lightgbm / transformers)を自動スキップする。ランナーが1プロセス=1 JSON を吐き、比較器が全 JSON を突き合わせて一致マトリクスと数値差分を出す。図と記事はその比較結果だけを入力にする。

**Tech Stack:** Python 3.11(一部 3.13 / 1.26系 numpy)、NumPy、scikit-learn、LightGBM、PyTorch、transformers(Qwen2.5-0.5B-Instruct)、Docker、matplotlib、手書き SVG

**Spec:** `docs/superpowers/specs/2026-10-14-randomness-reproducibility-design.md`

## Global Constraints

- **計測は必ずコンテナ内で行う。ホスト上での直接実行は不可。** ユニットテストも `docker run` 経由で回す。
- 計測ホストは4つ: MacBook(aarch64 / Docker Desktop)、anakin(x86_64 / AVX2まで / RTX 4090)、blade(x86_64 / AVX-512あり / RTX PRO 6000)、dgxspark(aarch64 / SVE2 / GB10)。ssh のホスト名はこの4つをそのまま使う。
- 共用マシンのため、各ホストの占有は実測10分以内・GPUメモリ8GB以下に収める。
- LLM モデルは `Qwen2.5-0.5B-Instruct` 固定。
- 成果物パス: 記事 `articles/reproducibility-of-randomness.md`、記事用PNG `images/reproducibility-of-randomness/`、ハーネス `reproducibility/`、調査レポート `reports/`、調査生ノート `research_notes/`(.gitignore)。
- 図版は **SVG を正**としてリポジトリに置き、記事には **PNG 変換版**を貼る。
- バージョンは全イメージで**完全ピン留め**(`==`)。ピン留めしない依存を入れない。
- **実測が仮説と食い違ったら、記事の筋の方を実測に合わせる。** 仮説に合う実測を選びに行かない。

## Review Focus

- **torch / lightgbm / transformers が存在しない環境(alpine、CPU専用イメージ)** — プローブの import 失敗が実行全体を巻き添えにしてはいけない。該当プローブだけが `skipped` として結果 JSON に残り、他は完走すること。→ Task 3 でテスト。
- **GPU が無いホストで GPU プローブを選択した場合** — エラー終了ではなく `skipped`(理由付き)。→ Task 8 でテスト。
- **プローブ集合が違う結果 JSON 同士の比較** — 片方にしか無いプローブは「比較不能」として表に残り、欠損を一致扱いしないこと。→ Task 11 でテスト。
- **dtype 違いのダイジェスト衝突** — 同じ数値列でも float32 と float64 は別ダイジェストになること(`tobytes()` だけだと取り違える)。→ Task 1 でテスト。
- **HF モデルのダウンロード失敗・モデルキャッシュ不在** — S9 だけが `skipped` になり、同じランの S1〜S8 の結果は保存されること。→ Task 9 でテスト。

---

## 並行トラック(オーケストレーターが Task 1 と同時に開始する)

- **R1 理論調査(deep-research)** — IEEE 754-2019 の correctly-rounded 操作と §9.2 recommended operations / NumPy NEP 19 / ACM Artifact Review バッジ定義 / Mersenne Twister 原論文 / PCG / Philox(Random123) / PyTorch Reproducibility 公式ドキュメント / CUDA の非決定性と `CUBLAS_WORKSPACE_CONFIG` / LLM 推論の batch-invariance。成果物は `reports/` 配下の最終レポート。
- **R2 dgxspark 疎通確認** — GB10(sm_121)で PyTorch が動くかを最優先で確認する。動かない場合は GPU 構成を anakin + blade の2台に縮退し、spec のリスク欄どおり記事の GPU 比較を2世代で成立させる。

---

### Task 1: ダイジェストと数値比較の基盤

**Files:**
- Create: `reproducibility/reprobench/__init__.py`
- Create: `reproducibility/reprobench/digest.py`
- Test: `reproducibility/tests/test_digest.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `canonical_bytes(value: Any) -> bytes` — ndarray は `dtype.str` + `shape` + C連続バイト列を連結。int/float/str/bytes/list/tuple/dict も受ける。
  - `digest(value: Any) -> str` — `canonical_bytes` の SHA-256 hex
  - `encode_sample(arr: np.ndarray, n: int = 4096) -> str` — float64 に昇格し、固定インデックス(`np.linspace(0, size-1, n)` の整数化)で部分抽出して base64
  - `decode_sample(s: str) -> np.ndarray`
  - `ulp_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray` — float64 同士の ULP 距離
  - `compare_arrays(a: np.ndarray, b: np.ndarray) -> dict` — `{"bit_equal": bool, "max_abs": float, "max_rel": float, "max_ulp": int}`

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_digest_distinguishes_dtype():
    a = np.arange(8, dtype=np.float32)
    b = np.arange(8, dtype=np.float64)
    assert digest(a) != digest(b)

def test_digest_distinguishes_shape():
    a = np.arange(8, dtype=np.float64)
    assert digest(a) != digest(a.reshape(2, 4))

def test_digest_stable_for_non_contiguous_view():
    base = np.arange(16, dtype=np.float64)
    assert digest(base[::2]) == digest(np.ascontiguousarray(base[::2]))

def test_encode_decode_sample_roundtrip():
    arr = np.linspace(0, 1, 10_000, dtype=np.float32)
    out = decode_sample(encode_sample(arr, n=256))
    assert out.shape == (256,) and out.dtype == np.float64

def test_encode_sample_is_deterministic():
    arr = np.linspace(0, 1, 10_000, dtype=np.float64)
    assert encode_sample(arr, n=256) == encode_sample(arr, n=256)

def test_ulp_diff_of_adjacent_floats_is_one():
    a = np.array([1.0]); b = np.nextafter(a, 2.0)
    assert ulp_diff(a, b)[0] == 1

def test_compare_arrays_reports_bit_equal():
    a = np.linspace(0, 1, 100)
    r = compare_arrays(a, a.copy())
    assert r["bit_equal"] is True and r["max_ulp"] == 0

def test_compare_arrays_reports_difference():
    a = np.linspace(1, 2, 100)
    r = compare_arrays(a, np.nextafter(a, 3.0))
    assert r["bit_equal"] is False and r["max_ulp"] == 1 and r["max_rel"] > 0
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `cd reproducibility && docker build -f docker/Dockerfile.base -t repro-base . 2>/dev/null || true`
(この時点では Dockerfile が未作成なので、暫定で `docker run --rm -v "$PWD":/w -w /w python:3.11-slim sh -c "pip install -q numpy==2.2.6 pytest && pytest tests/test_digest.py -q"`)
Expected: FAIL(`ModuleNotFoundError: reprobench`)

- [ ] **Step 3: `digest.py` を実装**

ULP 距離は float64 のビットパターンを `int64` として見て、符号付きの順序に正規化してから差を取る(負数は `0x8000000000000000 - bits` に写す)。`max_rel` は分母0を避けるため `|b|` が0の要素では絶対差を使う。

- [ ] **Step 4: テストが通ることを確認**

Run: 上と同じコマンド
Expected: PASS(8 passed)

- [ ] **Step 5: コミット**

```bash
git add reproducibility/reprobench reproducibility/tests/test_digest.py
git commit -m "feat(repro): add digest and numeric comparison primitives"
```

---

### Task 2: 環境情報の採取

**Files:**
- Create: `reproducibility/reprobench/envinfo.py`
- Test: `reproducibility/tests/test_envinfo.py`

**Interfaces:**
- Consumes: なし
- Produces: `collect_env() -> dict` — 必須キー `host, arch, libc, kernel, python, packages, numpy_cpu_features, blas, thread_env, torch, cuda, gpus, image, config_label, pythonhashseed`。取得できない項目は `None`(キーは必ず存在する)。

- [ ] **Step 1: 失敗するテストを書く**

```python
REQUIRED = {"host","arch","libc","kernel","python","packages","numpy_cpu_features",
            "blas","thread_env","torch","cuda","gpus","image","config_label","pythonhashseed"}

def test_collect_env_has_all_required_keys():
    assert REQUIRED <= set(collect_env())

def test_collect_env_is_json_serializable():
    json.dumps(collect_env())

def test_collect_env_records_pythonhashseed_from_environment(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    assert collect_env()["pythonhashseed"] == "0"

def test_collect_env_survives_missing_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert collect_env()["torch"] is None
```

- [ ] **Step 2: テストが落ちることを確認** — Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: `collect_env()` を実装**

`numpy_cpu_features` は `numpy.core._multiarray_umath.__cpu_features__`(NumPy 2系では `numpy._core._multiarray_umath`)から取得し、両方を試して取れた方を使う。`blas` は `numpy.__config__.show(mode="dicts")` を優先し、無ければ `np.show_config()` の文字列。`thread_env` は `OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS, VECLIB_MAXIMUM_THREADS` を辞書で。`image` と `config_label` は環境変数 `REPRO_IMAGE` / `REPRO_CONFIG` から読む。

- [ ] **Step 4: テストが通ることを確認** — Expected: PASS(4 passed)

- [ ] **Step 5: コミット**

```bash
git add reproducibility/reprobench/envinfo.py reproducibility/tests/test_envinfo.py
git commit -m "feat(repro): capture execution environment metadata"
```

---

### Task 3: プローブ登録と契約

**Files:**
- Create: `reproducibility/reprobench/registry.py`
- Create: `reproducibility/reprobench/probes/__init__.py`
- Test: `reproducibility/tests/test_registry.py`

**Interfaces:**
- Consumes: `digest` (Task 1)
- Produces:
  - `@dataclass(frozen=True) class ProbeResult: probe: str; digest: str; scalars: dict[str, float]; sample_b64: str | None; meta: dict`
  - `@dataclass(frozen=True) class SkippedProbe: probe: str; reason: str`
  - `register(name: str, *, requires: tuple[str, ...] = ()) -> Callable` — `requires` の要素は `"torch" | "cuda" | "lightgbm" | "sklearn" | "transformers" | "hf_model"`
  - `all_probes() -> dict[str, Probe]`
  - `run_probe(name: str) -> ProbeResult | SkippedProbe` — `requires` を満たさなければ `SkippedProbe`、プローブ本体が例外を投げても `SkippedProbe(reason=repr(exc))` にして握る
  - `check_requirement(req: str) -> bool`

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_register_adds_probe_to_registry():
    @register("T_dummy")
    def _p(): return ProbeResult("T_dummy", "d", {}, None, {})
    assert "T_dummy" in all_probes()

def test_run_probe_skips_when_requirement_absent(monkeypatch):
    monkeypatch.setattr(registry, "check_requirement", lambda r: False)
    @register("T_needs_cuda", requires=("cuda",))
    def _p(): raise AssertionError("must not run")
    r = run_probe("T_needs_cuda")
    assert isinstance(r, SkippedProbe) and "cuda" in r.reason

def test_run_probe_converts_exception_to_skip():
    @register("T_boom")
    def _p(): raise RuntimeError("boom")
    r = run_probe("T_boom")
    assert isinstance(r, SkippedProbe) and "boom" in r.reason

def test_one_failing_probe_does_not_affect_others():
    @register("T_ok")
    def _ok(): return ProbeResult("T_ok", "d", {}, None, {})
    @register("T_bad")
    def _bad(): raise ImportError("no module named torch")
    assert isinstance(run_probe("T_bad"), SkippedProbe)
    assert isinstance(run_probe("T_ok"), ProbeResult)
```

- [ ] **Step 2: テストが落ちることを確認** — Expected: FAIL

- [ ] **Step 3: `registry.py` を実装**

`check_requirement` は importlib でモジュール存在を見る。`"cuda"` は `torch.cuda.is_available()`、`"hf_model"` は HF キャッシュにモデルが存在するか。**プローブ本体の import は関数内で行う**(モジュール先頭で `import torch` すると alpine で全体が落ちるため)。

- [ ] **Step 4: テストが通ることを確認** — Expected: PASS(4 passed)

- [ ] **Step 5: コミット**

```bash
git add reproducibility/reprobench/registry.py reproducibility/reprobench/probes/__init__.py reproducibility/tests/test_registry.py
git commit -m "feat(repro): add probe registry with graceful skipping"
```

---

### Task 4: S1 — PRNG 生ビット(L0 の証明)

**Files:**
- Create: `reproducibility/reprobench/probes/s1_prng.py`
- Test: `reproducibility/tests/test_s1_prng.py`

**Interfaces:**
- Consumes: `register`, `ProbeResult`, `digest`, `encode_sample`
- Produces: プローブ名 `S1_stdlib_mt19937`, `S1_numpy_randomstate`, `S1_numpy_mt19937_generator`, `S1_numpy_pcg64`, `S1_numpy_philox`, `S1_numpy_normal_randomstate`, `S1_numpy_normal_generator`, `S1_generator_random_float32`。すべて seed=20261014 固定、10,000 個生成。
- **`S1_numpy_mt19937_generator` は `np.random.Generator(np.random.MT19937(seed))`。** これで「同じ MT19937・同じ seed」の経路が3本揃う(CPython `random` / NumPy `RandomState` / NumPy `Generator(MT19937)`)。**3本とも違う値を出す**はずで、理由は seed→state の写像が3者で別物だから(CPython は参照実装の `init_by_array`、`RandomState` は独自の `mt19937_seed`、`Generator` は `SeedSequence`)。MT19937 の仕様が固定しているのは **state→出力の twist+tempering であって seed→state ではない**。
- **`S1_generator_random_float32` は `np.random.default_rng(seed).random(10_000, dtype=np.float32)`。** NumPy 1.22.0 で float32 uniform の最下位ビットのバグ修正によりストリームが変わったことが release note に明記されており、1.21.x と 2.x を跨ぐと**値が変わるはず**。NEP 19 の「Generator はストリーム互換を保証しない」の実証そのものになる。

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_stdlib_mt19937_matches_published_value():
    # MT19937 は公開仕様。CPython の random.seed(0) 直後の最初の値は
    # 実装を跨いで一致する既知値であり、これが L0(規格レベル)の実証になる。
    import random
    random.seed(0)
    assert random.random() == 0.8444218515250481

def test_prng_probe_records_seed_and_count():
    r = run_probe("S1_stdlib_mt19937")
    assert r.meta["seed"] == 20261014 and r.meta["n"] == 10_000

def test_prng_probes_are_deterministic_within_process():
    for name in ("S1_stdlib_mt19937","S1_numpy_randomstate","S1_numpy_pcg64","S1_numpy_philox"):
        assert run_probe(name).digest == run_probe(name).digest

def test_randomstate_and_generator_differ():
    assert run_probe("S1_numpy_randomstate").digest != run_probe("S1_numpy_pcg64").digest

def test_three_mt19937_paths_give_three_different_streams():
    # 同じ MT19937、同じ seed、3つの異なる seed→state 写像。
    # 仕様が固定しているのは state→出力であって seed→state ではない。
    ds = {run_probe(n).digest for n in
          ("S1_stdlib_mt19937", "S1_numpy_randomstate", "S1_numpy_mt19937_generator")}
    assert len(ds) == 3
```

- [ ] **Step 2: テストが落ちることを確認** — Expected: FAIL

- [ ] **Step 3: S1 プローブ群を実装**

各プローブは「seed を固定 → 指定の生成器で 10,000 個 → `digest` と `encode_sample`」。`scalars` には `first`, `last`, `mean` を入れる。`S1_numpy_randomstate` は `np.random.RandomState(seed).random_sample`、`S1_numpy_pcg64` は `np.random.Generator(np.random.PCG64(seed))`、`S1_numpy_philox` は `np.random.Generator(np.random.Philox(seed))`。正規分布版は **legacy と Generator で変換アルゴリズムが違う**ことを示すために両方置く(第4章 L2 の実例)。

- [ ] **Step 4: テストが通ることを確認** — Expected: PASS(3 passed)

- [ ] **Step 5: コミット**

```bash
git add reproducibility/reprobench/probes/s1_prng.py reproducibility/tests/test_s1_prng.py
git commit -m "feat(repro): add S1 PRNG bit-stream probes"
```

---

### Task 5: S2/S3/S4 — 縮約・BLAS・libm

**Files:**
- Create: `reproducibility/reprobench/probes/s2_reduction.py`
- Create: `reproducibility/reprobench/probes/s3_blas.py`
- Create: `reproducibility/reprobench/probes/s4_libm.py`
- Test: `reproducibility/tests/test_numeric_probes.py`

**Interfaces:**
- Produces:
  - S2: `S2_sum_float32`, `S2_sum_float64`, `S2_mean_float32`, `S2_sum_shuffled_float32`(同じ値を別順で足して非結合性を直接示す)。配列は 10,000,000 要素、`default_rng(20261014)` 生成。
  - S3: `S3_matmul_float32`, `S3_matmul_float64`。2048×2048、`default_rng(20261014)`。
  - S4: `S4_exp_float64`, `S4_log_float64`, `S4_sin_float64`。1,000,000 要素。

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_shuffled_sum_differs_from_ordered_sum_float32():
    a = run_probe("S2_sum_float32"); b = run_probe("S2_sum_shuffled_float32")
    assert a.scalars["total"] != b.scalars["total"]   # 非結合性の直接証明

def test_numeric_probes_deterministic_within_process():
    for name in ("S2_sum_float64","S3_matmul_float64","S4_exp_float64"):
        assert run_probe(name).digest == run_probe(name).digest

def test_matmul_probe_records_shape_and_dtype():
    r = run_probe("S3_matmul_float32")
    assert r.meta["shape"] == [2048, 2048] and r.meta["dtype"] == "float32"
```

- [ ] **Step 2: テストが落ちることを確認** — Expected: FAIL

- [ ] **Step 3: S2/S3/S4 を実装**

S3 は 2048² の積の全要素をダイジェストに掛けるとメモリと時間を食うので、**全体のダイジェスト**に加えて `scalars` に `trace`, `fro_norm` を入れ、`encode_sample` で 4096 点を保存する。S2 の `total` は float64 に昇格した値を `scalars` に入れる(JSON で精度を落とさないため hex 表現 `float.hex()` も `meta["total_hex"]` に残す)。

- [ ] **Step 4: テストが通ることを確認** — Expected: PASS(3 passed)

- [ ] **Step 5: コミット**

```bash
git add reproducibility/reprobench/probes/s2_reduction.py reproducibility/reprobench/probes/s3_blas.py reproducibility/reprobench/probes/s4_libm.py reproducibility/tests/test_numeric_probes.py
git commit -m "feat(repro): add S2/S3/S4 reduction, BLAS and libm probes"
```

---

### Task 6: S0 — 「インプットが全部同じ」が崩れるケース

**Files:**
- Create: `reproducibility/reprobench/probes/s0_hidden_inputs.py`
- Test: `reproducibility/tests/test_s0_hidden_inputs.py`

**Interfaces:**
- Produces: `S0_set_iteration_order`(文字列20個の `set` 反復順)、`S0_dict_from_set`(set 経由で作った辞書のキー順)、`S0_glob_order`(一時ディレクトリに100ファイル作って `glob.glob` の順)、`S0_hash_of_str`(`hash("reproducibility")`)、`S0_parallel_completion_order`(`ThreadPoolExecutor` に 32 ジョブを投げ、完了順に結果を連結)

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_set_iteration_probe_records_hash_seed():
    r = run_probe("S0_set_iteration_order")
    assert "pythonhashseed" in r.meta

def test_glob_order_probe_is_sorted_flag_recorded():
    r = run_probe("S0_glob_order")
    assert "sorted_equals_raw" in r.meta   # OS によって一致したりしなかったりする

def test_parallel_completion_order_probe_returns_result():
    assert isinstance(run_probe("S0_parallel_completion_order"), ProbeResult)
```

- [ ] **Step 2: テストが落ちることを確認** — Expected: FAIL

- [ ] **Step 3: S0 を実装**

`S0_parallel_completion_order` は `as_completed` の到着順に ID を連結する。**この値はランごとに変わるのが正しい挙動**なので、テストは一致を要求しない。`meta` に「同一プロセス内で2回実行して一致したか」を `stable_within_process` として記録する(G1-1a の材料)。

- [ ] **Step 4: テストが通ることを確認** — Expected: PASS(3 passed)

- [ ] **Step 5: コミット**

```bash
git add reproducibility/reprobench/probes/s0_hidden_inputs.py reproducibility/tests/test_s0_hidden_inputs.py
git commit -m "feat(repro): add S0 hidden-input-drift probes"
```

---

### Task 7: S5/S6 — scikit-learn と LightGBM

**Files:**
- Create: `reproducibility/reprobench/probes/s5_sklearn.py`
- Create: `reproducibility/reprobench/probes/s6_lightgbm.py`
- Test: `reproducibility/tests/test_ml_probes.py`

**Interfaces:**
- Produces:
  - S5(`requires=("sklearn",)`): `S5_logreg_coef`, `S5_random_forest_pred`, `S5_kmeans_centers`, `S5_train_test_split_index`。データは `default_rng(20261014)` で生成した 20,000×30 の分類データ、`random_state=20261014` 固定。
  - S6(`requires=("lightgbm",)`): `S6_lgbm_pred`, `S6_lgbm_tree_structure`。`num_threads` は環境変数 `REPRO_THREADS` に従う。

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_sklearn_probes_deterministic_within_process():
    for name in ("S5_logreg_coef","S5_kmeans_centers","S5_train_test_split_index"):
        assert run_probe(name).digest == run_probe(name).digest

def test_train_test_split_probe_is_pure_l0_on_indices():
    r = run_probe("S5_train_test_split_index")
    assert r.meta["n_train"] + r.meta["n_test"] == 20_000

def test_ml_probes_skip_without_library(monkeypatch):
    monkeypatch.setattr(registry, "check_requirement", lambda r: r != "lightgbm")
    assert isinstance(run_probe("S6_lgbm_pred"), SkippedProbe)
```

- [ ] **Step 2: テストが落ちることを確認** — Expected: FAIL

- [ ] **Step 3: S5/S6 を実装**

`S5_random_forest_pred` は `n_jobs` を `REPRO_THREADS` に従わせる(並列数で結果が変わるかを見るため)。`S5_logreg_coef` は `solver="lbfgs", max_iter=200` 固定。`S6_lgbm_tree_structure` は `booster.dump_model()` の JSON を正規化してダイジェスト化し、**予測値だけでなく木構造そのもの**が変わるかを見る。

- [ ] **Step 4: テストが通ることを確認** — Expected: PASS(3 passed)

- [ ] **Step 5: コミット**

```bash
git add reproducibility/reprobench/probes/s5_sklearn.py reproducibility/reprobench/probes/s6_lightgbm.py reproducibility/tests/test_ml_probes.py
git commit -m "feat(repro): add S5/S6 scikit-learn and LightGBM probes"
```

---

### Task 8: S7/S8 — PyTorch CPU / GPU

**Files:**
- Create: `reproducibility/reprobench/probes/s7_torch_cpu.py`
- Create: `reproducibility/reprobench/probes/s8_torch_gpu.py`
- Test: `reproducibility/tests/test_torch_probes.py`

**Interfaces:**
- Produces:
  - S7(`requires=("torch",)`): `S7_mlp_forward`, `S7_mlp_train10`(3層MLP、入力 4096×256、10 step、`torch.manual_seed(20261014)`)
  - S8(`requires=("torch","cuda")`): `S8_mlp_train10_cuda`, `S8_mlp_train10_cuda_deterministic`, `S8_matmul_tf32_on`, `S8_matmul_tf32_off`, `S8_mlp_train10_cuda_bf16`
  - **`S8_curand_randn_sizes`(最重要)** — `torch.manual_seed(20261014)` を固定したまま `torch.randn(n, device="cuda")` を n = 1024 / 65536 / 4194304 / 67108864 の4サイズで生成し、**サイズごとに別々のダイジェスト**を `meta["digests"]` に記録する。狙い: PyTorch の CUDA RNG は `curand_init` が Philox をスレッド index で seed するため、**スレッド→要素の割り当てが SM 数に依存する**。要素数がある閾値を超えると、同じ seed・同じ Philox でも **GPU モデルが違えば生成される乱数そのものが変わる**(PyTorch issue #194676)。4サイズ測れば、どのサイズから崩れるかを実測で特定できる。
    これは「浮動小数の丸めがズレる」ではなく「**乱数の値そのものが変わる**」ケースであり、記事の第6章 G4 で最も強い実例になる。**この挙動を公式ドキュメントに追記する PR(#196733)は 2026-10-14 時点で approve 済みだが未マージ** = 現時点で公式には文書化されていない。実測が一次情報になる。

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_torch_cpu_probe_deterministic_within_process():
    assert run_probe("S7_mlp_train10").digest == run_probe("S7_mlp_train10").digest

def test_gpu_probes_skip_without_cuda(monkeypatch):
    monkeypatch.setattr(registry, "check_requirement", lambda r: r != "cuda")
    for name in ("S8_mlp_train10_cuda","S8_matmul_tf32_on"):
        r = run_probe(name)
        assert isinstance(r, SkippedProbe) and "cuda" in r.reason

def test_curand_sizes_probe_records_per_size_digests():
    r = run_probe("S8_curand_randn_sizes")
    if isinstance(r, ProbeResult):
        assert set(r.meta["digests"]) == {1024, 65536, 4194304, 67108864}

def test_tf32_probes_record_their_flag_state():
    # CUDA が無い環境では skipped になるので、そのときはこのテストを skip
    r = run_probe("S8_matmul_tf32_on")
    if isinstance(r, ProbeResult):
        assert r.meta["tf32"] is True
```

- [ ] **Step 2: テストが落ちることを確認** — Expected: FAIL

- [ ] **Step 3: S7/S8 を実装**

`S8_mlp_train10_cuda_deterministic` は `torch.use_deterministic_algorithms(True)` と `CUBLAS_WORKSPACE_CONFIG=:4096:8` を設定した状態で実行する(環境変数はプロセス起動時に必要なので、`meta` に設定済みかどうかを記録し、未設定なら `SkippedProbe`)。TF32 は `torch.backends.cuda.matmul.allow_tf32` と `torch.backends.cudnn.allow_tf32` の両方を切り替える。**両フラグの既定値は非対称**(matmul 側は PyTorch 1.12 以降 False、cudnn 側は True のまま)なので、全 S8 プローブの `meta` に両フラグの**実際の値**と `torch.cuda.get_device_properties(0).multi_processor_count`(SM 数)を必ず記録する。SM 数は `S8_curand_randn_sizes` の結果を解釈するのに要る。

- [ ] **Step 4: テストが通ることを確認** — Expected: PASS(3 passed、CUDA 無し環境では2 passed 1 skipped 相当)

- [ ] **Step 5: コミット**

```bash
git add reproducibility/reprobench/probes/s7_torch_cpu.py reproducibility/reprobench/probes/s8_torch_gpu.py reproducibility/tests/test_torch_probes.py
git commit -m "feat(repro): add S7/S8 PyTorch CPU and GPU probes"
```

---

### Task 9: S9 — LLM 推論と temperature

**Files:**
- Create: `reproducibility/reprobench/probes/s9_llm.py`
- Test: `reproducibility/tests/test_s9_llm.py`

**Interfaces:**
- Consumes: `register`, `ProbeResult`, `digest`
- Produces(すべて `requires=("transformers","torch","hf_model")`、モデル `Qwen/Qwen2.5-0.5B-Instruct`、`max_new_tokens=128`、プロンプト固定):
  - `S9_greedy_batch1`, `S9_greedy_batch8` — 同一プロンプトを batch 1 / 8 で greedy 生成。batch8 は同じプロンプトを8本並べ、**先頭1本の出力**を取る。
    **greedy は `do_sample=False` で行う。`temperature=0` は使わない** — transformers に「temperature=0」という決定的モードは存在せず、`do_sample=False` が公式の決定的パス。`temperature=0` + `do_sample=True` は strictly positive を要求して弾かれる。
  - `S9_temperature_zero_api_semantics` — transformers に `temperature=0` を渡すと実際に何が起きるかを記録する(例外なら例外型とメッセージ、警告なら警告文、無視なら無視された事実)。vLLM が `temperature<1e-5` を greedy 分岐に落とし `0<t<0.01` を 0.01 にクランプするのとの対比として、記事第7章で使う。`meta` に生の挙動を文字列で残す。
  - `S9_sampling_t0_7_seed`, `S9_sampling_t1_2_seed` — `do_sample=True`, `temperature=0.7 / 1.2`, `torch.manual_seed(20261014)`
  - `S9_logit_margin` — 各生成ステップの top-1 と top-2 のロジット差を記録
  - `S9_greedy_fp32`, `S9_greedy_bf16`, `S9_greedy_fp16` — dtype だけ変える
- 共通ヘルパー `first_divergence(a: list[int], b: list[int]) -> int | None` — 最初にトークンが食い違った位置(無ければ `None`)

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_first_divergence_finds_position():
    assert first_divergence([1,2,3,4],[1,2,9,4]) == 2
    assert first_divergence([1,2,3],[1,2,3]) is None
    assert first_divergence([1,2],[1,2,3]) == 2

def test_llm_probes_skip_without_model(monkeypatch):
    monkeypatch.setattr(registry, "check_requirement", lambda r: r != "hf_model")
    r = run_probe("S9_greedy_batch1")
    assert isinstance(r, SkippedProbe) and "hf_model" in r.reason

def test_llm_probe_meta_records_tokens_and_margin_when_available():
    r = run_probe("S9_logit_margin")
    if isinstance(r, ProbeResult):
        assert len(r.meta["margins"]) == r.meta["n_tokens"]
```

- [ ] **Step 2: テストが落ちることを確認** — Expected: FAIL

- [ ] **Step 3: S9 を実装**

`meta` には必ず `token_ids`(生成トークン列)と `text` を入れる。比較器が `first_divergence` を使えるようにするため。`S9_logit_margin` は `output_scores=True, return_dict_in_generate=True` で各ステップの上位2ロジットを取り、差を float のリストにする。モデルのロードは関数内で行い、同一プロセス内では module-level キャッシュを使い回す(1ランで何度もロードしないため)。

- [ ] **Step 4: テストが通ることを確認** — Expected: PASS(3 passed)

- [ ] **Step 5: コミット**

```bash
git add reproducibility/reprobench/probes/s9_llm.py reproducibility/tests/test_s9_llm.py
git commit -m "feat(repro): add S9 LLM inference and temperature probes"
```

---

### Task 10: ランナー CLI

**Files:**
- Create: `reproducibility/reprobench/runner.py`
- Test: `reproducibility/tests/test_runner.py`

**Interfaces:**
- Consumes: `collect_env`, `all_probes`, `run_probe`
- Produces:
  - `run(selected: list[str] | None = None, repeat: int = 1) -> dict` — `{"schema": 1, "env": {...}, "probes": [ {...} | {"probe":..., "skipped": reason} ]}`
  - `main() -> None` — `--probes S1,S2`(前方一致でグループ選択可)、`--out PATH`、`--repeat N`(同一プロセス内で N 回実行し `probes` を N セット出す。G1-1a 用)

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_run_includes_env_and_probes():
    out = run(selected=["S1_stdlib_mt19937"])
    assert out["schema"] == 1 and "env" in out and len(out["probes"]) == 1

def test_run_prefix_selection_matches_group():
    out = run(selected=["S1"])
    assert all(p["probe"].startswith("S1") for p in out["probes"]) and len(out["probes"]) >= 4

def test_skipped_probes_are_recorded_not_dropped(monkeypatch):
    monkeypatch.setattr(registry, "check_requirement", lambda r: False)
    out = run(selected=["S8"])
    assert out["probes"] and all("skipped" in p for p in out["probes"])

def test_repeat_produces_independent_rounds():
    out = run(selected=["S0_parallel_completion_order"], repeat=2)
    assert len(out["rounds"]) == 2
```

- [ ] **Step 2: テストが落ちることを確認** — Expected: FAIL

- [ ] **Step 3: `runner.py` を実装**

`--repeat` が 1 より大きいときは `rounds` キーに各回の probes を入れ、`probes` は1回目のものを指す(後方互換のため両方持つ)。出力ファイル名は `--out` 未指定なら `results/{host}__{image}__{config}.json`。

- [ ] **Step 4: テストが通ることを確認** — Expected: PASS(4 passed)

- [ ] **Step 5: コミット**

```bash
git add reproducibility/reprobench/runner.py reproducibility/tests/test_runner.py
git commit -m "feat(repro): add probe runner CLI"
```

---

### Task 11: 比較器

**Files:**
- Create: `reproducibility/reprobench/compare.py`
- Test: `reproducibility/tests/test_compare.py`

**Interfaces:**
- Consumes: `decode_sample`, `compare_arrays` (Task 1)
- Produces:
  - `load_results(paths: Iterable[str]) -> list[dict]`
  - `agreement(results: list[dict], probe: str) -> dict` — `{"groups": {digest: [labels]}, "all_equal": bool, "missing": [labels], "numeric": {pair: {...}}}`
  - `build_report(results: list[dict]) -> dict` — 全プローブ × 全ラベル
  - `to_markdown(report: dict) -> str` — 記事に貼れる ○×表
  - ラベルは `env` から `{host}/{image}/{config_label}` で作る

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_agreement_groups_identical_digests():
    rs = [_res("a", {"S2_sum_float32":"dead"}), _res("b", {"S2_sum_float32":"dead"}),
          _res("c", {"S2_sum_float32":"beef"})]
    g = agreement(rs, "S2_sum_float32")
    assert g["all_equal"] is False and sorted(map(len, g["groups"].values())) == [1, 2]

def test_missing_probe_is_reported_not_treated_as_match():
    rs = [_res("a", {"S8_x":"dead"}), _res("b", {})]
    g = agreement(rs, "S8_x")
    assert g["missing"] == ["b"] and g["all_equal"] is True   # 存在する2件だけで判定

def test_numeric_delta_computed_from_samples():
    rs = [_res("a", {"S2_sum_float64":"d1"}, samples={"S2_sum_float64": _b64(np.arange(8.))}),
          _res("b", {"S2_sum_float64":"d2"}, samples={"S2_sum_float64": _b64(np.nextafter(np.arange(8.), 9.))})]
    g = agreement(rs, "S2_sum_float64")
    assert g["numeric"][("a","b")]["max_ulp"] == 1

def test_to_markdown_renders_one_row_per_probe():
    md = to_markdown(build_report([_res("a", {"S1_x":"d"}), _res("b", {"S1_x":"d"})]))
    assert md.count("\n| S1_x") == 1
```

- [ ] **Step 2: テストが落ちることを確認** — Expected: FAIL

- [ ] **Step 3: `compare.py` を実装**

`all_equal` は**存在するラベルのみ**で判定し、欠損は `missing` に出す(欠損を一致扱いしない)。`numeric` は `sample_b64` が両方にあるペアだけ計算する。

- [ ] **Step 4: テストが通ることを確認** — Expected: PASS(4 passed)

- [ ] **Step 5: コミット**

```bash
git add reproducibility/reprobench/compare.py reproducibility/tests/test_compare.py
git commit -m "feat(repro): add cross-environment result comparator"
```

---

### Task 12: Docker イメージ群と Makefile

**Files:**
- Create: `reproducibility/docker/Dockerfile.{base,py313,oldpy,oldlibs,alpine,mkl,gpu}`
- Create: `reproducibility/requirements/{base,py313,oldpy,oldlibs,alpine,gpu}.txt`
- Create: `reproducibility/Makefile`
- Create: `reproducibility/README.md`

**Interfaces:**
- Produces: `make test`(base イメージでユニットテスト)、`make build IMAGE=gpu`、`make build-all-cpu`、`make run IMAGE=base CONFIG=threads1`、`make report`(`results/*.json` → `results/report.json` と `results/matrix.md`)、`make figures`(`results/report.json` → 実測図 PNG)。`report` と `figures` も base イメージ内で実行する

- [ ] **Step 1: 6つの Dockerfile と requirements を書く**

`base`: `python:3.11-slim` + numpy 2.2.6 / scipy / scikit-learn 最新ピン / lightgbm / torch(CPU版)/ pytest。
`py313`: `python:3.13-slim`、同じライブラリ版。
`oldlibs`: **`python:3.10-slim` + numpy 1.21.6** / scikit-learn 1.0.2 / 対応する旧 torch。numpy は 1.21.6 でなければならない — **NumPy 1.22.0 の float32 uniform 最下位ビット修正によるストリーム変化を跨ぐため**。1.26 では跨げず、NEP 19 の実証ができない。numpy 1.21.6 は Python 3.10 までしか対応しないので Python も 3.10 に下げる。
`oldpy`: **`python:3.10-slim` + numpy 2.2.6**(base と同じライブラリ版)。**交絡を切るための対照イメージ** — `base`(3.11)vs `oldpy`(3.10)で Python 版だけが動き、`oldpy` vs `oldlibs` で numpy 版だけが動く。これが無いと 2a と 2b が分離できない。
`alpine`: `python:3.11-alpine` + numpy のみ(musl wheel がある範囲)。**torch / lightgbm は入れない。**
`mkl`: `mambaorg/micromamba` ベースで conda-forge から `libblas=*=*mkl` 付き numpy。**x86_64 専用**。
`gpu`: `nvidia/cuda:13.0.1-runtime-ubuntu24.04` + torch(cu130)+ transformers + accelerate。**CUDA 12 系は使えない** — dgxspark の GB10 は compute capability (12,1) で cu12 ビルドでは起動しない(3ホストともドライバ 580.x / CUDA 13.0 で動作確認済み)。

全イメージ共通で `ENV REPRO_IMAGE=<name>` を設定し、`WORKDIR /work`、`COPY reprobench/ ./reprobench/` と `COPY tests/ ./tests/`。

- [ ] **Step 2: base イメージをビルドしてテストが全部通ることを確認**

Run: `cd reproducibility && make test`
Expected: すべてのユニットテストが PASS(CUDA 依存は skipped)

- [ ] **Step 3: 全イメージがビルドできることを確認**

Run: `cd reproducibility && make build-all-cpu`
Expected: base / py313 / oldpy / oldlibs / alpine が成功。`mkl` は x86_64 のみ成功(Mac では skip される)。

- [ ] **Step 4: `.gitignore` に `research_notes/` を追加する**

spec §6 のとおり、調査の生ノートはコミットしない(最終レポート `reports/` はコミットする)。

- [ ] **Step 4: README に再実行手順を書く**

読者が `git clone` → `make build IMAGE=base` → `make run IMAGE=base` で同じ JSON を得られる手順。

- [ ] **Step 5: コミット**

```bash
git add .gitignore reproducibility/docker reproducibility/requirements reproducibility/Makefile reproducibility/README.md
git commit -m "feat(repro): add container images and reproduction entrypoint"
```

---

### Task 13: 配布・実行スクリプト

**Files:**
- Create: `reproducibility/scripts/run_on_host.sh`
- Create: `reproducibility/scripts/run_all.sh`
- Create: `reproducibility/scripts/fetch_model.sh`

**Interfaces:**
- Produces: `./scripts/run_on_host.sh <host> <image> <config>` — rsync → `docker build` → `docker run` → 結果を `results/` に回収。`<host>` が `local` のときは MacBook の Docker Desktop を使う。

- [ ] **Step 1: `run_on_host.sh` を書く**

`rsync -az --delete --exclude results/ ./ <host>:~/reprobench-run/` → ssh で `docker build` と `docker run` → `scp` で JSON を回収。config は `threads1 / threads8 / threadsmax / nosimd512 / hashseed0 / hashseedfree` を環境変数に展開する。

- [ ] **Step 2: ドライラン(`--dry-run`)で各ホストに到達できることを確認**

Run: `./scripts/run_all.sh --dry-run`
Expected: 4ホスト分の実行コマンドが表示され、ssh 疎通エラーが出ない

- [ ] **Step 3: `fetch_model.sh` を書く** — 各 GPU ホストに Qwen2.5-0.5B-Instruct を事前取得。失敗したホストは記録して続行。

- [ ] **Step 4: 1ホスト1イメージだけ本当に流して JSON が回収できることを確認**

Run: `./scripts/run_on_host.sh local base threadsmax`
Expected: `results/local__base__threadsmax.json` が生成され、`schema` が 1

- [ ] **Step 5: コミット**

```bash
git add reproducibility/scripts
git commit -m "feat(repro): add multi-host deployment and collection scripts"
```

---

### Task 14: 実測の実行(P2)

**Files:**
- Create: `reproducibility/results/*.json`(生成物)
- Create: `reproducibility/results/report.json` と `reproducibility/results/matrix.md`

- [ ] **Step 1: CPU 実測を4ホストで流す**

`base` を 4 ホスト × config(`threads1`, `threads8`, `threadsmax`, `hashseed0`, `hashseedfree`)、blade では追加で `nosimd512`。
anakin と blade は `threadsmax` の値が違うので、**`threads8` を揃えた比較が 3b の本命**になる。

- [ ] **Step 2: ソフトウェア差分を blade で流す** — `py313` / `oldpy` / `oldlibs` / `mkl` / `alpine`、および dgxspark で `alpine`。**2a(Python 版)は base vs oldpy、2b(numpy 版)は oldpy vs oldlibs で読む**

- [ ] **Step 3: GPU 実測を anakin / blade / dgxspark で流す** — `gpu` イメージ。dgxspark が動かない場合は記録して2台で続行

- [ ] **Step 4: 比較レポートを生成し、仮説との一致を確認**

Run: `cd reproducibility && make report`
Expected: `results/matrix.md` が生成される。**ここで spec §3 の L0〜L4 の予測と実測を突き合わせ、食い違いを一覧にする。** 食い違いがあれば記事の筋をそちらに合わせる(Global Constraints 参照)。

- [ ] **Step 5: コミット**

```bash
git add reproducibility/results
git commit -m "chore(repro): add measurement results from 4 hosts"
```

---

### Task 15: 原理図(実測非依存)

**Files:**
- Create: `reproducibility/images_src/{f1_layers,f2_divergence,f3_levels,f5_assoc,f6_simd,f7_gpu_reduce,f10_blas_tiles}.svg`
- Create: `reproducibility/scripts/svg_to_png.sh`
- Create: `images/reproducibility-of-randomness/*.png`

- [ ] **Step 1: 7枚の SVG を書く** — spec §5 の F1・F2・F3・F5・F6・F7・F10

ライトテーマ前提、日本語ラベル、横幅 960px 基準。色は意味ごとに固定(L0=濃緑 → L4=赤のグラデーション)。

- [ ] **Step 2: PNG 変換して記事用ディレクトリに出す**

Run: `./scripts/svg_to_png.sh`
Expected: `images/reproducibility-of-randomness/` に 7 枚の PNG(幅 1920px、2倍解像度)

- [ ] **Step 3: 全 PNG を目視確認** — 文字切れ・はみ出し・日本語フォント欠けがないこと

- [ ] **Step 4: コミット**

```bash
git add reproducibility/images_src reproducibility/scripts/svg_to_png.sh images/reproducibility-of-randomness
git commit -m "feat(repro): add conceptual diagrams"
```

---

### Task 16: 実測図(実測依存)

**Files:**
- Create: `reproducibility/reprobench/figures.py`
- Create: `reproducibility/images_src/{f4_matrix,f9_flowchart}.svg`
- Create: `images/reproducibility-of-randomness/{f4_matrix,f8_llm_divergence,f8_llm_margin,f9_flowchart}.png`

- [ ] **Step 1: `figures.py` を実装** — `results/report.json` を入力に、F4(レベル×条件の ○× マトリクス)と F8(LLM の margin 分布 + 温度別の分岐位置)を matplotlib で描く

- [ ] **Step 2: 図を生成** — Run: `make figures` / Expected: PNG 3枚が出力され、F4 の ○× が `results/matrix.md` と一致する

- [ ] **Step 3: F9(判定フローチャート + 文言テンプレ)を SVG で書き、PNG 化する** — 実測で確定したレベル境界を反映すること

- [ ] **Step 4: コミット**

```bash
git add reproducibility/reprobench/figures.py reproducibility/images_src images/reproducibility-of-randomness
git commit -m "feat(repro): add measurement-driven figures"
```

---

### Task 17: 記事の執筆

**Files:**
- Create: `articles/reproducibility-of-randomness.md`

- [ ] **Step 1: spec §2 の章立てで全章を書く**

frontmatter は既存記事に合わせる(`title / emoji / type: "tech" / topics / published: false`)。
材料は `reports/`(理論の一次情報)と `results/matrix.md`(実測)のみ。**本文中の数値は必ず `results/` の実測値から取る。**

- [ ] **Step 2: 全図版を本文に差し込む** — パスは `/images/reproducibility-of-randomness/<name>.png`

- [ ] **Step 3: 第8章に判定フローと納品文言テンプレートを書く** — L0〜L4 ごとに、そのまま納品文書に貼れる条件文を実測の裏づけ付きで

- [ ] **Step 4: 記事内の全数値を `results/` と突き合わせて検算** — 一致しない数値が1つも無いこと

- [ ] **Step 5: コミット**

```bash
git add articles/reproducibility-of-randomness.md
git commit -m "feat: add randomness reproducibility article"
```

---

### Task 18: 最終レビュー

- [ ] **Step 1: 事実確認レビュー** — 記事中の理論的主張を `reports/` の一次情報と1つずつ突き合わせる。裏づけの無い主張は削るか「筆者の推測」と明記する
- [ ] **Step 2: 実測主張の再現確認** — 記事の主要な実測主張を1つ選び、`make run` から再実行して同じ結論になることを確認する
- [ ] **Step 3: 読者視点レビュー** — 「低レイヤ未経験の Python ユーザー」として読み、説明なしで使われている用語を洗い出して補う
- [ ] **Step 4: ブランチを切って PR を作成**
