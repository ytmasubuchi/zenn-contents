# reprobench

A measurement harness for the article
[「乱数の再現性」](../articles/reproducibility-of-randomness.md): it runs the
same 53 probes (`reprobench/probes/`) across combinations of container
image and runtime configuration, and compares the results
(`reprobench/compare.py`) to produce the article's evidence table.

**Everything runs inside a container.** There is no supported way to run
the probes directly on a host; every target in this `Makefile` and every
script in `scripts/` builds and runs a Docker image.

## Reproducing from a plain clone

```sh
git clone <this repo>
cd <repo>/reproducibility
make build IMAGE=base      # builds reprobench:base
make run IMAGE=base CONFIG=threadsmax
#   -> results/local__base__threadsmax.json
make report                # results/*.json -> results/report.json + results/matrix.md
make figures                # results/report.json -> images/reproducibility-of-randomness/*.png
```

`make run` defaults to `REPRO_HOST=local` (this machine's Docker
Desktop); see `scripts/run_on_host.sh` to run the same image/config pair
on a remote host instead (`make run IMAGE=base CONFIG=threadsmax
REPRO_HOST=anakin`).

Run the unit suite (base image only):

```sh
make test
```

## Images

| image | base | moves relative to `base` | notes |
|---|---|---|---|
| `base` | `python:3.11-slim` | -- (the reference image) | runs on all 4 measurement hosts |
| `py313` | `python:3.13-slim` | Python version only | same library pins as `base` |
| `oldpy` | `python:3.10-slim` | Python version only | same library pins as `base` (numpy 2.2.6, not 1.21.6) -- a confound-breaking control, see below |
| `oldlibs` | `python:3.10-slim` | numpy/scikit-learn version only | numpy==1.21.6, scikit-learn==1.0.2; **no torch** (see "Known issues") |
| `alpine` | `python:3.11-alpine` | libc (musl vs. glibc) | numpy only -- no scipy/scikit-learn/lightgbm/torch |
| `mkl` | `mambaorg/micromamba` + conda-forge | BLAS implementation (MKL vs. OpenBLAS) | **x86_64 only**; `make build`/`make build-all-cpu` skip it on aarch64 |
| `gpu` | `nvidia/cuda:13.0.1-runtime-ubuntu24.04` | adds CUDA 13 + torch (cu130) + transformers | **unverified from this development host** (see below) |

`oldpy` exists specifically to break a confound: `base` (Python 3.11,
numpy 2.2.6) vs. `oldpy` (Python 3.10, numpy 2.2.6) isolates the
Python-version axis, and `oldpy` vs. `oldlibs` (Python 3.10, numpy
1.21.6) isolates the numpy-version axis. Comparing `base` directly
against `oldlibs` would move both at once and neither axis could be read
independently.

numpy is pinned at exactly `1.21.6` in `oldlibs`, not a newer `1.x`:
NumPy's documented stream-breaking changes (NEP 19) land at `1.19.0` and
`1.22.0`, and `1.21.6` is the last release before the `1.22.0` boundary --
anything at or above `1.22.0` would cross neither boundary and the image
would demonstrate nothing. `1.21.6` is also the reason `oldlibs` (and
`oldpy`, its Python-only counterpart) are pinned to `python:3.10-slim`:
numpy 1.21.6 has no wheel for a newer Python.

Every version in every image is pinned with `==` (`requirements/*.txt`
and the `pip install` lines in each `docker/Dockerfile.*`); none of them
is left for pip to resolve freely.

## Makefile targets

| target | does |
|---|---|
| `make test` | builds `base`, runs `pytest -q -W error` inside it |
| `make build IMAGE=<name>` | builds exactly one image |
| `make build-all-cpu` | builds `base py313 oldpy oldlibs alpine mkl` (skips `mkl` on non-x86_64) |
| `make run IMAGE=<name> CONFIG=<name> [REPRO_HOST=<host>]` | delegates to `scripts/run_on_host.sh` |
| `make report` | `results/*.json` -> `results/report.json` + `results/matrix.md`, inside `base` |
| `make figures` | `results/report.json` -> article PNGs, inside `base` |

`CONFIG` is one of `threads1 threads8 threadsmax nosimd512 hashseed0
hashseedfree` (`scripts/run_on_host.sh`'s header comment documents
exactly which environment variables each one sets and why).

`make report`/`make figures` both run inside the `base` image, per this
project's own design: they only need `reprobench.compare` (`report`) or
`reprobench.figures` (`figures`, Task 16's deliverable -- `make figures`
will fail with `No module named reprobench.figures` until that lands;
this is expected, not a bug in this Makefile).

## Multi-host scripts (`scripts/`)

- `run_on_host.sh <host> <image> <config> [--dry-run]` -- build one image
  and run one measurement container on one host (`local`, or an ssh host
  alias: `anakin`, `blade`, `dgxspark`), then collect the resulting JSON
  into `results/`. Runs the container under an 8-minute outer timeout
  (the runner has no in-process watchdog by design -- see
  `reprobench/runner.py`'s module docstring) and always cleans up the
  container it creates.
- `run_all.sh [--dry-run]` -- drives the full host/image/config matrix
  this project's design spec lays out, by calling `run_on_host.sh` once
  per combination. `--dry-run` is the pre-flight check: it reaches every
  host and prints every command it would run, without building, running,
  or copying anything.
- `fetch_model.sh [--dry-run]` -- pre-fetches `Qwen/Qwen2.5-0.5B-Instruct`
  (the fixed LLM this project measures) into each GPU host's Hugging Face
  cache, so a `gpu`-image run finds it already cached instead of
  attempting a ~1GB download inside the measurement window. A host that
  fails to fetch is recorded and the script continues to the rest.

These scripts are deliberately considerate of the fact that `anakin`,
`blade` and `dgxspark` are shared machines other people are actively
using: they reuse whatever was built/synced on a previous run rather than
re-pulling or rebuilding from scratch every time, and they always remove
the containers they start (never left running).

## Known issues / unverified

- **`gpu` was never built or run from this development host.** This
  project's own development machine is an aarch64 MacBook with no GPU
  reachable from a container. Only the pinned package versions were
  confirmed installable (as wheels, for the right Python/architecture
  combination) from here -- the Dockerfile's apt/venv steps and CUDA
  runtime behaviour are unexercised. Treat it as unverified until it is
  built and run on an actual GPU host.
- **`mkl` was never built, only dry-run solved.** `mkl` is x86_64-only
  and this host is aarch64; `micromamba create --dry-run` (run under
  linux/amd64 QEMU emulation) confirmed conda-forge resolves the exact
  pinned version set together, linked against MKL, but the image itself
  was never built.
- **`oldlibs` has no torch.** Every torch wheel available for Python 3.10
  (checked: 1.10.2 -- the oldest cp310 wheel that exists -- 1.12.1, and
  1.13.1) was compiled against a newer numpy C-API than numpy==1.21.6
  provides; `tensor.numpy()` raises `RuntimeError: Numpy is not
  available` at runtime as a result (verified directly against all
  three). Since numpy is fixed at 1.21.6 for this image (see above), there
  is no version of torch that is both installable for Python 3.10 and
  numpy-C-API-compatible with it. S7 (and any other torch-requiring
  probe) records a clean `missing requirement: torch` skip on this image
  instead of a misleading crash.
- **One pre-existing unit test conflicts with this image's own brief.**
  `tests/test_envinfo.py::test_collect_env_survives_torch_module_not_found_error`
  asserts that `collect_env()["torch"]` is `None` by relying on torch
  genuinely being absent from whatever environment runs the suite -- but
  the `base` image is required (by this same Task's brief) to include
  torch. `make test` against `base` therefore shows 178 passed, 1 failed,
  not a clean pass; see the Task 12/13 report for the full writeup. This
  is a reported conflict, not something silently patched over by editing
  that test.
