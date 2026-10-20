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

Prerequisites: `docker` and `make` (both required for every command
below); `rsync` and `ssh` are only needed for running on a remote host
(`scripts/run_on_host.sh`/`run_all.sh`/`fetch_model.sh`), not for the
`local` walkthrough here.

```sh
git clone git@github.com:ytmasubuchi/zenn-contents.git
cd zenn-contents/reproducibility
make build IMAGE=base      # builds reprobench:base
make run IMAGE=base CONFIG=threadsmax
#   -> results/local__base__threadsmax.json
make report                # results/*__*__*.json -> results/report.json + results/matrix.md
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
| `oldlibs` | `python:3.10-slim` | numpy/scikit-learn version only | numpy==1.21.6, scikit-learn==1.0.2, lightgbm==4.5.0; **no torch, by design** (see below) |
| `alpine` | `python:3.11-alpine` | libc (musl vs. glibc) | numpy only -- no scipy/scikit-learn/lightgbm/torch |
| `mkl` | `mambaorg/micromamba` + conda-forge | BLAS implementation (MKL vs. OpenBLAS) | **x86_64 only**; `make build`/`make build-all-cpu` skip it on aarch64 |
| `gpu` | `nvidia/cuda:13.0.1-runtime-ubuntu24.04` | adds CUDA 13 + torch (cu130) + transformers | built and run successfully on anakin's RTX 4090 (not this development host -- see below) |

`oldlibs`'s library-version condition is measured on numpy and
scikit-learn (and lightgbm), not torch. Its purpose is the NEP 19
stream-compatibility condition (NumPy's documented stream-breaking
changes at `1.19.0` and `1.22.0`), which S1 through S4 already cover with
numpy alone; S5 and S6 (scikit-learn, lightgbm) ride along because both
install and actually run against numpy==1.21.6/Python 3.10 without any
version friction (confirmed: a real `LogisticRegression` fit/predict and
a full `lgb.train()`/`predict()` round-trip, not just a clean import, on
both linux/aarch64 and linux/amd64). torch is left out on purpose: adding
an old torch here would measure torch's own version compatibility, which
the article makes no claim about, not the numpy/scikit-learn condition
this image exists for. See `docker/Dockerfile.oldlibs`'s comment for the
investigation into whether a compatible old torch exists at all (short
answer: the earlier claim that none does was an artifact of testing on
aarch64 -- on x86_64, the architecture `oldlibs` actually runs on
(blade), a numpy==1.21.6-compatible torch wheel does exist. That finding
doesn't change this image's contents; it's recorded for accuracy, not
because it changes the decision.)

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
| `make report` | `results/*__*__*.json` -> `results/report.json` + `results/matrix.md`, inside `base` |
| `make figures` | `results/report.json` -> article PNGs, inside `base` |

`CONFIG` is one of `threads1 threads8 threadsmax nosimd512 hashseed0
hashseedfree s10batch practice` (`scripts/run_on_host.sh`'s header
comment documents exactly which environment variables each one sets and
why; `s10batch` additionally restricts the run to the S10
batch-invariance probes and is only meaningful with `IMAGE=gpu`;
`practice` restricts the run to the S11-S15 probe group (Task 2) and,
unlike every other config, leaves `CUBLAS_WORKSPACE_CONFIG` unset on
`IMAGE=gpu` -- that opt-out is itself what S13 measures).

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
  container it creates. Every Dockerfile in this project runs its process
  as root, so the measurement container's writes into `results/` (and,
  for `gpu`, the Hugging Face cache) land root-owned on the host --
  confirmed directly on anakin, a real Linux host, where the invoking
  account couldn't even delete the resulting file without another
  container. `run_on_host.sh` fixes this itself after every run, via a
  throwaway container (the same image, already built) that chowns the
  mount back to the invoking user.
- `run_all.sh [--dry-run]` -- drives the full host/image/config matrix
  this project's design spec lays out, by calling `run_on_host.sh` once
  per combination. `--dry-run` is the pre-flight check: it reaches every
  host and prints every command it would run, without building, running,
  or copying anything.
- `fetch_model.sh [--dry-run]` -- pre-fetches `Qwen/Qwen2.5-0.5B-Instruct`
  (the fixed LLM this project measures) into each GPU host's Hugging Face
  cache, so a `gpu`-image run finds it already cached instead of
  attempting a ~1GB download inside the measurement window. A host that
  fails to fetch is recorded and the script continues to the rest. Fixes
  the same root-owned-cache-file problem `run_on_host.sh` does, for the
  same reason.
- `lib_run_container.sh` -- not invoked directly. The outer-timeout
  mechanism (`run_container`), factored out of `run_on_host.sh` so it is
  defined in exactly one place: shipped verbatim to a remote host via
  `declare -f` + ssh, and exercised directly by `test_run_container.sh`
  below, instead of risking two copies that drift apart.
- `test_run_container.sh` -- a regression test for `run_container`
  (entirely local docker, no remote host touched). Asserts both of the
  behaviours this mechanism has to have: returning promptly when a
  container finishes on its own, and killing a container that outlives
  its timeout and still returning promptly (not waiting out the
  container's full, much longer runtime). Each of fix rounds 2 and 3
  broke one of these while fixing the other; this test exists so that
  class of regression gets caught automatically instead of needing a
  human to think to check the one path nothing else exercises.

These scripts are deliberately considerate of the fact that `anakin`,
`blade` and `dgxspark` are shared machines other people are actively
using: they reuse whatever was built/synced on a previous run rather than
re-pulling or rebuilding from scratch every time, and they always remove
the containers they start (never left running).

## Known issues / unverified

- **`gpu` was built and run on anakin's RTX 4090 (CUDA 13, driver 580.x),
  not from this development host.** This project's own development
  machine is an aarch64 MacBook with no GPU reachable from a container,
  so the build/run itself had to happen on a real GPU host. Confirmed
  there: the build succeeds, `torch.cuda.is_available()` is `True`,
  `python -m reprobench.runner --probes S7,S8` completes and writes a
  well-formed results JSON (all ten probes recorded, ~2.2s wall time),
  and the env block correctly carries the torch/CUDA/cuDNN versions, both
  TF32 flags, the GPU name/SM count, and `CUBLAS_WORKSPACE_CONFIG`. **An
  image built from this Dockerfile is left cached on anakin** so Task 14
  does not have to re-pull the multi-gigabyte CUDA base image again;
  remove it once measurements there are finished.
- **`mkl` was never built as the real image, only dry-run solved and
  spot-checked with a throwaway Dockerfile.** `mkl` is x86_64-only and
  this host is aarch64; `micromamba create --dry-run` (run under
  linux/amd64 QEMU emulation) confirmed conda-forge resolves the exact
  pinned version set together, linked against MKL. Separately (fix round
  2): `mambaorg/micromamba`'s base image defaults to a non-root user,
  which broke both the measurement write into `/work/results` and the
  chown-after fixup -- fixed with an explicit `USER root`, verified (via
  the same throwaway-Dockerfile technique, not the real image) to still
  load numpy linked against MKL correctly afterward. The real
  `Dockerfile.mkl` (with its actual pins) has still never been built or
  run.
- **One pre-existing unit test conflicted with `base`'s own brief, now
  fixed.**
  `tests/test_envinfo.py::test_collect_env_survives_torch_module_not_found_error`
  asserted that `collect_env()["torch"]` is `None` by relying on torch
  genuinely being absent from whatever environment runs the suite -- but
  `base` is required (by this same Task's brief) to include torch. Fixed
  by skipping this test when torch is importable: the monkeypatched
  sibling test (`test_collect_env_survives_missing_torch`) already covers
  the same code path unconditionally, so this test now only has something
  left to prove on an image that genuinely lacks torch (`alpine`).
  `make test` against `base` passes cleanly as a result. See the Task
  12/13 report for the cross-task-edit note (this test file belongs to
  Task 2).
