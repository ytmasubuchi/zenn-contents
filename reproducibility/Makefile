# reproducibility/Makefile
#
# Entry points for building the measurement images and reproducing the
# article's evidence table from a plain `git clone` (see README.md for the
# reader-facing walkthrough). Every target here runs inside a container --
# nothing in this Makefile runs reprobench code directly on the host
# (Global Constraint: measurement must happen in a container).
#
# Recipes avoid `.ONESHELL` and heredocs on purpose: this project's own
# development host ships GNU Make 3.81 (macOS's bundled `make`), which
# predates `.ONESHELL` (added in GNU Make 3.82) -- it silently ignores the
# directive and falls back to running every recipe line as its own `sh -c`
# invocation, which breaks any construct that depends on shared shell
# state across lines (verified directly against this Make while writing
# this file). Multi-step shell logic below (the `if` guard in `build`, the
# `for` loop in `build-all-cpu`) therefore uses classic trailing-backslash
# line continuation, which every Make version supports; `report` avoids
# the issue entirely by keeping its Python one-liner to a single recipe
# line instead of a heredoc.
IMAGE ?=
CONFIG ?=
REPRO_HOST ?= local

RESULTS_DIR := $(CURDIR)/results
IMAGES_DIR := $(CURDIR)/../images/reproducibility-of-randomness
HOST_ARCH := $(shell uname -m)

CPU_IMAGES := base py313 oldpy oldlibs alpine

.PHONY: test build build-all-cpu run report figures

# `make test`: build the base image and run the unit suite inside it.
# This is the fully-provisioned counterpart to this project's bare
# verification command (`pip install numpy==2.2.6 pytest`): base also
# carries scipy/scikit-learn/lightgbm/torch, so tests that
# `pytest.importorskip` those libraries actually run here instead of
# skipping. Passes cleanly: 178 passed, 1 skipped (fix round 1 --
# test_envinfo.py::test_collect_env_survives_torch_module_not_found_error
# now skips when torch is importable, since it only has something to
# prove where torch is genuinely absent; see the Task 12/13 report).
test:
	docker build -f docker/Dockerfile.base -t reprobench:base .
	docker run --rm reprobench:base pytest -q -W error

# `make build IMAGE=<name>`: build exactly one image from
# docker/Dockerfile.<name>, tagged reprobench:<name>.
#
# mkl is skipped (not failed) on non-x86_64 hosts: conda-forge does not
# publish an MKL-linked libblas for linux-aarch64 at all, so attempting
# the build there would fail deep inside a conda solve with a confusing
# error instead of the clear message below (Task 12 resolution 5).
build:
	if [ -z "$(IMAGE)" ]; then \
		echo "Usage: make build IMAGE=<base|py313|oldpy|oldlibs|alpine|mkl|gpu>" >&2; \
		exit 1; \
	fi; \
	if [ "$(IMAGE)" = "mkl" ] && [ "$(HOST_ARCH)" != "x86_64" ] && [ "$(HOST_ARCH)" != "amd64" ]; then \
		echo "make: skipping IMAGE=mkl on host arch $(HOST_ARCH) -- mkl is x86_64-only (Task 12 resolution 5)"; \
		exit 0; \
	fi; \
	docker build -f docker/Dockerfile.$(IMAGE) -t reprobench:$(IMAGE) .

# `make build-all-cpu`: build every CPU-only image (not gpu, which needs
# an actual GPU host to be meaningful). mkl is included but silently
# skipped on non-x86_64 hosts via `build`'s own guard above, exactly as
# the Task 12 brief's Step 3 describes ("mkl は x86_64 のみ成功 -- Mac で
# は skip される").
build-all-cpu:
	for img in $(CPU_IMAGES) mkl; do \
		$(MAKE) build IMAGE=$$img || exit 1; \
	done

# `make run IMAGE=<name> CONFIG=<name>`: delegates to
# scripts/run_on_host.sh so the config-name -> environment-variable
# mapping (threads1/threads8/threadsmax/nosimd512/hashseed0/hashseedfree)
# has exactly one implementation, shared with the real multi-host
# orchestration in Task 13's scripts rather than duplicated here.
# REPRO_HOST defaults to "local" (this machine's Docker Desktop); pass
# REPRO_HOST=anakin (etc.) to run on a remote measurement host instead.
run:
	if [ -z "$(IMAGE)" ] || [ -z "$(CONFIG)" ]; then \
		echo "Usage: make run IMAGE=<name> CONFIG=<threads1|threads8|threadsmax|nosimd512|hashseed0|hashseedfree>" >&2; \
		exit 1; \
	fi; \
	./scripts/run_on_host.sh $(REPRO_HOST) $(IMAGE) $(CONFIG)

# `make report`: results/*__*__*.json (the runner's
# `{host}__{image}__{config}.json` files) -> results/report.json +
# results/matrix.md, via
# reprobench.compare's public API (load_results/build_report/to_markdown).
# Runs inside the base image, per the Task 12 brief ("report と figures
# も base イメージ内で実行する").
#
# The glob is `*__*__*.json`, not `*.json`: a bare `*.json` would also
# match results/report.json itself on a second run, feeding the report
# its own previous output as if it were a measurement file.
#
# This is a single recipe line (not a heredoc) on purpose -- see the
# file-level comment above on why this Makefile avoids heredocs.
#
# The final line chowns $(RESULTS_DIR) back to the invoking user (fix
# round 2, small finding): the container above runs as root like every
# image in this project, so report.json/matrix.md would otherwise land
# root-owned inside a directory the reader's own clone created -- the
# same bug class fixed in scripts/run_on_host.sh for the measurement
# containers, missed here originally.
report:
	docker build -f docker/Dockerfile.base -t reprobench:base .
	mkdir -p $(RESULTS_DIR)
	docker run --rm -v $(RESULTS_DIR):/work/results reprobench:base python3 -c "import glob, json, sys; from reprobench.compare import build_report, load_results, to_markdown; paths = sorted(glob.glob('results/*__*__*.json')) or sys.exit('make report: no results/*__*__*.json files found (run make run for at least one host/image/config first)'); results = load_results(paths); report = build_report(results); json.dump(report, open('results/report.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False, allow_nan=False); open('results/matrix.md', 'w', encoding='utf-8').write(to_markdown(report)); print(f'make report: wrote results/report.json and results/matrix.md from {len(paths)} file(s)')"
	docker run --rm -v $(RESULTS_DIR):/fix reprobench:base chown -R $$(id -u):$$(id -g) /fix

# `make figures`: results/report.json -> article figures (F4, F8; PNGs
# land in images/reproducibility-of-randomness/, the repo-root sibling of
# this directory -- see spec section 6). Runs inside the base image, same
# as `report`.
#
# reprobench/figures.py is Task 16's deliverable, not Task 12's -- this
# target is wired in advance per the Task 12 brief's resolution 8 (the
# brief lists `report` and `figures` as Makefile targets to make real).
# Until figures.py lands this target fails with
# `No module named reprobench.figures`, which is expected, not a bug in
# this Makefile. The images directory is bind-mounted at
# /images/reproducibility-of-randomness, mirroring its real path one
# level up from WORKDIR /work (../images/reproducibility-of-randomness),
# so that whatever figures.py writes there lands in the real repo-root
# images/ directory on the host.
# The final two lines chown $(RESULTS_DIR)/$(IMAGES_DIR) back to the
# invoking user, for the same reason as `report`'s chown above.
figures:
	docker build -f docker/Dockerfile.base -t reprobench:base .
	mkdir -p $(RESULTS_DIR)
	mkdir -p $(IMAGES_DIR)
	docker run --rm -v $(RESULTS_DIR):/work/results -v $(IMAGES_DIR):/images/reproducibility-of-randomness reprobench:base python -m reprobench.figures
	docker run --rm -v $(RESULTS_DIR):/fix reprobench:base chown -R $$(id -u):$$(id -g) /fix
	docker run --rm -v $(IMAGES_DIR):/fix reprobench:base chown -R $$(id -u):$$(id -g) /fix
