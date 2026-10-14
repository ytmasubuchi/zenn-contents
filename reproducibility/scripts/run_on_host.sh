#!/usr/bin/env bash
# run_on_host.sh <host> <image> <config> [--dry-run]
#
# Build one reprobench image and run one measurement container on one
# host, then collect the resulting JSON back into reproducibility/results/.
#
#   host:   local | anakin | blade | dgxspark
#           ("local" means this machine's Docker Desktop -- no rsync/ssh;
#           anything else is treated as an ssh host alias, used exactly
#           as given, per the Global Constraint that ssh hostnames are
#           these four names verbatim)
#   image:  base | py313 | oldpy | oldlibs | alpine | mkl | gpu
#   config: threads1 | threads8 | threadsmax | nosimd512 | hashseed0 | hashseedfree
#
# --dry-run prints every command this script would run (resolved config
# env vars included) and checks host reachability (ssh for a remote host,
# `docker info` for local), but builds nothing, runs nothing, and copies
# nothing back. This is the primitive `run_all.sh --dry-run` is built on.
#
# Written in bash (not POSIX sh) for arrays; written to run under bash
# 3.2 specifically, because that is what ships as /bin/bash on this
# project's own development Mac (verified: $BASH_VERSION is 3.2.57 here,
# vs. 5.1 on the Linux measurement hosts) -- no associative arrays, no
# `${var,,}`, nothing newer than 3.2 is used anywhere in this file.
set -euo pipefail

# --- outer timeout (Task 13 resolution 10) ---------------------------------
# The runner has no in-process watchdog by design (see reprobench/runner.py's
# module docstring): one hung probe would otherwise stall the whole process
# forever on a shared host. 480s (8 minutes) is chosen so that a single
# container run cannot consume the entire 10-minute-per-host budget the
# Global Constraints set, while still leaving headroom above the slowest
# probe set actually measured for this project (S3's 2048x2048 matmuls,
# S7/S8's MLP training loops, S9's 11 LLM generations of 128 tokens each).
# A run that is still going at 480s is treated as hung and killed; the
# real cost of that, per runner.py's own docstring, is one re-run.
TIMEOUT_SECS=480

# AVX-512 family feature names numpy 2.2.6 actually recognises on x86_64,
# confirmed via `numpy._core._multiarray_umath.__cpu_dispatch__` under
# linux/amd64 (Task 12): AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX
# AVX512_CLX AVX512_CNL AVX512_ICL. NPY_DISABLE_CPU_FEATURES takes this as
# one space-separated value, so it must be passed as a single `-e` flag
# with the spaces intact, not split into multiple flags.
NOSIMD512_FEATURES="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"

# cuBLAS deterministic workspace config, set unconditionally whenever
# IMAGE=gpu regardless of which of the six named CONFIGs is chosen (see
# below): reprobench/probes/s8_torch_gpu.py's
# S8_mlp_train10_cuda_deterministic records a `missing env var:` skip
# unless CUBLAS_WORKSPACE_CONFIG was already set when the process started
# (cuBLAS reads it at init; setting it later does nothing -- resolution
# 13). None of the six configs name this axis, so without this, that one
# probe would skip on every single GPU run. ":4096:8" is one of the two
# values CUDA's cuBLAS accepts (the other is ":16:8"); this project does
# not vary this value, so one fixed choice is used everywhere.
GPU_CUBLAS_WORKSPACE_CONFIG=":4096:8"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DRY_RUN=0
POSITIONAL=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) POSITIONAL+=("$arg") ;;
  esac
done

if [ "${#POSITIONAL[@]}" -ne 3 ]; then
  echo "Usage: $0 <local|anakin|blade|dgxspark> <image> <config> [--dry-run]" >&2
  exit 1
fi

HOST="${POSITIONAL[0]}"
IMAGE="${POSITIONAL[1]}"
CONFIG="${POSITIONAL[2]}"
CNAME="reprobench_${HOST}_${IMAGE}_${CONFIG}_$$"

# --- host_nproc: resolve "threadsmax" to an actual integer ------------------
# Local (this Mac) has no `nproc`; remote hosts are all Linux and do.
host_nproc() {
  if [ "$HOST" = "local" ]; then
    if command -v nproc >/dev/null 2>&1; then
      nproc
    else
      sysctl -n hw.ncpu
    fi
  else
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" 'nproc' < /dev/null
  fi
}

# --- config -> docker run `-e` flags (Task 13 resolution 12) ---------------
ENV_FLAGS=()
add_env() { ENV_FLAGS+=("-e" "$1=$2"); }

THREAD_VARS="OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS REPRO_THREADS"
# REPRO_THREADS is not one of the five library thread-count variables the
# brief names, but reprobench/probes/s5_sklearn.py and s6_lightgbm.py read
# it directly (not OMP_NUM_THREADS) to set n_jobs/num_threads -- without
# also setting it, threads1/threads8/threadsmax would leave S5/S6 running
# single-threaded regardless of the requested config, defeating the point
# of those three configs for those two probes.

case "$CONFIG" in
  threads1)
    for v in $THREAD_VARS; do add_env "$v" "1"; done
    ;;
  threads8)
    for v in $THREAD_VARS; do add_env "$v" "8"; done
    ;;
  threadsmax)
    if [ "$DRY_RUN" -eq 1 ] && [ "$HOST" != "local" ]; then
      # Avoid an extra ssh round-trip during a dry-run preview; the real
      # run resolves this for real.
      n="<host's nproc>"
    else
      n="$(host_nproc)"
    fi
    for v in $THREAD_VARS; do add_env "$v" "$n"; done
    ;;
  nosimd512)
    add_env "NPY_DISABLE_CPU_FEATURES" "$NOSIMD512_FEATURES"
    ;;
  hashseed0)
    add_env "PYTHONHASHSEED" "0"
    ;;
  hashseedfree)
    : # deliberately leave PYTHONHASHSEED unset
    ;;
  *)
    echo "run_on_host.sh: unknown CONFIG '$CONFIG' (expected one of: threads1 threads8 threadsmax nosimd512 hashseed0 hashseedfree)" >&2
    exit 1
    ;;
esac

# Every run must carry these three regardless of CONFIG (resolution 12):
# they are how the comparator labels and groups results.
add_env "REPRO_HOST" "$HOST"
add_env "REPRO_IMAGE" "$IMAGE"
add_env "REPRO_CONFIG" "$CONFIG"

# "--gpus all" is appended directly onto ENV_FLAGS (not kept in its own
# array) on purpose: ENV_FLAGS is guaranteed non-empty (REPRO_HOST/IMAGE/
# CONFIG are always added above), whereas a separate GPU-only array would
# be genuinely empty for every non-gpu image. Expanding a genuinely empty
# array under `set -u` on bash 3.2 (this project's own development Mac --
# see the file-level comment) raises "unbound variable", and the usual
# `"${arr[@]-}"` workaround inserts a spurious empty-string argument into
# whatever it's concatenated with instead of zero arguments (verified
# directly while writing this script -- both failure modes were hit).
if [ "$IMAGE" = "gpu" ]; then
  add_env "CUBLAS_WORKSPACE_CONFIG" "$GPU_CUBLAS_WORKSPACE_CONFIG"
  ENV_FLAGS+=("--gpus" "all")
fi

# --- host-level GPU metadata (Task 13 resolution 11) ------------------------
# The driver version is a host attribute, not a per-process one: the
# per-process result JSON records GPU name/compute capability/SM count
# (reprobench/envinfo.py's `_gpus`) but not the driver version, and a
# driver difference is a candidate explanation for a cross-GPU
# disagreement that the per-process record alone cannot rule in or out.
capture_gpu_info() {
  local out="$REPRO_DIR/results/${HOST}__nvidia-smi.csv"
  if [ "$HOST" = "local" ]; then
    echo "run_on_host.sh: no GPU metadata on host 'local' (not reachable from a container on this host, per the Task 12 host table)" >&2
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] would run: ssh $HOST nvidia-smi --query-gpu=driver_version,name --format=csv,noheader > $out"
    return 0
  fi
  mkdir -p "$REPRO_DIR/results"
  if ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" 'command -v nvidia-smi >/dev/null 2>&1' < /dev/null; then
    ssh -o BatchMode=yes "$HOST" 'nvidia-smi --query-gpu=driver_version,name --format=csv,noheader' < /dev/null > "$out"
    echo "run_on_host.sh: wrote $out"
  else
    echo "run_on_host.sh: nvidia-smi not found on $HOST, skipping GPU metadata" >&2
  fi
}

# --- run_container: docker run under an outer timeout, with guaranteed
# cleanup (Task 13 resolutions 10 and 15). Lives in its own sourced file,
# lib_run_container.sh (fix round 3) -- see that file's header comment
# for the full history (fix rounds 2 and 3 each fixed one half of this
# mechanism by breaking the other) and for why it's sourced rather than
# defined inline: the same function is shipped to a remote host verbatim
# via `declare -f run_container` + ssh below, AND exercised directly by
# scripts/test_run_container.sh, so there is exactly one implementation
# instead of two that could silently drift apart.
source "$SCRIPT_DIR/lib_run_container.sh"

# --- dry-run: print the plan, check reachability, stop ----------------------
if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry-run] host=$HOST image=$IMAGE config=$CONFIG"
  if [ "$HOST" = "local" ]; then
    if docker info >/dev/null 2>&1; then
      echo "[dry-run] local docker: reachable"
    else
      echo "[dry-run] local docker: NOT reachable" >&2
      exit 1
    fi
  else
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" true < /dev/null 2>/dev/null; then
      echo "[dry-run] ssh $HOST: reachable"
    else
      echo "[dry-run] ssh $HOST: NOT reachable" >&2
      exit 1
    fi
    echo "[dry-run] would run: rsync -az --delete --exclude results/ --exclude hf-cache/ --exclude __pycache__/ --exclude '*.pyc' --exclude .pytest_cache/ $REPRO_DIR/ $HOST:~/reprobench-run/"
  fi
  capture_gpu_info
  echo "[dry-run] would run: docker build -f docker/Dockerfile.$IMAGE -t reprobench:$IMAGE ."
  echo "[dry-run] would run: docker run --name $CNAME ${ENV_FLAGS[*]} -v <results>:/work/results reprobench:$IMAGE python -m reprobench.runner"
  echo "[dry-run] outer timeout: ${TIMEOUT_SECS}s"
  exit 0
fi

capture_gpu_info

# Ownership fixup (Task 13 resolution 15, fix round 1): every Dockerfile
# in this project runs its process as root (no `USER` instruction), so a
# container writing into a bind-mounted directory leaves root-owned files
# on the host side. On this Mac's Docker Desktop that happens not to be
# visible (its VM remaps container-root writes to the invoking macOS
# user), but on a real Linux host it is a genuine problem: confirmed
# directly on anakin, the host account could not even `rm` the resulting
# file without another container. Two fixes were possible here --
# running the measurement container itself as `--user $(id -u):$(id -g)`,
# or chowning the mount afterward with a throwaway root container. The
# former was rejected: it would also require reworking the `gpu` image's
# Hugging Face cache path, since `/root/.cache/huggingface` is not
# traversable by a non-root uid (`/root` itself is mode 700), and that
# risk didn't seem worth taking for every image just to avoid one extra
# `docker run` per invocation. So: chown-after, via a throwaway container
# using the image that was just built (already present locally, nothing
# extra to pull), run as its default root user specifically so it has
# permission to hand ownership back to the host user -- a regular user
# cannot chown a file to themselves that root already owns. On the
# remote path this now runs AFTER collecting the result (fix round 2,
# Important 1) and never aborts the script on its own (both chown calls
# below are `||`-guarded): a transient chown failure must not strand an
# already-successful measurement, or take down the rest of a multi-row
# campaign over a cosmetic ownership problem.
if [ "$HOST" = "local" ]; then
  cd "$REPRO_DIR"
  docker build -f "docker/Dockerfile.$IMAGE" -t "reprobench:$IMAGE" .
  mkdir -p results
  RUN_FLAGS=("${ENV_FLAGS[@]}" "-v" "$REPRO_DIR/results:/work/results")
  if [ "$IMAGE" = "gpu" ]; then
    mkdir -p "$REPRO_DIR/hf-cache"
    RUN_FLAGS+=("-v" "$REPRO_DIR/hf-cache:/root/.cache/huggingface")
  fi

  run_exit=0
  run_container "$IMAGE" "$CNAME" "$TIMEOUT_SECS" "${RUN_FLAGS[@]}" || run_exit=$?

  # fix round 2, Important 1: these must never abort the script under
  # `set -e` -- a transient chown failure is not worth losing a
  # successful measurement over, and locally there is no separate
  # collection step for it to get stranded ahead of, but the principle
  # (cleanup is best-effort, never fatal) should hold here too.
  uid_gid="$(id -u):$(id -g)"
  docker run --rm -v "$REPRO_DIR/results:/fix" "reprobench:$IMAGE" chown -R "$uid_gid" /fix \
    || echo "run_on_host.sh: chown fixup for results/ failed (non-fatal)" >&2
  if [ "$IMAGE" = "gpu" ]; then
    docker run --rm -v "$REPRO_DIR/hf-cache:/fix" "reprobench:$IMAGE" chown -R "$uid_gid" /fix \
      || echo "run_on_host.sh: chown fixup for hf-cache/ failed (non-fatal)" >&2
  fi

  if [ "$run_exit" -ne 0 ]; then
    echo "run_on_host.sh: measurement container exited non-zero ($run_exit)" >&2
    exit "$run_exit"
  fi
  echo "run_on_host.sh: wrote results/${HOST}__${IMAGE}__${CONFIG}.json (if the runner did not refuse/skip -- see the log above)"
else
  echo "run_on_host.sh: syncing to $HOST:~/reprobench-run/ ..."
  # fix round 2, CRITICAL: hf-cache/ was missing from this exclude list.
  # It lives inside this same ~/reprobench-run/ tree on the remote host
  # but does not exist in this local source tree, so without excluding
  # it, `--delete` wiped the ~1GB Qwen2.5-0.5B-Instruct snapshot
  # fetch_model.sh had just fetched on every single subsequent
  # run_on_host.sh invocation against that host -- the GPU measurement
  # would then either re-download it inside the 480-second timeout or
  # record `missing requirement: hf_model`, defeating the prefetch step
  # outright and re-pulling a multi-gigabyte download exactly where
  # resolution 15 forbids it.
  rsync -az --delete --exclude 'results/' --exclude 'hf-cache/' --exclude '__pycache__/' --exclude '*.pyc' --exclude '.pytest_cache/' "$REPRO_DIR/" "$HOST:~/reprobench-run/"

  ssh "$HOST" "mkdir -p ~/reprobench-run/results ~/reprobench-run/hf-cache && cd ~/reprobench-run && docker build -f docker/Dockerfile.$IMAGE -t reprobench:$IMAGE ." < /dev/null

  # Resolve ~ and the invoking user's uid:gid to concrete values on the
  # remote side BEFORE building the run_container call: that call is
  # shipped through printf %q below, which protects a literal "$HOME"
  # from ever being expanded remotely (its entire job is to make sure the
  # string it quotes is reproduced byte-for-byte, not re-interpreted) --
  # so "$HOME" would reach the remote docker run as the literal four
  # characters "$HOME", not the remote user's home directory.
  remote_home="$(ssh -o BatchMode=yes "$HOST" 'echo $HOME' < /dev/null)"
  remote_repro_dir="$remote_home/reprobench-run"
  remote_uid_gid="$(ssh -o BatchMode=yes "$HOST" 'echo "$(id -u):$(id -g)"' < /dev/null)"

  RUN_FLAGS=("${ENV_FLAGS[@]}" "-v" "$remote_repro_dir/results:/work/results")
  if [ "$IMAGE" = "gpu" ]; then
    RUN_FLAGS+=("-v" "$remote_repro_dir/hf-cache:/root/.cache/huggingface")
  fi

  # Ship run_container's definition verbatim (declare -f) and call it with
  # every argument individually shell-quoted (printf %q), so values that
  # contain spaces -- NPY_DISABLE_CPU_FEATURES's feature list is a single
  # space-separated value -- survive the trip through ssh's remote command
  # string intact instead of being split into extra arguments.
  remote_call="run_container"
  for a in "$IMAGE" "$CNAME" "$TIMEOUT_SECS" "${RUN_FLAGS[@]}"; do
    remote_call="$remote_call $(printf '%q' "$a")"
  done

  run_exit=0
  ssh "$HOST" "cd ~/reprobench-run && $(declare -f run_container); $remote_call" < /dev/null || run_exit=$?

  # fix round 2, Important 1: collect BEFORE chowning, and make the
  # chown best-effort (never fatal under `set -e`). The previous order
  # ran the chown fixups first -- a transient failure there aborted the
  # script after a successful measurement and before the JSON was ever
  # copied back, stranding a real result on the remote host while the
  # rest of a `run_all.sh` campaign had already died: a shared-host slot
  # spent for nothing that a plain reordering avoids.
  mkdir -p "$REPRO_DIR/results"
  scp_exit=0
  scp "$HOST:~/reprobench-run/results/${HOST}__${IMAGE}__${CONFIG}.json" "$REPRO_DIR/results/" < /dev/null || scp_exit=$?

  # Ownership fixup on the remote host -- see the file-level comment
  # above this if/else for why this is a separate, root-run throwaway
  # container rather than running the measurement container as a
  # non-root user. Best-effort: a failure here is reported, not fatal.
  ssh "$HOST" "docker run --rm -v $remote_repro_dir/results:/fix reprobench:$IMAGE chown -R $remote_uid_gid /fix" < /dev/null \
    || echo "run_on_host.sh: chown fixup for results/ on $HOST failed (non-fatal)" >&2
  if [ "$IMAGE" = "gpu" ]; then
    ssh "$HOST" "docker run --rm -v $remote_repro_dir/hf-cache:/fix reprobench:$IMAGE chown -R $remote_uid_gid /fix" < /dev/null \
      || echo "run_on_host.sh: chown fixup for hf-cache/ on $HOST failed (non-fatal)" >&2
  fi

  if [ "$run_exit" -ne 0 ]; then
    echo "run_on_host.sh: measurement container exited non-zero ($run_exit)" >&2
    exit "$run_exit"
  fi
  if [ "$scp_exit" -ne 0 ]; then
    echo "run_on_host.sh: failed to collect results/${HOST}__${IMAGE}__${CONFIG}.json from $HOST (scp exit $scp_exit)" >&2
    exit "$scp_exit"
  fi
  echo "run_on_host.sh: collected results/${HOST}__${IMAGE}__${CONFIG}.json from $HOST"
fi
