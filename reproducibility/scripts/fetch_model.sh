#!/usr/bin/env bash
# fetch_model.sh [--dry-run]
#
# Pre-fetch Qwen2.5-0.5B-Instruct (the fixed LLM this project measures --
# reprobench.registry.HF_MODEL_ID) into each GPU host's Hugging Face
# cache, so a later `run_on_host.sh <gpu-host> gpu <config>` finds it
# already cached (reprobench.registry.check_requirement("hf_model") looks
# for exactly this) instead of attempting a ~1GB download from inside the
# 10-minute measurement window itself (spec section 8 risk register).
#
# A host that fails to fetch is recorded and the script continues to the
# remaining hosts (Task 13 Step 3) -- S9 will simply skip with
# "missing requirement: hf_model" on whichever host failed, which is a
# clean, well-understood skip, not a crash.
#
# GPU_HOSTS excludes "local": the Task 12 host table marks the MacBook's
# GPU as not reachable from a container, so there is nothing to fetch
# against on this machine.
#
# Same bash-3.2-compatible, ssh-stdin-safe style as run_on_host.sh /
# run_all.sh -- see those files' header comments for why.
set -euo pipefail

GPU_HOSTS="anakin blade dgxspark"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *)
      echo "fetch_model.sh: unknown argument '$arg' (only --dry-run is accepted)" >&2
      exit 1
      ;;
  esac
done

FAILED=""

# The model id is read from reprobench.registry.HF_MODEL_ID inside the
# container (not duplicated as a string literal here): that module's own
# docstring calls it "the single source of truth" for the measured model,
# specifically so nothing else in the project names it independently.
FETCH_PY='from huggingface_hub import snapshot_download; from reprobench.registry import HF_MODEL_ID; snapshot_download(repo_id=HF_MODEL_ID); print("fetch_model: cached", HF_MODEL_ID)'

for host in $GPU_HOSTS; do
  echo "=== $host ==="

  # < /dev/null on every ssh call: without it, ssh would inherit this
  # for-loop's stdin and, on a multi-host loop like this one, could
  # consume input meant for later iterations (see run_all.sh for the
  # directly-observed version of this bug).
  if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" true < /dev/null 2>/dev/null; then
    echo "fetch_model.sh: $host unreachable, skipping" >&2
    FAILED="$FAILED $host"
    continue
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] ssh $host: reachable"
    echo "[dry-run] would run: rsync -az --delete --exclude results/ --exclude __pycache__/ --exclude '*.pyc' --exclude .pytest_cache/ $REPRO_DIR/ $host:~/reprobench-run/"
    echo "[dry-run] would run: ssh $host 'mkdir -p ~/reprobench-run/hf-cache && cd ~/reprobench-run && docker build -f docker/Dockerfile.gpu -t reprobench:gpu .'"
    echo "[dry-run] would run: ssh $host \"docker run --rm -v ~/reprobench-run/hf-cache:/root/.cache/huggingface reprobench:gpu python3 -c '$FETCH_PY'\""
    continue
  fi

  if ! rsync -az --delete --exclude 'results/' --exclude '__pycache__/' --exclude '*.pyc' --exclude '.pytest_cache/' "$REPRO_DIR/" "$host:~/reprobench-run/"; then
    echo "fetch_model.sh: rsync to $host failed, skipping" >&2
    FAILED="$FAILED $host"
    continue
  fi

  if ! ssh "$host" "mkdir -p ~/reprobench-run/hf-cache && cd ~/reprobench-run && docker build -f docker/Dockerfile.gpu -t reprobench:gpu ." < /dev/null; then
    echo "fetch_model.sh: docker build on $host failed, skipping" >&2
    FAILED="$FAILED $host"
    continue
  fi

  if ! ssh "$host" "docker run --rm -v ~/reprobench-run/hf-cache:/root/.cache/huggingface reprobench:gpu python3 -c '$FETCH_PY'" < /dev/null; then
    echo "fetch_model.sh: model download on $host failed, skipping" >&2
    FAILED="$FAILED $host"
    continue
  fi

  echo "fetch_model.sh: $host OK"
done

echo ""
if [ -n "$FAILED" ]; then
  echo "fetch_model.sh: failed host(s):$FAILED" >&2
  exit 1
fi
echo "fetch_model.sh: all GPU hosts fetched successfully"
