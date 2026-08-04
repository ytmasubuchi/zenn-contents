#!/usr/bin/env bash
# Runs the GPU experiments (E8a-E8e) and writes JSON results into results/.
#
# Separate from run_all.sh (which is CPU-only, E1-E6) on purpose: this
# script needs `--gpus all` on every `docker run`, and depends on host
# state (an NVIDIA GPU + nvidia-container-runtime) that run_all.sh's CI/
# reproduction story never assumed.
#
# IMPORTANT: this host's GPU is shared with a resident vLLM server that
# holds ~23/24.5GB of VRAM at all times. This script must NEVER attempt to
# stop/kill that process. Tensors here are kept deliberately tiny (512x512
# matmul, small MLP/conv, etc.) to fit in whatever VRAM remains. If even
# that is insufficient, the Python scripts catch the CUDA RuntimeError
# themselves (see scripts/_common.py::run_with_fallback) and record the
# failure in the result JSON instead of crashing -- so this script itself
# does not need (and must not do) any special OOM handling.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
RESULTS_DIR="$(pwd)/results"
RAW_DIR="$RESULTS_DIR/e8_raw"
mkdir -p "$RESULTS_DIR" "$RAW_DIR"/e8a_run1 "$RAW_DIR"/e8a_run2 "$RAW_DIR"/e8a_run3 \
  "$RAW_DIR"/e8b_run1 "$RAW_DIR"/e8b_run2 "$RAW_DIR"/e8b_run3 "$RAW_DIR"/e8c "$RAW_DIR"/e8d

IMG=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

log "Using pre-pulled image $IMG (torch 2.5.1+cu124, matches the CPU-side torch==2.5.1 used by E1-E7)."
docker image inspect "$IMG" >/dev/null 2>&1 || { log "ERROR: image $IMG not found locally. Pull it first."; exit 1; }

### --- E8a: repeated runs, default (non-deterministic) settings --------
log "E8a: GPU repeated reproducibility, default settings, 3 fresh containers..."
for i in 1 2 3; do
  docker run --rm --gpus all \
    -v "$(pwd)/scripts":/work/scripts:ro -v "$RESULTS_DIR":/work/results \
    -w /work/scripts "$IMG" \
    python e8a_gpu_repeat.py "/work/results/e8_raw/e8a_run${i}" \
    > "$RESULTS_DIR/e8a_run${i}.json"
done
docker run --rm --gpus all \
  -v "$(pwd)/scripts":/work/scripts:ro -v "$RESULTS_DIR":/work/results \
  -w /work/scripts "$IMG" \
  python compare.py "E8a_run1_vs_run2" /work/results/e8a_run1.json /work/results/e8a_run2.json \
  > "$RESULTS_DIR/e8a_compare_1v2.json"
docker run --rm --gpus all \
  -v "$(pwd)/scripts":/work/scripts:ro -v "$RESULTS_DIR":/work/results \
  -w /work/scripts "$IMG" \
  python compare.py "E8a_run1_vs_run3" /work/results/e8a_run1.json /work/results/e8a_run3.json \
  > "$RESULTS_DIR/e8a_compare_1v3.json"

### --- E8b: repeated runs, deterministic-mode settings -------------------
log "E8b: GPU repeated reproducibility, deterministic mode, 3 fresh containers..."
for i in 1 2 3; do
  docker run --rm --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -v "$(pwd)/scripts":/work/scripts:ro -v "$RESULTS_DIR":/work/results \
    -w /work/scripts "$IMG" \
    python e8b_gpu_deterministic.py "/work/results/e8_raw/e8b_run${i}" \
    > "$RESULTS_DIR/e8b_run${i}.json"
done
docker run --rm --gpus all \
  -v "$(pwd)/scripts":/work/scripts:ro -v "$RESULTS_DIR":/work/results \
  -w /work/scripts "$IMG" \
  python compare.py "E8b_run1_vs_run2" /work/results/e8b_run1.json /work/results/e8b_run2.json \
  > "$RESULTS_DIR/e8b_compare_1v2.json"
docker run --rm --gpus all \
  -v "$(pwd)/scripts":/work/scripts:ro -v "$RESULTS_DIR":/work/results \
  -w /work/scripts "$IMG" \
  python compare.py "E8b_run1_vs_run3" /work/results/e8b_run1.json /work/results/e8b_run3.json \
  > "$RESULTS_DIR/e8b_compare_1v3.json"

### --- E8c: CPU vs GPU, same seed / same input --------------------------
log "E8c: CPU vs GPU comparison..."
docker run --rm --gpus all \
  -v "$(pwd)/scripts":/work/scripts:ro -v "$RESULTS_DIR":/work/results \
  -w /work/scripts "$IMG" \
  python e8c_cpu_vs_gpu.py /work/results/e8_raw/e8c > "$RESULTS_DIR/e8c_cpu_vs_gpu.json"

### --- E8d: TF32 on vs off ------------------------------------------------
log "E8d: TF32 effect on float32 matmul..."
docker run --rm --gpus all \
  -v "$(pwd)/scripts":/work/scripts:ro -v "$RESULTS_DIR":/work/results \
  -w /work/scripts "$IMG" \
  python e8d_tf32.py /work/results/e8_raw/e8d > "$RESULTS_DIR/e8d_tf32.json"

### --- E8e: ULP/abs/rel diff stats for whatever mismatched --------------
log "E8e: computing diff statistics for mismatching items..."
docker run --rm --gpus all \
  -v "$(pwd)/scripts":/work/scripts:ro -v "$RESULTS_DIR":/work/results \
  -w /work/scripts "$IMG" \
  python e8e_ulp_diff.py /work/results > "$RESULTS_DIR/e8_ulp_diff.json"

log "Done. See $RESULTS_DIR for all raw + comparison JSON (e8*.json)."
