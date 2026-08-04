#!/usr/bin/env bash
# Runs experiments E1-E6 end to end and writes JSON results into results/.
#
# Per project convention, ALL Python execution happens inside Docker
# containers (never directly on the host), including the uv.lock
# generation step documented in the README.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
RESULTS_DIR="$(pwd)/results"
mkdir -p "$RESULTS_DIR"

IMG_BASE_AMD64=random-repro:base-amd64
IMG_BASE_ARM64=random-repro:base-arm64
IMG_OLD_AMD64=random-repro:old-amd64

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

### --- Build images -----------------------------------------------------
log "Building base (amd64) image..."
docker build --platform linux/amd64 -f docker/Dockerfile.base -t "$IMG_BASE_AMD64" . || exit 1

log "Building old (amd64, E4) image..."
docker build --platform linux/amd64 -f docker/Dockerfile.old -t "$IMG_OLD_AMD64" . || exit 1

log "Building base (arm64, E5) image -- this is slow under QEMU, be patient..."
if docker build --platform linux/arm64 -f docker/Dockerfile.base -t "$IMG_BASE_ARM64" .; then
  ARM64_OK=1
else
  log "WARNING: arm64 build failed; E5 will be skipped."
  ARM64_OK=0
fi

### --- E1: baseline reproducibility (same image, run 3x) ---------------
log "E1: baseline reproducibility, 3 independent container runs..."
for i in 1 2 3; do
  docker run --rm --platform linux/amd64 \
    -v "$(pwd)/scripts":/work/scripts:ro -w /work/scripts \
    "$IMG_BASE_AMD64" python e1_baseline.py > "$RESULTS_DIR/e1_run${i}.json"
done
docker run --rm --platform linux/amd64 \
  -v "$(pwd)/scripts":/work/scripts:ro -v "$RESULTS_DIR":/work/results \
  -w /work/scripts "$IMG_BASE_AMD64" \
  python compare.py "E1_run1_vs_run2" /work/results/e1_run1.json /work/results/e1_run2.json \
  > "$RESULTS_DIR/e1_compare_1v2.json"
docker run --rm --platform linux/amd64 \
  -v "$(pwd)/scripts":/work/scripts:ro -v "$RESULTS_DIR":/work/results \
  -w /work/scripts "$IMG_BASE_AMD64" \
  python compare.py "E1_run1_vs_run3" /work/results/e1_run1.json /work/results/e1_run3.json \
  > "$RESULTS_DIR/e1_compare_1v3.json"

### --- E2: PYTHONHASHSEED ------------------------------------------------
log "E2: PYTHONHASHSEED effects..."
for i in 1 2 3; do
  docker run --rm --platform linux/amd64 \
    -v "$(pwd)/scripts":/work/scripts:ro -w /work/scripts \
    "$IMG_BASE_AMD64" python e2_hashseed.py > "$RESULTS_DIR/e2_unset_run${i}.json"
done
for i in 1 2 3; do
  docker run --rm --platform linux/amd64 -e PYTHONHASHSEED=0 \
    -v "$(pwd)/scripts":/work/scripts:ro -w /work/scripts \
    "$IMG_BASE_AMD64" python e2_hashseed.py > "$RESULTS_DIR/e2_seed0_run${i}.json"
done

### --- E3: same image, separate containers ------------------------------
log "E3: same image / separate container instances..."
for i in 1 2 3; do
  docker run --rm --platform linux/amd64 \
    -v "$(pwd)/scripts":/work/scripts:ro -w /work/scripts \
    "$IMG_BASE_AMD64" python e1_baseline.py > "$RESULTS_DIR/e3_container${i}.json"
done
docker run --rm --platform linux/amd64 \
  -v "$(pwd)/scripts":/work/scripts:ro -v "$RESULTS_DIR":/work/results \
  -w /work/scripts "$IMG_BASE_AMD64" \
  python compare.py "E3_container1_vs_container2" /work/results/e3_container1.json /work/results/e3_container2.json \
  > "$RESULTS_DIR/e3_compare_1v2.json"
docker run --rm --platform linux/amd64 \
  -v "$(pwd)/scripts":/work/scripts:ro -v "$RESULTS_DIR":/work/results \
  -w /work/scripts "$IMG_BASE_AMD64" \
  python compare.py "E3_container1_vs_container3" /work/results/e3_container1.json /work/results/e3_container3.json \
  > "$RESULTS_DIR/e3_compare_1v3.json"

### --- E4: same machine, different image/library versions ---------------
log "E4: library-version comparison (old image vs new/base image)..."
docker run --rm --platform linux/amd64 \
  -v "$(pwd)/scripts":/work/scripts:ro -w /work/scripts \
  "$IMG_OLD_AMD64" python e4_version_stream.py > "$RESULTS_DIR/e4_old.json"
docker run --rm --platform linux/amd64 \
  -v "$(pwd)/scripts":/work/scripts:ro -w /work/scripts \
  "$IMG_BASE_AMD64" python e4_version_stream.py > "$RESULTS_DIR/e4_new.json"
docker run --rm --platform linux/amd64 \
  -v "$(pwd)/scripts":/work/scripts:ro -v "$RESULTS_DIR":/work/results \
  -w /work/scripts "$IMG_BASE_AMD64" \
  python compare.py "E4_old_vs_new" /work/results/e4_old.json /work/results/e4_new.json \
  > "$RESULTS_DIR/e4_compare.json"

### --- E5: x86_64 vs arm64 (QEMU) ---------------------------------------
if [ "$ARM64_OK" = "1" ]; then
  log "E5: x86_64 vs arm64 (QEMU)..."
  docker run --rm --platform linux/amd64 \
    -v "$(pwd)/scripts":/work/scripts:ro -w /work/scripts \
    "$IMG_BASE_AMD64" python e5_arch_compare.py > "$RESULTS_DIR/e5_amd64.json"
  docker run --rm --platform linux/arm64 \
    -v "$(pwd)/scripts":/work/scripts:ro -w /work/scripts \
    "$IMG_BASE_ARM64" python e5_arch_compare.py > "$RESULTS_DIR/e5_arm64.json"
  docker run --rm --platform linux/amd64 \
    -v "$(pwd)/scripts":/work/scripts:ro -v "$RESULTS_DIR":/work/results \
    -w /work/scripts "$IMG_BASE_AMD64" \
    python compare.py "E5_amd64_vs_arm64" /work/results/e5_amd64.json /work/results/e5_arm64.json \
    > "$RESULTS_DIR/e5_compare.json"
else
  log "E5 SKIPPED: arm64 image build failed (see docker build logs)."
fi

### --- E6: parallelism / thread-count non-determinism -------------------
log "E6: parallel / thread non-determinism..."
docker run --rm --platform linux/amd64 -e OMP_NUM_THREADS=1 \
  -v "$(pwd)/scripts":/work/scripts:ro -w /work/scripts \
  "$IMG_BASE_AMD64" python e6_parallel_nondeterminism.py > "$RESULTS_DIR/e6_omp1.json"
docker run --rm --platform linux/amd64 -e OMP_NUM_THREADS=8 \
  -v "$(pwd)/scripts":/work/scripts:ro -w /work/scripts \
  "$IMG_BASE_AMD64" python e6_parallel_nondeterminism.py > "$RESULTS_DIR/e6_omp8.json"
docker run --rm --platform linux/amd64 \
  -v "$(pwd)/scripts":/work/scripts:ro -v "$RESULTS_DIR":/work/results \
  -w /work/scripts "$IMG_BASE_AMD64" \
  python compare.py "E6_omp1_vs_omp8" /work/results/e6_omp1.json /work/results/e6_omp8.json \
  > "$RESULTS_DIR/e6_compare_omp.json"

log "Done. See $RESULTS_DIR for all raw + comparison JSON."
