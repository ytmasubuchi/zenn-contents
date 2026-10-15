#!/usr/bin/env bash
# run_all.sh [--dry-run]
#
# Pre-flight / orchestration driver for the full measurement campaign:
# iterates every (host, image, config) combination this project's design
# spec (section 4-6, "イメージ割り当て") and the Task 14 brief's own step
# descriptions lay out, calling scripts/run_on_host.sh once per
# combination.
#
# `--dry-run` is the pre-flight check Task 13 asks for: it reaches every
# host (an ssh connectivity check per unique host, done once rather than
# once per matrix row) and prints every command the real run would issue,
# without building, running, or copying anything.
#
# Without --dry-run this actually runs the campaign -- Task 12/13's own
# instructions are explicit that executing real measurements on the
# remote hosts is Task 14's job, not this one. This script exists so
# Task 14 has a working driver to call (or adjust) rather than
# re-deriving the host/image/config matrix from the spec by hand; the
# matrix below is this project's plan as of Task 12/13, not a guarantee
# that Task 14 will run it unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *)
      echo "run_all.sh: unknown argument '$arg' (only --dry-run is accepted)" >&2
      exit 1
      ;;
  esac
done

# --- the campaign matrix ----------------------------------------------------
# One "host image config" triple per line. See Task 14 brief Steps 1-3 and
# spec section 4-6 for where each row comes from:
#   - base on all 4 hosts x (threads1/threads8/threadsmax/hashseed0/
#     hashseedfree), plus nosimd512 on blade only (blade is the one host
#     with AVX-512; Task 14 Step 1).
#   - py313/oldpy/oldlibs/mkl/alpine on blade, alpine also on dgxspark
#     (Task 14 Step 2). These use threadsmax (no artificial thread cap)
#     since the axis under test here is Python/library/BLAS/libc, not
#     thread count -- the brief does not name a config for this step, so
#     this is this project's choice, not a measured requirement.
#   - gpu on anakin/blade/dgxspark (Task 14 Step 3), also threadsmax for
#     the same reason.
MATRIX="
local base threads1
local base threads8
local base threadsmax
local base hashseed0
local base hashseedfree
anakin base threads1
anakin base threads8
anakin base threadsmax
anakin base hashseed0
anakin base hashseedfree
blade base threads1
blade base threads8
blade base threadsmax
blade base hashseed0
blade base hashseedfree
blade base nosimd512
dgxspark base threads1
dgxspark base threads8
dgxspark base threadsmax
dgxspark base hashseed0
dgxspark base hashseedfree
blade py313 threadsmax
blade oldpy threadsmax
blade oldlibs threadsmax
blade mkl threadsmax
blade alpine threadsmax
dgxspark alpine threadsmax
anakin gpu threadsmax
blade gpu threadsmax
dgxspark gpu threadsmax
"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "run_all.sh: pre-flight check -- ssh reachability per unique host"
  # Process substitution (not a pipe): a pipe would run this while loop in
  # a subshell, where `exit 1` only exits the subshell and the pre-flight
  # check would silently keep going instead of aborting.
  while read -r host; do
    if [ "$host" = "local" ]; then
      if docker info >/dev/null 2>&1; then
        echo "  local: docker reachable"
      else
        echo "  local: docker NOT reachable" >&2
        exit 1
      fi
    else
      # `< /dev/null`: without this, ssh inherits this while-loop's own
      # stdin (the process-substitution feeding `read -r host`) and
      # consumes the rest of it for itself, silently truncating the loop
      # to a single host. Verified: this is exactly what happened before
      # this redirect was added (Task 13 report).
      if ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" true < /dev/null 2>/dev/null; then
        echo "  $host: ssh reachable"
      else
        echo "  $host: ssh NOT reachable" >&2
        exit 1
      fi
    fi
  done < <(echo "$MATRIX" | awk 'NF {print $1}' | sort -u)
  echo ""
  echo "run_all.sh: $(echo "$MATRIX" | awk 'NF' | wc -l | tr -d ' ') planned run(s):"
fi

# fix round 2, Important 4: this loop used to run every row under plain
# `set -e` with no per-row guard, so the first failing row killed the
# entire campaign outright -- and since remote results/ is excluded from
# `--delete` (it persists between invocations) and the runner refuses to
# overwrite an existing output file, a second `run_all.sh` after that
# failure died again immediately, on the FIRST already-completed row,
# before ever reaching the one that actually failed. Resuming would have
# meant hand-pruning JSONs across up to four machines. Fixed two ways at
# once: (a) skip a row outright if its output is already sitting in
# results/ locally -- the one place completion is unambiguous regardless
# of which host produced it -- and (b) record a row's failure and
# continue to the rest rather than aborting, the same contract
# fetch_model.sh already uses for its per-host loop.
#
# Known residual gap, not fixed here: if a row's measurement succeeded
# remotely but collection (scp) failed, the local results/ file never
# gets created, so a retry calls run_on_host.sh again -- which rsyncs,
# rebuilds, and reruns the container, and the runner then refuses to
# overwrite the STALE remote output file from the first attempt. Fix
# round 2's reordering in run_on_host.sh (collect before chown) makes
# this less likely by removing one cause of exactly that failure mode,
# but does not eliminate every way collection could fail after a
# successful measurement.
FAILED_ROWS=""
SKIPPED_ROWS=""

while read -r host image config; do
  [ -z "$host" ] && continue
  if [ "$DRY_RUN" -eq 1 ]; then
    "$SCRIPT_DIR/run_on_host.sh" "$host" "$image" "$config" --dry-run
    echo "---"
    continue
  fi

  out_file="$REPRO_DIR/results/${host}__${image}__${config}.json"
  if [ -f "$out_file" ]; then
    echo "run_all.sh: already have results/${host}__${image}__${config}.json, skipping"
    SKIPPED_ROWS="$SKIPPED_ROWS ${host}/${image}/${config}"
    echo "---"
    continue
  fi

  if "$SCRIPT_DIR/run_on_host.sh" "$host" "$image" "$config"; then
    :
  else
    row_exit=$?
    echo "run_all.sh: ${host}/${image}/${config} FAILED (exit $row_exit), continuing to the next row" >&2
    FAILED_ROWS="$FAILED_ROWS ${host}/${image}/${config}"
  fi
  echo "---"
done < <(echo "$MATRIX")

if [ "$DRY_RUN" -ne 1 ]; then
  if [ -n "$SKIPPED_ROWS" ]; then
    echo "run_all.sh: skipped (results already collected):$SKIPPED_ROWS"
  fi
  if [ -n "$FAILED_ROWS" ]; then
    echo "run_all.sh: FAILED row(s):$FAILED_ROWS" >&2
    exit 1
  fi
  echo "run_all.sh: campaign complete (no failures)"
fi
