#!/usr/bin/env bash
# test_run_container.sh
#
# Regression test for scripts/lib_run_container.sh's `run_container`,
# added in fix round 3. Each of the two fix rounds before this one fixed
# one of this mechanism's two required behaviours by breaking the other:
# round 2 fixed the stdio-leak-hangs-ssh bug but, in doing so, broke the
# kill itself (a `wait` on a PID that was never this shell's child, which
# fails immediately and silently skips `docker kill` every time); this
# round's fix restores the kill without reintroducing the leak. Neither
# earlier round's own verification caught the break, because neither
# exercised the timeout actually needing to fire -- every container
# tested finished on its own well inside the timeout. This test exercises
# both paths every time it's run, against the exact function
# `run_on_host.sh` ships to every measurement host (sourced from the same
# file, not a copy).
#
# Builds a tiny throwaway helper image (not one of the seven real
# measurement images): run_container's command line is hardcoded to
# `python -m reprobench.runner` inside `reprobench:<image>`, so this test
# needs something tagged `reprobench:<name>` with a `python` on PATH that
# behaves predictably -- sleeps for $TEST_SLEEP_SECS if set, exits
# immediately otherwise. Built from stdin so nothing extra is committed
# under docker/.
#
# Entirely local docker; does not touch any remote host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_run_container.sh
source "$SCRIPT_DIR/lib_run_container.sh"

TEST_TAG="testharness"
FAILED=0

cleanup() {
  docker rmi -f "reprobench:$TEST_TAG" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "building throwaway test harness image (reprobench:$TEST_TAG) ..."
docker build -q -t "reprobench:$TEST_TAG" - >/dev/null <<'DOCKERFILE'
FROM alpine:3.20
RUN printf '#!/bin/sh\nif [ -n "${TEST_SLEEP_SECS:-}" ]; then sleep "$TEST_SLEEP_SECS"; fi\nexit 0\n' > /usr/local/bin/python \
    && chmod +x /usr/local/bin/python
DOCKERFILE

assert() {
  local description="$1" ok="$2"
  if [ "$ok" -eq 0 ]; then
    echo "PASS: $description"
  else
    echo "FAIL: $description" >&2
    FAILED=1
  fi
}

# --- Test 1: container finishes well inside the timeout -------------------
# Must return promptly (not wait out the rest of a generous timeout).
echo "=== test 1: container finishes early, timeout is generous (10s) ==="
cname1="test_run_container_early_$$"
SECONDS=0
rc=0
run_container "$TEST_TAG" "$cname1" 10 || rc=$?
# run_container sets its own `trap ... EXIT` (to clean up its specific
# container even if something inside it dies unexpectedly); bash traps
# don't stack, so that registration replaces this script's own `cleanup`
# trap set above. Re-arm it immediately after every run_container call
# so the test harness image still gets removed when this script exits.
trap cleanup EXIT
elapsed1=$SECONDS
echo "  elapsed=${elapsed1}s exit_code=$rc"
# `if ... ; then ok=0; else ok=1; fi`, not a bare `[ ... ] && [ ... ]`
# statement: under `set -e`, a standalone compound command's own overall
# non-zero exit status aborts the script right here -- which would skip
# the `assert` call below (and its "FAIL: ..." message) on exactly the
# failure this test exists to catch. Wrapping it as an `if` condition is
# exempt from `set -e`, so a failed assertion is reported, not silently
# fatal.
if [ "$rc" -eq 0 ] && [ "$elapsed1" -le 5 ]; then ok=0; else ok=1; fi
assert "finishes-early case returns promptly (<=5s) with exit 0" "$ok"

# --- Test 2: container outlives the timeout and must be killed ------------
# Must return close to the timeout (not the container's full 30s sleep),
# and the exit code must reflect a kill (137 = 128+SIGKILL, docker kill's
# default signal), confirming `docker kill` actually ran -- this is
# exactly the call fix round 2's bug silently skipped every time.
echo "=== test 2: container outlives timeout (3s timeout, 30s sleep) ==="
cname2="test_run_container_hang_$$"
SECONDS=0
rc=0
run_container "$TEST_TAG" "$cname2" 3 -e TEST_SLEEP_SECS=30 || rc=$?
trap cleanup EXIT  # re-arm, see the comment after test 1's run_container call
elapsed2=$SECONDS
echo "  elapsed=${elapsed2}s exit_code=$rc"
if [ "$rc" -eq 137 ] && [ "$elapsed2" -ge 2 ] && [ "$elapsed2" -le 10 ]; then ok=0; else ok=1; fi
assert "hanging case is killed (exit 137) and returns promptly (2-10s, not 30s)" "$ok"

echo ""
if [ "$FAILED" -ne 0 ]; then
  echo "test_run_container.sh: FAILED" >&2
  exit 1
fi
echo "test_run_container.sh: all tests passed"
