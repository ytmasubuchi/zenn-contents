# lib_run_container.sh -- sourced, not executed directly.
#
# `run_container`: docker run under an outer timeout, with guaranteed
# cleanup (Task 13 resolutions 10 and 15). Kept in its own sourced file
# (rather than inline in run_on_host.sh) for two reasons: `run_on_host.sh`
# ships this function's definition verbatim to a remote host via
# `declare -f run_container` + ssh (so it exists in exactly one place,
# never re-typed inside an ssh heredoc), and `scripts/test_run_container.sh`
# sources this same file to test it directly, so the function under test
# and the function actually shipped to measurement hosts can never drift
# apart into two different implementations.
#
# fix round 2, CRITICAL (partially fixed, see fix round 3 below): the
# original version backgrounded
# `( sleep "$timeout_secs" && docker kill ... ) &` without redirecting its
# descriptors, and `kill "$watcher"` only killed that subshell, not the
# `sleep` it was blocked on. On a remote run, that subshell inherits
# sshd's pipe back to the calling script, so the orphaned `sleep` held it
# open for up to the full timeout even after the measurement container
# had long since finished. Reviewer-verified with the identical pattern
# piped to `cat`: 6s elapsed with the pipe vs. 0s without.
#
# fix round 3, CRITICAL: fix round 2's own fix for the above introduced a
# second, worse bug. It tried to keep a "direct handle" on the `sleep`
# PID by backgrounding `sleep` separately from the subshell that waits on
# it:
#
#   sleep "$timeout_secs" &
#   sleep_pid=$!
#   ( wait "$sleep_pid" && docker kill ... ) &
#
# `wait` can only wait on a DIRECT CHILD of the shell calling it. The
# subshell forked by `( ... ) &` is a sibling of the backgrounded `sleep`
# (both are children of `run_container`'s own shell), not its parent --
# so inside the subshell, `wait "$sleep_pid"` fails immediately with
# "not a child of this shell", `&&` short-circuits, and `docker kill`
# NEVER runs, regardless of whether the timeout actually elapses.
# Reviewer-reproduced twice, including end to end: an 8-second container
# against a 5-second timeout was never killed; the full 8 seconds
# elapsed. Combined with fix round 2's other change (foreground
# `docker run`, no `-d`/`docker wait`), a genuinely hung container now
# blocked INDEFINITELY -- strictly worse than the bug being fixed, since
# the original 480s cap was gone entirely. Neither fix round 2's own A/B
# test nor its real end-to-end run caught this, because both only
# exercised the "container finishes before the timeout" path; nothing
# exercised the timeout actually needing to fire. See
# scripts/test_run_container.sh, added in this round specifically to
# close that gap: it asserts both behaviours (returns promptly when the
# container finishes early; kills and returns promptly when it does not)
# every time it's run, not just once by hand.
#
# The fix is to go back to the single-subshell shape and keep only the
# part that was actually needed out of fix round 2's change -- the
# descriptor redirection. In `( sleep X && docker kill ... ) &`, `sleep`
# is a real child of THIS subshell (bash forks it to exec the external
# `sleep` binary), so the subshell's own implicit wait on its child works
# correctly with no explicit `wait` builtin needed, and the redirection
# alone (applied to the subshell before backgrounding it, so it's
# inherited by everything the subshell itself forks) is what closes the
# descriptor leak -- regardless of whether the orphan eventually gets
# killed on the early-exit path. `kill "$watcher"` on that path still
# only terminates the subshell shell process, not a now-orphaned `sleep`
# if the container finished first -- but since the descriptors are
# already redirected to /dev/null, that orphan no longer holds anything
# open, so the only cost is a harmless background `sleep` running out
# its remaining duration before exiting on its own.
#
# fix round 2, Important 3: `--rm` added to `docker run`. This does not
# make the watchdog survive an interrupted ssh session (the watcher is
# still a child of the remote shell and dies with it on SIGHUP, which is
# a deliberately deferred, documented residual gap -- see the fix-round-2
# report), but it guarantees a container that outlives its watcher at
# least doesn't accumulate as a stopped container on a shared host
# afterward.
#
# fix round 2: also changed from `docker run -d` + `docker wait` +
# `docker logs` to a single foreground (attached) `docker run`: with `-d`
# and `--rm` together, there is a race between `docker wait` returning
# (the container has exited) and the daemon's own auto-removal
# completing -- `docker logs` right after could then intermittently fail
# to find the container at all, silently losing the one thing it exists
# to capture. Attached mode sidesteps this entirely: output streams live
# as it happens, and `docker run`'s own exit status already mirrors the
# container's, so neither a separate `docker wait` nor `docker logs` is
# needed. `docker kill` from the watcher below still reaches the
# container by name regardless of how the client is attached.
run_container() {
  local image="$1"; shift
  local cname="$1"; shift
  local timeout_secs="$1"; shift
  # remaining args ($@): full extra `docker run` flags (-e / -v / --gpus)
  trap "docker rm -f $cname >/dev/null 2>&1 || true" EXIT

  ( sleep "$timeout_secs" && docker kill "$cname" >/dev/null 2>&1 ) </dev/null >/dev/null 2>&1 &
  local watcher=$!

  local exit_code=0
  docker run --rm --name "$cname" "$@" "reprobench:$image" python -m reprobench.runner || exit_code=$?

  kill "$watcher" >/dev/null 2>&1 || true
  return "$exit_code"
}
