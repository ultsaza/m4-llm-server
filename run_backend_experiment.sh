#!/bin/sh
# Root bootstraps an owned temporary Interactive LaunchAgent. Never run directly.
set -eu
root=${MACQWEN_ROOT:-"$HOME/llm/macqwen"}
candidate=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
python=$candidate/venv/bin/python
label=org.macqwen.m4
domain=gui/$(id -u)
agent="$HOME/Library/LaunchAgents/$label.plist"
usage() {
  cat <<'EOF'
Usage: run_backend_experiment.sh --owner-label LABEL --owner-plist PATH MODE [args]
MODE: preflight | build | benchmark | api
preflight: safe imports/check-only; does not stop production
build: upstream builder args (e.g. --pool --checkpoint ABS --out ABS --report results/...)
benchmark: benchmark_backend.py args (--mode throughput|coverage|quality, --output-dir ABS)
api: backend_api.py args (--condition off|on, --output-dir ABS)
Only an owned org.macqwen.m4.backend-* Interactive LaunchAgent can run hardware modes.
Production is restored and authenticated ready on exit, including child failure/signals.
EOF
}
if [ "${1-}" = '--help' ]; then usage; exit 0; fi
[ "${1-}" = '--owner-label' ] || { usage >&2; exit 2; }
owner=$2
shift 2
[ "${1-}" = '--owner-plist' ] || { usage >&2; exit 2; }
owner_plist=$2
shift 2
mode=${1-}
[ -n "$mode" ] || { usage >&2; exit 2; }
shift
case "$mode" in preflight|build|benchmark|api) ;; *) usage >&2; exit 2 ;; esac
if [ "$mode" = preflight ]; then
  cd "$candidate"
  "$python" benchmark_backend.py --mode preflight "$@"
  "$python" candidate_start.py --condition off --sidecar "$root/cache/pool-small-sidecar-20261012" --check-only
  exit 0
fi
case "$owner" in org.macqwen.m4.backend-*) ;; *) echo 'invalid owned job label' >&2; exit 2 ;; esac
# Accept direct shell or caffeinate parent, but prove this shell belongs to the job.
"$python" - "$owner" "$owner_plist" "$$" <<'PY'
import os, pathlib, plistlib, re, subprocess, sys
label, path, shell_pid = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3])
config = plistlib.loads(path.read_bytes())
assert config['Label'] == label and config['ProcessType'] == 'Interactive', 'owned job must be Interactive'
state = subprocess.check_output(['launchctl', 'print', 'gui/' + str(os.getuid()) + '/' + label], text=True)
match = re.search(r'^\s*pid = (\d+)', state, re.M)
assert match, 'owned job is not running'
owner_pid = int(match.group(1))
pid = shell_pid
for _ in range(8):
    if pid == owner_pid:
        break
    pid = int(subprocess.check_output(['ps', '-p', str(pid), '-o', 'ppid='], text=True).strip())
else:
    raise SystemExit('wrapper is not descended from its owned job')
assert path.stat().st_uid == os.getuid(), 'owned plist must belong to current user'
PY
restore_needed=0
child_pid=
health() {
  "$python" - <<'PY'
import os, pathlib, requests
path = (pathlib.Path(os.environ.get("MACQWEN_ROOT",str(pathlib.Path.home()/"llm"/"macqwen")))/"config/api-key")
assert path.stat().st_mode & 0o077 == 0, 'key must have mode 0600'
response = requests.get('http://127.0.0.1:8080/health', headers={'Authorization': 'Bearer ' + path.read_text().strip()}, timeout=15)
response.raise_for_status()
assert response.json()['status'] == 'ready', 'service is not ready'
PY
}
restore() {
  status=$?
  trap - EXIT
  # Recovery must finish even if another stop signal arrives during cleanup.
  trap '' HUP INT TERM
  if [ -n "$child_pid" ]; then
    # Only the isolated group created by owned_run belongs to this wrapper.
    # A helper that dies unexpectedly cannot leave its model descendants
    # running when the original service is bootstrapped.
    if ! "$python" - "$child_pid" <<'PY_OWNED_GROUP'
import os, signal, subprocess, sys, time
pgid = int(sys.argv[1])
assert pgid > 1 and pgid != os.getpgrp(), 'refuse unrelated wrapper process group'
try:
    leader_group = os.getpgid(pgid)
except ProcessLookupError:
    leader_group = pgid  # The leader may have exited while descendants remain.
if leader_group != pgid:
    raise SystemExit('helper PID does not own the isolated group; refusing unrelated cleanup')

def living():
    state = subprocess.check_output(['ps', '-axo', 'pid=,pgid=,stat='], text=True)
    return [int(fields[0]) for line in state.splitlines()
            if len(fields := line.split()) >= 3 and int(fields[1]) == pgid
            and not fields[2].startswith('Z')]

for sig, timeout in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
    if not living():
        break
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        break
    until = time.monotonic() + timeout
    while living() and time.monotonic() < until:
        time.sleep(.1)
    if not living():
        break
else:
    raise SystemExit('owned descendants still alive; production restoration refused')
assert not living(), 'owned descendants still alive'
PY_OWNED_GROUP
    then
      echo 'owned descendant cleanup failed; original service was not restored' >&2
      exit 1
    fi
    # The bounded group cleanup has proved the leader exited; reap it now.
    wait "$child_pid" 2>/dev/null || true
  fi
  if [ "$restore_needed" -eq 1 ]; then
    if ! launchctl print "$domain/$label" >/dev/null 2>&1; then
      launchctl bootstrap "$domain" "$agent" || status=1
    fi
    attempt=0
    until health >/dev/null 2>&1; do
      if [ "$attempt" -ge 240 ]; then
        echo 'original service failed readiness after restore' >&2
        status=1
        break
      fi
      attempt=$((attempt + 1))
      sleep 1
    done
  fi
  exit "$status"
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
health
launchctl print "$domain/$label" >/dev/null
# Save the original PID before bootout, then wait for its actual process exit.
old_pid=$(launchctl print "$domain/$label" | sed -n 's/^[[:space:]]*pid = \([0-9][0-9]*\).*/\1/p' | head -n 1)
[ -n "$old_pid" ] || { echo 'original service PID missing' >&2; exit 1; }
restore_needed=1
launchctl bootout "$domain/$label"
attempt=0
while kill -0 "$old_pid" 2>/dev/null; do
  [ "$attempt" -lt 60 ] || { echo 'original process did not exit' >&2; exit 1; }
  attempt=$((attempt + 1))
  sleep 1
done
cd "$candidate"
owned_exec='import os, signal, sys; os.setsid(); [signal.signal(s, signal.SIG_DFL) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)]; os.execv(sys.argv[1], sys.argv[1:])'
owned_run() {
  # Finish recording the isolated helper PID before shell traps may unwind.
  launch_signal=0
  trap 'launch_signal=129' HUP
  trap 'launch_signal=130' INT
  trap 'launch_signal=143' TERM
  "$python" -c "$owned_exec" "$python" "$@" &
  child_pid=$!
  trap 'exit 130' INT
  trap 'exit 143' TERM HUP
  [ "$launch_signal" -eq 0 ] || exit "$launch_signal"
}
case "$mode" in
  build)
    cd "$candidate/src"
    owned_run -u -m models.flashnext.tests.bench.build_small_sidecar "$@"
    ;;
  benchmark) owned_run -u "$candidate/benchmark_backend.py" "$@" ;;
  api) owned_run -u "$candidate/backend_api.py" "$@" ;;
esac
wait "$child_pid"
# Keep the group identity until EXIT verifies that no owned descendants remain.
