#!/bin/sh
# Run only from the temporary Interactive benchmark LaunchAgent.
set -eu
root=${MACQWEN_ROOT:-"$HOME/llm/macqwen"}
release=${MACQWEN_RELEASE:-"$root/releases/api"}
label=org.macqwen.m4
domain=gui/$(id -u)
agent="$HOME/Library/LaunchAgents/$label.plist"
restore_needed=0
helper=benchmark_performance.py
if [ "${1-}" = "--profile" ]; then
  helper=profile_backend.py
  shift
fi
restore() {
  if [ "$restore_needed" -eq 1 ] && ! launchctl print "$domain/$label" >/dev/null 2>&1; then
    launchctl bootstrap "$domain" "$agent"
  fi
}
trap restore EXIT
trap 'exit 1' HUP INT TERM
"$release/venv/bin/python" -c 'import json, os, pathlib, requests; key=(pathlib.Path(os.environ.get("MACQWEN_ROOT",str(pathlib.Path.home()/"llm"/"macqwen")))/"config/api-key").read_text().strip(); response=requests.get("http://127.0.0.1:8080/health",headers={"Authorization":"Bearer "+key},timeout=15); response.raise_for_status(); assert response.json()["status"] == "ready", "service is busy"'
launchctl print "$domain/$label" >/dev/null
restore_needed=1
launchctl bootout "$domain/$label"
attempt=0
while launchctl print "$domain/$label" >/dev/null 2>&1; do
  [ "$attempt" -lt 30 ] || exit 1
  attempt=$((attempt + 1))
  sleep 1
done
cd "$release"
"$release/venv/bin/python" -u "$helper" "$@"
