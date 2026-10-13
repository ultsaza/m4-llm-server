#!/bin/sh
set -eu
root=${MACQWEN_ROOT:-"$HOME/llm/macqwen"}
release=${MACQWEN_RELEASE:-"$root/releases/api"}
label=org.macqwen.m4
owner_uid=$(id -u)
agent="$HOME/Library/LaunchAgents/$label.plist"
saved="$root/config/disabled-launch-agents/$label.plist"
daemon=/Library/LaunchDaemons/$label.plist
if [ -e "$agent" ] && [ -e "$saved" ]; then
  printf '%s\n' "Existing LaunchAgent backup must be resolved before switching services" >&2
  exit 1
fi
sudo -v
if [ -e "$daemon" ] || sudo launchctl print "system/$label" >/dev/null 2>&1; then
  printf '%s\n' "An existing system service must be updated separately" >&2
  exit 1
fi
wait_unloaded() {
  wait_attempt=0
  while sudo launchctl print "$1" >/dev/null 2>&1; do
    [ "$wait_attempt" -lt 30 ] || return 1
    wait_attempt=$((wait_attempt + 1))
    sleep 1
  done
}
agent_was_loaded=0
agent_moved=0
daemon_installed=0
rollback() {
  if [ "$daemon_installed" -eq 1 ]; then
    if sudo launchctl print "system/$label" >/dev/null 2>&1; then
      sudo launchctl bootout "system/$label" || true
    fi
    sudo rm -f "$daemon"
    wait_unloaded "system/$label" || return 1
  fi
  if [ "$agent_moved" -eq 1 ]; then
    mv "$saved" "$agent"
  fi
  if [ "$agent_was_loaded" -eq 1 ] && ! sudo launchctl print "gui/$owner_uid/$label" >/dev/null 2>&1; then
    sudo launchctl bootstrap "gui/$owner_uid" "$agent"
  fi
}
trap rollback EXIT
trap 'exit 1' HUP INT TERM
daemon_installed=1
sudo install -o root -g wheel -m 0644 "$release/org.macqwen.m4.daemon.plist" "$daemon"
if sudo launchctl print "gui/$owner_uid/$label" >/dev/null 2>&1; then
  agent_was_loaded=1
  sudo launchctl bootout "gui/$owner_uid/$label"
fi
if ! wait_unloaded "gui/$owner_uid/$label"; then
  printf '%s\n' "LaunchAgent is still loaded; refusing to start a second service" >&2
  exit 1
fi
if [ -e "$agent" ]; then
  mkdir -p "$(dirname "$saved")"
  mv "$agent" "$saved"
  agent_moved=1
fi
sudo launchctl bootstrap system "$daemon"
trap - EXIT HUP INT TERM
sudo launchctl print "system/$label"
