# Setup

Run on an Apple Silicon Mac with CPython 3.12.15 already installed. Reserve roughly 130 GB for the model and environment; experimental sidecars need additional space. Commands below use `~/llm/macqwen`.

```sh
python3 scripts/prepare-runtime.py
python3 scripts/setup-release.py --python /path/to/python3.12
export MACQWEN_ROOT="$HOME/llm/macqwen"
runtime="$MACQWEN_ROOT/releases/api"
"$runtime/venv/bin/hf" download TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP \
  --revision 2b170fa6309d5d1ee380b35636075fac7945f286 \
  --local-dir "$MACQWEN_ROOT/models/qwen38-flash-next-4bit"
"$runtime/venv/bin/python" "$runtime/verify_model.py"
"$runtime/venv/bin/python" "$runtime/start.py"
```

The verifier checks the pinned revision, shard sizes and index coverage; it does not independently hash every downloaded shard. The server binds to loopback port 8080. To start after login, render and install a LaunchAgent:

```sh
python3 scripts/render-launchagent.py \
  --output "$HOME/Library/LaunchAgents/org.macqwen.m4.plist"
launchctl bootstrap "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/org.macqwen.m4.plist"
```

Stop a foreground server before bootstrapping. Log output rotates under the runtime's `logs` directory. Login-before-start is required; unattended startup with FileVault was not qualified.

For remote access, connect both machines to Tailscale and run `tailscale serve --bg 8080` on the server. Use its HTTPS hostname as the client base URL. Health is `/health`; model discovery is `/v1/models`. Both require the bearer key in `config/api-key`.

On the client, save that key in a private file and save the authenticated model-discovery response as `catalog.json`. Then install without placing the key in arguments:

```sh
python3 codex/install-client.py --catalog catalog.json \
  --base-url https://YOUR-SERVER.YOUR-TAILNET.ts.net/v1 < /path/to/private-key
cd /path/to/project
codex-m4
# Resume an existing m4 session:
codex-m4 resume
```

The installer preserves the base Codex config and auth, uses command-backed authentication and refuses unrelated or locally edited managed files. Keep the same model ID and key to resume encrypted compaction after restarting the server.
