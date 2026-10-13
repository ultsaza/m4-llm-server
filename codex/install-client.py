#!/usr/bin/env python3
"""Install the isolated m4 client; read its credential from stdin only.

Usage: python3 codex/install-client.py --catalog /path/to/models.json < key-file
The catalog must come from the reviewed adapter's authenticated /v1/models.
Unrelated or locally edited destination files are refused. Managed updates are
backed up alongside the originals. No auth.json or base config is modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import tomllib
import uuid
from urllib.parse import urlsplit

SOURCE = Path(__file__).resolve().parent
MANIFEST = ".codex/m4-qwen.install.json"
CONTEXT_THRESHOLDS = {4096: 2500, 8192: 5000, 16384: 10000}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_catalog(path):
    payload = json.loads(path.read_text())
    cards = payload.get("models", payload.get("data", []))
    cards = [card for card in cards if card.get("slug") == "macqwen-flashnext"]
    if len(cards) != 1:
        raise ValueError("catalog must contain exactly one macqwen-flashnext model")
    card = cards[0]
    required = {"input_modalities": ["text"],
                "shell_type": "shell_command", "default_reasoning_level": "none",
                "supports_reasoning_summaries": False, "supports_parallel_tool_calls": False}
    if (type(card.get("context_window")) is not int
            or card["context_window"] not in CONTEXT_THRESHOLDS
            or any(card.get(key) != value for key, value in required.items())):
        raise ValueError("catalog does not match the reviewed text-only context settings")
    capabilities = payload.get("capabilities", {})
    if (not isinstance(capabilities, dict)
            or capabilities.get("remote_compaction", "unsupported") not in ("unsupported", "v2")):
        raise ValueError("catalog advertises an unsupported compaction protocol")
    return (json.dumps({"models": cards}, indent=2) + "\n").encode()


def render(home, context_window=4096, remote_compaction="unsupported", base_url="http://127.0.0.1:8080/v1"):
    if type(context_window) is not int or context_window not in CONTEXT_THRESHOLDS:
        raise ValueError("unsupported client context window")
    if remote_compaction not in ("unsupported", "v2"):
        raise ValueError("unsupported client compaction protocol")
    paths = {"CATALOG": ".codex/m4-qwen.models.json",
             "INSTRUCTIONS": ".codex/m4-qwen.instructions.md",
             "COMPACT": ".codex/m4-qwen.compact-prompt.txt",
             "KEY": ".config/m4-qwen/api-key"}
    text = (SOURCE / "m4-qwen.config.toml.in").read_text()
    text = text.replace("@CONTEXT@", str(context_window))
    text = text.replace("@AUTO_COMPACT@", str(CONTEXT_THRESHOLDS[context_window]))
    text = text.replace("@REMOTE_COMPACTION@", json.dumps(remote_compaction))
    text = text.replace("@BASE_URL@", json.dumps(base_url))
    for name, path in paths.items():
        text = text.replace("@" + name + "@", json.dumps(str(home / path), ensure_ascii=False))
    names = {"openaiDeveloperDocs", "node_repl", "notion"}
    base = home / ".codex/config.toml"
    configured = {}
    if base.exists():
        configured = tomllib.loads(base.read_text()).get("mcp_servers", {})
        names.update(configured)
    for name in sorted(names):
        # Disabled profile entries still need a valid transport in CLI 0.162.1.
        # Match the base transport kind to avoid a merged command+URL pair.
        url_transport = ("url" in configured[name] if name in configured
                         else name in {"openaiDeveloperDocs", "notion"})
        transport = 'url = "https://disabled.invalid/mcp"' if url_transport else 'command = "/bin/false"'
        text += f"\n[mcp_servers.{json.dumps(name, ensure_ascii=False)}]\nenabled = false\n{transport}\n"
    return text.encode()


def validate_destination(home, path):
    for current in (path, *path.parents):
        if current.is_symlink():
            raise ValueError("refusing symlink in client destination")
        if current != path and current.exists() and not current.is_dir():
            raise ValueError("refusing non-directory ancestor in client destination")
        if current == home:
            break


def write_file(path, data, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".m4-qwen-", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as output:
            output.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def install(home, catalog_path, key, base_url="http://127.0.0.1:8080/v1"):
    key = key.rstrip("\r\n")
    if not key or any(ord(char) <= 32 or ord(char) >= 127 for char in key):
        raise ValueError("stdin must contain one nonempty ASCII credential without whitespace")
    endpoint = urlsplit(base_url)
    if (endpoint.scheme not in {"http", "https"} or not endpoint.hostname
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
            or any(char.isspace() for char in base_url)):
        raise ValueError("base URL must be an HTTP(S) endpoint without embedded credentials")
    catalog = load_catalog(catalog_path)
    window = json.loads(catalog)["models"][0]["context_window"]
    protocol = json.loads(catalog_path.read_text()).get("capabilities", {}).get("remote_compaction", "unsupported")
    files = {".codex/m4-qwen.config.toml": (render(home, window, protocol, base_url), 0o600),
             ".codex/m4-qwen.models.json": (catalog, 0o600),
             ".codex/m4-qwen.instructions.md": ((SOURCE / "instructions.md").read_bytes(), 0o600),
             ".codex/m4-qwen.compact-prompt.txt": ((SOURCE / "compact-prompt.txt").read_bytes(), 0o600),
             ".config/m4-qwen/api-key": ((key + "\n").encode(), 0o600),
             ".local/bin/codex-m4": ((SOURCE / "codex-m4.py").read_bytes(), 0o755)}
    manifest_path = home / MANIFEST
    validate_destination(home, manifest_path)
    managed = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("installer") != "m4-qwen-v1" or not isinstance(manifest.get("files"), dict):
            raise ValueError("refusing unrelated client manifest")
        managed = manifest["files"]
    # Check every target before making any changes, including permission repairs.
    for name in files:
        path = home / name
        validate_destination(home, path)
        if path.exists() and (not path.is_file() or managed.get(name) != digest(path.read_bytes())):
            raise ValueError("refusing unrelated or edited client file: " + name)
    for name, (data, mode) in files.items():
        path = home / name
        if path.exists() and path.read_bytes() == data:
            path.chmod(mode)
            continue
        if path.exists():
            backup = path.with_name(path.name + ".bak-" + uuid.uuid4().hex)
            write_file(backup, path.read_bytes(), mode)
        write_file(path, data, mode)
    data = (json.dumps({"installer": "m4-qwen-v1", "files": {
        name: digest(value[0]) for name, value in files.items()}}, indent=2) + "\n").encode()
    if not manifest_path.exists() or manifest_path.read_bytes() != data:
        write_file(manifest_path, data, 0o600)
    else:
        manifest_path.chmod(0o600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path.home(), help="installation home (default: user home)")
    parser.add_argument("--catalog", type=Path, required=True, help="reviewed adapter model discovery JSON")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="server Responses base URL")
    args = parser.parse_args()
    try:
        install(args.home.expanduser().resolve(), args.catalog, sys.stdin.read(), args.base_url)
    except (OSError, ValueError, TypeError) as exc:
        # Input parser errors can include source text; never expose any credential.
        if isinstance(exc, (json.JSONDecodeError, tomllib.TOMLDecodeError)):
            print("m4-qwen installation refused: invalid metadata/config", file=sys.stderr)
        elif isinstance(exc, OSError):
            print("m4-qwen installation refused: filesystem operation failed", file=sys.stderr)
        else:
            print("m4-qwen installation refused: " + str(exc), file=sys.stderr)
        return 1
    print("Installed dedicated m4-qwen client; base configuration and authentication preserved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
