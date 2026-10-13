"""Start the pinned, native MLX server without modifying its environment."""
import hashlib
import importlib.metadata
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get("MACQWEN_ROOT", str(Path.home() / "llm" / "macqwen")))
RELEASE = Path(__file__).resolve().parent


def _read_server_settings(deployment):
    settings = deployment.get("server", {"context_limit": 4096, "remote_compaction": False})
    if (not isinstance(settings, dict)
            or set(settings) != {"context_limit", "remote_compaction"}
            or type(settings["context_limit"]) is not int
            or settings["context_limit"] not in (4096, 8192, 16384)
            or type(settings["remote_compaction"]) is not bool):
        raise SystemExit("invalid reviewed server context/compaction settings")
    return dict(settings)


def _read_memory_settings(deployment):
    memory = deployment.get("memory")
    if not isinstance(memory, dict) or set(memory) != {"expert_pool_gb", "mlx_cache_mib", "wired_gb"}:
        raise SystemExit("memory settings must specify expert_pool_gb, mlx_cache_mib and wired_gb")
    for name, maximum in (("expert_pool_gb", 8), ("wired_gb", 12)):
        value = memory[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
            raise SystemExit(f"{name} must be a finite number between 0 and {maximum}")
    cache = memory["mlx_cache_mib"]
    if cache is not None and (type(cache) is not int or not 0 <= cache <= 2048):
        raise SystemExit("mlx_cache_mib must be null or an integer between 0 and 2048")
    return {"expert_pool_gb": float(memory["expert_pool_gb"]), "mlx_cache_mib": cache,
            "wired_gb": float(memory["wired_gb"])}


def _read_key(secret):
    if secret.stat().st_mode & 0o077:
        raise SystemExit("API key file must have mode 0600")
    key = secret.read_text().strip()
    if not key:
        raise SystemExit("API key file must not be empty")
    return key


def _configure_logging():
    logger = logging.getLogger("macqwen-service")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(ROOT / "logs/server.log", maxBytes=10 * 1024 * 1024,
                                  backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)

    class LogStream:
        encoding = "utf-8"

        def write(self, text):
            for line in text.splitlines():
                if line.strip():
                    logger.info(line)
            return len(text)

        def flush(self):
            handler.flush()

        def isatty(self):
            return False

    sys.stdout = sys.stderr = LogStream()


def main():
    check_only = "--check-only" in sys.argv
    if not check_only:
        _configure_logging()
    deployment = json.loads((RELEASE / "deployment.json").read_text())
    memory = _read_memory_settings(deployment)
    server = _read_server_settings(deployment)
    api_dependencies = deployment.get("api_dependencies", {})
    if api_dependencies:
        if api_dependencies != {"cryptography": "50.0.2"}:
            raise SystemExit("unreviewed API dependency settings")
        api_path = RELEASE / "api-deps"
        if (not api_path.is_dir() or api_path.is_symlink()
                or {p.name for p in api_path.iterdir()} !=
                   {"cryptography", "cryptography-50.0.2.dist-info"}):
            raise SystemExit("isolated API dependency directory is invalid")
        sys.path.insert(0, str(api_path))
    if server["remote_compaction"] and not api_dependencies:
        raise SystemExit("remote compaction requires the isolated API crypto dependency")
    if sys.version_info[:3] != (3, 12, 15):
        raise SystemExit("the service requires its dedicated CPython 3.12.15")
    for package, expected in (deployment["dependencies"] | api_dependencies).items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise SystemExit(f"dependency mismatch: {package} {actual}, expected {expected}")
    source = RELEASE / "src"
    adapter_hash = hashlib.sha256((source / "macqwen/server.py").read_bytes()).hexdigest()
    if adapter_hash != deployment["server_sha256"]:
        raise SystemExit("HTTP adapter differs from the verified release")
    sys.path.insert(0, str(source))
    from macqwen.cli import _python_runtime_error

    error = _python_runtime_error(Path(sys.executable), "flashnext")
    if error:
        raise SystemExit(f"runtime compatibility check failed: {error}")
    complete = (ROOT / "config/model-manifest.json").is_file()
    prefs = json.loads((ROOT / "config/preferences.json").read_text())
    if prefs.get("flashnext_gpu_keepwarm") is not True:
        raise SystemExit("this service requires flashnext_gpu_keepwarm=true")
    if check_only:
        print(json.dumps({"python": sys.version.split()[0], "dependencies": "compatible",
                          "model_download_complete": complete, "source_revision": deployment["source_revision"],
                          "memory": memory, "server": server, "api_dependencies": api_dependencies}))
        return 0
    if not complete:
        raise SystemExit("the pinned model download has not been verified")
    os.environ["MACQWEN_SERVER_API_KEY"] = _read_key(ROOT / "config/api-key")
    os.environ["MACQWEN_SERVER_CONTEXT_LIMIT"] = str(server["context_limit"])
    os.environ["MACQWEN_REMOTE_COMPACTION"] = "1" if server["remote_compaction"] else "0"
    os.environ["FLASHNEXT_NATIVE_MTP"] = "off"
    os.environ["FLASHNEXT_EXPERT_POOL_GB"] = str(memory["expert_pool_gb"])
    os.environ["FLASHNEXT_WIRED_GB"] = str(memory["wired_gb"])
    os.environ["FLASHNEXT_GPU_KEEPWARM"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if memory["mlx_cache_mib"] is not None:
        import mlx.core as mx

        target = memory["mlx_cache_mib"] * 1024 ** 2
        mx.set_cache_limit(target)
        if mx.set_cache_limit(target) != target:
            raise SystemExit("MLX allocator cache limit did not apply")
    print(json.dumps({"event": "memory_policy", **memory}))
    from models.flashnext.settings.launch import apply_chat_environment

    apply_chat_environment(os.environ)
    os.chdir(source)
    sys.argv = ["macqwen-session", "--model", "flashnext", "--profile", "plain", "--server",
                "--checkpoint", str(ROOT / "models/qwen38-flash-next-4bit"),
                "--exact-quality", "--no-think", "--hide-think", "--max-tokens", "512",
                "--preferences-file", str(ROOT / "config/preferences.json"),
                "--api-keys-file", str(ROOT / "config/client-keys.json"),
                "--workspace", str(ROOT / "workspace"),
                "--session-dir", str(ROOT / "cache/sessions"),
                "--host", "127.0.0.1", "--port", "8080"]
    from macqwen.session import main as session_main

    return session_main()


if __name__ == "__main__":
    raise SystemExit(main())
