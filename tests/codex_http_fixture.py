"""Model-free native CLI fixture against the reviewed Responses HTTP adapter.

Only generation is synthetic. The installed CLI performs the real shell command,
and the real adapter validates the subsequent stateless function output replay.
No user config/auth, shared daemon, remote host or actual model is used.
"""
import contextlib
import http.client
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading

from macqwen.sampling import Sampling
from macqwen.server import MacqwenHTTPServer, MacqwenHandler, ModelService
from macqwen.tests.test_server import FakeSession

KEY = "literal-nonsecret-fixture-key"
OUTPUT = "M4_NATIVE_TOOL_OUTPUT_731942\n"
CALL_ID = "call_m4_owned_fixture"
FINAL = "NATIVE_CLI_FIXTURE_COMPLETED"


def native_cli_outcome(cli_events, returncode, expected_final=FINAL):
    """Retain native errors and require this fixture's completed CLI outcome."""
    error_events, warnings, unexpected = [], [], []
    finals = []
    terminals = []
    for event in cli_events:
        kind = event.get("type")
        item = event.get("item", {})
        if kind in {"error", "turn.failed"} or item.get("type") == "error":
            error_events.append(event)
            if item.get("type") == "error":
                message = str(item.get("message", event))
            else:
                message = str(event.get("message", event.get("error", {}).get("message", event)))
            if kind == "item.completed" and item.get("type") == "error" and re.fullmatch(
                r"Exceeded skills context budget\. All skill descriptions were removed and "
                r"[0-9]+ additional skills were not included in the model-visible skills list\.",
                message,
            ):
                warnings.append(message)
            else:
                unexpected.append(message)
        if kind in {"turn.completed", "turn.failed"}:
            terminals.append(kind)
        if kind == "item.completed" and item.get("type") == "agent_message":
            finals.append(item.get("text"))
    turn_completed = (terminals == ["turn.completed"] and bool(cli_events)
                      and cli_events[-1].get("type") == "turn.completed")
    final = finals[-1] if finals else None
    final_matches = final == expected_final
    return {
        "native_error_events": error_events,
        "native_informational_warnings": warnings,
        "native_unexpected_errors": unexpected,
        "native_turn_completed": turn_completed,
        "native_final_answer": final,
        "native_final_answer_matches": final_matches,
        "native_success": returncode == 0 and not unexpected and turn_completed and final_matches,
    }


def exercise_cli(home, codex, artifact_root=None, *, observe_mcp=False):
    if artifact_root:
        Path(artifact_root).mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="native-cli-", dir=artifact_root or home))
    owned = home / "fixture-work"
    owned.mkdir()
    (owned / "fixture.txt").write_text(OUTPUT)
    requests, events, auth_checks, paths = [], [], [], []
    adapter_log = io.StringIO()
    session = FakeSession([])
    session.preferences["thinking_enabled"] = False
    session.backend.context_window = 4096
    session.backend.sampling = Sampling.greedy_settings()

    original_generate = session.backend.generate

    def generate(max_tokens, out=None, should_cancel=None, on_prefill_progress=None):
        if on_prefill_progress:
            on_prefill_progress(0, len(session.backend.pending))
        return original_generate(max_tokens, out)

    session.backend.generate = generate

    class CaptureHandler(MacqwenHandler):
        def _complete(self, *args, **kwargs):
            with contextlib.redirect_stdout(adapter_log):
                return super()._complete(*args, **kwargs)

        def _authorized(self):
            valid = super()._authorized()
            auth_checks.append(valid)
            paths.append(self.path)
            return valid

        def _payload(self):
            payload = super()._payload()
            requests.append(payload)
            if len(requests) == 1:
                names = {tool.get("name") for tool in payload.get("tools", [])}
                if "exec_command" in names:
                    name, arguments = "exec_command", {"cmd": "cat fixture.txt", "max_output_tokens": 64}
                elif "shell_command" in names:
                    name, arguments = "shell_command", {"command": "cat fixture.txt"}
                else:
                    raise ValueError("CLI did not advertise a supported shell function")
                session.backend.pieces = ["<tool_call>" + json.dumps({
                    "id": CALL_ID, "name": name, "arguments": arguments}) + "</tool_call>"]
            elif len(requests) == 2:
                outputs = [item for item in payload.get("input", [])
                           if item.get("type") == "function_call_output"]
                if not any(item.get("call_id") == CALL_ID and OUTPUT.strip() in item.get("output", "")
                           for item in outputs):
                    raise ValueError("CLI did not replay matched actual command output")
                session.backend.pieces = [FINAL]
            else:
                raise ValueError("unexpected extra inference request")
            return payload

        def _error(self, message, status=400):
            events.append({"type": "fixture.http_error", "status": status, "message": str(message)})
            return super()._error(message, status)

        def _sse(self, value, event=None):
            events.append(value)
            return super()._sse(value, event)

    server = MacqwenHTTPServer(("127.0.0.1", 0), ModelService(session), KEY)
    server.RequestHandlerClass = CaptureHandler
    thread = threading.Thread(target=server.serve_forever,
                              kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    env = dict(os.environ, CODEX_HOME=str(home / ".codex"))
    env["PATH"] = str(Path(codex).parent) + os.pathsep + env["PATH"]
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    cmd = [str(home / ".local/bin/codex-m4"), "--strict-config", "exec",
           "--skip-git-repo-check", "--ephemeral", "--json", "-C", str(owned),
           "-c", "model_providers.m4_qwen.base_url=" + json.dumps(endpoint),
           "Read fixture.txt with the available shell function, then report completion."]
    mcp_marker = owned / "mcp-child-attempted"
    if observe_mcp:
        for name in ("node_repl", "custom_fixture"):
            cmd.extend(["-c", f"mcp_servers.{name}.command=" + json.dumps(sys.executable),
                        "-c", f"mcp_servers.{name}.args=" + json.dumps([
                            "-c", "from pathlib import Path; Path(" + repr(str(mcp_marker)) + ").write_text('MCP child attempted')"])])
        for name in ("openaiDeveloperDocs", "notion"):
            cmd.extend(["-c", f"mcp_servers.{name}.url=" + json.dumps(endpoint + "/mcp-attempt")])
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", "/v1/models", headers={"Authorization": "Bearer " + KEY})
        response = connection.getresponse()
        discovered = json.loads(response.read())
        connection.close()
        (artifacts / "http-model-discovery.json").write_text(json.dumps(discovered, indent=2))
        run = subprocess.run(cmd, env=env, cwd=owned, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        (artifacts / "requests.json").write_text(json.dumps(requests, indent=2))
        (artifacts / "response-events.json").write_text(json.dumps(events, indent=2))
    # exec supports strict-config; debug/features do not in 0.162.1.
    strict = run
    strict_cmd = cmd
    (artifacts / "strict-config-stdout.txt").write_text(strict.stdout)
    (artifacts / "strict-config-stderr.txt").write_text(strict.stderr)
    (artifacts / "adapter-completions.jsonl").write_text(adapter_log.getvalue())
    (artifacts / "cli-stdout.jsonl").write_text(run.stdout)
    (artifacts / "cli-stderr.txt").write_text(run.stderr)
    for name in ("m4-qwen.config.toml", "m4-qwen.models.json", "m4-qwen.instructions.md",
                 "m4-qwen.compact-prompt.txt"):
        (artifacts / name).write_bytes((home / ".codex" / name).read_bytes())
    cli_events = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    commands = [event["item"] for event in cli_events
                if event.get("type") == "item.completed" and event.get("item", {}).get("type") == "command_execution"]
    completed = [event for event in events if event.get("type") == "response.completed"]
    output_replays = [item for request in requests for item in request.get("input", [])
                      if item.get("type") == "function_call_output"]
    emitted = [item for event in completed for item in event.get("response", {}).get("output", [])
               if item.get("type") == "function_call"]
    version = subprocess.run([codex, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    evidence = {
        **native_cli_outcome(cli_events, run.returncode),
        "cli_version": version, "returncode": run.returncode,
        "strict_config_returncode": strict.returncode, "requests": len(requests),
        "request_paths": paths, "authentication_verified": bool(auth_checks) and all(auth_checks),
        "actual_command_output": commands[0].get("aggregated_output") if commands else None,
        "actual_command_exit_code": commands[0].get("exit_code") if commands else None,
        "emitted_call_id": emitted[0]["call_id"] if emitted else None,
        "replayed_call_id": output_replays[0]["call_id"] if output_replays else None,
        "stream_completed": len(completed) == 2 and all(event["response"]["status"] == "completed" for event in completed),
        "local_catalog_accepted": bool(requests) and "unknown model" not in (run.stdout + run.stderr).lower()
                                  and "not found in" not in (run.stdout + run.stderr).lower(),
        "http_discovery_by_cli": paths.count("/v1/models") > 1,
        "catalog_discovered": bool(requests) and all(request["model"] == "macqwen-flashnext" for request in requests)
                              and discovered.get("models", [{}])[0].get("slug") == "macqwen-flashnext",
        "tools": [{"type": tool.get("type"), "name": tool.get("name")} for tool in requests[0].get("tools", [])] if requests else [],
        "parallel_tool_calls": requests[0].get("parallel_tool_calls") if requests else None,
        "model_visible_text_characters": sum(len(part.get("text", "")) for item in requests[0].get("input", []) for part in item.get("content", [])) if requests else 0,
        "skill_block_characters": sum(len(part.get("text", "")) for item in requests[0].get("input", []) for part in item.get("content", []) if "<skills_instructions>" in part.get("text", "")) if requests else 0,
        "mcp_observation_enabled": observe_mcp,
        "mcp_child_attempted": mcp_marker.exists(),
        "mcp_http_attempts": [path for path in paths if "mcp-attempt" in path],
        "strict_config_command": strict_cmd, "cli_command": cmd,
        "artifact_directory": str(artifacts), "stderr": run.stderr,
    }
    (artifacts / "summary.json").write_text(json.dumps(evidence, indent=2))
    return evidence
