"""Exercise native TUI-only namespace tools with synthetic generation."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest

from macqwen.sampling import Sampling
from macqwen.server import MacqwenHTTPServer, MacqwenHandler, ModelService, model_card
from macqwen.tests.test_server import FakeSession


@unittest.skipUnless(shutil.which("codex") and importlib.util.find_spec("pexpect"),
                     "installed Codex and pexpect are required for native TUI qualification")
class NativeTuiNamespace(unittest.TestCase):
    def test_builtin_namespace_tool_is_executed_and_replayed_by_the_real_tui(self):
        import pexpect
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="m4-owned-tui-") as temporary:
            home = Path(temporary)
            work = home/"work"
            work.mkdir()
            subprocess.run(["git", "init", "-q", str(work)], check=True, capture_output=True)
            config = home/".codex/config.toml"
            config.parent.mkdir()
            config.write_text('[projects.'+json.dumps(str(work))+']\ntrust_level = "trusted"\n')
            before = config.read_bytes()
            session = FakeSession([])
            session.preferences["thinking_enabled"] = False
            session.backend.sampling = Sampling.greedy_settings()
            original_generate = session.backend.generate

            def generate(max_tokens, out=None, should_cancel=None, on_prefill_progress=None):
                if on_prefill_progress:
                    on_prefill_progress(0, len(session.backend.pending))
                return original_generate(max_tokens, out)

            session.backend.generate = generate
            catalog = home/"catalog.json"
            card = model_card(ModelService(session))
            card["context_window"] = 4096
            catalog.write_text(json.dumps({"models": [card]}))
            installed = subprocess.run([shutil.which("python3"), str(root/"codex/install-client.py"),
                                        "--home", str(home), "--catalog", str(catalog)],
                                       input="owned-tui-key\n", text=True, capture_output=True)
            self.assertEqual(installed.returncode, 0, installed.stderr)
            state = {"requests": [], "events": [], "errors": [], "title_errors": [], "result": None}
            completed = threading.Event()
            log = io.StringIO()
            call_id, final = "call_owned_tui_namespace", "OWNED_TUI_NAMESPACE_COMPLETED"

            class Handler(MacqwenHandler):
                def _payload(self):
                    payload = super()._payload()
                    metadata = json.loads(payload.get("client_metadata", {}).get("x-codex-turn-metadata", '{}'))
                    self.title_request = metadata.get("thread_source") == "thread_title"
                    if self.title_request:
                        # Auto-title generation is a separate, optional request.
                        # Let the real adapter reject unsupported structured output.
                        return payload
                    state["requests"].append(payload)
                    if len(state["requests"]) == 1:
                        namespace = next(t for t in payload["tools"]
                                         if t.get("type") == "namespace" and t.get("name") == "mcp__codex_worktrees")
                        assert any(t["name"] == "list_worktrees" for t in namespace["tools"])
                        session.backend.pieces = ['<tool_call>'+json.dumps({
                            "id": call_id, "name": "mcp__codex_worktrees.list_worktrees",
                            "arguments": {}})+'</tool_call>']
                    elif len(state["requests"]) == 2:
                        inputs = payload["input"]
                        call = next(i for i in inputs if i.get("type") == "function_call" and i.get("call_id") == call_id)
                        assert call["name"] == "list_worktrees" and call["namespace"] == "mcp__codex_worktrees"
                        result = next(i for i in inputs if i.get("type") == "function_call_output" and i.get("call_id") == call_id)
                        self_test_output = result["output"]
                        assert isinstance(self_test_output, list)
                        assert all(part.get("type") == "input_text" for part in self_test_output)
                        state["result"] = ''.join(part["text"] for part in self_test_output)
                        assert state["result"].strip(), 'the TUI must return its real tool result'
                        session.backend.pieces = [final]
                    else:
                        raise AssertionError("unexpected additional request")
                    return payload

                def _complete(self, *args, **kwargs):
                    with contextlib.redirect_stdout(log):
                        return super()._complete(*args, **kwargs)

                def _sse(self, value, event=None):
                    state["events"].append(value)
                    super()._sse(value, event)
                    if value.get("type") == "response.completed" and len(state["requests"]) == 2:
                        completed.set()

                def _error(self, message, status=400):
                    destination = "title_errors" if getattr(self, "title_request", False) else "errors"
                    state[destination].append(str(message))
                    super()._error(message, status)

            server = MacqwenHTTPServer(("127.0.0.1", 0), ModelService(session), "owned-tui-key")
            server.RequestHandlerClass = Handler
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
            env = dict(os.environ, CODEX_HOME=str(home/".codex"), TERM="xterm-256color")
            env["PATH"] = str(Path(shutil.which("codex")).parent)+os.pathsep+env["PATH"]
            args = ["--strict-config", "--no-alt-screen", "-C", str(work), "-c",
                    'model_providers.m4_qwen.base_url='+json.dumps(f"http://127.0.0.1:{server.server_port}/v1"),
                    "Inspect the worktree list once and report completion. Do not read or change files."]
            child = pexpect.spawn(str(home/".local/bin/codex-m4"), args, env=env,
                                  encoding="utf-8", echo=False, dimensions=(40, 120))
            thread.start()
            until = time.monotonic()+45
            try:
                while not completed.is_set() and time.monotonic() < until:
                    match = child.expect(["\x1b\\[6n", pexpect.EOF, pexpect.TIMEOUT], timeout=1)
                    if match == 0:
                        child.send("\x1b[1;1R")
                    elif match == 1:
                        break
                self.assertTrue(completed.is_set(), {"requests":len(state["requests"]),
                                                    "errors":state["errors"],
                                                    "terminal_tail":str(child.before)[-1200:]})
                child.expect(final, timeout=5)
            finally:
                child.terminate(force=True)
                child.close(force=True)
                server.shutdown()
                server.server_close()
                thread.join(2)
            self.assertEqual(state["errors"], [])
            _, separator, result = state["result"].partition('Output:')
            self.assertTrue(separator, state["result"])
            self.assertEqual(json.loads(result), {"data": [], "nextCursor": None})
            self.assertTrue(all(error == "structured text formats are unsupported"
                                for error in state["title_errors"]), state["title_errors"])
            self.assertEqual(len(state["requests"]), 2)
            self.assertEqual(config.read_bytes(), before)
            self.assertEqual([e["type"] for e in state["events"] if e["type"] in
                              {"response.completed", "response.failed"}],
                             ["response.completed", "response.completed"])
