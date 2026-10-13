"""Dedicated client installation and native CLI transport contracts."""
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "codex/install-client.py"
KEY = "literal-nonsecret-fixture-key"


def catalog():
    from macqwen.server import ModelService, model_card
    from macqwen.tests.test_server import FakeSession
    card = model_card(ModelService(FakeSession(["hello"])))
    card["context_window"] = 4096
    return {"models": [card]}


class ClientInstallation(unittest.TestCase):
    def setUp(self):
        self.assertTrue(INSTALLER.is_file(), "dedicated profile installer is not implemented")
        self.temp = tempfile.TemporaryDirectory(prefix="m4-client-unit-", dir=os.environ.get("TASK2_ARTIFACT_DIR"))
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name) / "owned home"
        self.home.mkdir()
        self.catalog = Path(self.temp.name) / "catalog.json"
        self.catalog.write_text(json.dumps(catalog()))
        base = self.home / ".codex/config.toml"
        base.parent.mkdir()
        base.write_text('sandbox_mode = "danger-full-access"\napproval_policy = "never"\n'
                        '[mcp_servers.custom_fixture]\ncommand = "/missing/never-run"\n'
                        '[mcp_servers.openaiDeveloperDocs]\nurl = "https://docs.fixture.invalid/mcp"\n'
                        '[mcp_servers.node_repl]\ncommand = "/missing/never-run"\n'
                        '[mcp_servers.notion]\nurl = "https://notion.fixture.invalid/mcp"\n')
        self.base_bytes = base.read_bytes()
        self.auth_bytes = b'{"fixture": "literal-nonsecret-auth-preserved"}\n'
        (self.home / ".codex/auth.json").write_bytes(self.auth_bytes)

    def install(self, key=KEY, base_url=None):
        command = [sys.executable, str(INSTALLER), "--home", str(self.home),
                   "--catalog", str(self.catalog)]
        if base_url is not None:
            command.extend(["--base-url", base_url])
        return subprocess.run(command, input=key + "\n",
                              text=True, capture_output=True)

    def assert_installed(self):
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(KEY, result.stdout + result.stderr)
        return tomllib.loads((self.home / ".codex/m4-qwen.config.toml").read_text())

    def test_separate_profile_renders_provider_paths_without_changing_base(self):
        cfg = self.assert_installed()
        self.assertEqual((self.home / ".codex/config.toml").read_bytes(), self.base_bytes)
        self.assertEqual((self.home / ".codex/auth.json").read_bytes(), self.auth_bytes)
        self.assertEqual(cfg["model"], "macqwen-flashnext")
        self.assertEqual(cfg["model_provider"], "m4_qwen")
        provider = cfg["model_providers"]["m4_qwen"]
        self.assertEqual(provider["base_url"], "http://127.0.0.1:8080/v1")
        self.assertEqual(provider["wire_api"], "responses")
        self.assertEqual(provider["auth"]["command"], "/bin/cat")
        self.assertEqual(provider["auth"]["args"], [str(self.home / ".config/m4-qwen/api-key")])
        self.assertEqual(cfg["model_context_window"], 4096)
        self.assertEqual(cfg["model_auto_compact_token_limit"], 2500)
        self.assertNotIn("sandbox_mode", cfg)
        self.assertNotIn("approval_policy", cfg)
        self.assertFalse(cfg["mcp_servers"]["custom_fixture"]["enabled"])
        self.assertEqual(json.loads(Path(cfg["model_catalog_json"]).read_text())["models"][0]["slug"], "macqwen-flashnext")
        self.assertTrue(Path(cfg["model_instructions_file"]).is_file())
        self.assertTrue(Path(cfg["experimental_compact_prompt_file"]).is_file())

    def test_key_is_private_and_never_printed_even_when_repaired(self):
        self.assert_installed()
        key_path = self.home / ".config/m4-qwen/api-key"
        self.assertEqual(key_path.read_text(), KEY + "\n")
        self.assertEqual(stat.S_IMODE(key_path.stat().st_mode), 0o600)
        key_path.chmod(0o644)
        self.assert_installed()
        self.assertEqual(stat.S_IMODE(key_path.stat().st_mode), 0o600)

    def test_configurable_https_endpoint_and_embedded_credentials_are_handled_privately(self):
        result = self.install(base_url="https://server.example.invalid/v1")
        self.assertEqual(result.returncode, 0, result.stderr)
        cfg = tomllib.loads((self.home/".codex/m4-qwen.config.toml").read_text())
        self.assertEqual(cfg["model_providers"]["m4_qwen"]["base_url"], "https://server.example.invalid/v1")
        self.assertEqual((self.home/".codex/config.toml").read_bytes(), self.base_bytes)
        before = (self.home/".codex/m4-qwen.config.toml").read_bytes()
        result = self.install(base_url="https://user:fixture-private-value@server.example.invalid/v1")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("fixture-private-value", result.stdout+result.stderr)
        self.assertNotIn(KEY, result.stdout+result.stderr)
        self.assertEqual((self.home/".codex/m4-qwen.config.toml").read_bytes(), before)

    def test_reinstall_is_idempotent_and_backs_up_owned_updates(self):
        self.assert_installed()
        before = {p: p.stat().st_mtime_ns for p in self.home.rglob("*") if p.is_file()}
        self.assert_installed()
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in self.home.rglob("*") if p.is_file()})
        result = self.install("new-nonsecret-fixture-key")
        self.assertEqual(result.returncode, 0, result.stderr)
        backups = list((self.home / ".config/m4-qwen").glob("api-key.bak-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), KEY + "\n")
        self.assertEqual(stat.S_IMODE(backups[0].stat().st_mode), 0o600)

    def test_unrelated_targets_are_refused_before_any_install(self):
        path = self.home / ".codex/m4-qwen.config.toml"
        path.write_text("unrelated profile")
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(path.read_text(), "unrelated profile")
        self.assertFalse((self.home / ".config/m4-qwen/api-key").exists())
        self.assertNotIn(KEY, result.stdout + result.stderr)

    def test_unrelated_key_or_launcher_is_refused_without_partial_profile(self):
        for relative in (".config/m4-qwen/api-key", ".local/bin/codex-m4"):
            with self.subTest(target=relative):
                path = self.home / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("unrelated fixture data")
                result = self.install()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(path.read_text(), "unrelated fixture data")
                self.assertFalse((self.home / ".codex/m4-qwen.config.toml").exists())
                self.assertNotIn(KEY, result.stdout + result.stderr)
                path.unlink()

    def test_ancestor_obstruction_leaves_no_partial_files_and_retry_succeeds(self):
        original_home = self.home
        for index, relative in enumerate((".codex", ".config", ".config/m4-qwen",
                                          ".local", ".local/bin")):
            with self.subTest(ancestor=relative):
                self.home = original_home / f"ancestor-case-{index}"
                self.home.mkdir()
                obstruction = self.home / relative
                obstruction.parent.mkdir(parents=True, exist_ok=True)
                obstruction.write_text("unrelated ancestor fixture")
                obstruction.chmod(0o640)
                before = {p.relative_to(self.home): (p.read_bytes(), p.stat().st_mode,
                                                    p.stat().st_mtime_ns)
                          for p in self.home.rglob("*") if p.is_file()}
                result = self.install()
                self.assertNotEqual(result.returncode, 0)
                after = {p.relative_to(self.home): (p.read_bytes(), p.stat().st_mode,
                                                   p.stat().st_mtime_ns)
                         for p in self.home.rglob("*") if p.is_file()}
                self.assertEqual(after, before, "ancestor conflict left partial client files")
                self.assertNotIn(KEY, result.stdout + result.stderr)
                obstruction.unlink()
                self.assert_installed()
        self.home = original_home

    def test_ancestor_obstruction_is_rejected_before_existing_key_permission_repair(self):
        self.assert_installed()
        key_path = self.home / ".config/m4-qwen/api-key"
        key_path.chmod(0o644)
        bin_path = self.home / ".local/bin"
        (bin_path / "codex-m4").unlink()
        bin_path.rmdir()
        bin_path.write_text("unrelated ancestor fixture")
        before = {p.relative_to(self.home): (p.read_bytes(), p.stat().st_mode,
                                            p.stat().st_mtime_ns)
                  for p in self.home.rglob("*") if p.is_file()}
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        after = {p.relative_to(self.home): (p.read_bytes(), p.stat().st_mode,
                                           p.stat().st_mtime_ns)
                 for p in self.home.rglob("*") if p.is_file()}
        self.assertEqual(after, before, "ancestor conflict changed existing client files or permissions")
        bin_path.unlink()
        self.assert_installed()
        self.assertEqual(stat.S_IMODE(key_path.stat().st_mode), 0o600)
        self.assertTrue((bin_path / "codex-m4").is_file())

    def test_modified_managed_file_and_symlink_are_refused(self):
        self.assert_installed()
        path = self.home / ".codex/m4-qwen.instructions.md"
        path.write_text("user changed this")
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(path.read_text(), "user changed this")
        path.unlink()
        outside = Path(self.temp.name) / "outside"
        outside.write_text("untouched")
        path.symlink_to(outside)
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(outside.read_text(), "untouched")

    def test_invalid_key_and_incompatible_catalog_leave_no_client(self):
        for key in ("", "two\nlines", "tab\tkey"):
            with self.subTest(key=repr(key)):
                result = self.install(key)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.home / ".local/bin/codex-m4").exists())
        bad = catalog()
        bad["models"][0]["context_window"] = 131072
        self.catalog.write_text(json.dumps(bad))
        self.assertNotEqual(self.install().returncode, 0)
        self.assertFalse((self.home / ".local/bin/codex-m4").exists())

    def test_launcher_forwards_literal_arguments_with_no_shell_interpolation(self):
        self.assert_installed()
        bin_path = Path(self.temp.name) / "fixture-bin"
        bin_path.mkdir()
        stub = bin_path / "codex"
        stub.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
        stub.chmod(0o700)
        marker = self.home / "SHOULD_NOT_EXIST"
        args = ["exec", "a b", 'quote"value', f"$(touch {marker})", "--json"]
        env = dict(os.environ, PATH=str(bin_path) + os.pathsep + os.environ["PATH"])
        result = subprocess.run([str(self.home / ".local/bin/codex-m4"), *args],
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), ["--no-daemon", "--profile", "m4-qwen", *args])
        self.assertFalse(marker.exists())

    @unittest.skipUnless(shutil.which("codex"), "installed Codex CLI is required")
    def test_disabled_mcp_transports_make_no_http_or_child_attempts(self):
        self.assert_installed()
        from codex_http_fixture import exercise_cli
        evidence = exercise_cli(self.home, shutil.which("codex"),
                                os.environ.get("TASK2_ARTIFACT_DIR"), observe_mcp=True)
        self.assertEqual(evidence["returncode"], 0, evidence)
        self.assertTrue(evidence["native_success"], evidence)
        self.assertEqual(evidence["mcp_http_attempts"], [])
        self.assertFalse(evidence["mcp_child_attempted"])
        self.assertEqual(evidence["requests"], 2)

    @unittest.skipUnless(shutil.which("codex"), "installed Codex CLI is required")
    def test_exact_profile_strict_config_and_native_http_tool_replay(self):
        self.assert_installed()
        from codex_http_fixture import exercise_cli
        evidence = exercise_cli(self.home, shutil.which("codex"),
                                os.environ.get("TASK2_ARTIFACT_DIR"))
        self.assertEqual(evidence["returncode"], 0, evidence)
        self.assertTrue(evidence["native_success"], evidence)
        self.assertEqual(evidence["requests"], 2)
        self.assertEqual(evidence["actual_command_output"], "M4_NATIVE_TOOL_OUTPUT_731942\n")
        self.assertEqual(evidence["replayed_call_id"], evidence["emitted_call_id"])
        self.assertTrue(evidence["stream_completed"])
        self.assertTrue(evidence["catalog_discovered"])
        self.assertTrue(evidence["local_catalog_accepted"])
        self.assertFalse(evidence["http_discovery_by_cli"])
        self.assertEqual({tool["name"] for tool in evidence["tools"]},
                         {"exec_command", "write_stdin", "request_user_input"})
        self.assertTrue(evidence["authentication_verified"])
        self.assertEqual(evidence["actual_command_exit_code"], 0)
        self.assertEqual(evidence["strict_config_returncode"], 0)


class NativeTranscriptAcceptance(unittest.TestCase):
    """Model-free native outcome regressions; real tool replay stays above."""

    WARNING = ("Exceeded skills context budget. All skill descriptions were removed and "
               "36 additional skills were not included in the model-visible skills list.")

    def transcript(self):
        from codex_http_fixture import FINAL, OUTPUT
        return [
            {"type": "thread.started", "thread_id": "owned-transcript"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {
                "type": "command_execution", "status": "completed", "exit_code": 0,
                "aggregated_output": OUTPUT}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": FINAL}},
            {"type": "turn.completed"},
        ]

    def outcome(self, events, returncode=0):
        from codex_http_fixture import native_cli_outcome
        return native_cli_outcome(events, returncode)

    def test_success_without_warning_is_accepted(self):
        outcome = self.outcome(self.transcript())
        self.assertTrue(outcome["native_success"])
        self.assertEqual(outcome["native_error_events"], [])
        self.assertEqual(outcome["native_informational_warnings"], [])
        self.assertTrue(outcome["native_turn_completed"])
        self.assertTrue(outcome["native_final_answer_matches"])

    def test_exact_known_skills_warning_is_accepted(self):
        events = self.transcript()
        warning = {"type": "item.completed", "item": {"type": "error", "message": self.WARNING}}
        events.insert(2, warning)
        outcome = self.outcome(events)
        self.assertTrue(outcome["native_success"])
        self.assertEqual(outcome["native_error_events"], [warning])
        self.assertEqual(outcome["native_informational_warnings"], [self.WARNING])
        self.assertEqual(outcome["native_unexpected_errors"], [])

    def test_appended_or_unknown_warning_is_rejected(self):
        for message in (self.WARNING + " Unexpected native tool subsystem failure",
                        self.WARNING + "\n", "Unknown skills warning"):
            with self.subTest(message=repr(message)):
                events = self.transcript()
                events.insert(2, {"type": "item.completed", "item": {
                    "type": "error", "message": message}})
                self.assertFalse(self.outcome(events)["native_success"])

    def test_unexpected_exit_zero_item_error_is_rejected(self):
        events = self.transcript()
        error = {"type": "item.completed", "item": {
            "type": "error", "message": "Unexpected native tool subsystem failure"}}
        events.append(error)
        outcome = self.outcome(events)
        self.assertFalse(outcome["native_success"])
        self.assertEqual(outcome["native_error_events"], [error])
        self.assertEqual(outcome["native_unexpected_errors"], [error["item"]["message"]])

    def test_top_level_error_is_rejected_even_with_known_message(self):
        for message in ("Unexpected native API failure", self.WARNING):
            with self.subTest(message=message):
                events = self.transcript()
                events.insert(2, {"type": "error", "message": message})
                self.assertFalse(self.outcome(events)["native_success"])

    def test_unfinished_error_item_is_not_a_known_warning(self):
        events = self.transcript()
        events.insert(2, {"type": "item.started", "item": {
            "type": "error", "message": self.WARNING}})
        self.assertFalse(self.outcome(events)["native_success"])

    def test_outer_message_cannot_hide_an_unknown_item_error(self):
        events = self.transcript()
        events.insert(2, {"type": "item.completed", "message": self.WARNING, "item": {
            "type": "error", "message": "Unexpected native tool subsystem failure"}})
        self.assertFalse(self.outcome(events)["native_success"])

    def test_failed_turn_is_rejected_with_or_without_completed_turn(self):
        for also_completed in (False, True):
            with self.subTest(also_completed=also_completed):
                events = self.transcript()
                failed = {"type": "turn.failed", "error": {"message": "native turn failed"}}
                if also_completed:
                    events.insert(-1, failed)
                else:
                    events[-1] = failed
                self.assertFalse(self.outcome(events)["native_success"])

    def test_missing_terminal_completion_is_rejected(self):
        self.assertFalse(self.outcome(self.transcript()[:-1])["native_success"])

    def test_missing_or_incorrect_final_answer_is_rejected(self):
        for final in (None, "unexpected answer", "NATIVE_CLI_FIXTURE_COMPLETED extra"):
            with self.subTest(final=final):
                events = self.transcript()
                if final is None:
                    del events[-2]
                else:
                    events[-2]["item"]["text"] = final
                self.assertFalse(self.outcome(events)["native_success"])

    def test_nonzero_process_is_rejected_despite_completed_events(self):
        self.assertFalse(self.outcome(self.transcript(), returncode=1)["native_success"])


if __name__ == "__main__":
    unittest.main()
