"""Model-free lifecycle/resource checks using real owned subprocesses and HTTP."""
import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import backend_api as api
import benchmark_backend as bench

REAL_POPEN, REAL_RUN = subprocess.Popen, subprocess.run


@contextmanager
def sleeper(seconds=30):
    process = REAL_POPEN([sys.executable, '-c', f'import time; time.sleep({seconds})'])
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)


def args_for(directory):
    return SimpleNamespace(mode='throughput', feature='slru', source_revision=bench.SLRU_REVISION,
        source=directory, checkpoint=directory, slru_protected_fraction=0.7,
        sidecar=directory, output_dir=directory, tokens=128, pairs=6, quality_pairs=1, seed=1,
        timeout=30, max_swap_mib=6144)


def resource_command(command, *, timeout_mode=False, delay=0, **kwargs):
    footprint = command[0] == '/usr/bin/footprint'
    text = 'phys_footprint: 1M' if footprint else 'vm.swapusage: used = 0M'
    code = f'import time; print({text!r}, flush=True); time.sleep({delay if footprint else 0})'
    if timeout_mode and footprint:
        code = 'import time; print("partial footprint", flush=True); time.sleep(1)'
        kwargs['timeout'] = .05
    return REAL_RUN([sys.executable, '-c', code], **kwargs)


class ChildLifetime(unittest.TestCase):
    def restore_fixture(self, *, isolated=True, repeated_signal=False):
        wrapper = Path(__file__).with_name('run_backend_experiment.sh').read_text()
        # Run the actual restoration function and traps, including their ordering.
        restore = wrapper.split('restore() {\n', 1)[1].split('\nhealth\nlaunchctl', 1)[0]
        with tempfile.TemporaryDirectory() as temporary, sleeper(seconds=90) as unrelated:
            directory = Path(temporary)
            ready, restored = directory / 'ready.json', directory / 'restored.json'
            helper_code = ('import json,os,pathlib,signal,time; '
                           + ('os.setsid(); ' if isolated else '')
                           + 'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                           + f'pathlib.Path({str(ready)!r}).write_text(json.dumps('
                             '{"pid":os.getpid(),"ppid":os.getppid(),"pgid":os.getpgrp()})); '
                           + 'time.sleep(90)')
            # Stub only external production operations. Bootstrap itself checks
            # that the directly owned leader has been removed and reaped first.
            bootstrap_code = ('import json,pathlib,subprocess,sys; '
                              'pid=int(sys.argv[1]); '
                              'state=subprocess.run(["ps","-p",str(pid),"-o","stat="],'
                              'capture_output=True,text=True); '
                              'assert not state.stdout.strip(), "helper still present at restoration"; '
                              + f'pathlib.Path({str(restored)!r}).write_text(json.dumps('
                                '{"owned_pid":pid,"removed_and_reaped":True}))')
            script = ('set -eu\npython=' + shlex.quote(sys.executable)
                      + '\nrestore_needed=1\nchild_pid=\ndomain=fixture\nlabel=original\nagent=fixture\n'
                      + 'health() { test -f ' + shlex.quote(str(restored)) + '; }\n'
                      + 'launchctl() {\n case "$1" in\n print) return 1 ;;\n bootstrap) '
                      + shlex.quote(sys.executable) + ' -c ' + shlex.quote(bootstrap_code)
                      + ' "$child_pid" ;;\n *) return 2 ;;\n esac\n}\n'
                      + 'restore() {\n' + restore + '\n'
                      + shlex.quote(sys.executable) + ' -c ' + shlex.quote(helper_code)
                      + ' >/dev/null 2>&1 &\nchild_pid=$!\nwait "$child_pid"\n')
            process = REAL_POPEN(['/bin/sh', '-c', script], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, start_new_session=True)
            child = None
            try:
                until = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < until and process.poll() is None:
                    time.sleep(.01)
                self.assertTrue(ready.exists(), 'direct helper did not become ready')
                child = json.loads(ready.read_text())
                self.assertEqual(child['ppid'], process.pid, 'fixture must be a direct shell child')
                self.assertEqual(child['pgid'], child['pid'] if isolated else process.pid)
                self.assertNotEqual(os.getpgid(unrelated.pid), child['pgid'])
                os.kill(process.pid, signal.SIGTERM)
                if repeated_signal:
                    time.sleep(.5)
                    os.kill(process.pid, signal.SIGTERM)
                try:
                    stdout, stderr = process.communicate(timeout=35)
                except subprocess.TimeoutExpired:
                    self.fail('restoration never reached bounded group cleanup for a TERM-ignoring helper')
                self.assertIsNone(unrelated.poll(), 'unrelated process must survive restoration')
                if isolated:
                    self.assertEqual(process.returncode, 143, stdout + stderr)
                    self.assertTrue(restored.exists(), 'original service restoration was skipped')
                    self.assertEqual(json.loads(restored.read_text()),
                                     {'owned_pid': child['pid'], 'removed_and_reaped': True})
                else:
                    self.assertEqual(process.returncode, 1, stdout + stderr)
                    self.assertIn('refusing unrelated cleanup', stderr)
                    self.assertFalse(restored.exists(), 'unsafe cleanup must refuse production restoration')
                    os.kill(child['pid'], 0)  # Fail-closed cleanup leaves this invalid group untouched.
            finally:
                # Kill only fixture-owned groups, even when the pre-fix wrapper hangs.
                if child and isolated:
                    try:
                        os.killpg(child['pgid'], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate(timeout=5)

    def test_restore_bounds_term_ignoring_direct_helper_and_reaps_before_bootstrap(self):
        self.restore_fixture()

    def test_restore_survives_a_second_signal_during_group_cleanup(self):
        self.restore_fixture(repeated_signal=True)

    def test_restore_refuses_bootstrap_when_helper_group_is_not_isolated(self):
        self.restore_fixture(isolated=False)

    def exercise(self, fault):
        children = []
        def spawn(_command, **kwargs):
            child = REAL_POPEN([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
            children.append(child)
            if fault in (signal.SIGTERM, signal.SIGHUP):
                # A real signal arrives while Popen's result is still unassigned.
                os.kill(os.getpid(), fault)
            return child
        previous = {sig: signal.signal(sig, bench.interrupted) for sig in (signal.SIGTERM, signal.SIGHUP)}
        try:
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                with patch.object(bench.subprocess, 'Popen', spawn), patch.object(
                        bench.threading.Thread, 'start', side_effect=RuntimeError('injected sampler startup failure')):
                    outcome = bench.run_child(args_for(directory), 'lifecycle', 'off')
                self.assertTrue(outcome['failure'])
                self.assertIsNotNone(children[0].returncode, 'child must be waited/reaped before wrapper can restore')
                self.assertEqual(children[0].wait(timeout=1), children[0].returncode)
                self.assertTrue((directory / 'lifecycle.outcome.json').is_file())
                self.assertTrue((directory / 'lifecycle.resources.json').is_file())
                self.assertTrue((directory / 'lifecycle.footprint.json').is_file())
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait()

    def test_sampler_start_failure_reaps_child(self):
        self.exercise('thread-start')

    def test_sigterm_during_child_creation_reaps_child(self):
        self.exercise(signal.SIGTERM)

    def test_sighup_during_child_creation_reaps_child(self):
        self.exercise(signal.SIGHUP)

    def test_child_spawn_failure_preserves_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            def missing(_command, **kwargs):
                return REAL_POPEN(['/definitely-absent-backend-command'], **kwargs)
            with patch.object(bench.subprocess, 'Popen', missing):
                outcome = bench.run_child(args_for(directory), 'spawn-failure', 'off')
            self.assertIsNone(outcome['returncode'])
            self.assertEqual(outcome['failure']['type'], 'FileNotFoundError')

    def test_wrapper_group_fallback_removes_only_owned_descendants(self):
        wrapper = Path(__file__).with_name('run_backend_experiment.sh').read_text()
        cleanup = wrapper.split("<<'PY_OWNED_GROUP'\n", 1)[1].split('\nPY_OWNED_GROUP', 1)[0]
        owned_exec = wrapper.split("owned_exec='", 1)[1].split("'\n", 1)[0]
        leader = None
        with tempfile.TemporaryDirectory() as temporary, sleeper() as unrelated:
            marker = Path(temporary) / 'child.json'
            child_code = ('import json, os, pathlib, subprocess, sys; '
                          'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
                          f'pathlib.Path({str(marker)!r}).write_text(json.dumps({{"pid":child.pid,"pgid":os.getpgrp()}}))')
            try:
                leader = REAL_POPEN([sys.executable, '-c', owned_exec, sys.executable, '-c', child_code])
                self.assertEqual(leader.wait(timeout=5), 0)  # Unexpected helper exit, descendant still alive.
                child = json.loads(marker.read_text())
                self.assertEqual(child['pgid'], leader.pid)
                self.assertEqual(os.getpgid(child['pid']), leader.pid)
                self.assertNotEqual(os.getpgid(unrelated.pid), leader.pid)
                result = REAL_RUN([sys.executable, '-', str(leader.pid)], input=cleanup,
                                  capture_output=True, text=True, timeout=35)
                self.assertEqual(result.returncode, 0, result.stderr)
                state = REAL_RUN(['ps', '-p', str(child['pid']), '-o', 'stat='], capture_output=True, text=True)
                self.assertTrue(not state.stdout.strip() or state.stdout.strip().startswith('Z'),
                                'restore boundary must have no running owned descendant')
                self.assertIsNone(unrelated.poll(), 'unrelated process must survive owned-group cleanup')
            finally:
                if leader:
                    try:
                        os.killpg(leader.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


class Resources(unittest.TestCase):
    def test_timeout_is_persisted_and_fails_resource_gate(self):
        with tempfile.TemporaryDirectory() as temporary, sleeper() as process:
            directory = Path(temporary)
            sampler = api.ResourceSampler(process, directory, time.monotonic(), swap_interval=.05)
            def injected(command, **kwargs):
                return resource_command(command, timeout_mode=True, **kwargs)
            with patch.object(api.subprocess, 'run', injected):
                sampler.start()
                self.assertTrue(sampler.failed.wait(3))
                gate = sampler.stop()
            self.assertFalse(gate['passed'])
            self.assertTrue(gate['failures'])
            raw = [json.loads(line) for line in (directory / 'physical_footprint.jsonl').read_text().splitlines()]
            self.assertEqual(raw[0]['failure']['type'], 'TimeoutExpired')
            self.assertIn('partial footprint', raw[0]['stdout'])
            with self.assertRaisesRegex(RuntimeError, 'required resource collection failed'):
                sampler.require_healthy()

    def test_footprint_does_not_block_swap_cadence(self):
        with tempfile.TemporaryDirectory() as temporary, sleeper() as process:
            sampler = api.ResourceSampler(process, Path(temporary), time.monotonic(),
                                          swap_interval=.04, footprint_interval=15)
            footprint_started, footprint_finished = threading.Event(), threading.Event()
            def injected(command, **kwargs):
                if command[0] == '/usr/bin/footprint':
                    footprint_started.set()
                    result = resource_command(command, delay=.35, **kwargs)
                    footprint_finished.set()
                    return result
                return resource_command(command, **kwargs)
            with patch.object(api.subprocess, 'run', injected):
                sampler.start()
                self.assertTrue(footprint_started.wait(2))
                self.assertTrue(footprint_finished.wait(3))
                gate = sampler.stop()
            self.assertTrue(gate['passed'], gate)
            self.assertGreaterEqual(len(sampler.rows['global_swap']), 3,
                                    'swap must keep collecting while footprint blocks')


@contextmanager
def api_fixture(directory):
    release = directory / 'release'
    release.mkdir()
    for name in ('verify_live.py', 'candidate_start.py', 'start.py', 'deployment.json', 'backend_api.py', 'benchmark_backend.py'):
        (release / name).write_text('model-free fixture\n')
    root = directory / 'root'
    (root / 'config').mkdir(parents=True)
    key = root / 'config/api-key'
    key.write_text('local-fixture-key')
    key.chmod(0o600)
    children = []
    def spawn(command, **kwargs):
        server = any(str(part).endswith('candidate_start.py') for part in command)
        verifier = any(str(part).endswith('verify_live.py') for part in command)
        if not server and not verifier:
            return REAL_POPEN(command, **kwargs)
        child = REAL_POPEN([sys.executable, '-c', 'import time; time.sleep(30)' if server else 'pass'], **kwargs)
        children.append(child)
        return child
    with patch.object(api, '__file__', str(release / 'backend_api.py')), patch.object(api, 'ROOT', root), \
            patch.object(api.subprocess, 'Popen', spawn), patch.object(sys, 'argv', ['backend_api.py',
                '--condition', 'off', '--output-dir', str(directory / 'output')]):
        try:
            yield children
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait()


class ApiEvidence(unittest.TestCase):
    def test_sampler_start_failure_cannot_pass_and_reaps_server(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with api_fixture(directory) as children, patch.object(api.threading.Thread, 'start',
                    side_effect=RuntimeError('injected API sampler startup failure')):
                with self.assertRaisesRegex(RuntimeError, 'injected API sampler'):
                    api.main()
                self.assertIsNotNone(children[0].returncode)
                report = json.loads((directory / 'output/api-run.json').read_text())
                self.assertFalse(report['passed'])
                self.assertFalse(report['resource_gate']['passed'])

    def test_collector_timeout_cannot_pass_complete_api_run(self):
        import requests
        ready, code = requests.Response(), requests.Response()
        ready.status_code, ready._content = 200, b'{"status":"ready"}'
        code.status_code, code._content = 200, b'{"choices":[{"message":{"content":"model-free code"}}]}'
        def injected(command, **kwargs):
            return resource_command(command, timeout_mode=True, **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with api_fixture(directory) as children, patch.object(api.requests, 'get', return_value=ready), \
                    patch.object(api.requests, 'post', return_value=code), patch.object(api.subprocess, 'run', injected):
                with self.assertRaises(RuntimeError):
                    api.main()
                report = json.loads((directory / 'output/api-run.json').read_text())
                self.assertFalse(report['passed'])
                self.assertFalse(report['resource_gate']['passed'])
                self.assertTrue(any(row.get('failure', {}).get('type') == 'TimeoutExpired'
                                    for row in report['resource_failures']))
                self.assertTrue(all(child.returncode is not None for child in children))

    def test_non_json_http_failure_retains_status_and_raw_body(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status":"ready"}')
            def do_POST(self):
                self.send_response(502)
                self.end_headers()
                self.wfile.write(b'<h1>injected backend unavailable</h1>')
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        real_get, real_post = api.requests.get, api.requests.post
        endpoint = f'http://127.0.0.1:{server.server_port}'
        def get(url, **kwargs):
            return real_get(url.replace('http://127.0.0.1:8080', endpoint), **kwargs)
        def post(url, **kwargs):
            return real_post(url.replace('http://127.0.0.1:8080', endpoint), **kwargs)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                with api_fixture(directory) as children, patch.object(api.requests, 'get', get), \
                        patch.object(api.requests, 'post', post), patch.object(api.subprocess, 'run', resource_command):
                    with self.assertRaises(ValueError):
                        api.main()
                    report = json.loads((directory / 'output/api-run.json').read_text())
                    self.assertEqual(report['code']['status'], 502)
                    self.assertEqual(report['code']['raw_text'], '<h1>injected backend unavailable</h1>')
                    self.assertTrue(report['code']['parse_failure'])
                    self.assertFalse(report['passed'])
                    self.assertTrue(all(child.returncode is not None for child in children))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    unittest.main(argv=[sys.argv[0]], verbosity=2)
