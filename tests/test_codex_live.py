"""Owned model-free fixtures; native CLI uses isolated home, never m4/user auth."""
import contextlib
import base64
import http.client
import http.server
import importlib.util
import json
import os
import signal
import socket
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('live', ROOT / 'verify_codex_live.py')
live = importlib.util.module_from_spec(spec)
if spec.loader and (ROOT / 'verify_codex_live.py').exists():
    spec.loader.exec_module(live)

KEY = 'literal-nonsecret-owned-key'
MARKER = 'OWNED_FRESH_READ_VALUE'


def command_events(output=MARKER, exit_code=0):
    return [
        {'type': 'item.completed', 'item': {'type': 'command_execution',
         'command': 'cat fixture.txt', 'aggregated_output': output, 'exit_code': exit_code, 'status': 'completed'}},
        {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': output}},
        {'type': 'turn.completed'},
    ]


def exchanges():
    return [
        {'path': '/v1/responses', 'request': {'model': 'macqwen-flashnext', 'input': [], 'tools': []}, 'status': 200,
         'events': [{'type': 'response.completed', 'sequence_number': 0, 'response': {'status': 'completed',
          'usage': {'input_tokens': 100, 'output_tokens': 20},
          'output': [{'type': 'function_call', 'call_id': 'call_owned', 'name': 'exec_command', 'arguments': '{"cmd":"cat fixture.txt"}'}]}}]},
        {'path': '/v1/responses', 'request': {'model': 'macqwen-flashnext', 'input': [
          {'type': 'function_call_output', 'call_id': 'call_owned', 'output': MARKER}], 'tools': []}, 'status': 200,
         'events': [{'type': 'response.completed', 'sequence_number': 0, 'response': {'status': 'completed',
          'usage': {'input_tokens': 120, 'output_tokens': 10}, 'output': []}}]},
    ]


def compact_setup(work, name='next.txt'):
    text = name + '\n' + ''.join(f'status {index:03d}: local fixture is unchanged; next action follows the first line\n' for index in range(64))
    (work / 'fixture.txt').write_text(text)
    (work / name).write_text(MARKER + '\n')
    return text


def compact_commands(text):
    return [{'type': 'item.completed', 'item': {'type': 'command_execution', 'command': 'cat fixture.txt',
             'aggregated_output': text, 'exit_code': 0, 'status': 'completed'}}, *command_events()]


def compact_records():
    first_text = 'next.txt\n' + ''.join(f'status {index:03d}: local fixture is unchanged; next action follows the first line\n' for index in range(64))
    first_call = {'type': 'function_call', 'call_id': 'call_first_owned', 'name': 'exec_command', 'arguments': '{"cmd":"cat fixture.txt"}'}
    first = {'path': '/v1/responses', 'status': 200, 'request': {'model': 'macqwen-flashnext', 'input': []},
        'events': [{'type': 'response.completed', 'sequence_number': 0,
                    'response': {'status': 'completed', 'output': [first_call]}}]}
    summary = {'path': '/v1/responses', 'status': 200,
        'request': {'model': 'macqwen-flashnext', 'input': [first_call,
            {'type': 'function_call_output', 'call_id': 'call_first_owned', 'output': first_text},
            {'role': 'user', 'content': [{'type': 'input_text', 'text': live.COMPACT_PROMPT}]}]},
        'events': [{'type': 'response.completed', 'sequence_number': 0, 'response': {'status': 'completed',
            'output': [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'First read is complete. NEXT: read next.txt.'}]}]}}]}
    return [first, summary, *exchanges()]


class ValidationTests(unittest.TestCase):
    def test_requires_observed_command_and_matched_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / 'fixture.txt').write_text(MARKER + '\n')
            errors, details = live.validate_case('read', Path(temp), MARKER, command_events(), exchanges(), False)
            self.assertEqual(errors, [])
            self.assertEqual(details['matched_call_ids'], ['call_owned'])
            forged = command_events()[1:]
            self.assertTrue(live.validate_case('read', Path(temp), MARKER, forged, exchanges(), False)[0])
            wrong = exchanges()
            wrong[1]['request']['input'][0]['call_id'] = 'forged'
            self.assertTrue(live.validate_case('read', Path(temp), MARKER, command_events(), wrong, False)[0])

    def test_direct_requires_real_command_and_write_independently_executes(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            (work / 'hello.py').write_text("print('M4_QWEN_WRITE_OK')\n")
            errors, details = live.validate_case('write', work, None, command_events('M4_QWEN_WRITE_OK'), [], True)
            self.assertEqual(errors, [])
            self.assertEqual(details['independent_stdout'], 'M4_QWEN_WRITE_OK\n')
            (work / 'hello.py').write_text("print('forged')\n")
            self.assertTrue(live.validate_case('write', work, None, command_events('M4_QWEN_WRITE_OK'), [], True)[0])

    def test_error_and_incomplete_stream_are_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / 'fixture.txt').write_text(MARKER + '\n')
            records = exchanges()
            records[-1]['events'] = [{'type': 'response.failed', 'sequence_number': 0, 'response': {'status': 'failed'}}]
            events = command_events() + [{'type': 'error', 'message': 'new model error'}]
            self.assertTrue(live.validate_case('read', Path(temp), MARKER, events, records, False)[0])

    def test_compaction_requires_observed_summary_then_continuation(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            text = compact_setup(work)
            self.assertTrue(live.validate_case('compact', work, MARKER, compact_commands(text), exchanges(), False, expected_first=text)[0])
            self.assertEqual(live.validate_case('compact', work, MARKER, compact_commands(text), compact_records(), False, expected_first=text)[0], [])

    def test_v2_compaction_requires_same_packet_replay_before_continued_tool(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);text=compact_setup(work);records=compact_records()
            packet={'type':'compaction','id':'cmp_owned','encrypted_content':'owned-opaque-token'}
            record=records[1]
            record['request']['input'][-1]={'type':'compaction_trigger'}
            record['events']=[{'type':'response.output_item.done','sequence_number':1,'item':packet},
                {'type':'response.completed','sequence_number':2,'response':{'status':'completed','output':[packet]}}]
            records[2]['request']['input']=[packet]
            errors,details=live.validate_case('compact',work,MARKER,compact_commands(text),records,False,expected_first=text)
            self.assertEqual(errors,[])
            self.assertEqual(details['v2_compaction_request_indices'],[1])
            records[2]['request']['input']=[dict(packet,encrypted_content='different')]
            self.assertTrue(live.validate_case('compact',work,MARKER,compact_commands(text),records,False,expected_first=text)[0])

    def test_outer_known_warning_cannot_hide_unknown_item_error(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);(work/'fixture.txt').write_text(MARKER+'\n')
            forged={'type':'item.completed','message':'Exceeded skills context budget. All skill descriptions were removed and 38 additional skills were not included in the model-visible skills list.',
                    'item':{'type':'error','message':'Unexpected tool subsystem failure'}}
            self.assertTrue(live.validate_case('read',work,MARKER,command_events()+[forged],exchanges(),False)[0])

    def test_compact_rejects_truncated_first_read_and_changed_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            text = compact_setup(work)
            self.assertTrue(live.validate_case('compact', work, MARKER, command_events(), compact_records(), False, expected_first=text)[0])
            (work / 'fixture.txt').write_text(text + 'changed\n')
            self.assertTrue(live.validate_case('compact', work, MARKER, compact_commands(text), compact_records(), False, expected_first=text)[0])

    def test_compact_binds_full_first_result_to_that_summary_and_prior_call(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            text = compact_setup(work)
            for forge in ('missing', 'wrong_id', 'partial', 'out_of_order', 'empty_id'):
                with self.subTest(forge=forge):
                    records = compact_records()
                    if forge == 'missing':
                        records[1]['request']['input'].pop(1)
                    elif forge == 'wrong_id':
                        records[1]['request']['input'][1]['call_id'] = 'call_forged'
                    elif forge == 'partial':
                        records[1]['request']['input'][1]['output'] = 'next.txt\n'
                    elif forge == 'out_of_order':
                        records[0], records[1] = records[1], records[0]
                    else:
                        records[0]['events'][0]['response']['output'][0]['call_id'] = None
                        records[1]['request']['input'][1]['call_id'] = None
                    errors, _ = live.validate_case('compact', work, MARKER, compact_commands(text), records, False, expected_first=text)
                    self.assertTrue(errors, 'forged first result qualified compaction')

    def test_read_requires_independent_unchanged_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / 'fixture.txt').write_text('changed\n')
            self.assertTrue(live.validate_case('read', Path(temp), MARKER, command_events(), exchanges(), False)[0])

    def test_only_exact_known_skill_warning_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / 'fixture.txt').write_text(MARKER + '\n')
            known = {'type': 'item.completed', 'item': {'type': 'error', 'message':
                'Exceeded skills context budget. All skill descriptions were removed and 38 additional skills were not included in the model-visible skills list.'}}
            self.assertEqual(live.validate_case('read', Path(temp), MARKER, command_events() + [known], exchanges(), False)[0], [])
            known['item']['message'] += ' New failure.'
            self.assertTrue(live.validate_case('read', Path(temp), MARKER, command_events() + [known], exchanges(), False)[0])

    def test_standard_compaction_warning_is_exact_and_compact_only(self):
        warning = 'Heads up: Long threads and multiple compactions can cause the model to be less accurate. Start a new thread when possible to keep threads small and targeted.'
        item = {'type': 'item.completed', 'item': {'type': 'error', 'message': warning}}
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            text = compact_setup(work)
            records = compact_records()
            errors, details = live.validate_case('compact', work, MARKER, compact_commands(text) + [item], records, False, expected_first=text)
            self.assertEqual(errors, [])
            self.assertIn(warning, details['known_warnings'])
            item['item']['message'] += ' New failure.'
            self.assertTrue(live.validate_case('compact', work, MARKER, compact_commands(text) + [item], records, False, expected_first=text)[0])
            item['item']['message'] = warning
            (work / 'fixture.txt').write_text(MARKER + '\n')
            self.assertIn('CLI error: ' + warning, live.validate_case('read', work, MARKER, command_events() + [item], exchanges(), False)[0])

    def test_evidence_redacts_key_and_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            writer = live.EvidenceWriter(Path(temp), KEY)
            writer.write('body.txt', 'prefix ' + KEY + ' suffix')
            self.assertNotIn(KEY, (Path(temp) / 'body.txt').read_text())
            self.assertTrue(writer.credential_found)
            keyfile = Path(temp) / 'key'
            keyfile.write_text(KEY)
            keyfile.chmod(0o644)
            with self.assertRaises(ValueError):
                live.read_key(keyfile)
            keyfile.chmod(0o600)
            self.assertEqual(live.read_key(keyfile), KEY)

    def test_failed_launcher_retains_evidence_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            key = root / 'key'
            key.write_text(KEY)
            key.chmod(0o600)
            launcher = root / 'launcher'
            launcher.write_text('#!/bin/sh\nif [ "$1" = --version ]; then echo owned-fixture; exit; fi\necho failure >&2\nexit 7\n')
            launcher.chmod(0o700)
            output = root / 'evidence'
            args = live.parser().parse_args(['--case', 'read', '--output', str(output), '--launcher', str(launcher),
                                            '--key-file', str(key), '--direct', '--timeout', '5'])
            summary = live.run_case(args)
            self.assertFalse(summary['passed'])
            self.assertEqual(summary['returncode'], 7)
            self.assertEqual((output / 'cli-stderr.txt').read_text(), 'failure\n')
            self.assertTrue((output / 'summary.json').exists())
            self.assertNotIn(KEY, ''.join(p.read_text() for p in output.iterdir() if p.is_file()))
            with self.assertRaises(ValueError):
                live.run_case(args)


class LifecycleTests(unittest.TestCase):
    def prepare(self, root):
        key = root / 'key'
        key.write_text(KEY)
        key.chmod(0o600)
        launcher = root / 'launcher'
        launcher.write_text('#!/bin/sh\nif [ "$1" = --version ]; then echo owned-fixture; exit; fi\nexit 7\n')
        launcher.chmod(0o700)
        return live.parser().parse_args(['--case', 'read', '--direct', '--output', str(root / 'evidence'),
                '--launcher', str(launcher), '--key-file', str(key), '--timeout', '5'])

    def test_direct_nondefault_url_fails_before_launcher_and_keeps_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.prepare(Path(temp))
            args.base_url = 'https://other-owned.invalid/v1'
            result = live.run_case(args)
            self.assertFalse(result['passed'])
            self.assertNotEqual(result.get('upstream_https_url'), args.base_url)
            self.assertTrue((args.output / 'summary.json').exists())
            self.assertFalse((args.output / 'cli-version.txt').exists())

    def test_startup_invalid_key_mode_keeps_failed_summary_without_credential(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.prepare(Path(temp))
            args.key_file.chmod(0o644)
            result = live.run_case(args)
            self.assertFalse(result['passed'])
            self.assertTrue(result['errors'])
            self.assertTrue((args.output / 'summary.json').exists())
            self.assertNotIn(KEY, (args.output / 'summary.json').read_text())
            before = (args.output / 'summary.json').read_bytes()
            with self.assertRaises(ValueError):
                live.run_case(args)
            self.assertEqual((args.output / 'summary.json').read_bytes(), before)

    def test_corrupted_or_unreadable_model_file_keeps_failed_summary(self):
        for corruption in ('non_utf8', 'unreadable'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                args = self.prepare(root)
                args.case = 'write'
                contents = b'\xff' + KEY.encode() if corruption == 'non_utf8' else b"print('M4_QWEN_WRITE_OK')\n"
                args.launcher.write_text('#!' + sys.executable + '\n' +
                    'import json, sys\nfrom pathlib import Path\n' +
                    "if '--version' in sys.argv:\n print('owned-fixture'); sys.exit()\n" +
                    'Path("hello.py").write_bytes(' + repr(contents) + ')\n' +
                    'events=' + repr(command_events('M4_QWEN_WRITE_OK')) + '\n' +
                    'for event in events: print(json.dumps(event))\n')
                original = Path.read_bytes
                def read_bytes(file):
                    if corruption == 'unreadable' and file.name == 'hello.py':
                        raise PermissionError('owned simulated read denial')
                    return original(file)
                with mock.patch.object(Path, 'read_bytes', read_bytes):
                    summary = live.run_case(args)
                self.assertFalse(summary['passed'])
                self.assertTrue((args.output / 'summary.json').exists())
                self.assertTrue(any('collection' in error.lower() for error in summary['errors']))
                self.assertNotIn(KEY, ''.join(p.read_text() for p in args.output.iterdir() if p.is_file()))
                if corruption == 'non_utf8':
                    binary = json.loads((args.output / 'hello.py.bytes.json').read_text())
                    retained = base64.b64decode(binary['data'])
                    self.assertTrue(retained.startswith(b'\xff'))
                    self.assertNotIn(KEY.encode(), retained)
                else:
                    self.assertEqual(summary['independent_returncode'], 0, 'independent verification was removed')

    def test_sigterm_and_keyboard_interrupt_reap_owned_child_and_preserve_partial_output(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=signum), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                args = self.prepare(root)
                pidfile = root / 'owned-child.pid'
                args.launcher.write_text('#!' + sys.executable + '\n' +
                    'import json, os, sys, time\nfrom pathlib import Path\n' +
                    "if '--version' in sys.argv:\n print('owned-fixture'); sys.exit()\n" +
                    "print(json.dumps({'type':'thread.started','thread_id':'owned-interrupted'}), flush=True)\n" +
                    'Path(' + repr(str(pidfile)) + ').write_text(str(os.getpid()))\n' +
                    'time.sleep(30)\n')
                command = [sys.executable, str(ROOT / 'verify_codex_live.py'), '--case', 'read', '--direct',
                    '--output', str(args.output), '--launcher', str(args.launcher), '--key-file', str(args.key_file), '--timeout', '30']
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    deadline = time.monotonic() + 5
                    while not pidfile.exists() and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertTrue(pidfile.exists(), 'owned CLI child did not start')
                    child_pid = int(pidfile.read_text())
                    process.send_signal(signum)
                    stdout, stderr = process.communicate(timeout=5)
                    self.assertEqual(process.returncode, 1, stdout + stderr)
                    summary = json.loads((args.output / 'summary.json').read_text())
                    self.assertFalse(summary['passed'])
                    self.assertTrue(any('interrupt' in error.lower() for error in summary['errors']))
                    self.assertIn('owned-interrupted', (args.output / 'cli-stdout.jsonl').read_text())
                    with self.assertRaises(ProcessLookupError):
                        os.kill(child_pid, 0)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()
                    if pidfile.exists():
                        with contextlib.suppress(ProcessLookupError):
                            os.kill(int(pidfile.read_text()), signal.SIGKILL)


class RelayTests(unittest.TestCase):
    def test_shutdown_interrupts_stalled_upstream_and_finalizes_records(self):
        release = threading.Event()
        class Upstream(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(b'data: {"type":"response.created","sequence_number":0}\n\n')
                self.wfile.flush()
                release.wait(5)
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        fallback = threading.Timer(1.0, release.set)
        fallback.start()
        client = None
        try:
            relay = live.CaptureRelay(f'http://127.0.0.1:{server.server_port}/v1', KEY, 20)
            with relay:
                client = http.client.HTTPConnection('127.0.0.1', relay.port, timeout=2)
                client.request('POST', '/v1/responses', b'{}', {'Authorization': 'Bearer ' + KEY})
                response = client.getresponse()
                self.assertTrue(response.readline().startswith(b'data:'))
                started = time.monotonic()
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, .5, 'shutdown waited for upstream rather than interrupting owned socket')
            record = relay.records[0]
            self.assertTrue(record['finalized'])
            self.assertIn('response.created', record['response_body'])
            saved = json.dumps(relay.records)
            time.sleep(.02)
            self.assertEqual(json.dumps(relay.records), saved)
        finally:
            fallback.cancel()
            release.set()
            if client:
                client.close()
            server.shutdown()
            server.server_close()
            thread.join()

    def test_shutdown_interrupts_owned_tls_handshake_without_disabling_verification(self):
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        accepted, release = threading.Event(), threading.Event()
        def stall():
            connection, _ = listener.accept()
            with connection:
                connection.recv(4096)  # The owned client's TLS ClientHello, never an HTTP body/key.
                accepted.set()
                release.wait(5)
        upstream = threading.Thread(target=stall, daemon=True)
        upstream.start()
        fallback = threading.Timer(1.0, release.set)
        fallback.start()
        client = None
        finished = threading.Event()
        try:
            relay = live.CaptureRelay(f'https://127.0.0.1:{listener.getsockname()[1]}/v1', KEY, 20)
            with relay:
                client = http.client.HTTPConnection('127.0.0.1', relay.port, timeout=3)
                def request():
                    try:
                        client.request('POST', '/v1/responses', b'{}', {'Authorization': 'Bearer ' + KEY})
                        client.getresponse()
                    except (OSError, http.client.HTTPException):
                        pass
                    finally:
                        finished.set()
                downstream = threading.Thread(target=request, daemon=True)
                downstream.start()
                self.assertTrue(accepted.wait(1))
                started = time.monotonic()
            self.assertLess(time.monotonic() - started, .5)
            self.assertTrue(finished.wait(1))
            self.assertTrue(relay.records[0]['finalized'])
            self.assertNotIn(KEY, json.dumps(relay.records))
            https = next(handler for handler in relay.opener.handlers if isinstance(handler, live.urllib.request.HTTPSHandler))
            self.assertTrue(https._context.check_hostname)
            self.assertEqual(https._context.verify_mode, live.ssl.CERT_REQUIRED)
        finally:
            fallback.cancel()
            release.set()
            if client:
                client.close()
            upstream.join(2)
            listener.close()

    def test_streams_before_upstream_finishes_and_forwards_exact_payload(self):
        first = threading.Event()
        release = threading.Event()
        seen = []
        class Upstream(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                seen.append((self.path, self.rfile.read(int(self.headers['Content-Length'])),
                             self.headers.get('Authorization'), self.headers.get('X-Owned')))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(b'event: response.created\ndata: {"type":"response.created","sequence_number":0}\n\n')
                self.wfile.flush()
                first.set()
                release.wait(3)
                self.wfile.write(b'event: response.completed\ndata: {"type":"response.completed","sequence_number":1,"response":{"status":"completed","output":[]}}\n\n')
                self.wfile.flush()
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with live.CaptureRelay(f'http://127.0.0.1:{server.server_port}/v1', KEY, 5) as relay:
                client = http.client.HTTPConnection('127.0.0.1', relay.port, timeout=2)
                body = b'{ "model": "macqwen-flashnext", "input": [] }'
                client.request('POST', '/v1/responses', body, {'Authorization': 'Bearer ' + KEY,
                               'Content-Type': 'application/json', 'X-Owned': 'yes'})
                response = client.getresponse()
                self.assertTrue(first.wait(1))
                self.assertEqual(response.readline(), b'event: response.created\n')
                release.set()
                response.read()
                client.close()
                self.assertEqual(seen, [('/v1/responses', body, 'Bearer ' + KEY, 'yes')])
                record = relay.records[0]
                self.assertEqual([e['type'] for e in record['events']], ['response.created', 'response.completed'])
                self.assertNotIn(KEY, json.dumps(relay.records))
                denied = http.client.HTTPConnection('127.0.0.1', relay.port, timeout=2)
                denied.request('POST', '/v1/responses/compact', b'{}', {'Authorization': 'Bearer ' + KEY})
                self.assertEqual(denied.getresponse().status, 404)
                self.assertEqual(relay.records[-1]['path'], '/v1/responses/compact')
                self.assertEqual(relay.records[-1]['status'], 404)
                denied.close()
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            thread.join()


@unittest.skipUnless(importlib.util.find_spec('macqwen') and shutil.which('codex'),
                     'native fixture requires existing reviewed source PYTHONPATH and installed CLI')
class NativeCompactionTests(unittest.TestCase):
    def test_native_2500_compaction_retains_whole_first_read_then_continues(self):
        from macqwen.sampling import Sampling
        from macqwen.server import MacqwenHTTPServer, MacqwenHandler, ModelService, model_card
        from macqwen.tests.test_server import FakeSession
        session = FakeSession([])
        session.preferences['thinking_enabled'] = False
        session.backend.context_window = 4096
        session.backend.sampling = Sampling.greedy_settings()
        original = session.backend.generate
        stats = {'input': 1787, 'output': 40}
        def generate(max_tokens, out=None, should_cancel=None, on_prefill_progress=None):
            if on_prefill_progress:
                on_prefill_progress(0, len(session.backend.pending))
            text, unused = original(max_tokens, out)
            return text, SimpleNamespace(finish='stop', prompt_tokens=stats['input'], tokens=stats['output'])
        session.backend.generate = generate
        records, state = [], {'reads': 0, 'summaries': 0}
        first_value, marker = 'next-owned.txt', 'OWNED_NATIVE_COMPACT_FINAL'
        first_text = first_value + '\n' + ''.join(f'status {index:03d}: local fixture is unchanged; next action follows the first line\n' for index in range(64))
        class Handler(MacqwenHandler):
            def log_message(self, *args): pass
            def _payload(self):
                payload = super()._payload()
                record = {'path': '/v1/responses', 'request': payload, 'status': 200, 'events': []}
                records.append(record)
                if live.COMPACT_PROMPT in json.dumps(payload, ensure_ascii=False):
                    if not any(item.get('type') == 'function_call_output' and first_text in item.get('output', '')
                               for item in payload.get('input', [])):
                        raise ValueError('compaction discarded actual first read result')
                    state['summaries'] += 1
                    session.backend.pieces = ['We read fixture.txt. Its contents are next-owned.txt. Next read next-owned.txt and report its contents.']
                    stats.update(input=900, output=40)
                elif state['reads'] == 0:
                    state['reads'] += 1
                    session.backend.pieces = ['<tool_call>' + json.dumps({'id': 'call_compact_owned_first',
                        'name': 'exec_command', 'arguments': {'cmd': 'cat fixture.txt', 'max_output_tokens': 1600}}) + '</tool_call>']
                    stats.update(input=1787, output=40)
                elif state['reads'] == 1:
                    if state['summaries'] != 1:
                        raise ValueError('expected one compaction after first read')
                    state['reads'] += 1
                    session.backend.pieces = ['<tool_call>' + json.dumps({'id': 'call_compact_owned_second',
                        'name': 'exec_command', 'arguments': {'cmd': 'cat next-owned.txt', 'max_output_tokens': 64}}) + '</tool_call>']
                    stats.update(input=2084, output=40)
                else:
                    if not any(item.get('call_id') == 'call_compact_owned_second' and marker in item.get('output', '')
                               for item in payload.get('input', [])):
                        raise ValueError('missing actual dependent second read output')
                    session.backend.pieces = [marker]
                    stats.update(input=2200, output=10)
                session.backend.tokenizer.rendered = 'x' * stats['input']
                return payload
            def _sse(self, value, event=None):
                records[-1]['events'].append(value)
                return super()._sse(value, event)
        service = ModelService(session)
        server = MacqwenHTTPServer(('127.0.0.1', 0), service, KEY)
        server.RequestHandlerClass = Handler
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            home = root / 'home'
            home.mkdir()
            (home / '.codex').mkdir()
            (home / '.codex/config.toml').write_text('sandbox_mode="danger-full-access"\napproval_policy="never"\n')
            cat = root / 'catalog.json'
            cat.write_text(json.dumps({'models': [model_card(service)]}))
            install = subprocess.run([sys.executable, str(ROOT / 'codex/install-client.py'), '--home', str(home),
                                      '--catalog', str(cat)], input=KEY + '\n', text=True, capture_output=True)
            self.assertEqual(install.returncode, 0, install.stderr)
            workspace = root / 'work'
            workspace.mkdir()
            (workspace / 'fixture.txt').write_text(first_text)
            (workspace / first_value).write_text(marker + '\n')
            command = [str(home / '.local/bin/codex-m4'), '--strict-config', 'exec', '--skip-git-repo-check', '--json',
                '-c', f'model_auto_compact_token_limit={live.COMPACT_LIMIT}', '-c',
                'model_providers.m4_qwen.base_url=' + json.dumps(f'http://127.0.0.1:{server.server_port}/v1'),
                live.compact_prompt()]
            env = dict(os.environ, CODEX_HOME=str(home / '.codex'))
            thread.start()
            try:
                with contextlib.redirect_stdout(__import__('io').StringIO()):
                    result = subprocess.run(command, cwd=workspace, env=env, text=True, capture_output=True,
                                            timeout=30, stdin=subprocess.DEVNULL)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
            if os.environ.get('TASK3_ARTIFACT_DIR'):
                artifacts = Path(tempfile.mkdtemp(prefix='native-compact-', dir=os.environ['TASK3_ARTIFACT_DIR']))
                writer = live.EvidenceWriter(artifacts, KEY)
                writer.write('cli-stdout.jsonl', result.stdout)
                writer.write('cli-stderr.txt', result.stderr)
                writer.write('requests-events.json', records)
                writer.write('command.json', command)
                writer.write('summary.json', {'synthetic_model_and_usage': True, 'real_native_shell_execution': True,
                    'returncode': result.returncode, 'state': state, 'compact_limit': live.COMPACT_LIMIT})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr + json.dumps([r['events'] for r in records]))
            events = [json.loads(line) for line in result.stdout.splitlines()]
            errors, details = live.validate_case('compact', workspace, marker, events, records, False, expected_first=first_text)
            self.assertEqual(errors, [])
            self.assertEqual(state, {'reads': 2, 'summaries': 1})
            self.assertIn('call_compact_owned_second', details['matched_call_ids'])


if __name__ == '__main__':
    unittest.main()
