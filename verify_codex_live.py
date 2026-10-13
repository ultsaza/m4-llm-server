#!/usr/bin/env python3
"""Qualify the installed m4 launcher; never install, configure or fabricate replies."""
import argparse
import base64
import contextlib
import http.server
import http.client
import json
import os
from pathlib import Path
import secrets
import re
import signal
import socket
import ssl
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

COMPACT_PROMPT = (Path(__file__).resolve().parent / 'codex/compact-prompt.txt').read_text().strip()
MODEL = 'macqwen-flashnext'
PRODUCTION_URL = os.environ.get("MACQWEN_BASE_URL", "https://server.example.invalid/v1")
COMPACT_LIMIT = 2500
COMPACTION_WARNING = ('Heads up: Long threads and multiple compactions can cause the model to be less accurate. '
                      'Start a new thread when possible to keep threads small and targeted.')


def compact_fixture_text(filename, rows=64):
    return filename + '\n' + ''.join(
        f'status {index:03d}: local fixture is unchanged; next action follows the first line\n'
        for index in range(rows))


def compact_prompt(output_tokens=1600):
    return (f'Read ALL of fixture.txt by running exactly `cat fixture.txt` in ONE shell function call '
            f'with max_output_tokens={output_tokens} so its whole contents are returned. '
            'Return the raw full output; do not use head, tail, line numbering, cat -A, or any output truncation. '
            'Its FIRST LINE contains the name of a second file; the remaining lines are nonsecret status context. '
            'Then read that second file in a SEPARATE shell function call and put its exact contents in your final answer. '
            'This task requires exactly TWO successful shell commands: the full first read and the second read. '
            'Do not run metadata, checksum, byte-inspection, or further verification commands. '
            'Do not guess the filename or contents. If context compaction occurs, treat verified reads recorded in its summary '
            'as completed and follow its NEXT action; never repeat completed reads. Once both reads are complete, '
            'answer immediately with only the second file contents and no further tool calls.')


class EvidenceWriter:
    def __init__(self, directory, key):
        self.directory, self.key, self.credential_found = directory, key, False

    def write(self, name, value):
        text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
        (self.directory / name).write_text(self.redact(text))

    def redact(self, value):
        for representation in (self.key, json.dumps(self.key)[1:-1]):
            if isinstance(value, bytes):
                representation = representation.encode()
            if representation and representation in value:
                self.credential_found = True
                replacement = b'[REDACTED_CREDENTIAL]' if isinstance(value, bytes) else '[REDACTED_CREDENTIAL]'
                value = value.replace(representation, replacement)
        return value

    def write_binary(self, name, value):
        self.write(name, {'encoding': 'base64', 'data': base64.b64encode(self.redact(value)).decode('ascii')})


def read_key(path):
    with path.open() as file:
        if stat.S_IMODE(os.fstat(file.fileno()).st_mode) != 0o600:
            raise ValueError('credential file must have mode 0600')
        key = file.read().strip()
    if not key or '\n' in key or '\r' in key:
        raise ValueError('credential file must contain one nonempty line')
    return key


class CaptureRelay:
    """Body-only qualification capture. urllib verifies production TLS certificates."""
    def __init__(self, upstream, key, timeout):
        self.upstream, self.key, self.timeout = upstream.rstrip('/'), key, timeout
        self.records = []
        self.stopping = threading.Event()
        relay = self
        self.connection_lock = threading.Lock()
        self.connections, self.response_sockets = set(), set()

        class OwnedResponse(http.client.HTTPResponse):
            def __init__(self, sock, *args, **kwargs):
                super().__init__(sock, *args, **kwargs)
                self.owned_socket = sock
                with relay.connection_lock:
                    relay.response_sockets.add(sock)
                    stopped = relay.stopping.is_set()
                if stopped:
                    relay.shutdown_socket(sock)

            def close(self):
                try:
                    super().close()
                finally:
                    with relay.connection_lock:
                        relay.response_sockets.discard(self.owned_socket)

        class OwnedHTTPConnection(http.client.HTTPConnection):
            response_class = OwnedResponse
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                with relay.connection_lock:
                    if relay.stopping.is_set():
                        raise OSError('qualification relay stopped')
                    relay.connections.add(self)

        class OwnedHTTPSConnection(http.client.HTTPSConnection):
            response_class = OwnedResponse
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                with relay.connection_lock:
                    if relay.stopping.is_set():
                        raise OSError('qualification relay stopped')
                    relay.connections.add(self)

            def connect(self):
                http.client.HTTPConnection.connect(self)
                hostname = self._tunnel_host or self.host
                self.sock = self._context.wrap_socket(self.sock, server_hostname=hostname, do_handshake_on_connect=False)
                if relay.stopping.is_set():
                    relay.shutdown_socket(self.sock)
                self.sock.do_handshake()

        tls_context = ssl.create_default_context()
        class HTTPHandler(urllib.request.HTTPHandler):
            def http_open(self, request):
                return self.do_open(OwnedHTTPConnection, request)
        class HTTPSHandler(urllib.request.HTTPSHandler):
            def https_open(self, request):
                return self.do_open(OwnedHTTPSConnection, request, context=tls_context)
        self.opener = urllib.request.build_opener(HTTPHandler(), HTTPSHandler(context=tls_context))

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.forward()

            def do_POST(self):
                self.forward()

            def forward(self):
                if self.headers.get('Authorization') != 'Bearer ' + relay.key:
                    self.send_error(401, 'authentication required')
                    return
                body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
                record = {'path': self.path, 'method': self.command, 'request_body': body.decode('utf-8'),
                          'events': [], 'event_times': [], 'response_body': '', 'authentication_verified': True}
                relay.records.append(record)
                if (self.command, self.path) not in {('GET', '/v1/models'), ('POST', '/v1/responses')}:
                    record['status'] = 404
                    record['error'] = 'unknown qualification route'
                    record['finalized'] = True
                    self.send_error(404, 'unknown qualification route')
                    return
                started = time.monotonic()
                chunks, pending = [], b''
                try:
                    record['request'] = json.loads(body) if body else None
                    headers = {name: value for name, value in self.headers.items()
                               if name.lower() not in {'host', 'connection', 'content-length', 'transfer-encoding'}}
                    request = urllib.request.Request(relay.upstream + self.path[len('/v1'):],
                              data=body if self.command == 'POST' else None, headers=headers, method=self.command)
                    try:
                        upstream = relay.opener.open(request, timeout=relay.timeout)
                    except urllib.error.HTTPError as error:
                        upstream = error
                    with upstream:
                        record['status'] = upstream.status
                        self.send_response(upstream.status)
                        for name, value in upstream.headers.items():
                            if name.lower() in {'content-type', 'retry-after'}:
                                self.send_header(name, value)
                        self.send_header('Connection', 'close')
                        self.end_headers()
                        is_sse = 'text/event-stream' in upstream.headers.get('Content-Type', '')
                        while True:
                            if relay.stopping.is_set():
                                record['error'] = 'qualification relay interrupted'
                                break
                            chunk = upstream.read1(65536)
                            if not chunk:
                                if relay.stopping.is_set():
                                    record['error'] = 'qualification relay interrupted'
                                break
                            chunks.append(chunk)
                            if is_sse:
                                pending += chunk.replace(b'\r\n', b'\n')
                                while b'\n\n' in pending:
                                    block, pending = pending.split(b'\n\n', 1)
                                    data = b'\n'.join(line[5:].lstrip() for line in block.splitlines() if line.startswith(b'data:'))
                                    if data and data != b'[DONE]':
                                        record['events'].append(json.loads(data))
                                        record['event_times'].append(time.monotonic() - started)
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        if is_sse and pending.strip():
                            record['error'] = 'incomplete SSE frame'
                except Exception as error:
                    record['error'] = str(error)
                finally:
                    record['response_body'] = b''.join(chunks).decode('utf-8', errors='replace')
                    record['wall_seconds'] = time.monotonic() - started
                    record['finalized'] = True
                    self.close_connection = True

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = False
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .05}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    @staticmethod
    def shutdown_socket(sock):
        # urllib closes the socket object while its response file keeps the FD alive.
        # Duplicate only this owned FD to bypass SSL close-state guards; shutdown
        # interrupts read1 without acquiring the buffered-reader lock.
        with contextlib.suppress(OSError):
            descriptor = os.dup(sock.fileno())
            try:
                cancel_socket = socket.socket(fileno=descriptor)
            except OSError:
                os.close(descriptor)
                raise
            with cancel_socket:
                cancel_socket.shutdown(socket.SHUT_RDWR)

    def __exit__(self, *args):
        self.stopping.set()
        with self.connection_lock:
            owned_sockets = set(self.response_sockets)
            owned_sockets.update(connection.sock for connection in self.connections if connection.sock is not None)
        for sock in owned_sockets:
            self.shutdown_socket(sock)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def validate_case(case, workspace, marker, cli_events, records, direct, *, expected_first=None):
    errors = []
    commands = [event['item'] for event in cli_events if event.get('type') == 'item.completed'
                and event.get('item', {}).get('type') == 'command_execution']
    finals = [event['item'].get('text', '') for event in cli_events if event.get('type') == 'item.completed'
              and event.get('item', {}).get('type') == 'agent_message']
    warnings = []
    for event in cli_events:
        kind = event.get('type', '')
        item = event.get('item', {})
        if kind in {'error', 'turn.failed'} or item.get('type') == 'error':
            message = str(item.get('message', item) if item.get('type') == 'error'
                          else event.get('message', event))
            if kind == 'item.completed' and (
                re.fullmatch(r'Exceeded skills context budget\. All skill descriptions were removed and [0-9]+ additional skills were not included in the model-visible skills list\.', message)
                or (case == 'compact' and message == COMPACTION_WARNING)
            ):
                warnings.append(message)
            else:
                errors.append('CLI error: ' + message)
    if not any(event.get('type') == 'turn.completed' for event in cli_events):
        errors.append('CLI turn did not complete')
    successful = [item for item in commands if item.get('exit_code') == 0 and item.get('status') == 'completed']
    expected = marker if case in {'read', 'compact'} else 'M4_QWEN_WRITE_OK'
    observed = [item for item in successful if expected in item.get('aggregated_output', '')]
    if not observed:
        errors.append('no successful CLI command with actual expected output')
    if not finals or expected not in finals[-1]:
        errors.append('final answer does not contain expected observed value')
    compact_name = None
    if case == 'compact' and not any(expected_first and expected_first in item.get('aggregated_output', '') for item in successful):
        errors.append('no successful CLI command returning the whole first fixture')
    if case in {'read', 'compact'}:
        try:
            fixture = workspace / 'fixture.txt'
            if fixture.is_symlink():
                raise ValueError('fixture changed to symlink')
            value_file = fixture
            if case == 'compact':
                contents = fixture.read_text()
                if expected_first is None or contents != expected_first:
                    raise ValueError('first fixture contents changed')
                compact_name = contents.splitlines()[0]
                if Path(compact_name).name != compact_name:
                    raise ValueError('fixture must identify an owned filename')
                value_file = workspace / compact_name
            if value_file.is_symlink() or value_file.read_text() != marker + '\n':
                raise ValueError('read fixture contents changed')
        except Exception as error:
            errors.append('independent read verification failed: ' + str(error))
    details = {'commands': commands, 'final_messages': finals, 'known_warnings': warnings,
               'wire_ids_captured': not direct, 'matched_call_ids': [], 'requests': []}
    if not direct:
        emitted, replays, summaries, retained_summaries, v2_summaries = [], [], [], [], []
        for index, record in enumerate(records):
            request = record.get('request') or {}
            if record.get('error') or record.get('status', 0) >= 400:
                errors.append('relay request failed: ' + str(record.get('error', record.get('status'))))
            if record['path'] != '/v1/responses':
                continue
            events = record.get('events', [])
            terminals = [event for event in events if event.get('type') in {'response.completed', 'response.failed', 'response.incomplete', 'error'}]
            if not terminals or terminals[-1].get('type') != 'response.completed' or terminals[-1].get('response', {}).get('status') != 'completed':
                errors.append('Responses stream did not complete successfully')
            numbers = [event['sequence_number'] for event in events if 'sequence_number' in event]
            if any(a >= b for a, b in zip(numbers, numbers[1:])):
                errors.append('SSE sequence numbers not strictly increasing')
            details['requests'].append({'index': index, 'model': request.get('model'),
                'max_output_tokens': request.get('max_output_tokens', 512), 'tools': request.get('tools', []),
                'input_structure': [{'type': item.get('type'), 'role': item.get('role')} for item in request.get('input', [])],
                'terminals': [{'type': event.get('type'), 'usage': event.get('response', {}).get('usage')} for event in terminals]})
            if request.get('model') not in {None, MODEL}:
                errors.append('unexpected model alias')
            v2 = any(item.get('type') == 'compaction_trigger' for item in request.get('input', []))
            if v2 or COMPACT_PROMPT in json.dumps(request, ensure_ascii=False):
                summaries.append(index)
                summary_text = '\n'.join(part.get('text', '') for terminal in terminals
                    for output in terminal.get('response', {}).get('output', []) for part in output.get('content', []))
                first_outputs = [item['aggregated_output'] for item in successful
                    if expected_first and expected_first in item.get('aggregated_output', '')]
                first_replays = [item for item in request.get('input', [])
                    if item.get('type') == 'function_call_output'
                    and any(output in item.get('output', '') for output in first_outputs)
                    and any(emit_index < index and call.get('call_id') and call['call_id'] == item.get('call_id')
                        and call.get('name') in {'exec_command', 'shell_command', 'write_stdin'}
                        for emit_index, call in emitted)]
                packets = [output for terminal in terminals
                    for output in terminal.get('response', {}).get('output', []) if output.get('type') == 'compaction']
                done = [event.get('item') for event in events if event.get('type') == 'response.output_item.done'
                        and event.get('item', {}).get('type') == 'compaction']
                if v2:
                    if len(packets) == 1 and done == packets and first_replays and packets[0].get('encrypted_content'):
                        token = packets[0]['encrypted_content']
                        replayed = any(later > index and any(item.get('type') == 'compaction'
                            and item.get('encrypted_content') == token for item in later_record.get('request', {}).get('input', []))
                            for later, later_record in enumerate(records))
                        if replayed:
                            retained_summaries.append(index)
                            v2_summaries.append(index)
                elif compact_name and compact_name in summary_text and first_replays:
                    retained_summaries.append(index)
            for event in terminals:
                for item in event.get('response', {}).get('output', []):
                    if item.get('type') == 'function_call':
                        emitted.append((index, item))
            for item in request.get('input', []):
                if item.get('type') == 'function_call_output':
                    replays.append((index, item))
        matched_indices = []
        for emit_index, call in emitted:
            if call.get('name') not in {'exec_command', 'shell_command', 'write_stdin'}:
                continue
            for replay_index, replay in replays:
                if replay_index > emit_index and call.get('call_id') and call['call_id'] == replay.get('call_id'):
                    if any(item['aggregated_output'].strip() and item['aggregated_output'].strip() in replay.get('output', '') for item in observed):
                        details['matched_call_ids'].append(call['call_id'])
                        matched_indices.append(emit_index)
        if not details['matched_call_ids']:
            errors.append('no emitted call ID replayed with observed command result')
        details['summarization_request_indices'] = summaries
        details['summaries_retaining_filename'] = retained_summaries
        details['v2_compaction_request_indices'] = v2_summaries
        if case == 'compact' and not any(emit_index > summary for summary in retained_summaries for emit_index in matched_indices):
            errors.append('no observed summarization followed by continued tool loop')
    elif case == 'compact':
        errors.append('direct mode cannot qualify compaction request shape')
    if case == 'write':
        file = workspace / 'hello.py'
        try:
            if file.is_symlink() or not file.is_file():
                raise ValueError('hello.py must be an owned regular file')
            details['independent_source'] = file.read_text()
            result = subprocess.run([sys.executable, str(file)], cwd=workspace, capture_output=True,
                                    text=True, timeout=10, stdin=subprocess.DEVNULL)
            details.update(independent_stdout=result.stdout, independent_stderr=result.stderr,
                           independent_returncode=result.returncode)
            if result.returncode != 0 or result.stdout != 'M4_QWEN_WRITE_OK\n' or result.stderr:
                errors.append('independent hello.py execution did not match exact behavior')
        except Exception as error:
            errors.append('independent write verification failed: ' + str(error))
    return errors, details



def run_owned_cli(command, workspace, timeout, errors):
    process = subprocess.Popen(command, cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               stdin=subprocess.DEVNULL, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except (Exception, KeyboardInterrupt) as error:
        reason = 'CLI timeout' if isinstance(error, subprocess.TimeoutExpired) else (
            'CLI interrupted: ' + str(error) if isinstance(error, KeyboardInterrupt) else 'CLI process failure: ' + str(error))
        errors.append(reason)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=5)
    return stdout, stderr, process.returncode


@contextlib.contextmanager
def preserve_on_signal():
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(signum, frame):
        # A second interrupt must not interrupt owned-child cleanup/evidence persistence.
        for number in previous:
            signal.signal(number, signal.SIG_IGN)
        raise KeyboardInterrupt('interrupted by ' + signal.Signals(signum).name)
    for signum in previous:
        signal.signal(signum, interrupted)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--case', choices=['read', 'write', 'compact'], required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--launcher', type=Path, default=Path.home() / '.local/bin/codex-m4')
    result.add_argument('--key-file', type=Path, default=Path.home() / '.config/m4-qwen/api-key')
    result.add_argument('--base-url', default=PRODUCTION_URL, help='relay upstream; direct mode requires production default')
    result.add_argument('--direct', action='store_true')
    result.add_argument('--timeout', type=int, default=1200)
    result.add_argument('--compact-limit', type=int, help='optional qualification-only threshold; otherwise use installed profile')
    result.add_argument('--compact-rows', type=int, default=64)
    result.add_argument('--tool-output-tokens', type=int, default=1600)
    return result


def run_case(args):
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError('output must be absent or an empty owned directory')
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {'case': args.case, 'mode': 'direct' if args.direct else 'qualification-relay',
               'passed': False, 'errors': [], 'returncode': None}
    try:
        if not 1 <= args.timeout <= 3600:
            raise ValueError('timeout must be between 1 and 3600 seconds')
        if (not 1 <= args.compact_rows <= 2048 or not 64 <= args.tool_output_tokens <= 16000
                or args.compact_limit is not None and not 500 <= args.compact_limit <= 16384):
            raise ValueError('invalid bounded compaction fixture settings')
        url = urllib.parse.urlsplit(args.base_url)
        if url.scheme != 'https' or not url.netloc or url.username or url.password or url.query or url.fragment or url.path.rstrip('/') != '/v1':
            raise ValueError('upstream must be credential-free HTTPS /v1 URL')
        if args.direct and args.base_url.rstrip('/') != PRODUCTION_URL:
            raise ValueError('direct mode requires the configured production URL; base-url only selects relay upstream')
        key = read_key(args.key_file)
    except (OSError, ValueError):
        # Do not retain user inputs or parser exception text before the secret is known.
        summary['errors'].append('startup validation failed; check production HTTPS URL, timeout and 0600 credential file')
        EvidenceWriter(args.output, '').write('summary.json', summary)
        return summary
    writer = EvidenceWriter(args.output, key)
    summary['upstream_https_url'] = args.base_url
    stdout, stderr, records, events = '', '', [], []
    with tempfile.TemporaryDirectory(prefix='m4-qwen-live-') as temp:
        workspace = Path(temp)
        marker = None
        if args.case in {'read', 'compact'}:
            marker = 'M4_QWEN_READ_' + secrets.token_hex(16)
            if args.case == 'compact':
                second_name = 'second-' + secrets.token_hex(8) + '.txt'
                (workspace / 'fixture.txt').write_text(compact_fixture_text(second_name, args.compact_rows))
                (workspace / second_name).write_text(marker + '\n')
            else:
                (workspace / 'fixture.txt').write_text(marker + '\n')
            prompt = 'Read fixture.txt using the available shell function. Put the exact file contents in your final answer. Do not guess the contents.'
            if args.case == 'compact':
                prompt = compact_prompt(args.tool_output_tokens)
        else:
            prompt = 'Create hello.py in the current directory that prints exactly M4_QWEN_WRITE_OK. Run it using the available shell function and include its actual output in your final answer.'
        command = [str(args.launcher.resolve()), '--strict-config', 'exec', '--skip-git-repo-check', '--json']
        if args.case != 'compact':
            command.append('--ephemeral')
        elif args.compact_limit is not None:
            command.extend(['-c', f'model_auto_compact_token_limit={args.compact_limit}'])
        started = time.monotonic()
        try:
            version = subprocess.run([str(args.launcher.resolve()), '--version'], capture_output=True, text=True,
                                     timeout=10, stdin=subprocess.DEVNULL)
            writer.write('cli-version.txt', version.stdout + version.stderr)
            summary['cli_version'] = version.stdout.strip()
            if version.returncode:
                summary['errors'].append('launcher version command failed')
            relay_context = contextlib.nullcontext(None) if args.direct else CaptureRelay(args.base_url, key, args.timeout)
            with relay_context as relay:
                if relay:
                    records = relay.records
                    command.extend(['-c', 'model_providers.m4_qwen.base_url=' + json.dumps(f'http://127.0.0.1:{relay.port}/v1')])
                command.append(prompt)
                summary['cli_command'] = command
                stdout, stderr, summary['returncode'] = run_owned_cli(command, workspace, args.timeout, summary['errors'])
            if summary['returncode'] != 0:
                summary['errors'].append('CLI returned nonzero status')
            for line in stdout.splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    summary['errors'].append('non-JSON CLI stdout')
            errors, details = validate_case(args.case, workspace, marker, events, records, args.direct,
                expected_first=compact_fixture_text(second_name, args.compact_rows) if args.case == 'compact' else None)
            summary.update(details)
            summary['errors'].extend(errors)
            # Stderr is not ignored when the native process returns zero.
            if any(word in stderr.lower() for word in ('error', 'failed', 'panic', 'disconnect')):
                summary['errors'].append('unexpected diagnostic on CLI stderr')
        except KeyboardInterrupt as error:
            summary['errors'].append('qualification interrupted: ' + str(error))
        except Exception as error:
            summary['errors'].append(str(error))
        finally:
            summary['wall_seconds'] = time.monotonic() - started
            writer.write('cli-stdout.jsonl', stdout)
            writer.write('cli-stderr.txt', stderr)
            writer.write('relay-exchanges.json', records)
            writer.write('cli-events.json', events)
            writer.write('command.json', command)
            names = ['fixture.txt', 'hello.py']
            if args.case == 'compact':
                names.append(second_name)
            for name in names:
                file = workspace / name
                try:
                    if file.is_file() and not file.is_symlink():
                        data = file.read_bytes()
                        try:
                            writer.write(name, data.decode('utf-8'))
                        except UnicodeDecodeError:
                            summary['errors'].append('owned file collection failed UTF-8 decoding: ' + name)
                            writer.write_binary(name + '.bytes.json', data)
                except OSError as error:
                    summary['errors'].append('owned file collection failed: ' + name + ' (' + type(error).__name__ + ')')
            writer.write('summary.json', summary)
            if writer.credential_found:
                summary['errors'].append('credential occurrence redacted; qualification rejected')
            summary['passed'] = not summary['errors']
            writer.write('summary.json', summary)
    return summary


def main():
    try:
        with preserve_on_signal():
            summary = run_case(parser().parse_args())
    except Exception:
        print('Qualification could not start; check owned output, HTTPS URL, launcher and 0600 credential file.', file=sys.stderr)
        return 1
    print(json.dumps({'passed': summary['passed'], 'mode': summary['mode'], 'errors_count': len(summary['errors'])}))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
