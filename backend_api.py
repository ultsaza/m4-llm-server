"""Own one candidate server and run unchanged live checks plus a code request."""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import threading
import time

import requests

from benchmark_backend import (FEATURES, S3FIFO_FEATURES, S3FIFO_REVISION, s3fifo_settings, validate_s3fifo_diagnostics, defer_signals, FRACTION_REVISION, SESSION_REVISION, SESSION_RESET_FEATURE, FRACTION_FEATURE, CACHE_FEATURE,
                               validated_fraction, validate_source_selection, feature_environment)

ROOT = Path(os.environ.get("MACQWEN_ROOT", str(Path.home() / "llm" / "macqwen")))


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class ResourceSampler:
    """Persist independent required swap/footprint streams and their failures."""
    def __init__(self, process, output_dir, began, *, swap_interval=3, footprint_interval=15):
        self.process, self.output_dir, self.began = process, output_dir, began
        self.stopped, self.failed = threading.Event(), threading.Event()
        self.rows = {'global_swap': [], 'physical_footprint': []}
        self.failures, self.threads = [], []
        self.specs = (
            ('global_swap', ['sysctl', 'vm.swapusage'], swap_interval, 10, 'used ='),
            ('physical_footprint', ['/usr/bin/footprint', '-p', str(process.pid)],
             footprint_interval, 20, 'phys_footprint'),
        )

    def collect(self, kind, command, interval, timeout, required_text):
        while not self.stopped.is_set():
            row = {'seconds': time.monotonic() - self.began, 'pid': self.process.pid}
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
                row.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
                if result.returncode or required_text not in result.stdout:
                    raise RuntimeError('required resource command failed or measurement unavailable')
            except Exception as error:
                row['failure'] = {'type': type(error).__name__, 'message': str(error)}
                if isinstance(error, subprocess.TimeoutExpired):
                    for name in ('stdout', 'stderr'):
                        partial = getattr(error, name, None)
                        row[name] = partial.decode(errors='replace') if isinstance(partial, bytes) else partial
            self.rows[kind].append(row)
            try:
                with (self.output_dir / (kind + '.jsonl')).open('a') as handle:
                    handle.write(json.dumps(row) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception as error:
                row['persistence_failure'] = {'type': type(error).__name__, 'message': str(error)}
            if 'failure' in row or 'persistence_failure' in row:
                self.failures.append({'kind': kind, **row})
                self.failed.set()
                return
            self.stopped.wait(interval)

    def start(self):
        for spec in self.specs:
            thread = threading.Thread(target=self.collect, args=spec, daemon=True)
            self.threads.append(thread)  # Own partial startup before Thread.start.
            thread.start()

    def require_healthy(self):
        if self.failed.is_set():
            raise RuntimeError('required resource collection failed; see resource JSONL evidence')

    def stop(self):
        self.stopped.set()
        for thread in self.threads:
            if thread.ident is not None:
                thread.join(35)
        alive = any(thread.is_alive() for thread in self.threads)
        return {'passed': not self.failures and not alive
                         and len(self.threads) == len(self.specs)
                         and all(self.rows.values()),
                'sample_counts': {kind: len(rows) for kind, rows in self.rows.items()},
                'failures': self.failures, 'threads_still_alive': alive}


def candidate_command(args, release):
    command = [sys.executable, '-u', str(release / 'candidate_start.py'),
               '--condition', args.condition, '--feature', args.feature,
               '--sidecar', str(args.sidecar), '--slru-protected-fraction', str(args.slru_protected_fraction)]
    if getattr(args, 'slru_reset_on_session', False):
        command += ['--slru-reset-on-session']
    if args.source_revision is not None:
        command += ['--source-revision', args.source_revision]
    return command


def candidate_cache_events(log, args, require_runtime=False):
    if args.source_revision not in (FRACTION_REVISION, SESSION_REVISION, S3FIFO_REVISION):
        return []
    selected = feature_environment(args.feature, args.condition, args.sidecar, args.slru_protected_fraction, getattr(args, 'slru_reset_on_session', False))
    expected_policy = selected.get(CACHE_FEATURE, 'lru')
    expected_fraction = float(selected.get(FRACTION_FEATURE, 0.7))
    events = []
    for line in log.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get('event') == 'candidate_cache_validation':
            if row.get('policy') != expected_policy or (args.source_revision != S3FIFO_REVISION and row.get('protected_fraction') != expected_fraction):
                raise RuntimeError('candidate actual cache policy/fraction event mismatch')
            if args.source_revision == S3FIFO_REVISION and row.get('phase') == 'runtime' and expected_policy == 's3fifo':
                validate_s3fifo_diagnostics(row.get('s3fifo'), row.get('slots'))
            if args.source_revision == SESSION_REVISION:
                expected_reset = SESSION_RESET_FEATURE in selected
                count = row.get('session_reset_count')
                if row.get('slru_session_reset') is not expected_reset or type(count) is not int or count < 0:
                    raise RuntimeError('candidate actual session reset event mismatch')
            events.append(row)
    phases = {row.get('phase') for row in events}
    if 'startup' not in phases or (require_runtime and 'runtime' not in phases):
        raise RuntimeError('candidate lacks actual startup/runtime cache validation events')
    return events


def interrupted(signum, _frame):
    raise SystemExit(128 + signum)


def main():
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature', choices=FEATURES, default='sidecar')
    parser.add_argument('--slru-protected-fraction', type=validated_fraction, default=0.7)
    parser.add_argument('--slru-reset-on-session', action='store_true')
    parser.add_argument('--source-revision')
    parser.add_argument('--condition', choices=('off', 'on'), required=True)
    parser.add_argument('--sidecar', type=Path, default=ROOT / 'cache/pool-small-sidecar-20261012')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ready-timeout', type=float, default=600)
    args = parser.parse_args()
    release = Path(__file__).resolve().parent
    if args.feature in S3FIFO_FEATURES or args.slru_reset_on_session or args.slru_protected_fraction != 0.7 or args.source_revision is not None:
        try:
            validate_source_selection(release / 'src', args.source_revision, args.feature, args.slru_protected_fraction, getattr(args, 'slru_reset_on_session', False))
        except ValueError as error:
            parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    key_path = ROOT / 'config/api-key'
    if key_path.stat().st_mode & 0o077:
        raise RuntimeError('key must have mode 0600')
    headers = {'Authorization': 'Bearer ' + key_path.read_text().strip()}
    report = {'s3fifo_settings': s3fifo_settings() if args.source_revision == S3FIFO_REVISION else None,
              'cache_policy_arms': {c: feature_environment(args.feature, c, args.sidecar, args.slru_protected_fraction, args.slru_reset_on_session).get(CACHE_FEATURE, 'lru') for c in ('off', 'on')},
              'slru_reset_on_session': args.slru_reset_on_session,
              'slru_session_reset_arms': {c: SESSION_RESET_FEATURE in feature_environment(args.feature, c, args.sidecar, args.slru_protected_fraction, args.slru_reset_on_session) for c in ('off', 'on')},
              'slru_protected_fraction': args.slru_protected_fraction,
              'source_revision': args.source_revision, 'condition': args.condition, 'feature': args.feature, 'sidecar': str(args.sidecar), 'passed': False,
              'helpers': {n: hashlib.sha256((release / n).read_bytes()).hexdigest()
                          for n in ('verify_live.py', 'candidate_start.py', 'start.py', 'deployment.json', 'backend_api.py', 'benchmark_backend.py')},
              'resources': []}
    save(args.output_dir / 'api-run.json', report)
    process, live_process, sampler, api_passed = None, None, None, False
    began = time.monotonic()
    with (args.output_dir / 'candidate-server.log').open('w') as log:
        try:
            with defer_signals():
                process = subprocess.Popen(candidate_command(args, release),
                                           cwd=release, stdout=log, stderr=subprocess.STDOUT)
            sampler = ResourceSampler(process, args.output_dir, began)
            report['resources'] = sampler.rows
            report['resource_failures'] = sampler.failures
            sampler.start()
            until = time.monotonic() + args.ready_timeout
            while True:
                sampler.require_healthy()
                if process.poll() is not None:
                    raise RuntimeError('candidate exited before readiness; see candidate-server.log')
                try:
                    response = requests.get('http://127.0.0.1:8080/health', headers=headers, timeout=3)
                    if response.status_code == 200 and response.json()['status'] == 'ready':
                        break
                except requests.RequestException:
                    pass
                if time.monotonic() > until:
                    raise RuntimeError('candidate readiness timeout')
                time.sleep(1)
            report['cache_validation_events'] = candidate_cache_events(
                args.output_dir / 'candidate-server.log', args)
            report['startup_to_ready_seconds'] = time.monotonic() - began
            report['initial_health'] = response.json()
            save(args.output_dir / 'api-run.json', report)
            command = [sys.executable, '-u', str(release / 'verify_live.py'),
                       '--output', str(args.output_dir / 'live-api.json')]
            with (args.output_dir / 'live-api.log').open('w') as live_log:
                with defer_signals():
                    live_process = subprocess.Popen(command, stdout=live_log, stderr=subprocess.STDOUT)
                until = time.monotonic() + 2400
                while live_process.poll() is None:
                    sampler.require_healthy()
                    if time.monotonic() > until:
                        raise RuntimeError('unchanged live verifier timed out')
                    try:
                        live_process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
            report['verify_live_exit'] = live_process.returncode
            save(args.output_dir / 'api-run.json', report)
            if live_process.returncode:
                raise RuntimeError('unchanged verify_live.py failed')
            payload = {'model': 'macqwen-flashnext', 'max_tokens': 128, 'temperature': 0,
                       'messages': [{'role': 'user', 'content':
                                    'Write a Python function stable_unique(items) that removes duplicate hashable items while preserving order. Include a short example.'}]}
            started = time.monotonic()
            response = requests.post('http://127.0.0.1:8080/v1/chat/completions', headers=headers,
                                     json=payload, timeout=(15, 1200))
            report['code'] = {'payload': payload, 'status': response.status_code,
                              'raw_text': response.text, 'seconds': time.monotonic() - started}
            save(args.output_dir / 'api-run.json', report)  # Preserve before JSON parsing/assertions.
            try:
                report['code']['response'] = response.json()
            except ValueError as error:
                report['code']['parse_failure'] = {'type': type(error).__name__, 'message': str(error)}
                save(args.output_dir / 'api-run.json', report)
                raise
            save(args.output_dir / 'api-run.json', report)
            response.raise_for_status()
            if not report['code']['response']['choices'][0]['message']['content']:
                raise RuntimeError('empty code response')
            health = requests.get('http://127.0.0.1:8080/health', headers=headers, timeout=15)
            health.raise_for_status()
            report['code']['health'] = health.json()
            sampler.require_healthy()
            report['cache_validation_events'] = candidate_cache_events(
                args.output_dir / 'candidate-server.log', args, require_runtime=True)
            api_passed = True
        except BaseException as error:
            report['failure'] = {'type': type(error).__name__, 'message': str(error)}
            raise
        finally:
            # Repeated handled signals must not interrupt owned-child cleanup.
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK,
                {signal.SIGTERM, signal.SIGHUP, signal.SIGINT})
            try:
                if live_process and live_process.poll() is None:
                    live_process.terminate()
                    try:
                        live_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        live_process.kill()
                        live_process.wait()
                report['resource_gate'] = sampler.stop() if sampler else {'passed': False, 'reason': 'sampler not initialized'}
                if process:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    else:
                        process.wait()
                report['server_exit'] = process.returncode if process else None
                report['passed'] = api_passed and report['resource_gate']['passed']
                if not report['resource_gate']['passed']:
                    report.setdefault('failure', {'type': 'ResourceCollectionFailure',
                                                  'message': 'required resource evidence incomplete'})
                save(args.output_dir / 'api-run.json', report)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    if not report['passed']:
        raise RuntimeError('API/resource gates failed; see api-run.json')
    print(json.dumps({'passed': report['passed'], 'condition': args.condition,
                      'startup_to_ready_seconds': report['startup_to_ready_seconds']}))


if __name__ == '__main__':
    main()
