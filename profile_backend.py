"""Attribute backend waits using existing instrumentation, without changing math."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

SOURCE = Path(os.environ.get("MACQWEN_SOURCE", str(Path(__file__).resolve().parent / "src")))
CHECKPOINT = Path(os.environ.get("MACQWEN_ROOT", str(Path.home() / "llm" / "macqwen"))) / 'models/qwen38-flash-next-4bit'
POLICY = {'pool_gb': 6, 'cache_mib': 512, 'wired_gb': 0,
          'pool_lock': False, 'stream_pack_chunk': 2, 'io_workers': 8}
DEPENDENCIES = tuple('models/flashnext/tests/bench/' + name for name in (
    'bench_split_p15.py', 'bench_long_states.py', 'gpu_pstates.py'))
COVERAGE = {
    'pool_preadv': 'module-local diagnostic proxy; same buffers, offsets, and syscall',
    'moe_issue': 'incomplete: expert-pool Metal issue is not timed',
    'router_sync': 'dependency/synchronization wait, not pure router or GPU cost',
    'critical_queue': 'last-completing task queue wait; not a removable-overhead estimate',
}


def provenance(tokens):
    from models.flashnext.tests.bench import bench_production as bench
    from models.flashnext.tests.bench.bench_long_states import PROMPT

    return {**bench.benchmark_provenance(CHECKPOINT), 'configuration': POLICY,
            'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'wrapper_sha256': hashlib.sha256(Path(__file__).with_name('run_performance.sh').read_bytes()).hexdigest(),
            'diagnostic_dependencies': {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                                        for name in DEPENDENCIES},
            'prompt': PROMPT, 'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
            'coverage': COVERAGE,
            'purpose': 'attribution only; profiler adds overhead; second pass has a warm expert pool',
            'token_limit': tokens, 'sampling': 'greedy'}


class PoolReadProbe:
    """Replace only expert_pool.os; leave the real os module/syscall unchanged."""

    def __init__(self):
        self.lock = threading.Lock()
        self.calls = self.bytes = 0
        self.sizes = {}

    def __getattr__(self, name):
        return getattr(os, name)

    def preadv(self, fd, buffers, offset):
        from models.flashnext.store import _profiled_preadv

        count = _profiled_preadv(fd, buffers, offset)
        with self.lock:
            self.calls += 1
            self.bytes += count
            self.sizes[count] = self.sizes.get(count, 0) + 1
        return count

    def reset(self):
        with self.lock:
            self.calls = self.bytes = 0
            self.sizes.clear()


def child(args):
    sys.path.insert(0, str(SOURCE))
    from models.flashnext.settings.launch import apply_chat_environment

    apply_chat_environment(os.environ)
    os.environ.update(FLASHNEXT_NATIVE_MTP='off', FLASHNEXT_EXPERT_POOL_GB='6',
                      FLASHNEXT_WIRED_GB='0', FLASHNEXT_EXPERT_POOL_LOCK='0',
                      FLASHNEXT_STREAM_PACK_CHUNK='2', FLASHNEXT_IO_WORKERS='8',
                      FLASHNEXT_PROFILE_IO='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    with tempfile.TemporaryDirectory(prefix='macqwen-profile-pins-') as private:
        pins = Path(private) / 'pins.json'
        pins.write_text('{}')
        os.environ['FLASHNEXT_PIN_CACHE'] = str(pins)
        import mlx.core as mx

        mx.set_cache_limit(512 * 1024 ** 2)
        assert mx.set_cache_limit(512 * 1024 ** 2) == 512 * 1024 ** 2
        from macqwen.backends.flashnext import FlashNextBackend
        from models.flashnext import expert_cache, expert_pool
        from models.flashnext.checkpoint_policy import resolve_resident_experts
        from models.flashnext.diskio import ReadMeter
        from models.flashnext.tests.bench import bench_production as bench
        from models.flashnext.tests.bench.bench_long_states import Clocks
        from models.flashnext.tests.bench.bench_split_p15 import BUCKETS, decode

        child_provenance = provenance(args.tokens)
        (args.output_dir / 'child-provenance.json').write_text(json.dumps(child_provenance, indent=2) + '\n')
        assert child_provenance == json.loads((args.output_dir / 'parent-provenance.json').read_text())
        backend = FlashNextBackend(model_path=str(CHECKPOINT),
                                   resident_experts=resolve_resident_experts(None, str(CHECKPOINT)) or 8,
                                   threshold=1.0, routing_profile='exact-quality', native_mtp='off')
        clocks, meter = Clocks(), ReadMeter()
        for name, profiling in (('reference', False), ('profiled', True)):
            probe = PoolReadProbe()
            original_os, original_reset = expert_pool.os, expert_cache.reset_profile
            boundary = {}

            def reset_profile():
                original_reset()
                probe.reset()
                pool = backend.store._flashnext_expert_pool
                boundary.update(hits=pool.hits, misses=pool.misses)

            if profiling:
                expert_pool.os = probe
                expert_cache.reset_profile = reset_profile
            expert_cache.set_profile(profiling)
            try:
                row = decode(backend, args.tokens, clocks, meter)
            finally:
                expert_cache.set_profile(False)
                expert_pool.os, expert_cache.reset_profile = original_os, original_reset
            row['ids'] = list(backend.tape[-row['tokens']:]) if row['tokens'] else []
            row['token_sha256'] = hashlib.sha256(str(row['ids']).encode()).hexdigest()
            row['configuration'] = POLICY
            row['effective_pool_bytes'] = len(backend.store._flashnext_expert_pool.array)
            row['pool_locked'] = backend.store._flashnext_expert_pool.locked
            if profiling:
                totals = expert_cache.profile_totals()
                row['totals_seconds_or_counts'] = totals
                pool = backend.store._flashnext_expert_pool
                misses, hits = pool.misses - boundary['misses'], pool.hits - boundary['hits']
                row['pool_read_probe'] = {'calls': probe.calls, 'bytes': probe.bytes,
                                          'sizes': probe.sizes, 'misses': misses, 'hits': hits}
                row['coverage'] = COVERAGE
                row['split_ms_per_token'] = {key: totals[key] / max(1, row['tokens']) * 1000
                                             for key in BUCKETS}
                row['split_ms_per_token']['remainder'] = (
                    row['ms_per_token'] - sum(row['split_ms_per_token'].values()))
            path = args.output_dir / f'{name}.json'
            path.write_text(json.dumps(row, indent=2) + '\n')
            if profiling:
                assert probe.calls == 9 * misses
                assert probe.sizes == {819200: 3 * misses, 102400: 6 * misses}
                assert probe.bytes == 3072000 * misses
                assert totals['pread_calls'] >= probe.calls and totals['pread_bytes'] >= probe.bytes
            row['metal_runtime'] = bench.inspect_metal_runtime(backend, True)
            assert provenance(args.tokens) == child_provenance
            assert row['tokens'] == args.tokens
            assert abs(row['effective_pool_bytes'] / 1e9 - 6) < .01 and not row['pool_locked']
            path.write_text(json.dumps(row, indent=2) + '\n')
            print(name, json.dumps({k: row[k] for k in ('gen_rate', 'mb_per_token', 'token_sha256')}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--tokens', default=128, type=int)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.child:
        child(args)
        return
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(SOURCE))
    from macqwen.measurement import MeasurementRun
    from models.flashnext.tests.bench import bench_production as bench

    metadata = provenance(args.tokens)
    (args.output_dir / 'parent-provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for name, expected in metadata['diagnostic_dependencies'].items():
        target = args.output_dir / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((SOURCE / name).read_bytes())
        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
    for source, name, expected in ((Path(__file__), 'harness.py', metadata['harness_sha256']),
                                   (Path(__file__).with_name('run_performance.sh'), 'wrapper.sh', metadata['wrapper_sha256'])):
        target = args.output_dir / name
        target.write_bytes(source.read_bytes())
        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
    record = MeasurementRun(args.output_dir / 'record.jsonl', runtime='flashnext',
                            experiment='backend-profile', metadata=metadata)
    record.start()
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--child',
               '--tokens', str(args.tokens), '--output-dir', str(args.output_dir)]
    samples, process = [], None
    try:
        with (args.output_dir / 'child.log').open('w') as log:
            began = time.monotonic()
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    swap = subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip()
                    samples.append({'seconds': time.monotonic() - began, 'swap': swap})
                    if time.monotonic() - began > 240 or float(swap.split('used = ')[1].split('M')[0]) > 6144:
                        raise RuntimeError('profile exceeded latency or swap limit')
                    time.sleep(3)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                (args.output_dir / 'resources.json').write_text(json.dumps(samples, indent=2) + '\n')
        rows = {}
        for name in ('reference', 'profiled'):
            path = args.output_dir / f'{name}.json'
            if path.is_file():
                rows[name] = row = json.loads(path.read_text())
                record.arm(arm_id=name, condition=name, round_index=0, command=command,
                           metrics=row, tokens=row['ids'], token_digest=row['token_sha256'])
        if process.returncode or set(rows) != {'reference', 'profiled'}:
            raise RuntimeError(f'profile child failed: exit {process.returncode}')
        if rows['reference']['ids'] != rows['profiled']['ids']:
            raise RuntimeError('profile changed the greedy trajectory')
        assert json.loads((args.output_dir / 'child-provenance.json').read_text()) == metadata
        assert provenance(args.tokens) == metadata
        for name in rows:
            record.validation(name, 'passed', exact_tokens=True, source_frozen=True)
        summary = {'all_tokens_equal': True, 'attribution_only': True,
                   'split_ms_per_token': rows['profiled']['split_ms_per_token'],
                   'coverage': COVERAGE, 'parent_child_provenance_equal': True,
                   'pool_read_probe': rows['profiled']['pool_read_probe'],
                   'reference_rate': rows['reference']['gen_rate'],
                   'profiled_rate': rows['profiled']['gen_rate']}
        record.finish('completed', **summary)
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        record.failure(str(error), error_type=type(error).__name__)
        record.finish('completed_with_failures')
        raise


if __name__ == '__main__':
    main()
