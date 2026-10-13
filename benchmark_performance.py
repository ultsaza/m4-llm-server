"""Compare memory settings in fresh processes without modifying the backend."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

SOURCE = Path(os.environ.get("MACQWEN_SOURCE", str(Path(__file__).resolve().parent / "src")))
CHECKPOINT = Path(os.environ.get("MACQWEN_ROOT", str(Path.home() / "llm" / "macqwen"))) / 'models/qwen38-flash-next-4bit'


def child(args):
    config = json.loads(args.config)
    sys.path.insert(0, str(SOURCE))
    from models.flashnext.settings.launch import apply_chat_environment

    apply_chat_environment(os.environ)
    os.environ.update(FLASHNEXT_NATIVE_MTP='off', FLASHNEXT_EXPERT_POOL_GB=str(config['pool_gb']),
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if config.get('wired_gb') is not None:
        os.environ['FLASHNEXT_WIRED_GB'] = str(config['wired_gb'])
    if 'pool_lock' in config:
        os.environ['FLASHNEXT_EXPERT_POOL_LOCK'] = '1' if config['pool_lock'] else '0'
    if 'stream_pack_chunk' in config:
        os.environ['FLASHNEXT_STREAM_PACK_CHUNK'] = str(config['stream_pack_chunk'])
    if 'io_workers' in config:
        os.environ['FLASHNEXT_IO_WORKERS'] = str(config['io_workers'])
    with tempfile.TemporaryDirectory(prefix='macqwen-perf-pins-') as private:
        pin = Path(private) / 'pins.json'
        pin.write_text('{}')
        os.environ['FLASHNEXT_PIN_CACHE'] = str(pin)
        import mlx.core as mx
        cache_limit = mx.set_cache_limit(0)
        mx.set_cache_limit(cache_limit)
        if config['cache_mib'] is not None:
            expected = config['cache_mib'] * 1024 ** 2
            mx.set_cache_limit(expected)
            assert mx.set_cache_limit(expected) == expected
            cache_limit = expected
        from macqwen.backends.flashnext import FlashNextBackend
        from models.flashnext.checkpoint_policy import resolve_resident_experts
        from models.flashnext.diskio import ReadMeter
        from models.flashnext.tests.bench import bench_production as bench
        from models.flashnext.tests.bench.bench_long_states import PROMPT

        provenance = bench.benchmark_provenance(CHECKPOINT)
        backend = FlashNextBackend(model_path=str(CHECKPOINT),
                                   resident_experts=resolve_resident_experts(None, str(CHECKPOINT)) or 8,
                                   threshold=1.0, routing_profile='exact-quality', native_mtp='off')
        raw_path = args.output_dir / args.arm_id

        def persist(row):
            raw_path.write_text(json.dumps(row, indent=2) + '\n')

        row = bench.arm(backend, args.tokens, ReadMeter(), time.perf_counter(),
                        condition={'FLASHNEXT_METAL_RUNTIME': '1'},
                        validate_metal_runtime=True, provenance=provenance,
                        prompt=PROMPT, on_raw_result=persist)
        pool = getattr(backend.store, '_flashnext_expert_pool', None)
        assert pool is not None, 'requested expert pool was not allocated'
        row['effective_pool_bytes'] = len(pool.array)
        assert abs(row['effective_pool_bytes'] / 1e9 - config['pool_gb']) < 0.01
        row['pool_hits'] = pool.hits
        row['pool_misses'] = pool.misses
        row['configuration'] = config
        row['effective_cache_limit_bytes'] = cache_limit
        from models.flashnext.loader import wired_gb

        row['effective_wired_gb'] = wired_gb()
        assert row['effective_wired_gb'] == (config.get('wired_gb') or 0)
        row['effective_environment'] = {k: v for k, v in os.environ.items() if k.startswith('FLASHNEXT_')}
        row['pool_locked'] = pool.locked
        persist(row)
        if 'pool_lock' in config:
            assert pool.locked == config['pool_lock'], 'requested pool lock did not apply'
        if 'stream_pack_chunk' in config:
            row['effective_stream_pack_chunk'] = int(os.environ['FLASHNEXT_STREAM_PACK_CHUNK'])
            assert row['effective_stream_pack_chunk'] == config['stream_pack_chunk']
        if 'io_workers' in config:
            from models.flashnext.expert_cache import _POOL

            row['effective_io_workers'] = _POOL._max_workers
            persist(row)
            assert row['effective_io_workers'] == config['io_workers'], 'requested I/O workers did not apply'
        row['thermal'] = subprocess.check_output(['pmset', '-g', 'therm'], text=True).strip()
        assert bench.runtime_source_fingerprints() == provenance['source_fingerprints'], 'runtime changed during arm'
        row['validated'] = True
        persist(row)
        print(json.dumps({k: row[k] for k in ('gen_rate', 'tail_rate', 'mb_per_token', 'token_sha256')}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=('cache', 'pool', 'wired', 'lock', 'chunk', 'workers'), default='cache')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--cache-mib', type=int, default=512)
    parser.add_argument('--wired-gb', type=float, default=12)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--arm-id')
    args = parser.parse_args()
    if args.child:
        child(args)
        return
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(SOURCE))
    from macqwen.measurement import MeasurementRun
    from models.flashnext.tests.bench import bench_production as bench
    from models.flashnext.tests.bench.bench_long_states import PROMPT

    if args.suite == 'cache':
        conditions = {'default': {'pool_gb': 6, 'cache_mib': None},
                      'cache512': {'pool_gb': 6, 'cache_mib': 512}}
    elif args.suite == 'wired':
        conditions = {'wired0': {'pool_gb': 6, 'cache_mib': args.cache_mib, 'wired_gb': 0},
                      'wired12': {'pool_gb': 6, 'cache_mib': args.cache_mib, 'wired_gb': args.wired_gb}}
    elif args.suite == 'lock':
        conditions = {'unlocked': {'pool_gb': 6, 'cache_mib': args.cache_mib, 'wired_gb': 0, 'pool_lock': False},
                      'locked': {'pool_gb': 6, 'cache_mib': args.cache_mib, 'wired_gb': 0, 'pool_lock': True}}
    elif args.suite == 'chunk':
        conditions = {'chunk2': {'pool_gb': 6, 'cache_mib': args.cache_mib, 'wired_gb': 0,
                                 'pool_lock': False, 'stream_pack_chunk': 2},
                      'chunk4': {'pool_gb': 6, 'cache_mib': args.cache_mib, 'wired_gb': 0,
                                 'pool_lock': False, 'stream_pack_chunk': 4}}
    elif args.suite == 'workers':
        conditions = {'workers8': {'pool_gb': 6, 'cache_mib': args.cache_mib, 'wired_gb': 0,
                                   'pool_lock': False, 'stream_pack_chunk': 2, 'io_workers': 8},
                      'workers16': {'pool_gb': 6, 'cache_mib': args.cache_mib, 'wired_gb': 0,
                                    'pool_lock': False, 'stream_pack_chunk': 2, 'io_workers': 16}}
    else:
        conditions = {'pool6': {'pool_gb': 6, 'cache_mib': args.cache_mib, 'wired_gb': args.wired_gb},
                      'pool8': {'pool_gb': 8, 'cache_mib': args.cache_mib, 'wired_gb': args.wired_gb}}
    metadata = {**bench.benchmark_provenance(CHECKPOINT), 'conditions': conditions,
                'prompt': PROMPT, 'token_limit': args.tokens, 'sampling': 'greedy',
                'cache_conditions': 'fresh child per arm; OS page cache retained; reversed pairs; no purge',
                'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    record = MeasurementRun(args.output_dir / 'record.jsonl', runtime='flashnext',
                            experiment=args.suite, metadata=metadata)
    record.start()
    rows, reference, effects = [], None, []
    names = list(conditions)
    try:
        for round_index in range(3):
            pair = {}
            for name in names if round_index % 2 == 0 else reversed(names):
                arm_id = f'{round_index + 1}-{name}.json'
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--child',
                           '--config', json.dumps(conditions[name]), '--arm-id', arm_id,
                           '--output-dir', str(args.output_dir), '--tokens', str(args.tokens)]
                print(f'round {round_index + 1} {name} start', flush=True)
                started = time.monotonic()
                samples = []
                with (args.output_dir / f'{arm_id}.log').open('w') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        while process.poll() is None:
                            swap = subprocess.check_output(['sysctl', 'vm.swapusage'], text=True).strip()
                            samples.append({'seconds': round(time.monotonic() - started, 2), 'swap': swap})
                            used_mib = float(swap.split('used = ')[1].split('M')[0])
                            if time.monotonic() - started > 300 or used_mib > 6144:
                                raise RuntimeError('arm exceeded latency or swap limit')
                            time.sleep(3)
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                        (args.output_dir / f'{arm_id}.resources').write_text(json.dumps(samples, indent=2) + '\n')
                raw = json.loads((args.output_dir / arm_id).read_text()) if (args.output_dir / arm_id).exists() else {}
                record.arm(arm_id=arm_id, condition=name, round_index=round_index, command=command,
                           metrics=raw, tokens=raw.get('ids'), token_digest=raw.get('token_sha256'))
                if process.returncode or not raw.get('validated'):
                    raise RuntimeError(f'child failed for {arm_id}: exit {process.returncode}')
                if raw['gen_tokens'] != args.tokens:
                    raise RuntimeError('different work: output stopped before the requested token limit')
                if reference is None:
                    reference = raw['ids']
                if raw['ids'] != reference:
                    raise RuntimeError('token mismatch: candidate rejected')
                record.validation(arm_id, 'passed', exact_tokens=True, source_frozen=True)
                raw['condition'] = name
                rows.append(raw)
                pair[name] = raw['gen_rate']
                print(f'{arm_id}: {raw["gen_rate"]:.3f} tok/s; {raw["mb_per_token"]:.1f} MB/token; '
                      f'cache {raw["mlx_cache_mb"]:.1f} MB; prefill {raw["prefill_seconds"]:.2f}s', flush=True)
            effects.append((pair[names[1]] / pair[names[0]] - 1) * 100)
        mean = statistics.mean(effects)
        band = 2 * statistics.stdev(effects) / math.sqrt(len(effects))
        summary = {'effects_percent': effects, 'mean_percent': mean, 'two_se_percent': band,
                   'all_tokens_equal': True, 'candidate_wins': all(e > 0 for e in effects) and mean > band,
                   'median_rates': {n: statistics.median(r['gen_rate'] for r in rows if r['condition'] == n)
                                    for n in names}}
        record.finish('completed', **summary)
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        record.failure(str(error), error_type=type(error).__name__)
        record.finish('completed_with_failures')
        raise


if __name__ == '__main__':
    main()
