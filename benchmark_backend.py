"""Fresh-process pool-sidecar comparison. Hardware runs belong to the controller.

Throughput is uninstrumented. Coverage records actual positioned reads and
routes separately. Preflight imports the protocol, but never constructs a model.
"""
import argparse
from contextlib import contextmanager
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time

SOURCE = Path(__file__).resolve().parent / 'src'
CHECKPOINT = Path(os.environ.get("MACQWEN_ROOT", str(Path.home() / "llm" / "macqwen"))) / 'models/qwen38-flash-next-4bit'
SIDECAR = Path(os.environ.get("MACQWEN_ROOT", str(Path.home() / "llm" / "macqwen"))) / 'cache/pool-small-sidecar-20261012'
FEATURE = 'FLASHNEXT_EXPERT_POOL_SMALL_SIDECAR'
NATIVE_FEATURE = 'FLASHNEXT_EXPERT_POOL_NATIVE_READ'
CACHE_FEATURE = 'FLASHNEXT_EXPERT_POOL_CACHE_POLICY'
FRACTION_FEATURE = 'FLASHNEXT_EXPERT_POOL_SLRU_PROTECTED_FRACTION'
SESSION_RESET_FEATURE = 'FLASHNEXT_EXPERT_POOL_SLRU_SESSION_RESET'
SESSION_REVISION = '161b1e2338817a27e02b4f7f52f5012cfd4c284f'
FRACTION_REVISION = 'e52692714b7f7cbbb8f22c7a4675c376a8b48abb'
S3FIFO_REVISION = 'f7b60e8312843ae27aee7fa3e48320cf6010ffe1'
S3FIFO_FEATURES = ('s3fifo', 'sidecar-s3fifo')
FEATURES = ('sidecar', 'native', 'combined', 'slru', 'sidecar-slru', *S3FIFO_FEATURES)
SLRU_REVISION = '4bbd79fff6d053e0d23c221a9eeb3e546ffbe088'
NATIVE_REVISION = 'e2f6f58956543c202390451b219a73365082c619'
SIDECAR_REVISION = 'cc1981e25a18c033630a8534a37ba55e0f50e8ea'
POLICY = dict(pool_gb=6, cache_mib=512, wired_gb=0, pool_lock=False,
              stream_pack_chunk=2, io_workers=8, native_mtp='off',
              routing_profile='exact-quality', threshold=1.0)
SAMPLED = dict(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5)
QUALITY = {
    'japanese': 'SSDに保存したMoEモデルを16GBのMacで推論する仕組みを、日本語の3文で説明してください。',
    'code': 'Write a Python function stable_unique(items) that removes duplicate hashable items while preserving order. Include a short example.',
}
HELPERS = ('benchmark_backend.py', 'run_backend_experiment.sh', 'backend_api.py',
           'candidate_start.py', 'start.py', 'deployment.json', 'verify_live.py')
SKIP = {'.git', '__pycache__', '.pytest_cache', 'results', 'venv', '.venv'}


def validated_fraction(value):
    try:
        fraction = float(value) if not isinstance(value, bool) else float('nan')
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('SLRU protected fraction must be finite and strictly between 0 and 1') from error
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('SLRU protected fraction must be finite and strictly between 0 and 1')
    return fraction


def feature_environment(feature, condition, sidecar, protected_fraction=0.7, reset_on_session=False):
    if feature not in FEATURES or condition not in ('off', 'on'):
        raise ValueError('unknown feature or condition')
    fraction = validated_fraction(protected_fraction)
    if fraction != 0.7 and feature not in ('slru', 'sidecar-slru'):
        raise ValueError('nondefault protected fraction requires an SLRU comparison')
    if reset_on_session and feature not in ('slru', 'sidecar-slru'):
        raise ValueError('session reset requires an SLRU comparison')
    environment = {}
    if feature in ('native', 'slru', 's3fifo') or condition == 'on':
        environment[FEATURE] = str(sidecar)
    if feature in ('native', 'combined') and condition == 'on':
        environment[NATIVE_FEATURE] = '1'
    if feature in ('slru', 'sidecar-slru') and condition == 'on':
        environment[CACHE_FEATURE] = 'slru'
        if reset_on_session:
            environment[SESSION_RESET_FEATURE] = '1'
        if fraction != 0.7:
            environment[FRACTION_FEATURE] = str(fraction)
    if feature in S3FIFO_FEATURES and condition == 'on':
        environment[CACHE_FEATURE] = 's3fifo'
    return environment


def validate_source_selection(source, revision, feature, protected_fraction=0.7, reset_on_session=False):
    fraction = validated_fraction(protected_fraction)
    if fraction != 0.7 and (revision not in (FRACTION_REVISION, SESSION_REVISION) or feature not in ('slru', 'sidecar-slru')):
        raise ValueError('nondefault protected fraction requires reviewed fraction source and SLRU comparison')
    if reset_on_session and (revision != SESSION_REVISION or feature not in ('slru', 'sidecar-slru')):
        raise ValueError('session reset requires reviewed session source and SLRU comparison')
    if feature in S3FIFO_FEATURES and revision != S3FIFO_REVISION:
        raise ValueError('S3-FIFO requires reviewed S3-FIFO source')
    native = (source / 'models/flashnext/pool_native.py').is_file()
    slru = (source / 'models/flashnext/expert_pool_policy.py').is_file()
    if revision not in (SIDECAR_REVISION, NATIVE_REVISION, SLRU_REVISION, FRACTION_REVISION, SESSION_REVISION, S3FIFO_REVISION):
        raise ValueError('unreviewed source revision')
    if native != (revision == NATIVE_REVISION) or slru != (revision in (SLRU_REVISION, FRACTION_REVISION, SESSION_REVISION, S3FIFO_REVISION)):
        raise ValueError('source and declared reviewed revision disagree about implementations')
    if revision in (FRACTION_REVISION, SESSION_REVISION) and FRACTION_FEATURE not in (source / 'models/flashnext/expert_pool_policy.py').read_text():
        raise ValueError('fraction source lacks protected fraction interface')
    s3fifo = slru and "'s3fifo'" in (source / 'models/flashnext/expert_pool_policy.py').read_text()
    if s3fifo != (revision == S3FIFO_REVISION):
        raise ValueError('source and declared revision disagree about S3-FIFO implementation')
    if revision == SESSION_REVISION and SESSION_RESET_FEATURE not in (source / 'models/flashnext/expert_pool_policy.py').read_text():
        raise ValueError('session source lacks reset interface')
    if feature in ('native', 'combined') and not native:
        raise ValueError('native and combined comparisons require reviewed native source')
    if feature in ('slru', 'sidecar-slru') and not slru:
        raise ValueError('SLRU comparisons require reviewed SLRU source')


def validate_cache_policy(store, pool, expected, require_explicit=False):
    # Reviewed old sources only implement LRU and lack this diagnostic field.
    if require_explicit and (not hasattr(store, '_flashnext_pool_cache_policy') or
                             (pool is not None and not hasattr(pool, 'cache_policy'))):
        raise RuntimeError('new source lacks actual startup/runtime cache policy diagnostics')
    startup = getattr(store, '_flashnext_pool_cache_policy', 'lru')
    runtime = getattr(pool, 'cache_policy', 'lru') if pool is not None else startup
    if startup != expected or runtime != expected:
        raise RuntimeError('actual startup/runtime cache policy mismatch')
    return runtime


def s3fifo_settings():
    slots = (int(POLICY['pool_gb'] * 1e9) - 4096) // 3072000
    small = max(1, int(0.1 * slots))
    return dict(small_fraction=0.1, minimum_small_target=1, small_target=small,
                main_target=slots - small, ghost_capacity=slots - small,
                resident_slots=slots, maximum_frequency=3, promotion_threshold=2,
                shared_capacity=True)


def validate_s3fifo_diagnostics(state, slots):
    if type(slots) is not int or slots < 1 or not isinstance(state, dict):
        raise RuntimeError('actual S3-FIFO pool geometry/diagnostics absent')
    small = max(1, int(0.1 * slots))
    for field, expected in dict(small_target=small, main_target=slots-small,
                                maximum_frequency=3, promotion_threshold=2).items():
        if type(state.get(field)) is not int or state[field] != expected:
            raise RuntimeError('actual S3-FIFO target/frequency mismatch')
    if state.get('policy') != 's3fifo':
        raise RuntimeError('actual S3-FIFO diagnostics policy mismatch')
    for queue in ('small', 'main', 'ghost'):
        count, oldest = state.get(queue + '_count'), state.get(queue + '_oldest')
        bound = slots-small if queue == 'ghost' else slots
        if (type(count) is not int or not 0 <= count <= bound or
                not isinstance(oldest, list) or len(oldest) > min(count, 8)):
            raise RuntimeError('actual S3-FIFO bounded queue diagnostics mismatch')
    # Targets select eviction paths; only the shared resident total is a hard bound.
    if state['small_count'] + state['main_count'] > slots:
        raise RuntimeError('actual S3-FIFO resident queues exceed shared slots')
    return state


def validate_cache_s3fifo(store, pool, expected):
    actual = validate_cache_policy(store, pool, expected, require_explicit=True)
    if pool is None or actual != 's3fifo':
        return None
    return validate_s3fifo_diagnostics(pool.policy_diagnostics(), pool.slots)


def validate_cache_fraction(store, pool, expected):
    startup = getattr(store, '_flashnext_pool_slru_protected_fraction', None)
    runtime = getattr(pool, 'slru_protected_fraction', None) if pool is not None else startup
    if startup != expected or runtime != expected:
        raise RuntimeError('actual startup/runtime protected fraction mismatch')
    if pool is not None:
        state = pool.policy_diagnostics()
        if (state.get('protected_fraction') != expected or
                state.get('protected_capacity') != int(expected * pool.slots)):
            raise RuntimeError('actual protected fraction/quota diagnostics mismatch')
    return runtime


def validate_cache_session_reset(store, pool, expected):
    startup = getattr(store, '_flashnext_pool_slru_session_reset', None)
    runtime = getattr(pool, 'slru_session_reset', None) if pool is not None else startup
    if type(startup) is not bool or type(runtime) is not bool or startup is not expected or runtime is not expected:
        raise RuntimeError('actual startup/runtime session reset boolean mismatch')
    if pool is not None:
        state = pool.policy_diagnostics()
        count = state.get('session_reset_count')
        if state.get('slru_session_reset') is not expected or type(count) is not int or count < 0:
            raise RuntimeError('actual session reset diagnostics mismatch')
    return runtime


def session_reset_boundary(store, pool, expected, expected_count=None):
    actual = validate_cache_session_reset(store, pool, expected)
    count = pool.policy_diagnostics()['session_reset_count'] if pool is not None else 0
    if expected_count is not None and count != expected_count:
        raise RuntimeError('actual session reset count at prefill boundary mismatch')
    return dict(slru_session_reset=actual, session_reset_count=count)


def decode_pool_deltas(pool, boundary):
    if pool is None or not all(k in boundary for k in ('hits', 'misses')):
        raise RuntimeError('decode cache evidence lacks actual prefill boundary/pool')
    result = {k: getattr(pool, k) - boundary[k] for k in ('hits', 'misses')}
    if any(v < 0 for v in result.values()):
        raise RuntimeError('decode cache counters precede prefill boundary')
    return result


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(source):
    # Include .c and native libraries, unlike upstream runtime fingerprint.
    return {p.relative_to(source).as_posix(): sha(p) for p in sorted(source.rglob('*'))
            if p.is_file() and not set(p.relative_to(source).parts) & SKIP
            and p.suffix not in {'.pyc', '.pyo'}}


def imports(args):
    sys.path.insert(0, str(args.source))
    from models.flashnext.tests.bench import bench_production as bench
    from models.flashnext.tests.bench.bench_long_states import PROMPT
    from models.flashnext import expert_pool, small_sidecar
    from macqwen.backends.flashnext import FlashNextBackend
    for module in (bench, expert_pool, small_sidecar, sys.modules[FlashNextBackend.__module__]):
        if not Path(module.__file__).resolve().is_relative_to(args.source):
            raise RuntimeError(f'import escaped requested source: {module.__name__}')
    if not hasattr(small_sidecar, 'PoolSmallSidecar'):
        raise RuntimeError('candidate is missing the reviewed pool-sidecar implementation')
    return bench, PROMPT


def provenance(args):
    bench, prompt = imports(args)
    versions = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    native = {}
    for name in ('mlx', 'mlx-metal'):
        dist = importlib.metadata.distribution(name)
        for item in dist.files or ():
            path = Path(dist.locate_file(item))
            if path.suffix in {'.so', '.dylib', '.metallib'} and path.is_file():
                native[str(path.resolve())] = sha(path)
    helpers = {name: sha(Path(__file__).with_name(name)) for name in HELPERS
               if Path(__file__).with_name(name).is_file()}
    pool_native = None
    if args.feature in ('native', 'combined'):
        from models.flashnext.pool_native import NativeReader
        pool_native = NativeReader().provenance()
    manifest = source_manifest(args.source)
    return {**bench.benchmark_provenance(args.checkpoint), 'source': str(args.source),
            'source_files': manifest,
            'source_snapshot_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
            'benchmark_dependencies': {n: h for n, h in manifest.items() if '/tests/bench/' in n},
            'native_source_and_libraries': {n: h for n, h in manifest.items()
                                           if Path(n).suffix in {'.c', '.cc', '.cpp', '.mm', '.so', '.dylib'}},
            'pool_native': pool_native, 'compared_feature': args.feature,
            'pinned_native_libraries': native, 'dependencies': dict(sorted(versions.items())),
            'python': sys.version, 'python_executable': sys.executable, 'helpers': helpers,
            'configuration': POLICY, 'prompt': prompt,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'quality_prompts': QUALITY, 'sampled_settings': SAMPLED, 'seed': args.seed,
            'token_limit': args.tokens, 'pairs': args.pairs, 'quality_pairs': args.quality_pairs, 'mode': args.mode,
            'cache_protocol': 'fresh process per arm; reversed pairs; OS cache retained; no purge',
            'reviewed_source_revision': args.source_revision,
            'sidecar_base_revision': SIDECAR_REVISION,
            'slru_reset_on_session': getattr(args, 'slru_reset_on_session', False),
            'slru_session_reset_arms': {c: SESSION_RESET_FEATURE in feature_environment(args.feature, c, args.sidecar, args.slru_protected_fraction, getattr(args, 'slru_reset_on_session', False)) for c in ('off', 'on')},
            'slru_protected_fraction': args.slru_protected_fraction,
            'slru_protected_fraction_arms': {c: float(feature_environment(args.feature, c, args.sidecar, args.slru_protected_fraction, getattr(args, 'slru_reset_on_session', False)).get(FRACTION_FEATURE, 0.7)) for c in ('off', 'on')},
            's3fifo_settings': s3fifo_settings() if args.source_revision == S3FIFO_REVISION else None,
            'cache_policy_arms': {c: feature_environment(args.feature, c, args.sidecar, args.slru_protected_fraction, getattr(args, 'slru_reset_on_session', False)).get(CACHE_FEATURE, 'lru') for c in ('off', 'on')},
            'hash_caveat': 'upstream runtime hash excludes .c and benches; full manifest and native digests above are authoritative'}


def preflight(args):
    meta = provenance(args)
    deployment = json.loads(Path(__file__).with_name('deployment.json').read_text())
    if sys.version_info[:3] != (3, 12, 15):
        raise RuntimeError('requires pinned Python 3.12.15')
    for name, expected in deployment['dependencies'].items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f'dependency mismatch: {name}')
    if sha(args.source / 'macqwen/server.py') != deployment['server_sha256']:
        raise RuntimeError('HTTP adapter differs from production')
    from models.flashnext.store import SafeTensorStore
    from models.flashnext.small_sidecar import _pool_contract
    from models.flashnext.slab_pack import get_slab_layout
    store = SafeTensorStore(str(args.checkpoint))  # Headers only; no tensor reads/maps.
    try:
        prefixes, experts, _ = _pool_contract(store, get_slab_layout(32))
        if set(prefixes) != set(range(1, 48)) or set(experts.values()) != {512}:
            raise RuntimeError('real checkpoint geometry differs from 47-layer/512-expert contract')
    finally:
        store.close()
    print(json.dumps({'preflight': 'passed', 'model_loaded': False,
                      'source_snapshot_sha256': meta['source_snapshot_sha256'],
                      'python': sys.version.split()[0], 'sidecar_exists': args.sidecar.is_dir(),
                      'bench_dependency_count': len(meta['benchmark_dependencies']),
                      'native_library_count': len(meta['pinned_native_libraries'])}), flush=True)


class ReadProbe:
    """Module-local os proxy forwarding original FD/iovecs/offset unchanged."""
    def __init__(self, original):
        self.original = original
        self.lock = threading.Lock()
        self.rows = []
        self.origins = {}

    def __getattr__(self, name):
        return getattr(self.original, name)

    @staticmethod
    def signature(info):
        return {'device': info.st_dev, 'inode': info.st_ino,
                'size': info.st_size, 'mode': info.st_mode}

    def remember(self, fd, path, flags):
        origin = {'fd': fd, 'path': str(Path(path).resolve()), 'flags': flags,
                  'fstat': self.signature(self.original.fstat(fd))}
        with self.lock:
            self.origins[fd] = origin

    def open(self, path, flags, *args, **kwargs):
        fd = self.original.open(path, flags, *args, **kwargs)
        try:
            self.remember(fd, path, flags)
        except BaseException:
            self.original.close(fd)
            raise
        return fd

    def remember_pool(self, pool):
        if pool is not None:
            for shard, fd in pool._fds.items():
                if fd not in self.origins:
                    self.remember(fd, Path(pool.store.dir) / shard,
                                  fcntl.fcntl(fd, fcntl.F_GETFL))

    def descriptor_origins(self):
        with self.lock:
            return dict(self.origins)

    def preadv(self, fd, buffers, offset):
        sizes = [len(b) for b in buffers]
        count = self.original.preadv(fd, buffers, offset)
        with self.lock:
            self.rows.append({'fd': fd, 'offset': offset, 'iovecs': sizes, 'returned': count})
        return count

    def reset(self):
        with self.lock:
            self.rows.clear()


def validate_weight_descriptors(reads, pool, probe):
    """Validate every actual FD, including legitimate duplicate lazy opens.

    The runtime cache can overwrite a concurrently opened FD; its final dict is
    not an inventory of all descriptors that read checkpoint bytes.
    """
    allowed = {str((Path(pool.store.dir) / ref.shard).resolve())
               for ref in pool.store.refs.values()}
    origins = probe.descriptor_origins()
    for fd in {read['fd'] for read in reads}:
        origin = origins.get(fd)
        if (origin is None or origin['path'] not in allowed or
                origin['flags'] & os.O_ACCMODE != os.O_RDONLY):
            raise RuntimeError('weight read FD lacks readonly checkpoint provenance')
        current = probe.signature(os.fstat(fd))
        source = probe.signature(Path(origin['path']).stat())
        if origin['fstat'] != current or current != source:
            raise RuntimeError('weight read FD no longer identifies its original checkpoint file')


def child(args):
    row = {'condition': args.condition, 'feature': args.feature, 'mode': args.mode, 'validated': False}
    raw = args.output_dir / f'{args.arm_id}.json'
    write(raw, row)
    try:
        from_source = args.source
        sys.path.insert(0, str(from_source))
        from models.flashnext.settings.launch import CHAT_ENV
        # Controlled preset, not inherited experimental FLASHNEXT overrides.
        for key in list(os.environ):
            if key.startswith('FLASHNEXT_'):
                del os.environ[key]
        os.environ.update(CHAT_ENV)
        os.environ.update(FLASHNEXT_NATIVE_MTP='off', FLASHNEXT_EXPERT_POOL_GB='6',
                          FLASHNEXT_WIRED_GB='0', FLASHNEXT_EXPERT_POOL_LOCK='0',
                          FLASHNEXT_STREAM_PACK_CHUNK='2', FLASHNEXT_IO_WORKERS='8',
                          FLASHNEXT_PROFILE_IO='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        selected = feature_environment(args.feature, args.condition, args.sidecar, args.slru_protected_fraction, getattr(args, 'slru_reset_on_session', False))
        os.environ.update(selected)
        meta = provenance(args)
        row['provenance'] = meta
        write(raw, row)
        if meta != json.loads((args.output_dir / 'parent-provenance.json').read_text()):
            raise RuntimeError('parent/child provenance mismatch')
        with tempfile.TemporaryDirectory(prefix='macqwen-backend-pins-') as private:
            pins = Path(private) / 'pins.json'
            pins.write_text('{}')
            os.environ['FLASHNEXT_PIN_CACHE'] = str(pins)
            import mlx.core as mx
            mx.set_cache_limit(512 * 1024 ** 2)
            if mx.set_cache_limit(512 * 1024 ** 2) != 512 * 1024 ** 2:
                raise RuntimeError('cache policy did not apply')
            from macqwen.backends.flashnext import FlashNextBackend
            from models.flashnext.checkpoint_policy import resolve_resident_experts
            from models.flashnext.diskio import ReadMeter
            from models.flashnext import expert_pool, small_sidecar, expert_cache
            from models.flashnext.loader import wired_gb
            from macqwen.sampling import Sampling
            bench, prompt = imports(args)
            started = time.perf_counter()
            backend = FlashNextBackend(model_path=str(args.checkpoint),
                                       resident_experts=resolve_resident_experts(None, str(args.checkpoint)) or 8,
                                       threshold=1.0, routing_profile='exact-quality', native_mtp='off')
            row['model_load_seconds'] = time.perf_counter() - started
            sidecar = getattr(backend.store, '_flashnext_pool_small_sidecar', None)
            row['sidecar'] = None if sidecar is None else {
                'directory': str(sidecar.directory), 'manifest_identity': sidecar.manifest_identity,
                'validation_seconds': sidecar.validation_seconds, 'validated_bytes': sidecar.validated_bytes,
                'layers': list(sidecar.layers), 'retained_fds': dict(sidecar._fds),
                'retained_fd_count': len(sidecar._fds)}
            row['startup_cache_policy'] = validate_cache_policy(backend.store, None, selected.get(CACHE_FEATURE, 'lru'),
                require_explicit=args.source_revision in (SLRU_REVISION, FRACTION_REVISION, SESSION_REVISION, S3FIFO_REVISION))
            if args.source_revision == S3FIFO_REVISION:
                validate_cache_s3fifo(backend.store, None, selected.get(CACHE_FEATURE, 'lru'))
            if args.source_revision in (FRACTION_REVISION, SESSION_REVISION):
                row['startup_slru_protected_fraction'] = validate_cache_fraction(backend.store, None,
                    float(selected.get(FRACTION_FEATURE, 0.7)))
            if args.source_revision == SESSION_REVISION:
                row['startup_slru_session_reset'] = validate_cache_session_reset(backend.store, None, SESSION_RESET_FEATURE in selected)
            write(raw, row)
            if (sidecar is not None) != (FEATURE in selected):
                raise RuntimeError('feature did not apply during startup')
            native_reader = getattr(backend.store, '_flashnext_pool_native_reader', None)
            if (native_reader is not None) != (NATIVE_FEATURE in selected):
                raise RuntimeError('native feature did not apply before readiness')
            if native_reader and native_reader.provenance() != meta['pool_native']:
                raise RuntimeError('runtime native artifact differs from frozen provenance')
            probes = [ReadProbe(expert_pool.os), ReadProbe(small_sidecar.os)]
            routes, boundary, latency = [], {}, {}
            original_generate = backend.generate
            pool = getattr(backend.store, '_flashnext_expert_pool', None)
            if args.mode == 'coverage':
                probes[0].remember_pool(pool)
            # Pool is normally created lazily during custom decode; intercept its
            # class assign only in coverage mode and only after model loading.
            original_assign = expert_pool.ExpertPool.assign
            original_os = expert_pool.os, small_sidecar.os
            active = False

            def assign(instance, layer, experts):
                requested = list(experts)
                assigned, misses = original_assign(instance, layer, requested)
                if active:
                    routes.append({'layer': int(layer), 'experts': [int(x) for x in requested],
                                   'slots': [[int(e), int(s)] for e, s in assigned.items()],
                                   'misses': [[int(e), int(s)] for e, s in misses]})
                return assigned, misses

            def generate(*pos, **kw):
                nonlocal active
                prefilled = kw.pop('on_prefilled', None)
                token = kw.pop('on_decode_token', None)
                began = time.perf_counter()
                def on_prefilled():
                    nonlocal active
                    if prefilled:
                        prefilled()
                    for probe in probes:
                        probe.reset()
                    if native_reader:
                        native_reader.reset_diagnostics(args.mode == 'coverage')
                    routes.clear()
                    p = getattr(backend.store, '_flashnext_expert_pool', None)
                    boundary.update(hits=p.hits if p else 0, misses=p.misses if p else 0)
                    if args.source_revision == SESSION_REVISION:
                        expected_reset = SESSION_RESET_FEATURE in selected
                        expected_count = (index if expected_reset else 0) if args.mode == 'quality' else None
                        boundary.update(session_reset_boundary(backend.store, p, expected_reset, expected_count))
                    if args.mode == 'coverage':
                        probes[0].remember_pool(p)
                        boundary.update(
                            initial_slot_of=[[int(layer), int(expert), int(slot)]
                                for (layer, expert), slot in p._slot_of.items()] if p else [],
                            initial_free_slots=list(p._free) if p else [],
                            initial_pool_capacity=p.slots if p else 0,
                            pool_created=p is not None)
                    latency['prefilled_seconds'] = time.perf_counter() - began
                    active = True
                def on_token(value, piece):
                    if piece and 'first_visible_seconds' not in latency:
                        latency['first_visible_seconds'] = time.perf_counter() - began
                    if token:
                        token(value, piece)
                return original_generate(*pos, on_prefilled=on_prefilled, on_decode_token=on_token, **kw)

            def persist(result):
                row.update(result)
                row['latency'] = latency
                if args.mode == 'coverage':
                    row['coverage'] = {'weight_reads': list(probes[0].rows),
                                       'sidecar_reads': list(probes[1].rows), 'routes': list(routes),
                                       'boundary': boundary,
                                       'native_reads': native_reader.diagnostics() if native_reader else [],
                                       'weight_fd_origins': probes[0].descriptor_origins()}
                write(raw, row)

            if args.mode == 'coverage':
                expert_pool.os, small_sidecar.os = probes
                expert_pool.ExpertPool.assign = assign
            backend.generate = generate
            try:
                if args.mode == 'quality':
                    quality = []
                    for index, (name, text_prompt) in enumerate(QUALITY.items()):
                        backend.reset()
                        backend.sampling = Sampling(**SAMPLED)
                        formatted = '<|im_start|>user\n' + text_prompt + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
                        backend.append_text(formatted)
                        latency.clear()
                        boundary.clear()
                        seed = args.seed + index
                        mx.random.seed(seed)  # Immediately before generation, both arms.
                        began = time.perf_counter()
                        text, stats = backend.generate(args.tokens)
                        ids = list(backend.tape[-stats.tokens:]) if stats.tokens else []
                        quality.append({'name': name, 'prompt': formatted, 'seed': seed,
                                        'sampling': dataclasses.asdict(backend.sampling), 'ids': ids,
                                        'text': text, 'gen_tokens': stats.tokens, 'finish': stats.finish,
                                        'gen_rate': stats.rate, 'wall': time.perf_counter() - began,
                                        'latency': dict(latency),
                                        'decode_pool_boundary': dict(boundary),
                                        'decode_pool': decode_pool_deltas(getattr(backend.store, '_flashnext_expert_pool', None), boundary)})
                        persist({'quality': quality})
                    row['metal_runtime'] = bench.inspect_metal_runtime(backend, True)
                else:
                    result = bench.arm(backend, args.tokens, ReadMeter(), time.perf_counter(),
                                       condition={'FLASHNEXT_METAL_RUNTIME': '1'},
                                       validate_metal_runtime=True, on_raw_result=persist,
                                       provenance=bench.benchmark_provenance(args.checkpoint), prompt=prompt)
                    persist(result)
            finally:
                backend.generate = original_generate
                expert_pool.os, small_sidecar.os = original_os
                expert_pool.ExpertPool.assign = original_assign
            pool = getattr(backend.store, '_flashnext_expert_pool', None)
            row['runtime_cache_policy'] = validate_cache_policy(backend.store, pool, selected.get(CACHE_FEATURE, 'lru'),
                require_explicit=args.source_revision in (SLRU_REVISION, FRACTION_REVISION, SESSION_REVISION, S3FIFO_REVISION))
            if args.source_revision == S3FIFO_REVISION:
                row['runtime_s3fifo'] = validate_cache_s3fifo(backend.store, pool, selected.get(CACHE_FEATURE, 'lru'))
            if args.source_revision in (FRACTION_REVISION, SESSION_REVISION):
                row['runtime_slru_protected_fraction'] = validate_cache_fraction(backend.store, pool,
                    float(selected.get(FRACTION_FEATURE, 0.7)))
            if args.source_revision == SESSION_REVISION:
                row['runtime_slru_session_reset'] = validate_cache_session_reset(backend.store, pool, SESSION_RESET_FEATURE in selected)
            row['cache_policy_diagnostics'] = pool.policy_diagnostics() if pool and hasattr(pool, 'policy_diagnostics') else None
            row.update(effective_pool_bytes=len(pool.array) if pool else None,
                       pool_locked=pool.locked if pool else None,
                       pool_hits=pool.hits if pool else None, pool_misses=pool.misses if pool else None,
                       effective_cache_limit_bytes=mx.set_cache_limit(512 * 1024 ** 2),
                       effective_wired_gb=wired_gb(), effective_io_workers=expert_cache._POOL._max_workers,
                       sampling=dataclasses.asdict(backend.sampling),
                       effective_environment={k: v for k, v in os.environ.items() if k.startswith('FLASHNEXT_')},
                       routing_profile=backend.routing_profile, threshold=backend.threshold,
                       native_mtp=backend.native_mtp)
            write(raw, row)
            if not pool or abs(len(pool.array) / 1e9 - 6) >= .01 or pool.locked:
                raise RuntimeError('pool policy mismatch')
            if getattr(pool, 'native_reader', None) is not native_reader:
                raise RuntimeError('runtime did not reuse startup native reader')
            if pool.sidecar is not sidecar:
                raise RuntimeError('runtime did not reuse startup-validated sidecar')
            if sidecar and (set(sidecar.layers) != set(range(1, 48)) or len(sidecar._fds) != 47):
                raise RuntimeError('sidecar layer/FD coverage mismatch')
            if row['effective_cache_limit_bytes'] != 512 * 1024 ** 2 or wired_gb() != 0 or expert_cache._POOL._max_workers != 8:
                raise RuntimeError('memory/worker policy mismatch')
            if backend.routing_profile != 'exact-quality' or backend.threshold != 1.0 or backend.native_mtp != 'off':
                raise RuntimeError('routing/MTP policy mismatch')
            if args.mode == 'coverage':
                misses = pool.misses - boundary['misses']
                weights, small = probes[0].rows, probes[1].rows
                if native_reader:
                    native_rows = native_reader.diagnostics()
                    if weights or small:
                        raise RuntimeError('native arm unexpectedly used Python positioned reads')
                    origins = probes[0].descriptor_origins()
                    weights = [r for r in native_rows if r['fd'] in origins]
                    small = [r for r in native_rows if sidecar and r['fd'] in sidecar._fds.values()]
                    if len(weights) + len(small) != len(native_rows):
                        raise RuntimeError('native reads escaped validated descriptors')
                expected = {819200: 3 * misses}
                if sidecar is None:
                    expected[102400] = 6 * misses
                observed = {}
                for read in weights:
                    observed[read['returned']] = observed.get(read['returned'], 0) + 1
                    if read['iovecs'] != [read['returned']]:
                        raise RuntimeError('weight/original metadata iovec mismatch')
                if observed != expected or len(small) != (misses if sidecar else 0):
                    raise RuntimeError('positioned-read count mismatch')
                validate_weight_descriptors(weights, pool, probes[0])
                if sidecar and any(r['fd'] not in sidecar._fds.values() for r in small):
                    raise RuntimeError('metadata reads escaped validated sidecar descriptors')
                if any(r['returned'] != 614400 or r['iovecs'] != [204800] * 3 for r in small):
                    raise RuntimeError('sidecar scatter mismatch')
                if sum(r['returned'] for r in weights + small) != 3072000 * misses:
                    raise RuntimeError('positioned-read bytes mismatch')
                if sum(len(r['misses']) for r in routes) != misses or set(r['layer'] for r in routes) != set(range(1, 48)):
                    raise RuntimeError('route coverage mismatch')
                row['coverage'].update(decode_misses=misses, actual_reads=len(weights) + len(small),
                                       actual_bytes=sum(r['returned'] for r in weights + small),
                                       reads_per_miss=4 if sidecar else 9, bytes_per_miss=3072000)
            if args.mode != 'quality' and (row['gen_tokens'] != args.tokens or not backend.sampling.greedy):
                raise RuntimeError('throughput/coverage requires full greedy token work')
            if args.mode == 'quality' and any(not r['ids'] or r['sampling'] != dataclasses.asdict(Sampling(**SAMPLED)) for r in row['quality']):
                raise RuntimeError('quality work/sampler mismatch')
            row['imported_source_paths'] = {n: str(Path(m.__file__).resolve())
                for n, m in sys.modules.copy().items()
                if (n == 'macqwen' or n.startswith(('macqwen.', 'models.flashnext.')))
                and getattr(m, '__file__', None)}
            write(raw, row)
            if any(not Path(p).is_relative_to(args.source) for p in row['imported_source_paths'].values()):
                raise RuntimeError('runtime import escaped requested source')
            if provenance(args) != meta:
                raise RuntimeError('source/dependencies/helpers changed during arm')
            row['validated'] = True
            write(raw, row)
    except BaseException as error:
        row['failure'] = {'type': type(error).__name__, 'message': str(error)}
        write(raw, row)
        raise


@contextmanager
def defer_signals():
    """Finish assigning a newly created child before handled signals unwind."""
    received = []
    previous = {sig: signal.getsignal(sig) for sig in
                (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    def remember(signum, frame):
        received.append((signum, frame))
    try:
        for sig, handler in previous.items():
            if handler != signal.SIG_IGN:
                signal.signal(sig, remember)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if received:
            signum, frame = received[0]
            handler = previous[signum]
            if callable(handler):
                handler(signum, frame)
            else:
                raise SystemExit(128 + signum)


def run_child(args, arm_id, condition):
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--child',
               '--mode', args.mode, '--feature', args.feature, '--source-revision', args.source_revision, '--source', str(args.source), '--checkpoint', str(args.checkpoint),
               '--slru-protected-fraction', str(args.slru_protected_fraction),
               '--sidecar', str(args.sidecar), '--output-dir', str(args.output_dir),
               '--tokens', str(args.tokens), '--pairs', str(args.pairs),
               '--quality-pairs', str(args.quality_pairs), '--seed', str(args.seed),
               '--arm-id', arm_id, '--condition', condition]
    if getattr(args, 'slru_reset_on_session', False):
        command += ['--slru-reset-on-session']
    samples, process, footprint_thread = [], None, None
    footprint_rows, stop_footprint = [], threading.Event()
    began = time.monotonic()
    failure = None
    with (args.output_dir / f'{arm_id}.log').open('w') as log:
        def footprints():
            while not stop_footprint.is_set() and process.poll() is None:
                try:
                    result = subprocess.run(['/usr/bin/footprint', '-p', str(process.pid)],
                                            capture_output=True, text=True, timeout=20)
                    footprint_rows.append({'seconds': time.monotonic() - began,
                                           'stdout': result.stdout, 'stderr': result.stderr,
                                           'returncode': result.returncode})
                except (OSError, subprocess.SubprocessError) as error:
                    footprint_rows.append({'seconds': time.monotonic() - began,
                                           'failure': str(error), 'type': type(error).__name__})
                stop_footprint.wait(15)
        try:
            # Defer handled signals through assignment, without passing a
            # blocked OS signal mask to the new child.
            with defer_signals():
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            footprint_thread = threading.Thread(target=footprints, daemon=True)
            footprint_thread.start()
            while process.poll() is None:
                elapsed = time.monotonic() - began
                sample = {'seconds': elapsed, 'pid': process.pid}
                swap = subprocess.run(['sysctl', 'vm.swapusage'], capture_output=True, text=True, timeout=10)
                sample['global_swap'] = {'stdout': swap.stdout, 'stderr': swap.stderr, 'returncode': swap.returncode}
                samples.append(sample)
                write(args.output_dir / f'{arm_id}.resources.json', samples)
                if elapsed > args.timeout:
                    raise RuntimeError('arm exceeded controller timeout')
                if swap.returncode == 0 and float(swap.stdout.split('used = ')[1].split('M')[0]) > args.max_swap_mib:
                    raise RuntimeError('global swap exceeded guard')
                time.sleep(3)
        except BaseException as error:
            failure = {'type': type(error).__name__, 'message': str(error)}
        finally:
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK,
                {signal.SIGTERM, signal.SIGHUP, signal.SIGINT})
            try:
                stop_footprint.set()
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    else:
                        process.wait()
                if footprint_thread is not None and footprint_thread.ident is not None:
                    footprint_thread.join(25)
                write(args.output_dir / f'{arm_id}.resources.json', samples)
                write(args.output_dir / f'{arm_id}.footprint.json', footprint_rows)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    path = args.output_dir / f'{arm_id}.json'
    raw = json.loads(path.read_text()) if path.exists() else {}
    outcome = {'command': command, 'returncode': process.returncode if process is not None else None, 'failure': failure, 'raw': raw}
    write(args.output_dir / f'{arm_id}.outcome.json', outcome)
    return outcome


def parent(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    meta = provenance(args)
    write(args.output_dir / 'parent-provenance.json', meta)
    if meta['pool_native']:
        artifact = Path(meta['pool_native']['library'])
        frozen = args.output_dir / artifact.name
        shutil.copy2(artifact, frozen)
        if sha(frozen) != meta['pool_native']['library_sha256']:
            raise RuntimeError('native library changed during freeze')

    with tarfile.open(args.output_dir / 'source-snapshot.tar.gz', 'w:gz') as archive:
        for name in meta['source_files']:
            archive.add(args.source / name, arcname=name, recursive=False)
    with tarfile.open(args.output_dir / 'source-snapshot.tar.gz') as archive:
        for name, expected in meta['source_files'].items():
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != expected:
                raise RuntimeError('source archive changed during freeze')
    write(args.output_dir / 'source-archive.json',
          {'sha256': sha(args.output_dir / 'source-snapshot.tar.gz'),
           'manifest_sha256': meta['source_snapshot_sha256']})
    for name in meta['helpers']:
        shutil.copy2(Path(__file__).with_name(name), args.output_dir / name)
    from macqwen.measurement import MeasurementRun
    record = MeasurementRun(args.output_dir / 'record.jsonl', runtime='flashnext',
                            experiment='pool-' + args.feature + '-' + args.mode, metadata=meta)
    record.start()
    pairs, rows, quality_pairs = [], [], []
    try:
        rounds = args.pairs if args.mode == 'throughput' else args.quality_pairs if args.mode == 'quality' else 1
        reference = quality_work = None
        for index in range(rounds):
            pair = {}
            for condition in ('off', 'on') if index % 2 == 0 else ('on', 'off'):
                arm_id = f'{index + 1}-{condition}'
                print(f'{arm_id}: fresh child start ({args.mode})', flush=True)
                outcome = run_child(args, arm_id, condition)
                raw = outcome['raw']
                record.arm(arm_id=arm_id, condition=condition, round_index=index,
                           command=outcome['command'], metrics=raw, tokens=raw.get('ids'),
                           token_digest=raw.get('token_sha256'))
                if outcome['returncode'] or outcome['failure'] or not raw.get('validated'):
                    raise RuntimeError(f'child failed for {arm_id}; see raw, log, outcome, resources')
                if raw['provenance'] != meta or provenance(args) != meta:
                    raise RuntimeError('source/dependencies changed during run')
                ids = [r['ids'] for r in raw['quality']] if args.mode == 'quality' else raw['ids']
                if reference is None:
                    reference = ids
                if ids != reference:
                    raise RuntimeError('actual generated token IDs differ')
                if args.mode != 'quality' and raw['gen_tokens'] != args.tokens:
                    raise RuntimeError('different token work')
                if args.mode == 'quality':
                    if ([r['name'] for r in raw['quality']] != list(QUALITY) or
                            any(not 0 < r['gen_tokens'] <= args.tokens or len(r['ids']) != r['gen_tokens']
                                for r in raw['quality'])):
                        raise RuntimeError('quality cases/token work mismatch')
                    if args.source_revision == SESSION_REVISION:
                        expected_reset = meta['slru_session_reset_arms'][condition]
                        for case_index, case in enumerate(raw['quality']):
                            boundary = case.get('decode_pool_boundary', {})
                            count = boundary.get('session_reset_count')
                            if (boundary.get('slru_session_reset') is not expected_reset or
                                    type(count) is not int or count != (case_index if expected_reset else 0)):
                                raise RuntimeError('quality lacks actual session reset prefill boundary evidence')
                    work = [(r['name'], r['text'], r['prompt'], r['seed'], r['sampling'], r['gen_tokens'], r['finish'])
                            for r in raw['quality']]
                    if quality_work is None:
                        quality_work = work
                    if work != quality_work:
                        raise RuntimeError('quality text/settings/work/finish differs')
                record.validation(arm_id, 'passed', exact_tokens=True, source_frozen=True)
                pair[condition] = raw
                rows.append(raw)
            if args.mode == 'throughput':
                pairs.append((pair['on']['gen_rate'] / pair['off']['gen_rate'] - 1) * 100)
            elif args.mode == 'quality':
                quality_pairs.append(pair)
        summary = {'mode': args.mode, 'feature': args.feature, 'all_tokens_equal': True, 'pairs': rounds,
                   'screening': args.mode == 'throughput' and (args.pairs != 6 or args.tokens != 128),
                   'operational_promotion': 'requires separate warm/long/API/resource controller evaluation'}
        if pairs:
            mean = statistics.mean(pairs)
            band = 2 * statistics.stdev(pairs) / math.sqrt(len(pairs)) if len(pairs) > 1 else None
            summary.update(effects_percent=pairs, mean_percent=mean, two_se_percent=band,
                           all_pairs_win=all(x > 0 for x in pairs),
                           decode_gate=(not summary['screening'] and mean >= 5 and band is not None
                                        and mean > band and all(x > 0 for x in pairs)))
        if quality_pairs:
            cases = {}
            for index, name in enumerate(QUALITY):
                rates, pools, effects, deltas = [], [], [], []
                for pair in quality_pairs:
                    rates.append({c: pair[c]['quality'][index]['gen_rate'] for c in ('off', 'on')})
                    pools.append({c: pair[c]['quality'][index]['decode_pool'] for c in ('off', 'on')})
                    effects.append((rates[-1]['on'] / rates[-1]['off'] - 1) * 100)
                    deltas.append({k: pools[-1]['on'][k] - pools[-1]['off'][k] for k in ('hits', 'misses')})
                cases[name] = {'paired_rates': rates, 'effects_percent': effects,
                               'mean_rates': {c: statistics.mean(r[c] for r in rates) for c in ('off', 'on')},
                               'mean_percent': statistics.mean(effects),
                               'two_se_percent': 2 * statistics.stdev(effects) / math.sqrt(rounds) if rounds > 1 else None,
                               'paired_decode_pool': pools, 'decode_pool_deltas': deltas,
                               'mean_decode_pool_delta': {k: statistics.mean(d[k] for d in deltas) for k in ('hits', 'misses')}}
            summary['quality_cases'] = cases
            summary['quality_gate_scope'] = 'paired sampled quality timing only; standard decode and operational gates remain independent'
        write(args.output_dir / 'summary.json', summary)
        record.finish('completed', **summary)
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        record.failure(str(error), error_type=type(error).__name__)
        record.finish('completed_with_failures')
        write(args.output_dir / 'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise


def interrupted(signum, _frame):
    raise SystemExit(128 + signum)


def main():
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature', choices=FEATURES, default='sidecar')
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--source-revision', choices=(SIDECAR_REVISION, NATIVE_REVISION, SLRU_REVISION, FRACTION_REVISION, SESSION_REVISION, S3FIFO_REVISION), default=NATIVE_REVISION)
    parser.add_argument('--slru-protected-fraction', type=validated_fraction, default=0.7)
    parser.add_argument('--slru-reset-on-session', action='store_true')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--sidecar', type=Path, default=SIDECAR)
    parser.add_argument('--mode', choices=('throughput', 'coverage', 'quality', 'preflight'), default='throughput')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--pairs', type=int, default=6, help='other counts are screening only')
    parser.add_argument('--quality-pairs', type=int, default=1, help='fresh reversed quality pairs; nondefault requires quality mode')
    parser.add_argument('--tokens', type=int, default=128, help='throughput gate requires 128')
    parser.add_argument('--seed', type=int, default=20261012)
    parser.add_argument('--timeout', type=float, default=1200)
    parser.add_argument('--max-swap-mib', type=float, default=6144)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--arm-id', help=argparse.SUPPRESS)
    parser.add_argument('--condition', choices=('off', 'on'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.source, args.checkpoint, args.sidecar = (p.expanduser().resolve() for p in (args.source, args.checkpoint, args.sidecar))
    try:
        validate_source_selection(args.source, args.source_revision, args.feature, args.slru_protected_fraction, getattr(args, 'slru_reset_on_session', False))
    except ValueError as error:
        parser.error(str(error))
    if args.pairs < 1 or args.tokens < 1:
        parser.error('pairs and tokens must be positive')
    if args.quality_pairs < 1:
        parser.error('quality pairs must be positive')
    if args.quality_pairs != 1 and args.mode != 'quality':
        parser.error('nondefault quality pairs require quality mode')
    if args.mode == 'preflight':
        preflight(args)
        return
    if args.output_dir is None:
        parser.error('--output-dir is required for hardware modes')
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.child:
        if not args.arm_id or not args.condition:
            parser.error('child requires arm-id and condition')
        child(args)
    else:
        parent(args)


if __name__ == '__main__':
    main()
