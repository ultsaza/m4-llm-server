"""Opt-in cache arms and measured cache evidence use real helper interfaces."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import benchmark_backend as bench


class SLRUHelpers(unittest.TestCase):
    def test_feature_arms_hold_native_off_and_control_sidecar_independently(self):
        side = 'FLASHNEXT_EXPERT_POOL_SMALL_SIDECAR'
        policy = 'FLASHNEXT_EXPERT_POOL_CACHE_POLICY'
        expected = {
            ('slru', 'off'): {side: '/side'},
            ('slru', 'on'): {side: '/side', policy: 'slru'},
            ('sidecar-slru', 'off'): {},
            ('sidecar-slru', 'on'): {side: '/side', policy: 'slru'},
        }
        for arms, env in expected.items():
            with self.subTest(arms=arms):
                self.assertEqual(bench.feature_environment(*arms, Path('/side')), env)

    def test_launcher_new_modes_clear_inherited_policy_and_native(self):
        with tempfile.TemporaryDirectory() as temporary:
            release = Path(temporary)
            for name in ('candidate_start.py', 'benchmark_backend.py'):
                shutil.copy2(ROOT / name, release / name)
            (release / 'deployment.json').write_text(json.dumps({'memory': dict(expert_pool_gb=6, mlx_cache_mib=512, wired_gb=0)}))
            (release / 'start.py').write_text('import json, os\ndef main():\n print(json.dumps(dict(os.environ)))\n')
            for feature, condition in (('slru', 'off'), ('slru', 'on'), ('sidecar-slru', 'off'), ('sidecar-slru', 'on')):
                result = subprocess.run([sys.executable, str(release / 'candidate_start.py'), '--feature', feature,
                    '--condition', condition, '--sidecar', '/side', '--check-only'], capture_output=True, text=True,
                    env=dict(os.environ, FLASHNEXT_EXPERT_POOL_NATIVE_READ='1', FLASHNEXT_EXPERT_POOL_CACHE_POLICY='typo'))
                self.assertEqual(result.returncode, 0, result.stderr)
                actual = {k: v for k, v in json.loads(result.stdout).items() if k in (
                    'FLASHNEXT_EXPERT_POOL_SMALL_SIDECAR', 'FLASHNEXT_EXPERT_POOL_NATIVE_READ', 'FLASHNEXT_EXPERT_POOL_CACHE_POLICY')}
                self.assertEqual(actual, bench.feature_environment(feature, condition, Path('/side')))

    def test_decode_cache_deltas_exclude_prefill_and_reject_missing_boundary(self):
        self.assertTrue(hasattr(bench, 'decode_pool_deltas'), 'actual boundary deltas are missing')
        pool = SimpleNamespace(hits=130, misses=42)
        self.assertEqual(bench.decode_pool_deltas(pool, {'hits': 100, 'misses': 40}), {'hits': 30, 'misses': 2})
        with self.assertRaises(RuntimeError):
            bench.decode_pool_deltas(pool, {})
        with self.assertRaises(RuntimeError):
            bench.decode_pool_deltas(pool, {'hits': 131, 'misses': 40})

    def test_cache_policy_assertion_uses_actual_startup_and_runtime_values(self):
        self.assertTrue(hasattr(bench, 'validate_cache_policy'), 'startup/runtime policy assertion is missing')
        for selected in ('lru', 'slru'):
            store = SimpleNamespace(_flashnext_pool_cache_policy=selected)
            pool = SimpleNamespace(cache_policy=selected)
            self.assertEqual(bench.validate_cache_policy(store, pool, selected), selected)
            pool.cache_policy = 'other'
            with self.assertRaises(RuntimeError):
                bench.validate_cache_policy(store, pool, selected)
        with self.assertRaises(RuntimeError):
            bench.validate_cache_policy(SimpleNamespace(), None, 'slru')
        self.assertEqual(bench.validate_cache_policy(SimpleNamespace(), None, 'lru'), 'lru')

    def test_new_source_requires_explicit_policy_diagnostics_even_for_lru(self):
        with self.assertRaises(RuntimeError):
            bench.validate_cache_policy(SimpleNamespace(), None, 'lru', require_explicit=True)
        store = SimpleNamespace(_flashnext_pool_cache_policy='lru')
        with self.assertRaises(RuntimeError):
            bench.validate_cache_policy(store, SimpleNamespace(), 'lru', require_explicit=True)
        self.assertEqual(bench.validate_cache_policy(store, SimpleNamespace(cache_policy='lru'),
            'lru', require_explicit=True), 'lru')

    def test_explicit_source_selector_preserves_old_sources_and_requires_slru_marker(self):
        self.assertTrue(hasattr(bench, 'validate_source_selection'), 'source selector is missing')
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            policy = source / 'models/flashnext/expert_pool_policy.py'
            policy.parent.mkdir(parents=True)
            for revision in (bench.SIDECAR_REVISION, bench.NATIVE_REVISION):
                native = source / 'models/flashnext/pool_native.py'
                if revision == bench.NATIVE_REVISION:
                    native.touch()
                self.assertIsNone(bench.validate_source_selection(source, revision, 'sidecar'))
            native.unlink()
            with self.assertRaises(ValueError):
                bench.validate_source_selection(source, bench.SLRU_REVISION, 'slru')
            policy.touch()
            for mode in ('slru', 'sidecar-slru', 'sidecar'):
                self.assertIsNone(bench.validate_source_selection(source, bench.SLRU_REVISION, mode))
            with self.assertRaises(ValueError):
                bench.validate_source_selection(source, bench.SLRU_REVISION, 'combined')


class FractionHelpers(unittest.TestCase):
    def test_half_fraction_only_reaches_slru_on_arm_and_preserves_old_defaults(self):
        for feature in ('slru', 'sidecar-slru'):
            for condition in ('off', 'on'):
                env = bench.feature_environment(feature, condition, Path('/side'), protected_fraction=0.5)
                if condition == 'on':
                    self.assertEqual(env['FLASHNEXT_EXPERT_POOL_SLRU_PROTECTED_FRACTION'], '0.5')
                else:
                    self.assertNotIn('FLASHNEXT_EXPERT_POOL_SLRU_PROTECTED_FRACTION', env)
                self.assertNotIn('FLASHNEXT_EXPERT_POOL_SLRU_PROTECTED_FRACTION',
                                 bench.feature_environment(feature, condition, Path('/side')))
        for value in (0, 1, float('nan'), float('inf'), 'bad', True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bench.feature_environment('slru', 'on', Path('/side'), protected_fraction=value)
        with self.assertRaises(ValueError):
            bench.feature_environment('sidecar', 'on', Path('/side'), protected_fraction=0.5)

    def test_nondefault_fraction_requires_new_reviewed_source_and_fraction_presence(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            file = source / 'models/flashnext/expert_pool_policy.py'
            file.parent.mkdir(parents=True)
            file.write_text('')
            for revision in (bench.SIDECAR_REVISION, bench.NATIVE_REVISION, bench.SLRU_REVISION):
                with self.subTest(revision=revision), self.assertRaises(ValueError):
                    bench.validate_source_selection(source, revision, 'slru', protected_fraction=0.5)
            with self.assertRaises(ValueError):
                bench.validate_source_selection(source, bench.FRACTION_REVISION, 'slru', protected_fraction=0.5)
            file.write_text("FRACTION_FLAG = 'FLASHNEXT_EXPERT_POOL_SLRU_PROTECTED_FRACTION'\n")
            for mode in ('slru', 'sidecar-slru'):
                self.assertIsNone(bench.validate_source_selection(source, bench.FRACTION_REVISION, mode, protected_fraction=0.5))
            with self.assertRaises(ValueError):
                bench.validate_source_selection(source, bench.FRACTION_REVISION, 'sidecar', protected_fraction=0.5)

    def test_actual_fraction_assertion_rejects_missing_mismatched_and_wrong_quota(self):
        store = SimpleNamespace(_flashnext_pool_slru_protected_fraction=0.5)
        pool = SimpleNamespace(slru_protected_fraction=0.5, slots=5,
            policy_diagnostics=lambda: dict(protected_fraction=0.5, protected_capacity=2))
        self.assertEqual(bench.validate_cache_fraction(store, None, 0.5), 0.5)
        self.assertEqual(bench.validate_cache_fraction(store, pool, 0.5), 0.5)
        with self.assertRaises(RuntimeError):
            bench.validate_cache_fraction(SimpleNamespace(), pool, 0.5)
        pool.slru_protected_fraction = 0.7
        with self.assertRaises(RuntimeError):
            bench.validate_cache_fraction(store, pool, 0.5)
        pool.slru_protected_fraction = 0.5
        pool.policy_diagnostics = lambda: dict(protected_fraction=0.5, protected_capacity=3)
        with self.assertRaises(RuntimeError):
            bench.validate_cache_fraction(store, pool, 0.5)


    def test_candidate_fraction_requires_reviewed_pin_and_clears_off_arm_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            release = Path(temporary)
            for name in ('candidate_start.py', 'benchmark_backend.py'):
                shutil.copy2(ROOT / name, release / name)
            (release / 'deployment.json').write_text(json.dumps({'memory': dict(expert_pool_gb=6, mlx_cache_mib=512, wired_gb=0)}))
            policy = release / 'src/models/flashnext/expert_pool_policy.py'
            policy.parent.mkdir(parents=True)
            policy.write_text("FRACTION_FLAG = 'FLASHNEXT_EXPERT_POOL_SLRU_PROTECTED_FRACTION'\n")
            (release / 'start.py').write_text('import json, os\ndef main():\n print(json.dumps(dict(os.environ)))\n')
            command = [sys.executable, str(release / 'candidate_start.py'), '--feature', 'slru',
                       '--sidecar', '/side', '--check-only', '--slru-protected-fraction', '0.5']
            denied = subprocess.run(command + ['--condition', 'on'], capture_output=True, text=True)
            self.assertNotEqual(denied.returncode, 0)
            for condition in ('off', 'on'):
                result = subprocess.run(command + ['--condition', condition, '--source-revision', bench.FRACTION_REVISION],
                    capture_output=True, text=True, env=dict(os.environ,
                    FLASHNEXT_EXPERT_POOL_SLRU_PROTECTED_FRACTION='0.9'))
                self.assertEqual(result.returncode, 0, result.stderr)
                actual = json.loads(result.stdout)
                if condition == 'on':
                    self.assertEqual(actual['FLASHNEXT_EXPERT_POOL_SLRU_PROTECTED_FRACTION'], '0.5')
                else:
                    self.assertNotIn('FLASHNEXT_EXPERT_POOL_SLRU_PROTECTED_FRACTION', actual)

    def test_candidate_validation_events_use_actual_startup_and_runtime_once(self):
        import candidate_start
        from contextlib import redirect_stdout
        import io
        class Pool:
            def __init__(self, store):
                self.store = store
                self.cache_policy = store._flashnext_pool_cache_policy
                self.slru_protected_fraction = store._flashnext_pool_slru_protected_fraction
                self.slots = 5
            def close(self):
                pass
            def policy_diagnostics(self):
                return dict(protected_fraction=self.slru_protected_fraction, protected_capacity=2)
        def initialize(store):
            store._flashnext_pool_cache_policy = 'slru'
            store._flashnext_pool_slru_protected_fraction = 0.5
        module = SimpleNamespace(initialize_sidecar=initialize, ExpertPool=Pool)
        output = io.StringIO()
        with redirect_stdout(output):
            candidate_start.install_cache_validation(module, 'slru', 0.5)
            store = SimpleNamespace()
            module.initialize_sidecar(store)
            module.ExpertPool(store)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([row['phase'] for row in events], ['startup', 'runtime'])
        self.assertEqual([row['protected_fraction'] for row in events], [0.5, 0.5])
        store._flashnext_pool_slru_protected_fraction = 0.7
        with self.assertRaises(RuntimeError):
            module.ExpertPool(store)

    def test_api_forwards_fraction_and_requires_both_actual_candidate_events(self):
        import backend_api
        args = SimpleNamespace(condition='on', feature='slru', sidecar=Path('/side'),
            slru_protected_fraction=0.5, source_revision=bench.FRACTION_REVISION)
        command = backend_api.candidate_command(args, Path('/release'))
        self.assertEqual(command[command.index('--slru-protected-fraction') + 1], '0.5')
        self.assertEqual(command[command.index('--source-revision') + 1], bench.FRACTION_REVISION)
        events = [dict(event='candidate_cache_validation', phase=p, policy='slru', protected_fraction=0.5)
                  for p in ('startup', 'runtime')]
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / 'candidate.log'
            log.write_text(''.join(json.dumps(row) + '\n' for row in events))
            self.assertEqual(backend_api.candidate_cache_events(log, args, require_runtime=True), events)
            log.write_text(json.dumps(events[0]) + '\n')
            with self.assertRaises(RuntimeError):
                backend_api.candidate_cache_events(log, args, require_runtime=True)
            events[1]['protected_fraction'] = 0.7
            log.write_text(''.join(json.dumps(row) + '\n' for row in events))
            with self.assertRaises(RuntimeError):
                backend_api.candidate_cache_events(log, args, require_runtime=True)


    def test_parent_provenance_freezes_fraction_and_child_command_preserves_it(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            args = SimpleNamespace(source=source, checkpoint=source, feature='slru',
                sidecar=Path('/side'), slru_protected_fraction=0.5, source_revision=bench.FRACTION_REVISION,
                seed=20261012, tokens=128, pairs=6, quality_pairs=1, mode='throughput', output_dir=source)
            production_bench = SimpleNamespace(benchmark_provenance=lambda checkpoint: {})
            with mock.patch.object(bench, 'imports', return_value=(production_bench, 'prompt')), \
                    mock.patch.object(bench.importlib.metadata, 'distributions', return_value=[]), \
                    mock.patch.object(bench.importlib.metadata, 'distribution', return_value=SimpleNamespace(files=[])):
                meta = bench.provenance(args)
            self.assertEqual(meta['slru_protected_fraction'], 0.5)
            self.assertEqual(meta['slru_protected_fraction_arms'], {'off': 0.7, 'on': 0.5})
            self.assertEqual(meta['cache_policy_arms'], {'off': 'lru', 'on': 'slru'})
            self.assertEqual(meta['configuration']['pool_gb'], 6)
            child = SimpleNamespace(returncode=0, pid=999, poll=lambda: 0, wait=lambda: 0)
            with mock.patch.object(bench.subprocess, 'Popen', return_value=child):
                outcome = bench.run_child(args, 'fraction', 'off')
            command = outcome['command']
            self.assertEqual(command[command.index('--slru-protected-fraction') + 1], '0.5')
            self.assertEqual(command[command.index('--condition') + 1], 'off')
            self.assertEqual(command[command.index('--source-revision') + 1], bench.FRACTION_REVISION)
