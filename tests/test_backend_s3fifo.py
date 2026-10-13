"""S3-FIFO selectors and evidence validate actual policy and shared capacity."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import benchmark_backend as b

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.runtime/variants/s3fifo'
REVISION = 'f7b60e8312843ae27aee7fa3e48320cf6010ffe1'


class S3FIFOHelpers(unittest.TestCase):
    def state(self, slots=16):
        return dict(policy='s3fifo', small_target=max(1, int(.1 * slots)),
            main_target=slots-max(1, int(.1 * slots)), small_count=16, main_count=0,
            ghost_count=2, maximum_frequency=3, promotion_threshold=2,
            small_oldest=[(1, i) for i in range(8)], main_oldest=[], ghost_oldest=[(1, 20), (1, 21)])

    def test_feature_arms_and_rejected_fraction_session_requests(self):
        for feature in ('s3fifo', 'sidecar-s3fifo'):
            self.assertEqual(b.feature_environment(feature, 'on', Path('/side')), {b.FEATURE: '/side', b.CACHE_FEATURE: 's3fifo'})
            self.assertEqual(b.feature_environment(feature, 'off', Path('/side')), {b.FEATURE: '/side'} if feature == 's3fifo' else {})
            for kwargs in (dict(protected_fraction=.5), dict(reset_on_session=True)):
                with self.assertRaises(ValueError): b.feature_environment(feature, 'on', Path('/side'), **kwargs)

    def test_selector_requires_new_engine_pin_and_preserves_old_contracts(self):
        self.assertEqual(getattr(b, 'S3FIFO_REVISION', None), REVISION)
        for feature in ('s3fifo', 'sidecar-s3fifo'):
            self.assertIsNone(b.validate_source_selection(SOURCE, REVISION, feature))
            for old in (b.SIDECAR_REVISION, b.NATIVE_REVISION, b.SLRU_REVISION, b.FRACTION_REVISION, b.SESSION_REVISION, None):
                with self.subTest(old=old), self.assertRaises(ValueError): b.validate_source_selection(SOURCE, old, feature)
            with self.assertRaises(ValueError): b.validate_source_selection(SOURCE, REVISION, feature, .5)
            with self.assertRaises(ValueError): b.validate_source_selection(SOURCE, REVISION, feature, reset_on_session=True)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory); policy = source / 'models/flashnext/expert_pool_policy.py'
            policy.parent.mkdir(parents=True); policy.write_text(b.FRACTION_FEATURE)
            self.assertIsNone(b.validate_source_selection(source, b.FRACTION_REVISION, 'slru', .5))
            with self.assertRaises(ValueError): b.validate_source_selection(source, REVISION, 's3fifo')

    def test_launcher_isolates_every_flag_on_both_arms_and_requires_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory)
            for name in ('candidate_start.py', 'benchmark_backend.py'): shutil.copy2(ROOT / name, release / name)
            (release / 'deployment.json').write_text(json.dumps({'memory': dict(expert_pool_gb=6, mlx_cache_mib=512, wired_gb=0)}))
            (release / 'start.py').write_text('import json, os\ndef main():\n print(json.dumps(dict(os.environ)))\n')
            policy = release / 'src/models/flashnext/expert_pool_policy.py'; policy.parent.mkdir(parents=True)
            shutil.copy2(SOURCE / 'models/flashnext/expert_pool_policy.py', policy)
            inherited = dict(os.environ, **{k:'bad' for k in (b.FEATURE,b.CACHE_FEATURE,b.NATIVE_FEATURE,b.FRACTION_FEATURE,b.SESSION_RESET_FEATURE)})
            for feature in ('s3fifo', 'sidecar-s3fifo'):
                command = [sys.executable, str(release / 'candidate_start.py'), '--feature', feature, '--sidecar', '/side', '--check-only']
                denied = subprocess.run(command + ['--condition', 'off'], capture_output=True, text=True, env=inherited)
                self.assertNotEqual(denied.returncode, 0)
                for condition in ('off', 'on'):
                    result = subprocess.run(command + ['--condition', condition, '--source-revision', REVISION], capture_output=True, text=True, env=inherited)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    env = json.loads(result.stdout)
                    actual = {k:v for k,v in env.items() if k in (b.FEATURE,b.CACHE_FEATURE,b.NATIVE_FEATURE,b.FRACTION_FEATURE,b.SESSION_RESET_FEATURE)}
                    self.assertEqual(actual, b.feature_environment(feature, condition, Path('/side')))

    def test_runtime_s3_validation_accepts_shared_targets_and_rejects_bad_bounds(self):
        self.assertTrue(hasattr(b, 'validate_cache_s3fifo'), 'actual S3 validation is absent')
        state = self.state()
        store = SimpleNamespace(_flashnext_pool_cache_policy='s3fifo')
        pool = SimpleNamespace(cache_policy='s3fifo', slots=16, policy_diagnostics=lambda:state)
        actual = b.validate_cache_s3fifo(store, pool, 's3fifo')
        self.assertEqual(actual, state)
        # Queue lengths may exceed individual eviction targets; total cannot exceed slots.
        for field, bad in (('small_target', 2), ('main_target', 14), ('small_count', 17), ('main_count', 1),
                           ('ghost_count', 16), ('small_count', True), ('main_count', -1), ('ghost_count', None),
                           ('maximum_frequency', 1), ('promotion_threshold', 1), ('policy', 'slru'),
                           ('small_oldest', list(range(9))), ('ghost_oldest', list(range(3)))):
            original = copy.deepcopy(state)
            state[field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(RuntimeError): b.validate_cache_s3fifo(store, pool, 's3fifo')
            state.clear(); state.update(original)
        for actual in ('lru', None):
            store._flashnext_pool_cache_policy=actual
            with self.assertRaises(RuntimeError): b.validate_cache_s3fifo(store, pool, 's3fifo')
        store._flashnext_pool_cache_policy='s3fifo'
        self.assertIsNone(b.validate_cache_s3fifo(store, None, 's3fifo'))
        with self.assertRaises(RuntimeError): b.validate_cache_s3fifo(SimpleNamespace(), None, 'lru')

    def test_parent_child_provenance_freezes_s3_targets_frequency_and_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            args = SimpleNamespace(source=source, checkpoint=source, feature='sidecar-s3fifo', sidecar=Path('/side'),
                slru_protected_fraction=.7, source_revision=REVISION, seed=20261012, tokens=128, pairs=6,
                quality_pairs=6, mode='quality', output_dir=source)
            with patch.object(b, 'imports', return_value=(SimpleNamespace(benchmark_provenance=lambda checkpoint: {}), 'prompt')), \
                    patch.object(b.importlib.metadata, 'distributions', return_value=[]), \
                    patch.object(b.importlib.metadata, 'distribution', return_value=SimpleNamespace(files=[])):
                first = b.provenance(args)
                args.condition='on'
                child = b.provenance(args)
            self.assertEqual(first, child)
            self.assertEqual(first['cache_policy_arms'], dict(off='lru', on='s3fifo'))
            self.assertEqual(first['s3fifo_settings'], dict(small_fraction=.1, minimum_small_target=1,
                small_target=195, main_target=1758, ghost_capacity=1758, resident_slots=1953,
                maximum_frequency=3, promotion_threshold=2, shared_capacity=True))
            self.assertEqual(first['slru_protected_fraction_arms'], dict(off=.7,on=.7))
            with patch.object(b.subprocess, 'Popen', return_value=SimpleNamespace(returncode=0,pid=999,poll=lambda:0,wait=lambda:0)):
                outcome = b.run_child(args, 's3', 'on')
            command = outcome['command']
            self.assertEqual(command[command.index('--feature')+1], 'sidecar-s3fifo')
            self.assertEqual(command[command.index('--source-revision')+1], REVISION)
            self.assertNotIn('--slru-reset-on-session', command)

    def test_candidate_events_assert_actual_s3_without_slru_fraction(self):
        import candidate_start, backend_api
        state = self.state()
        class Pool:
            def __init__(self, store): self.cache_policy='s3fifo'; self.slots=16; self.closed=False
            def close(self): self.closed=True
            def policy_diagnostics(self): return state
        module = SimpleNamespace(ExpertPool=Pool, initialize_sidecar=lambda store:setattr(store, '_flashnext_pool_cache_policy', 's3fifo'))
        output = io.StringIO()
        with redirect_stdout(output):
            candidate_start.install_cache_validation(module, 's3fifo', .7, expected_s3fifo=True)
            store = SimpleNamespace(); module.initialize_sidecar(store); module.ExpertPool(store)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([row['phase'] for row in events], ['startup','runtime'])
        self.assertNotIn('protected_fraction', events[0])
        self.assertEqual(events[1]['s3fifo']['small_target'], 1)
        args = SimpleNamespace(source_revision=REVISION,feature='s3fifo',condition='on',sidecar=Path('/side'),slru_protected_fraction=.7)
        with tempfile.TemporaryDirectory() as directory:
            log=Path(directory)/'candidate.log'
            log.write_text(output.getvalue())
            self.assertEqual(backend_api.candidate_cache_events(log,args,require_runtime=True),events)
            for mutation in ('startup','runtime','policy','bounds'):
                changed=copy.deepcopy(events)
                if mutation in ('startup','runtime'): changed=[row for row in changed if row['phase'] != mutation]
                elif mutation == 'policy': changed[0]['policy']='lru'
                else: changed[1]['s3fifo']['ghost_count']=16
                log.write_text(''.join(json.dumps(row)+'\n' for row in changed))
                with self.subTest(mutation=mutation), self.assertRaises(RuntimeError): backend_api.candidate_cache_events(log,args,require_runtime=True)
        state['small_target']=2
        with self.assertRaises(RuntimeError): module.ExpertPool(store)


if __name__ == '__main__': unittest.main()
