"""Session reset selectors, arm isolation, provenance and actual markers."""
from contextlib import redirect_stdout
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
FLAG = 'FLASHNEXT_EXPERT_POOL_SLRU_SESSION_RESET'
REVISION = '161b1e2338817a27e02b4f7f52f5012cfd4c284f'

class SessionHelpers(unittest.TestCase):
    def test_requested_reset_only_reaches_slru_on_and_requires_new_source(self):
        for feature in ('slru', 'sidecar-slru'):
            self.assertNotIn(FLAG, b.feature_environment(feature, 'off', Path('/side'), reset_on_session=True))
            self.assertEqual(b.feature_environment(feature, 'on', Path('/side'), reset_on_session=True)[FLAG], '1')
        with self.assertRaises(ValueError): b.feature_environment('sidecar', 'on', Path('/side'), reset_on_session=True)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory); policy = source / 'models/flashnext/expert_pool_policy.py'
            policy.parent.mkdir(parents=True); policy.write_text(b.FRACTION_FEATURE + '\n' + FLAG)
            for revision in (b.SLRU_REVISION, b.FRACTION_REVISION, b.SIDECAR_REVISION, b.NATIVE_REVISION):
                with self.subTest(revision=revision), self.assertRaises(ValueError):
                    b.validate_source_selection(source, revision, 'slru', reset_on_session=True)
            self.assertIsNone(b.validate_source_selection(source, REVISION, 'slru', reset_on_session=True))
            policy.write_text(b.FRACTION_FEATURE)
            with self.assertRaises(ValueError): b.validate_source_selection(source, REVISION, 'slru', reset_on_session=True)

    def test_candidate_cli_clears_inherited_override_and_rejects_old_source(self):
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory)
            for name in ('candidate_start.py', 'benchmark_backend.py'): shutil.copy2(ROOT / name, release / name)
            (release / 'deployment.json').write_text(json.dumps({'memory': dict(expert_pool_gb=6, mlx_cache_mib=512, wired_gb=0)}))
            (release / 'start.py').write_text('import json, os\ndef main():\n print(json.dumps(dict(os.environ)))\n')
            policy = release / 'src/models/flashnext/expert_pool_policy.py'; policy.parent.mkdir(parents=True)
            policy.write_text(b.FRACTION_FEATURE + '\n' + FLAG)
            command = [sys.executable, str(release / 'candidate_start.py'), '--feature', 'sidecar-slru', '--sidecar', '/side', '--check-only', '--slru-reset-on-session']
            denied = subprocess.run(command + ['--condition', 'on', '--source-revision', b.FRACTION_REVISION], capture_output=True, text=True)
            self.assertNotEqual(denied.returncode, 0)
            for condition in ('off', 'on'):
                result = subprocess.run(command + ['--condition', condition, '--source-revision', REVISION], capture_output=True, text=True, env=dict(os.environ, **{FLAG: 'bad'}))
                self.assertEqual(result.returncode, 0, result.stderr)
                env = json.loads(result.stdout)
                self.assertEqual(env.get(FLAG), '1' if condition == 'on' else None)

    def test_actual_marker_and_boundary_count_reject_missing_type_mismatch_and_wrong_count(self):
        store = SimpleNamespace(_flashnext_pool_slru_session_reset=True)
        pool = SimpleNamespace(slru_session_reset=True, policy_diagnostics=lambda: dict(slru_session_reset=True, session_reset_count=1))
        self.assertIs(b.validate_cache_session_reset(store, pool, True), True)
        self.assertEqual(b.session_reset_boundary(store, pool, True, 1), dict(slru_session_reset=True, session_reset_count=1))
        self.assertEqual(b.session_reset_boundary(store, None, True, 0), dict(slru_session_reset=True, session_reset_count=0))
        for actual in (None, 1, False):
            with self.subTest(actual=actual), self.assertRaises(RuntimeError):
                b.validate_cache_session_reset(SimpleNamespace(_flashnext_pool_slru_session_reset=actual), pool, True)
        for count in (True, -1, None, 0):
            pool.policy_diagnostics = lambda: dict(slru_session_reset=True, session_reset_count=count)
            with self.subTest(count=count), self.assertRaises(RuntimeError): b.session_reset_boundary(store, pool, True, 1)
        pool.slru_session_reset = False
        with self.assertRaises(RuntimeError): b.validate_cache_session_reset(store, pool, True)

    def test_parent_and_child_freeze_request_and_arm_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            args = SimpleNamespace(source=path, checkpoint=path, feature='sidecar-slru', sidecar=Path('/side'), slru_protected_fraction=0.7,
                slru_reset_on_session=True, source_revision=REVISION, seed=20261012, tokens=128, pairs=6, quality_pairs=1, mode='quality', output_dir=path)
            with patch.object(b, 'imports', return_value=(SimpleNamespace(benchmark_provenance=lambda checkpoint: {}), 'prompt')), \
                    patch.object(b.importlib.metadata, 'distributions', return_value=[]), \
                    patch.object(b.importlib.metadata, 'distribution', return_value=SimpleNamespace(files=[])):
                meta = b.provenance(args)
            self.assertIs(meta['slru_reset_on_session'], True)
            self.assertEqual(meta['slru_session_reset_arms'], {'off': False, 'on': True})
            with patch.object(b.subprocess, 'Popen', return_value=SimpleNamespace(returncode=0, pid=999, poll=lambda:0, wait=lambda:0)):
                outcome = b.run_child(args, 'reset', 'on')
            self.assertIn('--slru-reset-on-session', outcome['command'])
            import backend_api
            args.condition = 'on'
            self.assertIn('--slru-reset-on-session', backend_api.candidate_command(args, path))

    def test_candidate_events_assert_actual_boolean_and_runtime_count(self):
        import candidate_start, backend_api
        class Pool:
            def __init__(self, store):
                self.cache_policy='slru'; self.slru_protected_fraction=0.7; self.slru_session_reset=True; self.slots=5
            def close(self): pass
            def policy_diagnostics(self): return dict(protected_fraction=0.7, protected_capacity=3, slru_session_reset=True, session_reset_count=0)
        def initialize(store):
            store._flashnext_pool_cache_policy='slru'; store._flashnext_pool_slru_protected_fraction=0.7; store._flashnext_pool_slru_session_reset=True
        module=SimpleNamespace(initialize_sidecar=initialize, ExpertPool=Pool); output=io.StringIO()
        with redirect_stdout(output):
            candidate_start.install_cache_validation(module, 'slru', 0.7, expected_reset=True)
            store=SimpleNamespace(); module.initialize_sidecar(store); module.ExpertPool(store)
        events=[json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([row['slru_session_reset'] for row in events], [True, True])
        self.assertEqual(events[1]['session_reset_count'], 0)
        args=SimpleNamespace(source_revision=REVISION, feature='slru', condition='on', sidecar=Path('/side'), slru_protected_fraction=0.7, slru_reset_on_session=True)
        with tempfile.TemporaryDirectory() as directory:
            log=Path(directory)/'log'; log.write_text(output.getvalue())
            self.assertEqual(backend_api.candidate_cache_events(log,args,True),events)
            events[1]['slru_session_reset']=1
            log.write_text(''.join(json.dumps(row)+'\n' for row in events))
            with self.assertRaises(RuntimeError): backend_api.candidate_cache_events(log,args,True)

    def test_quality_parent_rejects_missing_or_wrong_actual_prefill_reset_evidence(self):
        from tests.test_backend_quality_pairs import args_for, frozen_provenance, quality_arm
        for defect in (None, 'missing', 'wrong_count', 'wrong_flag'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                args=args_for(Path(directory)); args.source_revision=REVISION; args.slru_reset_on_session=True
                args.slru_protected_fraction=0.7; meta=frozen_provenance(args)
                def run(args, arm_id, condition):
                    raw=quality_arm(meta)
                    for index, case in enumerate(raw['quality']):
                        case['decode_pool_boundary'].update(slru_session_reset=condition == 'on', session_reset_count=index if condition == 'on' else 0)
                    if condition == 'on' and defect:
                        boundary=raw['quality'][1]['decode_pool_boundary']
                        if defect == 'missing': boundary.pop('session_reset_count')
                        if defect == 'wrong_count': boundary['session_reset_count']=0
                        if defect == 'wrong_flag': boundary['slru_session_reset']=False
                    return dict(raw=raw, returncode=0, failure=None, command=['fixture'])
                with patch.object(b, 'provenance', return_value=meta), patch.object(b, 'run_child', side_effect=run), redirect_stdout(io.StringIO()):
                    if defect:
                        with self.assertRaises(RuntimeError): b.parent(args)
                    else: b.parent(args)
