"""Feature arms are independent and launcher defaults to its own release."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import benchmark_backend as bench


class Features(unittest.TestCase):
    def test_native_comparison_keeps_sidecar_equal_and_combined_toggles_both(self):
        self.assertTrue(hasattr(bench, 'feature_environment'), 'feature arm selection is missing')
        side = 'FLASHNEXT_EXPERT_POOL_SMALL_SIDECAR'
        native = 'FLASHNEXT_EXPERT_POOL_NATIVE_READ'
        expected = {
            ('sidecar', 'off'): {}, ('sidecar', 'on'): {side: '/side'},
            ('native', 'off'): {side: '/side'}, ('native', 'on'): {side: '/side', native: '1'},
            ('combined', 'off'): {}, ('combined', 'on'): {side: '/side', native: '1'},
        }
        for args, environment in expected.items():
            self.assertEqual(bench.feature_environment(*args, Path('/side')), environment)
        with self.assertRaises(ValueError):
            bench.feature_environment('typo', 'on', Path('/side'))

    def test_launcher_applies_feature_in_its_own_release_and_clears_inherited_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            release = Path(temporary)
            for name in ('candidate_start.py', 'benchmark_backend.py'):
                shutil.copy2(ROOT / name, release / name)
            (release / 'deployment.json').write_text(json.dumps({'memory': dict(expert_pool_gb=6, mlx_cache_mib=512, wired_gb=0)}))
            (release / 'start.py').write_text('import json, os\ndef main():\n print(json.dumps(dict(os.environ)))\n')
            result = subprocess.run([sys.executable, str(release / 'candidate_start.py'), '--feature', 'native',
                                     '--condition', 'off', '--sidecar', '/side', '--check-only'], capture_output=True, text=True,
                                    env=dict(os.environ, FLASHNEXT_EXPERT_POOL_NATIVE_READ='1'))
            self.assertEqual(result.returncode, 0, result.stderr)
            environment = json.loads(result.stdout)
            self.assertEqual(environment['FLASHNEXT_EXPERT_POOL_SMALL_SIDECAR'], '/side')
            self.assertNotIn('FLASHNEXT_EXPERT_POOL_NATIVE_READ', environment)

    def test_duplicate_checkpoint_opens_are_validated_by_origin_after_prefill_reset(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        from types import SimpleNamespace
        self.assertTrue(hasattr(bench, 'validate_weight_descriptors'), 'open-time FD provenance validation is missing')
        with tempfile.TemporaryDirectory() as temporary:
            shard = Path(temporary) / 'weights.bin'
            shard.write_bytes(b'abcdefgh')
            other = Path(temporary) / 'other.bin'
            other.write_bytes(b'ABCDEFGH')
            probe = bench.ReadProbe(os)
            pool = SimpleNamespace(store=SimpleNamespace(dir=temporary, refs={'weight': SimpleNamespace(shard='weights.bin')}), _fds={})
            barrier = threading.Barrier(2)
            def lazy_open():
                fd = pool._fds.get('weights.bin')
                if fd is None:
                    fd = probe.open(str(shard), os.O_RDONLY)
                    barrier.wait(timeout=5)
                    pool._fds['weights.bin'] = fd
                return fd
            with ThreadPoolExecutor(2) as workers:
                futures = [workers.submit(lazy_open) for _ in range(2)]
                fds = [future.result() for future in futures]
            wrong = probe.open(str(other), os.O_RDONLY)
            try:
                self.assertEqual(len(set(fds)), 2)
                self.assertEqual(len(pool._fds), 1)
                probe.reset()  # Actual on_prefilled resets counters, not FD origin.
                for fd in fds:
                    target = bytearray(8)
                    self.assertEqual(probe.preadv(fd, [target], 0), 8)
                    self.assertEqual(target, b'abcdefgh')
                bench.validate_weight_descriptors(probe.rows, pool, probe)
                probe.preadv(wrong, [bytearray(8)], 0)
                with self.assertRaisesRegex(RuntimeError, 'checkpoint'):
                    bench.validate_weight_descriptors(probe.rows, pool, probe)
            finally:
                for fd in fds + [wrong]:
                    os.close(fd)

    def test_wrapper_preflight_uses_own_release_not_the_first_sidecar_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            release = Path(temporary)
            shutil.copy2(ROOT / 'run_backend_experiment.sh', release)
            (release / 'venv/bin').mkdir(parents=True)
            python = release / 'venv/bin/python'
            python.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$*"\n')
            python.chmod(0o755)
            result = subprocess.run(['sh', str(release / 'run_backend_experiment.sh'), '--owner-label', 'test',
                                     '--owner-plist', '/test', 'preflight', '--feature', 'native'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines()[0], str(release.resolve()))
            self.assertIn('--feature native', result.stdout)


if __name__ == '__main__':
    unittest.main()
