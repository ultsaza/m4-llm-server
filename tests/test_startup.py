"""A missing/empty service key must never disable HTTP authentication."""
from pathlib import Path
import tempfile
import unittest

import start
from start import _read_key


class KeyContract(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.key = Path(self.directory.name) / "api-key"
        self.key.write_text("test-secret\n")
        self.key.chmod(0o600)

    def test_private_key_is_loaded(self):
        self.assertEqual(_read_key(self.key), "test-secret")

    def test_blank_key_is_rejected(self):
        self.key.write_text(" \n")
        with self.assertRaises(SystemExit):
            _read_key(self.key)

    def test_readable_key_is_rejected(self):
        self.key.chmod(0o644)
        with self.assertRaises(SystemExit):
            _read_key(self.key)


class MemoryContract(unittest.TestCase):
    def test_validated_policy_is_available_before_model_loading(self):
        policy = start._read_memory_settings({"memory": {
            "expert_pool_gb": 8, "mlx_cache_mib": 512, "wired_gb": 0,
        }})
        self.assertEqual(policy, {"expert_pool_gb": 8.0, "mlx_cache_mib": 512, "wired_gb": 0.0})

    def test_missing_or_misspelled_memory_settings_are_rejected(self):
        for memory in (None, {}, {"expert_pool_gb": 8, "cache_mb": 512, "wired_gb": 0}):
            with self.subTest(memory=memory), self.assertRaises(SystemExit):
                start._read_memory_settings({"memory": memory})

    def test_invalid_memory_budgets_cannot_reach_the_runtime(self):
        for key, invalid in (("expert_pool_gb", 9), ("expert_pool_gb", -1),
                             ("expert_pool_gb", True), ("expert_pool_gb", float("nan")),
                             ("wired_gb", 16), ("wired_gb", float("inf")),
                             ("mlx_cache_mib", 16384), ("mlx_cache_mib", True),
                             ("mlx_cache_mib", "512"), ("mlx_cache_mib", 512.5)):
            memory = {"expert_pool_gb": 8, "mlx_cache_mib": 512, "wired_gb": 0}
            memory[key] = invalid
            with self.subTest(key=key, invalid=invalid), self.assertRaises(SystemExit):
                start._read_memory_settings({"memory": memory})


if __name__ == "__main__":
    unittest.main()
