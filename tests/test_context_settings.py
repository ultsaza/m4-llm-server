"""Catalog-driven client and release context settings."""
import importlib.util
import json
from pathlib import Path
import tempfile
import tomllib
import unittest

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
installer=module('context_installer',ROOT/'codex/install-client.py')
startup=module('context_startup',ROOT/'start.py')


class ContextSettings(unittest.TestCase):
    def test_installer_uses_truthful_catalog_context_and_v2_capability(self):
        baseline=json.loads((ROOT/'tests/fixtures/catalog.json').read_text())
        for limit,threshold in [(8192,5000),(16384,10000)]:
            with self.subTest(limit=limit),tempfile.TemporaryDirectory() as temp:
                home=Path(temp);catalog=home/'catalog.json'
                data=json.loads(json.dumps(baseline));data['models'][0]['context_window']=limit
                data['capabilities']={'remote_compaction':'v2'};catalog.write_text(json.dumps(data))
                installer.install(home,catalog,'nonsecret-fixture-key')
                profile=tomllib.loads((home/'.codex/m4-qwen.config.toml').read_text())
                self.assertEqual(profile['model_context_window'],limit)
                self.assertEqual(profile['model_auto_compact_token_limit'],threshold)
                self.assertEqual(profile['model_providers']['m4_qwen']['capabilities']['remote_compaction'],'v2')
                self.assertEqual(json.loads((home/'.codex/m4-qwen.models.json').read_text())['models'][0]['context_window'],limit)

    def test_invalid_context_and_capability_refuse_before_client_writes(self):
        baseline=json.loads((ROOT/'tests/fixtures/catalog.json').read_text())
        for limit,cap in [(True,'v2'),(32768,'v2'),(8192,'invented')]:
            with self.subTest(limit=limit),tempfile.TemporaryDirectory() as temp:
                home=Path(temp);catalog=home/'catalog.json'
                data=json.loads(json.dumps(baseline));data['models'][0]['context_window']=limit
                data['capabilities']={'remote_compaction':cap};catalog.write_text(json.dumps(data))
                with self.assertRaises(ValueError):installer.install(home,catalog,'nonsecret-fixture-key')
                self.assertFalse((home/'.codex').exists())

    def test_old_catalog_keeps_old_context_and_local_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp)
            installer.install(home,ROOT/'tests/fixtures/catalog.json','nonsecret-fixture-key')
            profile=tomllib.loads((home/'.codex/m4-qwen.config.toml').read_text())
            self.assertEqual(profile['model_context_window'],4096)
            self.assertEqual(profile['model_providers']['m4_qwen']['capabilities']['remote_compaction'],'unsupported')

    def test_startup_validates_approved_limits_and_retains_legacy_default(self):
        self.assertEqual(startup._read_server_settings({}),{'context_limit':4096,'remote_compaction':False})
        self.assertEqual(startup._read_server_settings({'server':{'context_limit':16384,'remote_compaction':True}}),
                         {'context_limit':16384,'remote_compaction':True})
        for settings in [{'context_limit':True,'remote_compaction':True},
                         {'context_limit':32768,'remote_compaction':True},
                         {'context_limit':8192,'remote_compaction':'yes'},
                         {'context_limit':8192,'remote_compaction':True,'unknown':1}]:
            with self.subTest(settings=settings),self.assertRaises(SystemExit):startup._read_server_settings({'server':settings})
