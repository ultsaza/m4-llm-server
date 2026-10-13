"""Model-free audit fixtures enforce the prospective repeated quality protocol."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.recheck_backend import audit


def write_json(path, value):
    path.write_text(json.dumps(value) + '\n')


def quality_fixture(directory, *, tokens=128, seed=20261012, pairs=6,
                    rates=(11, 20), names=('japanese', 'code')):
    source = b'model-free frozen source\n'
    manifest = {'fixture.py': hashlib.sha256(source).hexdigest()}
    manifest_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    archive = directory / 'source-snapshot.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        entry = tarfile.TarInfo('fixture.py')
        entry.size = len(source)
        tar.addfile(entry, io.BytesIO(source))
    helper = b'model-free frozen helper\n'
    (directory / 'fixture-helper.py').write_bytes(helper)
    meta = dict(source_files=manifest, source_snapshot_sha256=manifest_sha,
                helpers={'fixture-helper.py': hashlib.sha256(helper).hexdigest()},
                reviewed_source_revision='local-fixture', quality_pairs=pairs,
                token_limit=tokens, seed=seed,
                quality_prompts={name: name + ' fixture prompt' for name in names},
                sampled_settings=dict(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5))
    write_json(directory / 'parent-provenance.json', meta)
    write_json(directory / 'source-archive.json',
               dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), manifest_sha256=manifest_sha))
    summary_cases = {}
    for index, name in enumerate(names):
        off, on = (10, 20)[index], rates[index]
        effect = (on / off - 1) * 100
        pool = {'hits': 100, 'misses': 20}
        summary_cases[name] = dict(paired_rates=[{'off': off, 'on': on}] * pairs,
            paired_decode_pool=[{'off': pool, 'on': pool}] * pairs,
            effects_percent=[effect] * pairs, decode_pool_deltas=[{'hits': 0, 'misses': 0}] * pairs,
            mean_rates={'off': off, 'on': on}, mean_percent=effect,
            two_se_percent=0.0 if pairs > 1 else None,
            mean_decode_pool_delta={'hits': 0, 'misses': 0})
    write_json(directory / 'summary.json',
               dict(mode='quality', feature='sidecar', pairs=pairs, quality_cases=summary_cases))
    for pair in range(1, pairs + 1):
        for condition in ('off', 'on'):
            quality = [dict(name=name,
                prompt='<|im_start|>user\n' + prompt
                    + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
                seed=seed + index, sampling={'min_p': 0.0, **meta['sampled_settings']},
                ids=[index + 1] * tokens, gen_tokens=tokens, text=name + ' actual fixture text',
                finish='length', gen_rate=rates[index] if condition == 'on' else (10, 20)[index],
                decode_pool={'hits': 100, 'misses': 20})
                for index, (name, prompt) in enumerate(meta['quality_prompts'].items())]
            row = dict(validated=True, provenance=meta, quality=quality,
                       effective_pool_bytes=5999624192, effective_cache_limit_bytes=536870912,
                       effective_wired_gb=0, effective_io_workers=8, pool_locked=False,
                       native_mtp='off', routing_profile='exact-quality', threshold=1.0,
                       model_load_seconds=1)
            stem = f'{pair}-{condition}'
            write_json(directory / (stem + '.json'), row)
            write_json(directory / (stem + '.outcome.json'), dict(returncode=0, failure=None, raw=row))
            write_json(directory / (stem + '.resources.json'),
                       [dict(global_swap=dict(returncode=0, stdout='used = 0M'))])
            write_json(directory / (stem + '.footprint.json'),
                       [dict(returncode=0, stdout='phys_footprint: 1M')])


class RepeatedWarmGate(unittest.TestCase):
    def check_gate(self, expected, **options):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            quality_fixture(directory, **options)
            result = audit(directory)
        self.assertTrue(result['passed'], 'noncanonical runs remain valid screening evidence')
        self.assertTrue(result['actual_tokens_equal'])
        self.assertEqual(result['repeated_warm_gate'], expected)
        self.assertNotIn('decode_gate', result, 'quality cannot acquire the independent standard decode gate')

    def test_canonical_six_pairs_with_nonnegative_case_means_pass(self):
        self.check_gate(True)

    def test_short_six_pair_run_remains_screening_and_cannot_pass(self):
        self.check_gate(False, tokens=4)

    def test_custom_seed_six_pair_run_remains_screening_and_cannot_pass(self):
        self.check_gate(False, seed=1)

    def test_code_must_use_20261013_and_japanese_20261012(self):
        self.check_gate(False, names=('code', 'japanese'))

    def test_negative_mean_in_either_case_fails(self):
        for rates in ((9, 20), (11, 19)):
            with self.subTest(rates=rates):
                self.check_gate(False, rates=rates)

    def test_fewer_than_six_pairs_remains_screening(self):
        self.check_gate(False, pairs=5)

    def test_later_pair_cannot_change_actual_tokens_or_fixed_seed(self):
        for field, value in (('ids', [9] * 128), ('seed', 20261014)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                quality_fixture(directory)
                row_path = directory / '6-on.json'
                row = json.loads(row_path.read_text())
                row['quality'][1][field] = value
                write_json(row_path, row)
                write_json(directory / '6-on.outcome.json', dict(returncode=0, failure=None, raw=row))
                with self.assertRaises(AssertionError):
                    audit(directory)

    def test_missing_resource_evidence_still_fails_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            quality_fixture(directory)
            write_json(directory / '6-on.resources.json', [])
            with self.assertRaisesRegex(AssertionError, 'resource evidence missing'):
                audit(directory)
