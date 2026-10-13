"""Repeated sampled measurements keep fixed work and paired case-level effects."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import benchmark_backend as bench

REAL_POPEN = subprocess.Popen


def args_for(directory, mode='quality', quality_pairs=1):
    return SimpleNamespace(source=directory / 'src', checkpoint=directory, sidecar=directory,
        source_revision=bench.FRACTION_REVISION, feature='sidecar-slru',
        slru_protected_fraction=0.5, mode=mode, output_dir=directory / 'output',
        tokens=128, pairs=6, quality_pairs=quality_pairs, seed=20261012,
        timeout=30, max_swap_mib=6144)


def frozen_provenance(args):
    production = SimpleNamespace(benchmark_provenance=lambda checkpoint: {})
    with patch.object(bench, 'imports', return_value=(production, 'prompt')), \
            patch.object(bench.importlib.metadata, 'distributions', return_value=[]), \
            patch.object(bench.importlib.metadata, 'distribution', return_value=SimpleNamespace(files=[])):
        return bench.provenance(args)


def quality_arm(meta, rates=(10, 20), pools=((100, 20), (80, 30))):
    cases = []
    for index, (name, prompt) in enumerate(bench.QUALITY.items()):
        cases.append(dict(name=name, prompt='<|im_start|>user\n' + prompt + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
            seed=20261012 + index, sampling={**bench.SAMPLED, 'min_p': 0.0},
            ids=[index + 1] * 128, text=name + ' actual decoded text',
            gen_tokens=128, finish='length', gen_rate=rates[index],
            wall=8, latency={}, decode_pool_boundary={'hits': 1000, 'misses': 2000},
            decode_pool=dict(zip(('hits', 'misses'), pools[index]))))
    return dict(provenance=meta, validated=True, quality=cases)


class QualityPairs(unittest.TestCase):
    def parse_dispatch(self, *options):
        captured = []
        with patch.object(sys, 'argv', ['benchmark_backend.py', '--output-dir', '/tmp/quality-test', *options]), \
                patch.object(bench, 'validate_source_selection'), \
                patch.object(bench, 'parent', side_effect=captured.append), \
                patch.object(bench, 'preflight', side_effect=captured.append), redirect_stderr(io.StringIO()):
            bench.main()
        return captured[0]

    def test_default_quality_stays_one_pair_and_throughput_stays_six(self):
        quality = self.parse_dispatch('--mode', 'quality')
        throughput = self.parse_dispatch()
        self.assertEqual(quality.quality_pairs, 1)
        self.assertEqual(quality.tokens, 128)
        self.assertEqual(quality.seed, 20261012)
        self.assertEqual(throughput.pairs, 6)
        self.assertEqual(throughput.quality_pairs, 1)
        self.assertEqual(self.parse_dispatch('--mode', 'quality', '--quality-pairs', '6').quality_pairs, 6)

    def test_nonpositive_or_noninteger_counts_and_nonquality_repetition_are_rejected(self):
        for value in ('0', '-2', '1.5'):
            with self.subTest(value=value), self.assertRaises(SystemExit) as error:
                self.parse_dispatch('--mode', 'quality', '--quality-pairs', value)
            self.assertEqual(error.exception.code, 2)
        for mode in ('throughput', 'coverage', 'preflight'):
            with self.subTest(mode=mode), self.assertRaises(SystemExit) as error:
                self.parse_dispatch('--mode', mode, '--quality-pairs', '6')
            self.assertEqual(error.exception.code, 2)
        self.assertEqual(self.parse_dispatch('--mode', 'coverage', '--quality-pairs', '1').quality_pairs, 1)

    def exercise_parent(self, directory, count, change=None):
        args = args_for(directory, quality_pairs=count)
        meta = frozen_provenance(args)
        order = []
        def run(_args, arm_id, condition):
            order.append(arm_id)
            index = int(arm_id.split('-')[0]) - 1
            # Hand-chosen effects: Japanese +10,-10,+20%; code -10,+10,0%.
            rates = ((11, 18), (9, 22), (12, 20))[index % 3] if condition == 'on' else (10, 20)
            pools = ((105, 15), (90, 20)) if condition == 'on' else ((100, 20), (80, 30))
            raw = quality_arm(meta, rates, pools)
            if change:
                change(arm_id, raw)
            return dict(raw=raw, returncode=0, failure=None, command=['model-free-fixture'])
        with patch.object(bench, 'provenance', return_value=meta), patch.object(bench, 'run_child', side_effect=run), \
                redirect_stdout(io.StringIO()):
            bench.parent(args)
        return order, json.loads((args.output_dir / 'summary.json').read_text())

    def test_repeated_arms_alternate_and_summary_uses_paired_case_rates_and_actual_deltas(self):
        with tempfile.TemporaryDirectory() as temporary:
            order, summary = self.exercise_parent(Path(temporary), 3)
        self.assertEqual(order, ['1-off', '1-on', '2-on', '2-off', '3-off', '3-on'])
        self.assertEqual(summary['pairs'], 3)
        self.assertTrue(summary['all_tokens_equal'])
        self.assertNotIn('decode_gate', summary)
        japanese, code = summary['quality_cases']['japanese'], summary['quality_cases']['code']
        for actual, expected in zip(japanese['effects_percent'], [10, -10, 20]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(japanese['mean_percent'], 20 / 3)
        self.assertAlmostEqual(japanese['two_se_percent'], 17.638342073763937)
        self.assertAlmostEqual(code['mean_percent'], 0)
        self.assertAlmostEqual(code['two_se_percent'], 11.547005383792516)
        self.assertEqual(japanese['paired_rates'], [{'off': 10, 'on': 11}, {'off': 10, 'on': 9}, {'off': 10, 'on': 12}])
        self.assertEqual(japanese['mean_rates'], {'off': 10, 'on': 32 / 3})
        self.assertEqual(japanese['decode_pool_deltas'], [{'hits': 5, 'misses': -5}] * 3)
        self.assertEqual(code['decode_pool_deltas'], [{'hits': 10, 'misses': -10}] * 3)
        self.assertEqual(code['mean_decode_pool_delta'], {'hits': 10, 'misses': -10})
        self.assertEqual(japanese['paired_decode_pool'][0], {'off': {'hits': 100, 'misses': 20}, 'on': {'hits': 105, 'misses': 15}})
        with tempfile.TemporaryDirectory() as temporary:
            order, summary = self.exercise_parent(Path(temporary), 6)
        self.assertEqual(order, ['1-off', '1-on', '2-on', '2-off', '3-off', '3-on',
                                 '4-on', '4-off', '5-off', '5-on', '6-on', '6-off'])
        self.assertEqual(summary['pairs'], 6)

    def test_default_parent_reports_one_pair_and_no_two_se(self):
        with tempfile.TemporaryDirectory() as temporary:
            order, summary = self.exercise_parent(Path(temporary), 1)
        self.assertEqual(order, ['1-off', '1-on'])
        self.assertEqual(summary['pairs'], 1)
        self.assertIsNone(summary['quality_cases']['code']['two_se_percent'])
        self.assertNotIn('decode_gate', summary)

    def test_matching_later_pair_cannot_replace_first_actual_outputs_or_work(self):
        for field, value in [('ids', [9] * 128), ('text', 'different text'), ('prompt', 'different prompt'),
                             ('seed', 1), ('sampling', {}), ('gen_tokens', 127), ('finish', 'stop'), ('name', 'other')]:
            def change(arm_id, raw):
                if arm_id.startswith('3-'):
                    raw['quality'][0][field] = value
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                with self.assertRaises(RuntimeError):
                    self.exercise_parent(directory, 3, change)
                self.assertFalse((directory / 'output/summary.json').exists())
                self.assertTrue((directory / 'output/failure.json').exists())

    def test_first_arm_requires_both_named_cases_and_consistent_bounded_token_work(self):
        changes = [lambda raw: raw['quality'].pop(),
                   lambda raw: raw['quality'][0].update(name='code'),
                   lambda raw: raw['quality'][0].update(gen_tokens=127),
                   lambda raw: raw['quality'][0].update(ids=[], gen_tokens=0),
                   lambda raw: raw['quality'][0].update(ids=[1] * 129, gen_tokens=129)]
        for mutate in changes:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(RuntimeError):
                    self.exercise_parent(Path(temporary), 1, lambda arm_id, raw: mutate(raw))

    def test_count_is_frozen_and_forwarded_to_real_child_argument_consumer(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = args_for(directory, quality_pairs=6)
            args.output_dir.mkdir()
            meta = frozen_provenance(args)
            self.assertEqual(meta['quality_pairs'], 6)
            def spawn(command, **kwargs):
                # Real process consumes forwarded argv; model loading alone is replaced.
                code = 'import json,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))'
                return REAL_POPEN([sys.executable, '-c', code, str(directory / 'argv.json'), *command[3:]], **kwargs)
            with patch.object(bench.subprocess, 'Popen', side_effect=spawn), \
                    patch.object(bench.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='used = 0M', stderr='')), \
                    patch.object(bench.time, 'sleep'):
                outcome = bench.run_child(args, '1-off', 'off')
            self.assertEqual(outcome['returncode'], 0)
            received = json.loads((directory / 'argv.json').read_text())
            self.assertEqual(received[received.index('--quality-pairs') + 1], '6')
            self.assertEqual(received[received.index('--tokens') + 1], '128')
            self.assertEqual(received[received.index('--seed') + 1], '20261012')
            with patch.object(sys, 'argv', ['benchmark_backend.py', *received]), \
                    patch.object(bench, 'validate_source_selection'), patch.object(bench, 'child') as child:
                bench.main()
            self.assertEqual(frozen_provenance(child.call_args.args[0]), meta)
