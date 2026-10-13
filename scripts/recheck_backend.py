"""Recompute frozen backend experiment checks locally, without model activity."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tarfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def audit(directory):
    meta = read(directory / 'parent-provenance.json')
    manifest = meta['source_files']
    calculated = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    assert calculated == meta['source_snapshot_sha256'], 'source manifest digest'
    archive = directory / 'source-snapshot.tar.gz'
    archive_record = read(directory / 'source-archive.json')
    assert digest(archive) == archive_record['sha256'], 'frozen source archive'
    assert calculated == archive_record['manifest_sha256']
    with tarfile.open(archive) as tar:
        assert {p.name for p in tar if p.isfile()} == set(manifest)
        for name, expected in manifest.items():
            assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == expected, name
    for name, expected in meta['helpers'].items():
        assert digest(directory / name) == expected, 'frozen helper ' + name
    native = meta.get('pool_native')
    if native:
        assert digest(directory / Path(native['library']).name) == native['library_sha256']
        assert manifest['models/flashnext/pool_native.c'] == native['source_sha256']
    summary = read(directory / 'summary.json')
    mode = summary['mode']
    session_source = meta['reviewed_source_revision'] == '161b1e2338817a27e02b4f7f52f5012cfd4c284f'
    s3_source = meta['reviewed_source_revision'] == 'f7b60e8312843ae27aee7fa3e48320cf6010ffe1'
    feature = summary.get('feature')
    if feature is None:
        # The immutable first helper predates the configurable-feature field.
        assert meta['helpers']['benchmark_backend.py'] == 'b1c17a9a5b783a877e4529ec09d0f77b9d55315eb2850e8c75d5bf788ae7b060'
        assert meta['reviewed_source_revision'] == 'cc1981e25a18c033630a8534a37ba55e0f50e8ea'
        feature = 'sidecar'
    if feature in ('s3fifo', 'sidecar-s3fifo'):
        assert s3_source and meta['cache_policy_arms'] == {'off': 'lru', 'on': 's3fifo'}
        assert meta['s3fifo_settings'] == dict(small_fraction=0.1, minimum_small_target=1,
            small_target=195, main_target=1758, ghost_capacity=1758, resident_slots=1953,
            maximum_frequency=3, promotion_threshold=2, shared_capacity=True)
    rounds = meta['pairs'] if mode == 'throughput' else meta.get('quality_pairs', 1) if mode == 'quality' else 1
    assert rounds > 0 and summary['pairs'] == rounds
    reference = None
    effects, arms, quality_pairs = [], [], []
    for index in range(1, rounds + 1):
        pair = {}
        for condition in ('off', 'on'):
            stem = f'{index}-{condition}'
            row = read(directory / (stem + '.json'))
            outcome = read(directory / (stem + '.outcome.json'))
            assert outcome['returncode'] == 0 and outcome['failure'] is None, stem
            assert outcome['raw'] == row and row['validated'], stem
            assert row['provenance'] == meta, stem + ' provenance'
            assert row['effective_pool_bytes'] == 5999624192
            assert row['effective_cache_limit_bytes'] == 512 * 1024 ** 2
            assert row['effective_wired_gb'] == 0 and row['effective_io_workers'] == 8
            assert row['pool_locked'] is False
            assert row['native_mtp'] == 'off' and row['routing_profile'] == 'exact-quality'
            assert row['threshold'] == 1.0
            if 'cache_policy_arms' in meta:
                expected_policy = meta['cache_policy_arms'][condition]
                assert row['startup_cache_policy'] == row['runtime_cache_policy'] == expected_policy
                assert row['cache_policy_diagnostics']['policy'] == expected_policy
            if 'slru_protected_fraction_arms' in meta and not s3_source:
                expected_fraction = meta['slru_protected_fraction_arms'][condition]
                assert row['startup_slru_protected_fraction'] == expected_fraction
                assert row['runtime_slru_protected_fraction'] == expected_fraction
                state = row['cache_policy_diagnostics']
                assert state['protected_fraction'] == expected_fraction
                # Fixed Q4/G32 layout: a 4096-byte header and 3,072,000 bytes/slot.
                assert row['effective_pool_bytes'] == 5999624192
                slots = (row['effective_pool_bytes'] - 4096) // 3072000
                assert slots == 1953 and state['protected_capacity'] == int(expected_fraction * slots)
            if s3_source:
                environment = row['effective_environment']
                assert not any(name in environment for name in (
                    'FLASHNEXT_EXPERT_POOL_NATIVE_READ', 'FLASHNEXT_EXPERT_POOL_SLRU_PROTECTED_FRACTION',
                    'FLASHNEXT_EXPERT_POOL_SLRU_SESSION_RESET'))
                state = row['cache_policy_diagnostics']
                if meta['cache_policy_arms'][condition] == 's3fifo':
                    assert row['runtime_s3fifo'] == state
                    assert environment['FLASHNEXT_EXPERT_POOL_CACHE_POLICY'] == 's3fifo'
                    assert state['small_target'] == 195 and state['main_target'] == 1758
                    assert state['maximum_frequency'] == 3 and state['promotion_threshold'] == 2
                    assert all(type(state[key]) is int and state[key] >= 0
                               for key in ('small_count', 'main_count', 'ghost_count'))
                    assert state['small_count'] + state['main_count'] == state['probation_count'] <= 1953
                    assert state['ghost_count'] <= 1758
                    assert all(isinstance(state[queue + '_oldest'], list)
                               and len(state[queue + '_oldest']) <= min(state[queue + '_count'], 8)
                               for queue in ('small', 'main', 'ghost'))
                else:
                    assert row['runtime_s3fifo'] is None
                    expected = meta['cache_policy_arms'][condition]
                    assert environment.get('FLASHNEXT_EXPERT_POOL_CACHE_POLICY', 'lru') == expected
                    assert 'small_target' not in state
            if session_source:
                expected_reset = meta['slru_session_reset_arms'][condition]
                assert type(expected_reset) is bool
                assert row['startup_slru_session_reset'] is expected_reset
                assert row['runtime_slru_session_reset'] is expected_reset
                state = row['cache_policy_diagnostics']
                assert state['slru_session_reset'] is expected_reset
                assert type(state['session_reset_count']) is int and state['session_reset_count'] >= 0
            if mode == 'quality':
                assert [r['name'] for r in row['quality']] == list(meta['quality_prompts'])
                assert all(r['prompt'] == '<|im_start|>user\n' + meta['quality_prompts'][r['name']]
                           + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
                           for r in row['quality'])
                actual = [(r['name'], r['prompt'], r['seed'], r['sampling'],
                           r['ids'], r['text'], r['gen_tokens'], r['finish']) for r in row['quality']]
                assert all(0 < r['gen_tokens'] == len(r['ids']) <= meta['token_limit']
                           for r in row['quality'])
                # Sampling(**SAMPLED) also records the pinned min_p default.
                expected_sampling = {'min_p': 0.0, **meta['sampled_settings']}
                assert all(r['seed'] == meta['seed'] + i and r['sampling'] == expected_sampling
                           for i, r in enumerate(row['quality']))
                if session_source:
                    for i, case in enumerate(row['quality']):
                        boundary = case['decode_pool_boundary']
                        assert boundary['slru_session_reset'] is expected_reset
                        assert type(boundary['session_reset_count']) is int
                        assert boundary['session_reset_count'] == (i if expected_reset else 0)
                    assert state['session_reset_count'] == (1 if expected_reset else 0)
            else:
                actual = row['ids']
                assert len(actual) == row['gen_tokens'] == meta['token_limit']
                # The upstream raw arm hashes Python's tuple representation.
                assert hashlib.sha256(str(tuple(actual)).encode()).hexdigest() == row['token_sha256']
            if reference is None:
                reference = actual
            assert actual == reference, stem + ' actual tokens/work'
            samples = read(directory / (stem + '.resources.json'))
            footprints = read(directory / (stem + '.footprint.json'))
            assert samples and footprints, stem + ' resource evidence missing'
            assert all(r.get('returncode') == 0 and 'phys_footprint' in r.get('stdout', '')
                       for r in footprints), stem + ' footprint collection'
            assert all(s['global_swap']['returncode'] == 0 for s in samples), stem + ' swap collection'
            used = [float(re.search(r'used = ([\d.]+)M', s['global_swap']['stdout']).group(1))
                    for s in samples]
            arm = dict(arm=stem, maximum_global_swap_mib=max(used),
                       model_load_seconds=row['model_load_seconds'])
            if mode == 'throughput':
                assert 'coverage' not in row, 'throughput contains diagnostic wrapper output'
                arm.update(gen_rate=row['gen_rate'], latency=row['latency'])
            arms.append(arm)
            pair[condition] = row
        if mode == 'throughput':
            effects.append((pair['on']['gen_rate'] / pair['off']['gen_rate'] - 1) * 100)
        elif mode == 'quality':
            quality_pairs.append(pair)
    result = dict(directory=str(directory), feature=feature, mode=mode,
                  actual_tokens_equal=True, source_snapshot_sha256=calculated,
                  reviewed_source_revision=meta['reviewed_source_revision'],
                  frozen_source_files=len(manifest), frozen_helpers=len(meta['helpers']), arms=arms)
    if 'slru_protected_fraction_arms' in meta and not s3_source:
        result['actual_protected_fraction_arms'] = meta['slru_protected_fraction_arms']
    if s3_source and 's3fifo' in meta['cache_policy_arms'].values():
        result['actual_s3fifo_settings'] = dict(small_target=195, main_target=1758,
                                              maximum_frequency=3, promotion_threshold=2)
    if session_source:
        result['actual_session_reset_arms'] = meta['slru_session_reset_arms']
    if quality_pairs:
        cases = {}
        for index, name in enumerate(meta['quality_prompts']):
            rates, pools, changes, deltas = [], [], [], []
            for pair in quality_pairs:
                rates.append({c: pair[c]['quality'][index]['gen_rate'] for c in ('off', 'on')})
                pools.append({c: pair[c]['quality'][index]['decode_pool'] for c in ('off', 'on')})
                assert all(rates[-1][c] > 0 for c in ('off', 'on'))
                assert all(isinstance(pools[-1][c][k], int) and pools[-1][c][k] >= 0
                           for c in ('off', 'on') for k in ('hits', 'misses'))
                changes.append((rates[-1]['on'] / rates[-1]['off'] - 1) * 100)
                deltas.append({k: pools[-1]['on'][k] - pools[-1]['off'][k] for k in ('hits', 'misses')})
            cases[name] = dict(paired_rates=rates, paired_decode_pool=pools,
                               effects_percent=changes, decode_pool_deltas=deltas,
                               mean_rates={c: statistics.mean(r[c] for r in rates) for c in ('off', 'on')},
                               mean_percent=statistics.mean(changes),
                               two_se_percent=2 * statistics.stdev(changes) / math.sqrt(rounds) if rounds > 1 else None,
                               mean_decode_pool_delta={k: statistics.mean(d[k] for d in deltas) for k in ('hits', 'misses')})
        if 'quality_pairs' in meta:
            assert cases == summary['quality_cases'], 'repeated quality summary differs from raw'
        assert 'decode_gate' not in summary, 'quality cannot claim standard decode gate'
        result['quality_cases'] = cases
        # Custom work/seeds remain screening evidence, not the prospective gate.
        # The checked per-case seed offsets require Japanese first, then code.
        result['repeated_warm_gate'] = (rounds == 6 and meta['token_limit'] == 128
            and meta['seed'] == 20261012 and list(cases) == ['japanese', 'code']
            and all(c['mean_percent'] >= 0 for c in cases.values()))
    if effects:
        mean = statistics.mean(effects)
        band = 2 * statistics.stdev(effects) / math.sqrt(len(effects)) if len(effects) > 1 else None
        full = rounds == 6 and meta['token_limit'] == 128
        gate = full and mean >= 5 and band is not None and mean > band and all(x > 0 for x in effects)
        assert effects == summary['effects_percent']
        assert mean == summary['mean_percent'] and band == summary['two_se_percent']
        assert gate == summary['decode_gate']
        result.update(effects_percent=effects, mean_percent=mean, two_se_percent=band,
                      decode_gate=gate, all_pairs_win=all(x > 0 for x in effects),
                      mean_off_rate=statistics.mean(r['gen_rate'] for r in arms if r['arm'].endswith('-off')),
                      mean_on_rate=statistics.mean(r['gen_rate'] for r in arms if r['arm'].endswith('-on')))
    result['passed'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = {'analysis_sha256': digest(__file__), 'experiments': [audit(p) for p in args.directories]}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps([{k: r[k] for k in ('directory', 'feature', 'mode', 'passed', 'actual_tokens_equal')}
                      for r in result['experiments']]))


if __name__ == '__main__':
    main()
