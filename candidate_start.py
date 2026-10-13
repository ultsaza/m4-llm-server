"""Candidate-only entrypoint reusing the unchanged production launcher."""
import argparse
import importlib.util
import os
import json
from pathlib import Path
import sys


def install_cache_validation(expert_pool, expected_policy, expected_fraction, expected_reset=None, expected_s3fifo=False):
    from benchmark_backend import validate_cache_policy, validate_cache_fraction, session_reset_boundary, validate_cache_s3fifo
    original_initialize = expert_pool.initialize_sidecar
    original_constructor = expert_pool.ExpertPool.__init__

    def validate(store, pool, phase):
        policy = validate_cache_policy(store, pool, expected_policy, require_explicit=True)
        event = dict(event='candidate_cache_validation', phase=phase, policy=policy)
        if expected_s3fifo:
            state = validate_cache_s3fifo(store, pool, expected_policy)
            if pool is not None and state is not None:
                event['s3fifo'] = state
        else:
            event['protected_fraction'] = validate_cache_fraction(store, pool, expected_fraction)
        if expected_reset is not None:
            event.update(session_reset_boundary(store, pool, expected_reset))
        if pool is not None:
            event['slots'] = pool.slots
            if not expected_s3fifo:
                event['protected_capacity'] = pool.policy_diagnostics()['protected_capacity']
        print(json.dumps(event), flush=True)

    def initialize(store):
        result = original_initialize(store)
        validate(store, None, 'startup')
        return result

    def constructor(pool, store, *args, **kwargs):
        original_constructor(pool, store, *args, **kwargs)
        try:
            validate(store, pool, 'runtime')
        except BaseException:
            pool.close()
            raise

    expert_pool.initialize_sidecar = initialize
    expert_pool.ExpertPool.__init__ = constructor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from benchmark_backend import (FEATURES, S3FIFO_FEATURES, S3FIFO_REVISION, FRACTION_REVISION, SESSION_REVISION, SESSION_RESET_FEATURE, validated_fraction,
                                   validate_source_selection, feature_environment)
    parser.add_argument('--feature', choices=FEATURES, default='sidecar')
    parser.add_argument('--slru-protected-fraction', type=validated_fraction, default=0.7)
    parser.add_argument('--slru-reset-on-session', action='store_true')
    parser.add_argument('--source-revision')
    parser.add_argument('--condition', choices=('off', 'on'), required=True)
    parser.add_argument('--sidecar', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    release = Path(__file__).resolve().parent
    if args.feature in S3FIFO_FEATURES or args.slru_reset_on_session or args.slru_protected_fraction != 0.7 or args.source_revision is not None:
        try:
            validate_source_selection(release / 'src', args.source_revision, args.feature,
                                      args.slru_protected_fraction, args.slru_reset_on_session)
        except ValueError as error:
            parser.error(str(error))
    memory = json.loads((release / 'deployment.json').read_text())['memory']
    if memory != dict(expert_pool_gb=6, mlx_cache_mib=512, wired_gb=0):
        raise RuntimeError('candidate requires unchanged pool6/cache512/wired0 memory policy')
    for key in list(os.environ):
        if key.startswith('FLASHNEXT_'):
            del os.environ[key]
    os.environ.update(FLASHNEXT_EXPERT_POOL_LOCK='0', FLASHNEXT_IO_WORKERS='8',
                      FLASHNEXT_STREAM_PACK_CHUNK='2', FLASHNEXT_PROFILE_IO='0')
    os.environ.update(feature_environment(args.feature, args.condition, args.sidecar.resolve(), args.slru_protected_fraction, args.slru_reset_on_session))
    if args.source_revision in (FRACTION_REVISION, SESSION_REVISION, S3FIFO_REVISION) and not args.check_only:
        sys.path.insert(0, str(release / 'src'))
        from models.flashnext import expert_pool
        selected = feature_environment(args.feature, args.condition, args.sidecar.resolve(), args.slru_protected_fraction, args.slru_reset_on_session)
        install_cache_validation(expert_pool, selected.get('FLASHNEXT_EXPERT_POOL_CACHE_POLICY', 'lru'),
            float(selected.get('FLASHNEXT_EXPERT_POOL_SLRU_PROTECTED_FRACTION', 0.7)),
            expected_reset=(SESSION_RESET_FEATURE in selected) if args.source_revision == SESSION_REVISION else None,
            expected_s3fifo=args.source_revision == S3FIFO_REVISION)
    spec = importlib.util.spec_from_file_location('candidate_production_launcher', release / 'start.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    # Preserve production logs: the owned job captures candidate stdout/stderr.
    launcher._configure_logging = lambda: None
    sys.argv = ['start.py'] + (['--check-only'] if args.check_only else [])
    return launcher.main()


if __name__ == '__main__':
    raise SystemExit(main())
