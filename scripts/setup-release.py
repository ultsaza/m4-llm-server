#!/usr/bin/env python3
"""Create a pinned macOS release. Model download and service startup are separate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True, help='CPython 3.12.15 executable')
    parser.add_argument('--root', type=Path, default=Path.home()/'llm/macqwen')
    args = parser.parse_args()
    version = subprocess.check_output([args.python,'-c',
                'import sys; print(".".join(map(str, sys.version_info[:3])))'],text=True).strip()
    if version != '3.12.15':
        raise SystemExit('CPython 3.12.15 is required')
    root = args.root.expanduser().resolve()
    release = root/'releases/api'
    if release.exists():
        raise SystemExit('Refusing to replace an existing release')
    source = PROJECT/'.runtime/api'
    if not source.is_dir():
        raise SystemExit('Run scripts/prepare-runtime.py first')
    for name in ['config','logs','models','cache','workspace','releases']:
        (root/name).mkdir(parents=True,exist_ok=True)
    release.mkdir()
    shutil.copytree(source,release/'src')
    for path in PROJECT.glob('*.py'):
        shutil.copy2(path,release/path.name)
    shutil.copytree(PROJECT/'codex',release/'codex')
    manifest = json.loads((PROJECT/'deployment.json').read_text())
    manifest['release'] = 'api'
    actual = hashlib.sha256((release/'src/macqwen/server.py').read_bytes()).hexdigest()
    if actual != manifest['server_sha256']:
        raise SystemExit('Rebuilt HTTP adapter differs from the pinned implementation')
    (release/'deployment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    subprocess.run([args.python,'-m','venv',str(release/'venv')],check=True)
    python = str(release/'venv/bin/python')
    subprocess.run([python,'-m','pip','install','--require-hashes','-r',str(PROJECT/'requirements.lock')],check=True)
    subprocess.run([python,'-m','pip','install','--no-deps','--target',str(release/'api-deps'),
                    'cryptography==50.0.2'],check=True)
    prefs = root/'config/preferences.json'
    if not prefs.exists():
        prefs.write_text(json.dumps({'flashnext_gpu_keepwarm':True})+'\n')
    key = root/'config/api-key'
    if not key.exists():
        descriptor = os.open(key,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'w') as output:
            output.write(secrets.token_urlsafe(32)+'\n')
    env = dict(os.environ, MACQWEN_ROOT=str(root))
    subprocess.run([python,str(release/'start.py'),'--check-only'],env=env,check=True)
    print(f'Prepared release: {release}. Download and verify the model before starting it.')


if __name__ == '__main__':
    main()
