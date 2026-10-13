#!/usr/bin/env python3
"""Rebuild a source variant from the public upstream pin and one verified patch."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

PROJECT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((PROJECT/'sources.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['all', *manifest['variants']], default='api')
    parser.add_argument('--upstream', type=Path, help='existing local upstream Git checkout')
    args = parser.parse_args()
    runtime = PROJECT/'.runtime'
    runtime.mkdir(exist_ok=True)
    upstream = args.upstream or runtime/'upstream'
    if not upstream.exists():
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',
                        manifest['upstream'],str(upstream)],check=True)
    base = manifest['upstream_revision']
    subprocess.run(['git','-C',str(upstream),'cat-file','-e',base+'^{commit}'],check=True)
    snapshot = subprocess.check_output(['git','-C',str(upstream),'archive',base])
    labels = list(manifest['variants']) if args.variant == 'all' else [args.variant]
    for label in labels:
        metadata = manifest['variants'][label]
        patch = PROJECT/metadata['patch']
        assert hashlib.sha256(patch.read_bytes()).hexdigest() == metadata['sha256']
        target = runtime/'api' if label == 'api' else runtime/'variants'/label
        if target.exists():
            if json.loads((target/'.source.json').read_text()) == metadata:
                print(f'Already prepared: {label}')
                continue
            raise SystemExit(f'Refusing to replace existing source: {target}')
        target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.prepare-',dir=target.parent) as temporary:
            stage = Path(temporary)
            with tarfile.open(fileobj=io.BytesIO(snapshot)) as archive:
                archive.extractall(stage,filter='data')
            subprocess.run(['git','init','-q',str(stage)],check=True)
            subprocess.run(['git','-C',str(stage),'apply','--check',str(patch)],check=True)
            subprocess.run(['git','-C',str(stage),'apply',str(patch)],check=True)
            shutil.rmtree(stage/'.git')
            (stage/'.source.json').write_text(json.dumps(metadata,indent=2)+'\n')
            stage.rename(target)
        print(f'Prepared: {label} -> {target}')


if __name__ == '__main__':
    main()
