#!/usr/bin/env python3
"""Render an uninstalled LaunchAgent for a prepared release."""
import argparse
import os
from pathlib import Path
import plistlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path.home()/'llm/macqwen')
parser.add_argument('--output',type=Path,required=True)
args = parser.parse_args()
root = args.root.expanduser().resolve()
release = root/'releases/api'
config = {
    'Label':'org.macqwen.m4', 'ProgramArguments':['/usr/bin/caffeinate','-is',
        str(release/'venv/bin/python'),'-u',str(release/'start.py')],
    'WorkingDirectory':str(release),'RunAtLoad':True,'KeepAlive':True,
    'ThrottleInterval':30,'ProcessType':'Interactive',
    'EnvironmentVariables':{'MACQWEN_ROOT':str(root),'PYTHONUNBUFFERED':'1',
        'PATH':'/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin'},
    'StandardOutPath':'/dev/null','StandardErrorPath':'/dev/null',
}
with args.output.expanduser().open('xb') as output:
    plistlib.dump(config,output)
print(f'Rendered: {args.output}; bootstrap with launchctl in gui/{os.getuid()}.')
