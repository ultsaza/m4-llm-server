"""Installed Codex v2 auto-compaction with synthetic model/usage and real tools."""
import contextlib
import http.client
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from types import SimpleNamespace

from macqwen.sampling import Sampling
from macqwen.server import MacqwenHTTPServer, MacqwenHandler, ModelService
from macqwen.tests.test_server import FakeSession
from codex_http_fixture import native_cli_outcome

ROOT=Path(__file__).resolve().parents[1]
KEY='owned-native-compaction-fixture-key'


class NativeCompaction(unittest.TestCase):
    def exercise(self, fallback=False, cycles=2):
        if not shutil.which('codex'):
            self.fail('installed Codex CLI is required for the native acceptance gate')
        session=FakeSession([])
        session.preferences['thinking_enabled']=False
        session.backend.sampling=Sampling.greedy_settings()
        stats={'input':11000,'output':40}
        original=session.backend.generate
        def generate(max_tokens,out=None,should_cancel=None,on_prefill_progress=None):
            if on_prefill_progress:on_prefill_progress(0,len(session.backend.pending))
            text,_=original(max_tokens,out)
            return text,SimpleNamespace(finish='stop',prompt_tokens=stats['input'],tokens=stats['output'])
        session.backend.generate=generate
        state={'reads':0,'compactions':0};records=[];wire_calls={}
        value_name='next-owned-value.txt';marker='NATIVE_V2_OWNED_FINAL_731942'
        first=value_name+'\nnonsecret owned status output\n'
        class Handler(MacqwenHandler):
            def _payload(self):
                payload=super()._payload()
                records.append({'path':self.path,'request':payload,'events':[]})
                inputs=payload.get('input',[])
                compact=any(i.get('type')=='compaction_trigger' for i in inputs)
                if compact:
                    state['compactions']+=1
                    call=wire_calls['first'] if state['reads']==1 else 'call_second'
                    value=value_name if state['reads']==1 else marker
                    if not any(i.get('type')=='function_call_output' and i.get('call_id')==call
                               and value in i.get('output','') for i in inputs):
                        raise ValueError('compaction did not receive the actual matched tool output')
                    session.backend.pieces=['b' if fallback else
                        'Task: read the pending file '+value_name+'. The first read is complete; preserve the observed value '+value+'. Continue the outstanding user request.']
                    stats.update(input=9000,output=40)
                elif state['reads']==0:
                    state['reads']=1
                    session.backend.pieces=['<tool_call><function=exec_command>'
                        '<parameter=cmd>cat fixture.txt</parameter>'
                        '<parameter=max_output_tokens>64</parameter></function></tool_call>']
                    stats.update(input=11000,output=40)
                elif state['reads']==1:
                    if state['compactions']!=1 or not any(i.get('type')=='compaction' for i in inputs):
                        raise ValueError('native automatic v2 compaction was not replayed')
                    state['reads']=2
                    session.backend.pieces=['<tool_call>'+json.dumps({'id':'call_second','name':'exec_command',
                        'arguments':{'cmd':'cat '+value_name,'max_output_tokens':64}})+'</tool_call>']
                    stats.update(input=11000 if cycles==2 else 2500,output=40)
                else:
                    if state['compactions']!=cycles:
                        raise ValueError('wrong automatic compaction count')
                    if not any(i.get('type')=='compaction' for i in inputs):
                        raise ValueError('missing continued encrypted history')
                    session.backend.pieces=[marker]
                    stats.update(input=2500,output=10)
                session.backend.tokenizer.rendered='x'*stats['input']
                return payload
            def _sse(self,value,event=None):
                records[-1]['events'].append(value)
                if value.get('type')=='response.output_item.done' and value.get('item',{}).get('type')=='function_call':
                    item=value['item']
                    if 'cat fixture.txt' in item['arguments']:
                        wire_calls['first']=item['call_id']
                return super()._sse(value,event)
        server=MacqwenHTTPServer(('127.0.0.1',0),ModelService(session),KEY,
                                context_limit=16384,compaction_enabled=True)
        server.RequestHandlerClass=Handler
        thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True)
        artifact_root=Path(os.environ.get('CONTEXT_ARTIFACT_DIR',ROOT/'.artifacts/native-compaction'))
        artifact_root.mkdir(parents=True,exist_ok=True)
        artifact=Path(tempfile.mkdtemp(prefix='native-v2-',dir=artifact_root))
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp)/'home';home.mkdir();(home/'.codex').mkdir()
            (home/'.codex/config.toml').write_text('sandbox_mode="danger-full-access"\napproval_policy="never"\n')
            work=Path(temp)/'work';work.mkdir();(work/'fixture.txt').write_text(first)
            (work/value_name).write_text(marker+'\n')
            thread.start()
            try:
                connection=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
                connection.request('GET','/v1/models',headers={'Authorization':'Bearer '+KEY})
                catalog=connection.getresponse();data=json.loads(catalog.read());connection.close()
                cat=Path(temp)/'catalog.json';cat.write_text(json.dumps(data))
                install=subprocess.run(['python3',str(ROOT/'codex/install-client.py'),'--home',str(home),
                    '--catalog',str(cat)],input=KEY+'\n',text=True,capture_output=True)
                self.assertEqual(install.returncode,0,install.stderr)
                command=[str(home/'.local/bin/codex-m4'),'--strict-config','exec','--skip-git-repo-check','--json',
                    '-c','model_providers.m4_qwen.base_url='+json.dumps(f'http://127.0.0.1:{server.server_port}/v1'),
                    'Read fixture.txt with the shell. Then read the file named in its first line. Return the actual second file contents.']
                environment=dict(os.environ,CODEX_HOME=str(home/'.codex'))
                with contextlib.redirect_stdout(io.StringIO()):
                    run=subprocess.run(command,cwd=work,env=environment,text=True,capture_output=True,
                                       timeout=40,stdin=subprocess.DEVNULL)
            finally:
                server.shutdown();server.server_close();thread.join(2)
            cli=[json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]
            outcome=native_cli_outcome(cli,run.returncode,expected_final=marker)
            (artifact/'cli-stdout.jsonl').write_text(run.stdout);(artifact/'cli-stderr.txt').write_text(run.stderr)
            (artifact/'requests-events.json').write_text(json.dumps(records,indent=2))
            packets=[i for r in records for e in r['events'] if e.get('type')=='response.completed'
                     for i in e['response'].get('output',[]) if i.get('type')=='compaction']
            notes=[server.compaction_codec.open(p['encrypted_content']) for p in packets]
            summary={'synthetic_model_and_usage':True,'real_native_shell_execution':True,
                     'state':state,'expected_compactions':cycles,'fallback':fallback,**outcome}
            (artifact/'summary.json').write_text(json.dumps(summary,indent=2))
            (artifact/'owned-note-audit.json').write_text(json.dumps(notes,indent=2))
            self.assertTrue(outcome['native_success'],run.stdout+run.stderr)
            self.assertEqual(state,{'reads':2,'compactions':cycles})
            self.assertEqual(len(packets),cycles)
            self.assertIn(value_name,notes[0]['summary'])
            self.assertIn(marker,notes[-1]['summary'] if cycles==2 else json.dumps(records))
            calls=[e['item'] for e in cli if e.get('type')=='item.completed'
                   and e.get('item',{}).get('type')=='command_execution']
            self.assertEqual(len(calls),2)
            self.assertTrue(all(i['exit_code']==0 for i in calls))
            self.assertIn(value_name,calls[0]['aggregated_output']);self.assertIn(marker,calls[1]['aggregated_output'])

    def test_native_two_automatic_compactions_continue_real_commands(self):self.exercise()
    def test_native_degenerate_summary_extracts_continue_real_commands(self):self.exercise(fallback=True,cycles=1)
