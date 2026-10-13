"""Native Codex v2 compaction and expanded context through actual HTTP."""
import json
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from macqwen.server import MacqwenHTTPServer, ModelService
import test_responses_contract as response_contract

ResponseSession = response_contract.ResponseSession
FUNCTION = response_contract.FUNCTION


class CompactionContracts(unittest.TestCase):
    connect = response_contract.ResponsesContracts.connect
    request = response_contract.ResponsesContracts.request
    events = response_contract.ResponsesContracts.events

    def setUp(self):
        self.session = ResponseSession()
        self.session.backend.pieces = ['TASK: read the pending second file next_731942.txt. ',
                                       'The first file was read successfully. NEXT: cat next_731942.txt.']
        self.service = ModelService(self.session)
        self.server = MacqwenHTTPServer(('127.0.0.1', 0), self.service, 'test-key', compaction_enabled=True)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.session.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(1)

    def payload(self, **overrides):
        payload = {'model': 'macqwen-flashnext', 'input': [
            {'role': 'developer', 'content': 'Keep the user requirements.'},
            {'role': 'user', 'content': 'Read fixture.txt and then the file it names; report its value.'},
            {'type': 'function_call', 'name': 'shell_command', 'call_id': 'call_first',
             'arguments': '{"command":"cat fixture.txt"}'},
            {'type': 'function_call_output', 'call_id': 'call_first',
             'output': 'Exit code: 0\nnext_731942.txt\nnonsecret status output'},
            {'type': 'compaction_trigger'}], 'tools': [FUNCTION],
            'parallel_tool_calls': True}
        payload.update(overrides)
        return payload

    def compact(self, **overrides):
        status, _, raw = self.request(self.payload(**overrides))
        self.assertEqual(status, 200, raw)
        response = json.loads(raw)
        self.assertEqual(response['status'], 'completed')
        self.assertEqual([item['type'] for item in response['output']], ['compaction'])
        return response['output'][0]

    def test_native_trigger_produces_one_encrypted_item_and_greedy_data_summary(self):
        original = self.session.backend.sampling
        item = self.compact(temperature=1.2)
        self.assertIsInstance(item['encrypted_content'], str)
        self.assertNotIn('next_731942.txt', item['encrypted_content'])
        note = self.server.compaction_codec.open(item['encrypted_content'])
        self.assertIn('next_731942.txt', note['summary'])
        self.assertIn('Exit code: 0', note['summary'])
        self.assertEqual(self.session.settings[-1].temperature, 0)
        self.assertIs(self.session.backend.sampling, original)
        rendered = self.session.backend.tokenizer.messages
        self.assertEqual([m['role'] for m in rendered], ['system', 'user'])
        self.assertIn('cat fixture.txt', rendered[1]['content'])
        self.assertTrue(all(not m.get('tool_calls') for m in rendered))
        self.assertTrue(all(t.name.startswith('macqwen-http-model') for t in self.session.threads))

    def test_stream_has_exactly_one_compaction_done_and_completed(self):
        events = self.events(self.payload(stream=True))
        self.assertEqual(events[-1]['type'], 'response.completed')
        done = [e['item'] for e in events if e['type'] == 'response.output_item.done']
        self.assertEqual([i['type'] for i in done], ['compaction'])
        self.assertEqual(events[-1]['response']['output'], done)
        self.assertEqual([e['sequence_number'] for e in events], list(range(1, len(events)+1)))
        self.assertFalse(any(e['type'] == 'response.output_text.delta' for e in events))

    def test_replay_restores_note_deduplicates_only_covered_prior_users(self):
        item = self.compact()
        old = self.payload()['input'][1]
        fresh = {'role': 'user', 'content': 'Continue and return the actual value.'}
        self.session.backend.pieces = ['continued']
        status, _, raw = self.request({'input': [old, item, fresh], 'max_output_tokens': 8})
        self.assertEqual(status, 200, raw)
        contents = [m['content'] for m in self.session.backend.tokenizer.messages]
        self.assertIn(fresh['content'], contents)
        self.assertNotIn(old['content'], contents)
        self.assertTrue(any('next_731942.txt' in value for value in contents))
        restored = next(value for value in contents if '[Prior conversation notes]' in value)
        self.assertLess(restored.index(old['content']), restored.index('TASK:'))
        self.request({'input': [item, old], 'max_output_tokens': 8})
        self.assertIn(old['content'], [m['content'] for m in self.session.backend.tokenizer.messages])
        changed = dict(old, content=old['content']+' Also inspect third.txt.')
        self.request({'input': [changed, item, fresh], 'max_output_tokens': 8})
        self.assertIn(changed['content'], [m['content'] for m in self.session.backend.tokenizer.messages])

    def test_stateless_packet_survives_new_server_with_same_key_and_repeated_cycles(self):
        item = self.compact()
        replacement = MacqwenHTTPServer(('127.0.0.1', 0), ModelService(ResponseSession()), 'test-key', compaction_enabled=True)
        try:
            self.assertEqual(replacement.compaction_codec.open(item['encrypted_content']),
                             self.server.compaction_codec.open(item['encrypted_content']))
        finally:
            replacement.server_close()
        second = self.compact(input=[self.payload()['input'][1], item,
                {'role': 'user', 'content': 'Now retain this new constraint.'},
                {'type': 'compaction_trigger'}])
        plain = self.server.compaction_codec.open(second['encrypted_content'])
        self.assertIn('Now retain this new constraint.', plain['users'])
        self.assertIn('next_731942.txt', plain['summary'])
        self.assertEqual(len(plain['covered']), 2)

    def test_degenerate_truncated_and_tool_summary_use_disclosed_extracts(self):
        for text in ['b', '<tool_call>{"name":"shell_command","arguments":{}}</tool_call>',
                     'The first file was read; continue with next_731942.txt.\n'
                     '<function><name>shell_command</name><arguments>{}</arguments></function>']:
            with self.subTest(text=text):
                self.session.backend.pieces = [text]
                item = self.compact()
                plain = self.server.compaction_codec.open(item['encrypted_content'])
                self.assertIn('next_731942.txt', plain['summary'])
                self.assertTrue(any('Read fixture.txt' in text for text in plain['users']))
                self.assertNotIn('<tool_call>', plain['summary'])
                self.assertNotIn('<function>', plain['summary'])
                self.assertIn('cat fixture.txt', plain['summary'])
                self.assertIn('Exit code: 0', plain['summary'])
                self.assertEqual(self.server.last_request['compaction']['method'], 'extractive')

    def test_fallback_keeps_all_user_constraints_or_their_original_messages(self):
        self.session.backend.pieces = ['b']
        users = [{'role':'user','content': f'constraint {i}: use Python3.12; preserve file{i}.py; '+'条件保持'*50}
                 for i in range(40)]
        item = self.compact(input=[*users, {'type':'compaction_trigger'}])
        plain = self.server.compaction_codec.open(item['encrypted_content'])
        self.assertLessEqual(sum(len(t.encode()) for t in plain['users']),6000)
        self.request({'input':[*users,item,{'role':'user','content':'Continue.'}], 'max_output_tokens':8})
        visible=[m['content'] for m in self.session.backend.tokenizer.messages]
        for user in users:
            self.assertTrue(user['content'] in visible or any(user['content'] in t for t in visible))

    def test_tampered_wrong_key_and_oversized_packets_fail_before_inference(self):
        item = self.compact()
        calls = self.session.calls
        other = MacqwenHTTPServer(('127.0.0.1', 0), ModelService(ResponseSession()), 'other-key', compaction_enabled=True)
        try:
            foreign = other.compaction_codec.seal('Pending task', [])
        finally:
            other.server_close()
        for token in [item['encrypted_content'][:-8]+'AAAAAAAA', foreign, 'x'*200000]:
            with self.subTest(kind=len(token)):
                status, _, raw = self.request({'input': [{'type': 'compaction', 'encrypted_content': token}]})
                self.assertEqual(status, 400, raw)
        self.assertEqual(self.session.calls, calls)

    def test_invalid_controls_and_covered_images_are_rejected(self):
        bad = [[], [{'type': 'compaction_trigger'}],
               [{'type': 'compaction_trigger'}, {'role': 'user', 'content': 'after'}],
               [{'role': 'user', 'content': 'hi'}, {'type': 'compaction_trigger'}, {'type': 'compaction_trigger'}],
               [{'role': 'user', 'content': 'hi'}, {'type': 'compaction_trigger', 'extra': True}]]
        for items in bad:
            with self.subTest(items=items):
                self.assertEqual(self.request(self.payload(input=items))[0], 400)
        self.assertEqual(self.session.calls, 0)
        item = self.compact()
        image = {'role': 'user', 'content': [{'type': 'input_image','image_url':'about:blank'}]}
        self.assertEqual(self.request({'input':[image,item]})[0], 400)

    def test_backend_failure_is_not_compaction_success(self):
        self.session.failure = 'controlled backend failure'
        events = self.events(self.payload(stream=True))
        self.assertEqual(events[-1]['type'], 'response.failed')
        self.assertFalse(any(e['type'] == 'response.completed' for e in events))

    def test_memory_measurements_run_on_the_same_model_owner(self):
        threads = []
        def peak_reset(): threads.append(threading.current_thread())
        def measure():
            threads.append(threading.current_thread()); return 123
        fake = SimpleNamespace(reset_peak_memory=peak_reset, get_peak_memory=measure,
                               get_active_memory=measure, get_cache_memory=measure)
        with patch.dict('sys.modules', {'mlx.core': fake}):
            self.compact()
        self.assertEqual(self.server.last_request['mlx_peak_bytes'], 123)
        self.assertTrue(all(t is self.session.threads[-1] for t in threads))


class LargerContextContracts(unittest.TestCase):
    def test_expanded_exact_boundaries_catalog_and_invalid_limits(self):
        for limit in (8192,16384):
            session = ResponseSession()
            service = ModelService(session)
            server = MacqwenHTTPServer(('127.0.0.1',0), service, 'test-key', context_limit=limit)
            try:
                self.assertEqual(server.context_limit,limit)
                self.assertEqual(service.context_limit,limit)
                service.output_budget=8
                self.assertFalse(service._adopt('p'*(limit-8)))
                with self.assertRaises(ValueError):service._adopt('p'*(limit-7))
            finally:server.server_close()
        for limit in (True,0,-1,'8192',262145):
            with self.subTest(limit=limit),self.assertRaises(ValueError):
                MacqwenHTTPServer(('127.0.0.1',0), ModelService(ResponseSession()), 'test-key', context_limit=limit)
