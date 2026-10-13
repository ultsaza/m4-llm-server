"""Responses HTTP contracts with only expensive generation replaced."""
import http.client
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from macqwen.sampling import Sampling
from macqwen.server import MAX_BODY, MacqwenHTTPServer, ModelService, _responses_messages
from macqwen.tests.test_server import FakeSession, FakeTokenizer


FUNCTION = {"type": "function", "name": "shell_command", "description": "Run a command",
            "strict": False, "parameters": {"type": "object", "properties": {
                "command": {"type": "string"}, "count": {"type": "integer"}},
                "required": ["command"]}}


class PinnedSystemTokenizer(FakeTokenizer):
    """Enforce the system-position rule captured from the pinned template."""
    def apply_chat_template(self, messages, **options):
        self.attempted_messages = messages
        for index, message in enumerate(messages):
            if message["role"] == "system" and index != 0:
                raise ValueError("System message must be at the beginning.")
        for message in messages:
            for call in message.get("tool_calls", []):
                function = call.get("function", call)
                arguments = function.get("arguments", "")
                if arguments != "" and not isinstance(arguments, dict):
                    raise TypeError("Can only get item pairs from a mapping.")
        return super().apply_chat_template(messages, **options)


class ResponseSession(FakeSession):
    def __init__(self):
        super().__init__(["hello ", "world"])
        self.preferences["thinking_enabled"] = False
        self.backend.tokenizer = PinnedSystemTokenizer()
        self.backend.sampling = Sampling.greedy_settings()
        self.backend.generate = self.generate
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.delay = False
        self.block_after_first = False
        self.failure = None
        self.calls = 0
        self.threads = []
        self.settings = []

    def wait(self, should_cancel):
        until = time.monotonic() + 3
        while time.monotonic() < until and not self.release.wait(.005):
            if should_cancel and should_cancel():
                self.cancelled.set()
                raise RuntimeError("generation cancelled")

    def generate(self, max_tokens, out=None, should_cancel=None, on_prefill_progress=None):
        self.calls += 1
        self.threads.append(threading.current_thread())
        self.settings.append(self.backend.sampling)
        self.started.set()
        if on_prefill_progress:
            on_prefill_progress(0, len(self.backend.pending))
        if self.delay:
            self.wait(should_cancel)
        for index, piece in enumerate(self.backend.pieces):
            if out:
                out(piece)
            if self.block_after_first and index == 0:
                self.wait(should_cancel)
        if self.failure:
            raise RuntimeError(self.failure)
        text = "".join(self.backend.pieces)
        self.backend.tape.extend(self.backend.pending)
        self.backend.pending = []
        self.backend.tape.extend(self.backend.encode(text))
        return text, SimpleNamespace(finish="stop", tokens=3, prompt_tokens=4)


class ResponsesContracts(unittest.TestCase):
    def setUp(self):
        self.session = ResponseSession()
        self.service = ModelService(self.session)
        self.server = MacqwenHTTPServer(("127.0.0.1", 0), self.service, "test-key")
        self.server.heartbeat_interval = .025
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.session.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(1)

    def payload(self, **kwargs):
        return dict({"model": "macqwen-flashnext", "input": "hi", "max_output_tokens": 8}, **kwargs)

    def connect(self, payload=None, path="/v1/responses", key="test-key", timeout=4, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=timeout)
        body = json.dumps(payload).encode() if payload is not None else None
        request_headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        request_headers.update(headers or {})
        connection.request("POST" if payload is not None else "GET", path, body, request_headers)
        return connection, connection.getresponse()

    def request(self, *args, **kwargs):
        connection, reply = self.connect(*args, **kwargs)
        result = reply.status, dict(reply.getheaders()), reply.read()
        connection.close()
        return result

    def events(self, payload):
        status, _, body = self.request(payload)
        self.assertEqual(status, 200, body)
        return [json.loads(line[6:]) for line in body.splitlines() if line.startswith(b"data: ")]

    def test_text_json_uses_full_prompt_usage_and_developer_content(self):
        payload = self.payload(instructions="Be brief", input=[
            {"role": "developer", "content": [{"type": "input_text", "text": "Policy"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Earlier"}]},
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]}])
        status, _, body = self.request(payload)
        self.assertEqual(status, 200, body)
        response = json.loads(body)
        self.assertEqual(response["output"][0]["content"][0]["text"], "hello world")
        self.assertEqual(response["usage"], {"input_tokens": 15, "output_tokens": 3, "total_tokens": 18})
        self.assertEqual([m["role"] for m in self.session.backend.tokenizer.messages],
                         ["system", "assistant", "user"])
        self.assertEqual(self.session.backend.tokenizer.messages[0]["content"], "Be brief\n\nPolicy")
        self.assertIsNotNone(self.server.last_request)

    def test_absent_input_can_use_nonempty_instructions(self):
        status, _, _ = self.request({"model": "macqwen-flashnext", "instructions": "Say hi"})
        self.assertEqual(status, 200)

    def test_ordered_text_sse_completes_with_consistent_ids_and_usage(self):
        events = self.events(self.payload(stream=True))
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds[:4], ["response.created", "response.in_progress",
                                  "response.output_item.added", "response.content_part.added"])
        self.assertEqual(kinds[-4:], ["response.output_text.done", "response.content_part.done",
                                    "response.output_item.done", "response.completed"])
        self.assertEqual([e["sequence_number"] for e in events], list(range(1, len(events) + 1)))
        self.assertEqual("".join(e["delta"] for e in events if e["type"] == "response.output_text.delta"), "hello world")
        final = events[-1]["response"]
        self.assertEqual(final["usage"]["input_tokens"], 15)
        self.assertEqual(final["output"][0]["id"], events[2]["item"]["id"])
        self.assertEqual(final["status"], "completed")

    def test_plain_text_with_tools_streams_before_generation_finishes(self):
        self.session.block_after_first = True
        connection, reply = self.connect(self.payload(stream=True, tools=[FUNCTION]), timeout=.5)
        try:
            self.assertEqual(reply.status, 200)
            observed = None
            try:
                while True:
                    line = reply.readline()
                    if line.startswith(b"data: "):
                        value = json.loads(line[6:])
                        if value["type"] == "response.output_text.delta":
                            observed = value["delta"]
                            break
            except (TimeoutError, socket.timeout):
                pass
            self.assertEqual(observed, "hello", "text held until generation ended")
        finally:
            self.session.release.set()
            reply.close()
            connection.close()

    def test_split_qwen_tags_are_hidden_and_typed_native_calls_replay_exactly(self):
        self.session.backend.pieces = ["Inspecting ", "<to", "ol_call><function=shell_command>",
            "<parameter=command>printf '&lt;x&gt;'</parameter>",
            "<parameter=count>3</parameter></function></tool_call>"]
        events = self.events(self.payload(stream=True, tools=[FUNCTION], parallel_tool_calls=False,
                                          tool_choice="auto", store=False))
        final = events[-1]["response"]
        self.assertEqual(final["output"][0]["content"][0]["text"], "Inspecting")
        visible = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
        self.assertEqual(visible, "Inspecting")
        call = final["output"][1]
        self.assertEqual(json.loads(call["arguments"]), {"command": "printf '<x>'", "count": 3})
        self.assertTrue(call["call_id"])
        self.assertNotEqual(call["id"], call["call_id"])
        args_events = [e for e in events if "function_call_arguments" in e["type"]]
        self.assertEqual([e["type"] for e in args_events], ["response.function_call_arguments.delta", "response.function_call_arguments.done"])
        self.assertEqual(args_events[0]["delta"], call["arguments"])
        self.session.backend.pieces = ["Done"]
        replay = [{"role": "user", "content": "Inspect"}, call,
                  {"type": "function_call_output", "call_id": call["call_id"], "output": " exact\noutput "}]
        status, _, body = self.request(self.payload(input=replay, tools=[FUNCTION]))
        self.assertEqual(status, 200, body)
        self.assertEqual(json.loads(body)["output"][0]["content"][0]["text"], "Done")
        history = self.session.backend.tokenizer.messages
        self.assertEqual(history[-2]["tool_calls"][0]["id"], call["call_id"])
        self.assertEqual(history[-2]["tool_calls"][0]["function"]["arguments"], {"command": "printf '<x>'", "count": 3})
        canonical = _responses_messages(self.payload(input=replay, tools=[FUNCTION]))
        self.assertEqual(canonical[-2]["tool_calls"][0]["function"]["arguments"], call["arguments"])
        self.assertEqual(history[-1], {"role": "tool", "tool_call_id": call["call_id"], "content": " exact\noutput "})

    def test_transport_hints_and_empty_tools_are_accepted(self):
        self.assertEqual(self.request(self.payload(tools=[], tool_choice="none", store=False,
            include=["reasoning.encrypted_content"], reasoning={"effort": "none", "summary": "auto"},
            prompt_cache_key="key", client_metadata={"session_id": "test"},
            background=False, truncation="disabled", previous_response_id=None))[0], 200)

    def test_malformed_and_unsupported_options_are_400_before_generation(self):
        invalid = [{"model": "other"}, {"stream": "true"}, {"max_output_tokens": 1025},
            {"max_output_tokens": True}, {"max_tokens": 8}, {"instructions": {}},
            {"input": {}}, {"input": []}, {"input": ""}, {"input": [None]},
            {"input": [{"type": "reasoning"}]}, {"input": [{"role": "tool", "content": "x"}]},
            {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}]},
            {"input": [{"role": "user", "content": [{"type": "input_text", "text": 42}]}]},
            {"input": [{"role": "user", "content": "   "}]},
            {"tools": [{"type": "function", "name": "x", "description": []}]},
            {"tools": [{"type": "function", "name": "x", "parameters": {"required": "x"}}]},
            {"tools": [{"type": "custom", "name": "x"}]}, {"tools": [{"type": "function", "name": "x", "parameters": []}]},
            {"tools": [{"type": "function", "name": "x", "parameters": {"properties": {"x": 3}}}]},
            {"tools": [{"type": "function", "name": "x", "strict": "false"}]},
            {"tool_choice": "required"}, {"tool_choice": {"type": "function", "name": "x"}},
            {"parallel_tool_calls": "true"}, {"parallel_tool_calls": 0}, {"store": True}, {"store": 0},
            {"include": "reasoning.encrypted_content"}, {"include": ["other"]},
            {"reasoning": {"effort": "high"}}, {"reasoning": []}, {"reasoning": {"summary": True}},
            {"reasoning": {"unknown": "x"}}, {"prompt_cache_key": []}, {"client_metadata": []},
            {"background": True}, {"previous_response_id": "resp_old"}, {"conversation": "old"},
            {"truncation": "auto"}, {"text": {"format": {"type": "json_schema"}}}, {"temperature": True}]
        for override in invalid:
            with self.subTest(override=override):
                status, _, body = self.request(self.payload(**override))
                self.assertEqual(status, 400, body)
        self.assertEqual(self.session.calls, 0)

    def test_unmatched_duplicate_and_invalid_replay_are_rejected(self):
        call = {"type": "function_call", "call_id": "call_1", "name": "shell_command", "arguments": '{ "command": "ls" }'}
        output = {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
        invalid = [[output], [call], [call, output, output], [call, call, output],
                   [dict(call, arguments="[]"), output], [dict(call, arguments="bad"), output],
                   [dict(call, call_id=""), output], [dict(call, name=1), output],
                   [call, dict(output, output={})]]
        for items in invalid:
            self.assertEqual(self.request(self.payload(input=items))[0], 400, items)
        self.assertEqual(self.session.calls, 0)

    def test_context_error_has_compaction_code_and_precedes_sse(self):
        self.session.backend.tokenizer.rendered = "p" * 4090
        for stream in (False, True):
            status, _, body = self.request(self.payload(stream=stream))
            self.assertEqual(status, 400, body)
            self.assertEqual(json.loads(body)["error"]["code"], "context_length_exceeded")
        self.assertEqual(self.session.calls, 0)
        self.session.backend.tokenizer.rendered = "p" * 4088
        self.assertEqual(self.request(self.payload())[0], 200)

    def test_auth_and_body_size_statuses_remain_distinct(self):
        self.assertEqual(self.request(self.payload(), key="wrong")[0], 401)
        self.assertEqual(self.request(self.payload(), headers={"Content-Length": str(MAX_BODY + 1)})[0], 413)
        self.assertEqual(self.session.calls, 0)

    def test_namespace_functions_keep_distinct_names_and_wire_identity(self):
        namespaces = [{"type": "namespace", "name": prefix, "tools": [FUNCTION]}
                      for prefix in ("worktrees", "other")]
        arguments = '{ "command": "ls", "count": 3 }'
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.session.backend.pieces = ['<tool_call>' + json.dumps({
                    "id": "call_namespace", "name": "worktrees.shell_command",
                    "arguments": {"command": "ls", "count": 3}}) + '</tool_call>']
                payload = self.payload(tools=[FUNCTION, *namespaces], stream=stream)
                before = json.loads(json.dumps(payload))
                if stream:
                    events = self.events(payload)
                    self.assertEqual(events[-1]["type"], "response.completed")
                    call = events[-1]["response"]["output"][0]
                    added = next(e["item"] for e in events if e["type"] == "response.output_item.added")
                    self.assertEqual(added["namespace"], "worktrees")
                    self.assertEqual(added["id"], call["id"])
                else:
                    status, _, body = self.request(payload)
                    self.assertEqual(status, 200, body)
                    call = json.loads(body)["output"][0]
                self.assertEqual(payload, before)
                self.assertEqual(call["name"], "shell_command")
                self.assertEqual(call["namespace"], "worktrees")
                self.assertEqual(call["call_id"], "call_namespace")
                self.assertEqual(json.loads(call["arguments"]), json.loads(arguments))
                self.assertEqual([t["function"]["name"] for t in self.session.backend.tokenizer.options["tools"]],
                                 ["shell_command", "worktrees.shell_command", "other.shell_command"])

    def test_namespaced_replay_preserves_exact_arguments_and_call_id(self):
        arguments = '{ "command": "ls", "count": 3 }'
        inputs = [{"role": "user", "content": "Inspect"},
                  {"type": "function_call", "call_id": "call_ns", "name": "shell_command",
                   "namespace": "worktrees", "arguments": arguments},
                  {"type": "function_call_output", "call_id": "call_ns", "output": "actual result\n"},
                  {"role": "user", "content": "Done?"}]
        before = json.loads(json.dumps(inputs))
        self.session.backend.pieces = ["Done"]
        status, _, body = self.request(self.payload(input=inputs, tools=[
            {"type": "namespace", "name": "worktrees", "tools": [FUNCTION]}]))
        self.assertEqual(status, 200, body)
        self.assertEqual(inputs, before)
        raw = _responses_messages({"input": inputs})
        call = raw[1]["tool_calls"][0]
        self.assertEqual(call["id"], "call_ns")
        self.assertEqual(call["function"], {"name": "worktrees.shell_command", "arguments": arguments})
        self.assertEqual(raw[2]["content"], "actual result\n")
        rendered = next(m["tool_calls"][0] for m in self.session.backend.tokenizer.messages
                        if m.get("tool_calls"))
        self.assertEqual(rendered["function"]["arguments"], json.loads(arguments))

    def test_namespace_validation_rejects_bad_members_and_ambiguous_names_before_generation(self):
        ns = {"type": "namespace", "name": "worktrees", "tools": [FUNCTION]}
        invalid = [[dict(ns, name="")], [dict(ns, name=3)], [dict(ns, description=3)],
                   [dict(ns, tools=[])], [dict(ns, tools={})],
                   [dict(ns, tools=[{"type": "custom", "name": "freeform"}])],
                   [dict(ns, tools=[ns])], [dict(ns, tools=[FUNCTION, FUNCTION])],
                   [ns, dict(FUNCTION, name="worktrees.shell_command")]]
        for tools in invalid:
            with self.subTest(tools=tools):
                self.assertEqual(self.request(self.payload(tools=tools))[0], 400)
        for namespace in ("", 3, "bad/name"):
            items = [{"type": "function_call", "call_id": "bad_ns", "name": "shell_command",
                      "namespace": namespace, "arguments": '{}'},
                     {"type": "function_call_output", "call_id": "bad_ns", "output": "result"}]
            self.assertEqual(self.request(self.payload(input=items))[0], 400)
        self.assertEqual(self.session.calls, 0)

    def test_namespace_call_cannot_fall_back_to_another_same_leaf_function(self):
        self.session.backend.pieces = ['<tool_call>' + json.dumps({
            "id": "call_wrong_namespace", "name": "other.shell_command",
            "arguments": {"command": "ls"}}) + '</tool_call>']
        tools = [{"type": "namespace", "name": "worktrees", "tools": [FUNCTION]}]
        status, _, body = self.request(self.payload(tools=tools))
        self.assertEqual(status, 500, body)

    def test_namespaced_text_content_result_blocks_preserve_literal_output(self):
        inputs = [{"role": "user", "content": "Inspect"},
                  {"type": "function_call", "call_id": "call_blocks", "name": "shell_command",
                   "namespace": "worktrees", "arguments": '{}'},
                  {"type": "function_call_output", "call_id": "call_blocks", "output": [
                      {"type": "input_text", "text": " exact\n"},
                      {"type": "input_text", "text": "result "}]},
                  {"role": "user", "content": "Done?"}]
        before = json.loads(json.dumps(inputs))
        self.session.backend.pieces = ["Done"]
        for stream in (False, True):
            payload = self.payload(input=inputs, stream=stream)
            if stream:
                self.assertEqual(self.events(payload)[-1]["type"], "response.completed")
            else:
                self.assertEqual(self.request(payload)[0], 200)
            self.assertEqual(inputs, before)
            result = next(m for m in self.session.backend.tokenizer.messages if m.get("role") == "tool")
            self.assertEqual(result["content"], " exact\nresult ")
            self.assertEqual(result["tool_call_id"], "call_blocks")
        self.session.calls = 0
        for output in ([], [{"type": "input_image", "image_url": "https://fixture.invalid/image"}],
                       [{"type": "input_text", "text": 3}]):
            bad = [*inputs[:2], dict(inputs[2], output=output), inputs[3]]
            self.assertEqual(self.request(self.payload(input=bad))[0], 400)
        self.assertEqual(self.session.calls, 0)

    def test_busy_rejects_second_request_and_health_stays_responsive(self):
        self.session.delay = True
        first = threading.Thread(target=lambda: self.request(self.payload()), daemon=True)
        first.start()
        self.assertTrue(self.session.started.wait(1))
        try:
            status, _, body = self.request(path="/health", timeout=.3)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["status"], "busy")
            status, headers, _ = self.request(self.payload(), timeout=.3)
            self.assertEqual(status, 503)
            self.assertEqual(headers["Retry-After"], "5")
        finally:
            self.session.release.set()
            first.join(1)

    def test_prefill_heartbeat_and_disconnect_cancellation_allow_recovery(self):
        self.session.delay = True
        connection, reply = self.connect(self.payload(stream=True), timeout=.5)
        try:
            self.assertEqual(reply.status, 200)
            while not reply.readline().startswith(b": generating"):
                pass
        finally:
            reply.close()
            connection.close()
        self.assertTrue(self.session.cancelled.wait(1))
        # Notification precedes handler cleanup releasing the inference lock.
        until = time.monotonic() + 1
        ready = False
        while time.monotonic() < until:
            status, _, body = self.request(path='/health', timeout=.25)
            if status == 200 and json.loads(body)['status'] == 'ready':
                ready = True
                break
            time.sleep(.005)
        self.assertTrue(ready, 'cancelled inference did not release the engine within one second')
        self.session.delay = False
        self.assertEqual(self.request(self.payload())[0], 200)

    def test_deadline_and_generation_failures_never_complete_streams(self):
        self.session.delay = True
        self.server.request_timeout = .05
        self.assertEqual(self.request(self.payload())[0], 408)
        events = self.events(self.payload(stream=True))
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertIn("deadline", events[-1]["response"]["error"]["message"])
        self.assertNotIn("response.completed", [e["type"] for e in events])
        self.session.delay = False
        self.session.failure = "backend broke"
        self.assertEqual(self.request(self.payload())[0], 500)
        events = self.events(self.payload(stream=True))
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertIn("backend broke", events[-1]["response"]["error"]["message"])
        self.assertEqual([e["sequence_number"] for e in events], list(range(1, len(events)+1)))
        self.session.failure = None
        self.assertEqual(self.request(self.payload())[0], 200)

    def test_sampling_restores_and_all_turns_use_the_same_model_owner_thread(self):
        original = self.session.backend.sampling
        self.assertEqual(self.request(self.payload(temperature=.3, top_p=.6))[0], 200)
        self.assertIs(self.session.backend.sampling, original)
        self.assertEqual(self.session.settings[-1].temperature, .3)
        self.assertEqual(self.request(self.payload())[0], 200)
        self.assertEqual(self.session.settings[-1].temperature, .7)
        self.assertIs(self.session.threads[0], self.session.threads[1])
        self.assertIsNot(self.session.threads[0], threading.current_thread())
        self.session.failure = "failure"
        self.assertEqual(self.request(self.payload(temperature=.2))[0], 500)
        self.assertIs(self.session.backend.sampling, original)

    def test_chat_tools_keep_native_calls_hidden_and_share_model_owner(self):
        self.session.backend.pieces = ["Ready ", "<to", "ol_call><function=shell_command>",
            "<parameter=command>ls</parameter></function></tool_call>"]
        status, _, body = self.request({"model": "macqwen-flashnext", "messages": [
            {"role": "user", "content": "Inspect"}], "tools": [{"type": "function", "function": {
                "name": "shell_command", "parameters": FUNCTION["parameters"]}}],
            "max_tokens": 8, "stream": True}, path="/v1/chat/completions")
        self.assertEqual(status, 200)
        chunks = [json.loads(line[6:]) for line in body.splitlines()
                  if line.startswith(b"data: ") and line != b"data: [DONE]"]
        self.assertEqual("".join(c["choices"][0]["delta"].get("content", "") for c in chunks), "Ready")
        call = next(c["choices"][0]["delta"]["tool_calls"][0] for c in chunks
                    if "tool_calls" in c["choices"][0]["delta"])
        self.assertEqual(json.loads(call["function"]["arguments"]), {"command": "ls"})
        self.session.backend.pieces = ["ok"]
        self.assertEqual(self.request(self.payload())[0], 200)
        self.assertIs(self.session.threads[0], self.session.threads[1])

    def test_replay_preserves_literal_json_arguments_and_call_identity(self):
        arguments = '{ "command" : "printf \\"x\\"", "count": 1 }'
        # Literal whitespace survives the transport and normalization unchanged.
        call = {"type": "function_call", "id": "fc_test", "call_id": "call_test",
                "name": "shell_command", "arguments": arguments}
        status, _, body = self.request(self.payload(input=[call, {"type": "function_call_output",
            "call_id": "call_test", "output": "\n exact \n"}], tools=[FUNCTION]))
        self.assertEqual(status, 200, body)
        history = self.session.backend.tokenizer.messages
        self.assertEqual(history[-2]["tool_calls"][0]["function"]["arguments"], {"command": 'printf "x"', "count": 1})
        canonical = _responses_messages(self.payload(input=[call, {"type": "function_call_output",
            "call_id": "call_test", "output": "\n exact \n"}], tools=[FUNCTION]))
        self.assertEqual(canonical[-2]["tool_calls"][0]["function"]["arguments"], arguments)
        self.assertEqual(history[-2]["tool_calls"][0]["id"], "call_test")
        self.assertEqual(history[-1]["content"], "\n exact \n")

    def test_xml_number_budget_retains_integer_json_for_native_cli(self):
        function = {'type': 'function', 'name': 'exec_command', 'parameters': {
            'type': 'object', 'properties': {'cmd': {'type': 'string'},
                                            'max_output_tokens': {'type': 'number'}}}}
        self.session.backend.pieces = ['<tool_call><function=exec_command>',
            '<parameter=cmd>cat fixture.txt</parameter>',
            '<parameter=max_output_tokens>12000</parameter></function></tool_call>']
        events = self.events({'input': 'Read the file.', 'tools': [function], 'stream': True})
        self.assertEqual(events[-1]['type'], 'response.completed')
        calls = [e['item'] for e in events if e['type'] == 'response.output_item.done'
                 and e['item']['type'] == 'function_call']
        arguments = json.loads(calls[0]['arguments'])
        self.assertIs(type(arguments['max_output_tokens']), int)
        self.assertEqual(arguments['max_output_tokens'], 12000)

    def test_incomplete_tool_protocol_fails_without_emitting_xml(self):
        self.session.backend.pieces = ["Before ", "<tool_call><function=shell_command>",
                                       "<parameter=command>ls"]
        events = self.events(self.payload(stream=True, tools=[FUNCTION]))
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertNotIn("response.completed", [e["type"] for e in events])
        visible = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
        self.assertNotIn("<tool_call>", visible)

    def test_function_definitions_keep_strict_schema_and_nested_compatibility(self):
        status, _, body = self.request(self.payload(tools=[FUNCTION]))
        self.assertEqual(status, 200, body)
        rendered = self.session.backend.tokenizer.options["tools"][0]["function"]
        self.assertIs(rendered["strict"], False)
        self.assertEqual(rendered["parameters"], FUNCTION["parameters"])
        nested = {"type": "function", "function": dict(FUNCTION)}
        self.assertEqual(self.request(self.payload(tools=[nested]))[0], 200)

    def test_explicit_nontext_instructions_and_invalid_replay_ids_are_rejected(self):
        nested_schema = {}
        for _ in range(80):
            nested_schema = {"properties": {"x": nested_schema}}
        for override in ({"tools": [dict(FUNCTION, parameters=nested_schema)]},
            {"input": [{"type": "function_call", "call_id": "call_1", "name": "x", "arguments": '{"x":NaN}'},
                       {"type": "function_call_output", "call_id": "call_1", "output": "ok"}]},
            {"instructions": None}, {"instructions": "   ", "input": []},
            {"input": [{"type": "function_call", "id": {}, "call_id": "call_1", "name": "x", "arguments": "{}"},
                       {"type": "function_call_output", "call_id": "call_1", "output": "ok"}]}):
            with self.subTest(override=override):
                self.assertEqual(self.request(self.payload(**override))[0], 400)

    def test_model_tool_output_without_tools_is_hidden_and_fails(self):
        self.session.backend.pieces = ["<tool_call>", '{"name":"shell_command","arguments":{"command":"ls"}}', "</tool_call>"]
        events = self.events(self.payload(stream=True))
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertNotIn("<tool_call>", "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta"))

    def test_invalid_generated_arguments_and_duplicate_ids_fail(self):
        bad_outputs = [
            '<tool_call>{"name":"shell_command","arguments":"[]"}</tool_call>',
            '<tool_call>{"id":"same","name":"shell_command","arguments":{}}</tool_call>' * 2,
            '<tool_call>{"name":"unknown","arguments":{}}</tool_call>']
        for text in bad_outputs:
            with self.subTest(text=text):
                self.session.backend.pieces = [text]
                events = self.events(self.payload(stream=True, tools=[FUNCTION], parallel_tool_calls=True))
                self.assertEqual(events[-1]["type"], "response.failed")
                self.assertNotIn("response.completed", [e["type"] for e in events])

    def test_true_parallel_permission_accepts_text_and_serializes_valid_multiple_calls(self):
        self.assertEqual(self.request(self.payload(parallel_tool_calls=True))[0], 200)
        self.session.backend.pieces = [
            '<tool_call>{"id":"one","name":"shell_command","arguments":{"command":"ls"}}</tool_call>',
            '<tool_call>{"id":"two","name":"shell_command","arguments":{"command":"pwd"}}</tool_call>']
        for stream in (False, True):
            with self.subTest(stream=stream):
                payload = self.payload(stream=stream, tools=[FUNCTION], parallel_tool_calls=True)
                if stream:
                    events = self.events(payload)
                    response = events[-1]["response"]
                    self.assertEqual(events[-1]["type"], "response.completed")
                    added = [e for e in events if e["type"] == "response.output_item.added"]
                    self.assertEqual([e["output_index"] for e in added], [0, 1])
                    self.assertEqual([e["sequence_number"] for e in events], list(range(1, len(events)+1)))
                else:
                    status, _, body = self.request(payload)
                    self.assertEqual(status, 200, body)
                    response = json.loads(body)
                self.assertEqual([item["call_id"] for item in response["output"]], ["one", "two"])
                self.assertEqual([json.loads(item["arguments"]) for item in response["output"]],
                                 [{"command": "ls"}, {"command": "pwd"}])
                self.assertEqual(len({item["id"] for item in response["output"]}), 2)

    def test_false_parallel_permission_rejects_multiple_model_calls(self):
        self.session.backend.pieces = [
            '<tool_call>{"id":"one","name":"shell_command","arguments":{}}</tool_call>',
            '<tool_call>{"id":"two","name":"shell_command","arguments":{}}</tool_call>']
        for options in ({}, {"parallel_tool_calls": False}):
            for stream in (False, True):
                with self.subTest(options=options, stream=stream):
                    payload = self.payload(stream=stream, tools=[FUNCTION], **options)
                    if stream:
                        events = self.events(payload)
                        self.assertEqual(events[-1]["type"], "response.failed")
                        self.assertIn("parallel", events[-1]["response"]["error"]["message"])
                        self.assertNotIn("response.completed", [e["type"] for e in events])
                    else:
                        self.assertEqual(self.request(payload)[0], 500)

    def test_invalid_selected_input_schema_is_rejected_before_generation(self):
        definitions = [dict(FUNCTION, input_schema=[]),
            dict(FUNCTION, input_schema={"properties": {"command": []}}),
            dict(FUNCTION, input_schema={"type": "array"})]
        for definition in definitions:
            for stream in (False, True):
                with self.subTest(definition=definition, stream=stream):
                    status, _, body = self.request(self.payload(stream=stream, tools=[definition]))
                    self.assertEqual(status, 400, body)
                    self.assertEqual(self.session.calls, 0)

    def test_valid_input_schema_alias_is_the_schema_used_for_native_types(self):
        definition = dict(FUNCTION, input_schema={"type": "object", "properties": {
            "count": {"type": "integer"}}}, parameters={"properties": {"count": {"type": "string"}}})
        self.session.backend.pieces = ['<tool_call><function=shell_command><parameter=count>3</parameter></function></tool_call>']
        status, _, body = self.request(self.payload(tools=[definition]))
        self.assertEqual(status, 200, body)
        self.assertEqual(json.loads(json.loads(body)["output"][0]["arguments"]), {"count": 3})
        schema = self.session.backend.tokenizer.options["tools"][0]["function"]["parameters"]
        self.assertEqual(schema, {"type": "object", "properties": {"count": {"type": "integer"}}})

    def test_every_malformed_complete_block_fails_responses(self):
        malformed = ['<tool_call>not valid protocol</tool_call>',
                     '<tool_call>{"arguments":{"command":"ls"}}</tool_call>',
                     '<tool_call>{"name":"","arguments":{}}</tool_call>']
        valid = '<tool_call>{"name":"shell_command","arguments":{"command":"ls"}}</tool_call>'
        for bad in malformed:
            for text in (bad, valid + bad, bad + valid):
                for stream in (False, True):
                    with self.subTest(text=text, stream=stream):
                        self.session.backend.pieces = [text]
                        payload = self.payload(stream=stream, tools=[FUNCTION], parallel_tool_calls=True)
                        if stream:
                            events = self.events(payload)
                            self.assertEqual(events[-1]["type"], "response.failed")
                            self.assertIn("protocol", events[-1]["response"]["error"]["message"])
                            self.assertNotIn("response.completed", [e["type"] for e in events])
                            self.assertNotIn("<tool_call>", "".join(e["delta"] for e in events
                                if e["type"] == "response.output_text.delta"))
                        else:
                            status, _, body = self.request(payload)
                            self.assertEqual(status, 500, body)
                            self.assertIn("protocol", json.loads(body)["error"]["message"])

    def test_unfinished_opening_marker_fails_json_and_sse_without_leaking(self):
        valid = '<tool_call>{"name":"shell_command","arguments":{"command":"ls"}}</tool_call>'
        for prefix in ("<tool_ca", "<tool_call", "<to"):
            for first in ("Before ", valid + "After "):
                for tools in ([], [FUNCTION]):
                    for stream in (False, True):
                        with self.subTest(prefix=prefix, first=first, tools=tools, stream=stream):
                            self.session.backend.pieces = [first, prefix]
                            payload = self.payload(stream=stream, tools=tools)
                            if stream:
                                events = self.events(payload)
                                self.assertEqual(events[-1]["type"], "response.failed")
                                self.assertIn("protocol", events[-1]["response"]["error"]["message"])
                                self.assertNotIn("response.completed", [e["type"] for e in events])
                                visible = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
                                self.assertNotIn(prefix, visible)
                            else:
                                status, _, body = self.request(payload)
                                self.assertEqual(status, 500, body)
                                self.assertIn("protocol", json.loads(body)["error"]["message"])

    def test_chat_tolerance_for_discarded_blocks_and_unfinished_prefix_is_preserved(self):
        for pieces, expected in ((["Before ", "<tool_call>not valid protocol</tool_call>"], "Before"),
                                 (["Before ", "<tool_ca"], "Before <tool_ca")):
            for stream in (False, True):
                with self.subTest(pieces=pieces, stream=stream):
                    self.session.backend.pieces = pieces
                    status, _, body = self.request({"model": "macqwen-flashnext", "messages": [
                        {"role": "user", "content": "Inspect"}], "tools": [{"type": "function", "function": {
                            "name": "shell_command", "parameters": FUNCTION["parameters"]}}],
                        "max_tokens": 8, "stream": stream}, path="/v1/chat/completions")
                    self.assertEqual(status, 200, body)
                    if stream:
                        self.assertIn(b"data: [DONE]", body)
                        self.assertNotIn(b'"error"', body)
                        chunks = [json.loads(line[6:]) for line in body.splitlines()
                                  if line.startswith(b"data: ") and line != b"data: [DONE]"]
                        visible = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
                        self.assertEqual(visible, expected)
                    else:
                        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], expected)

    def test_consecutive_native_developer_messages_render_one_leading_system(self):
        inputs = [{"type": "message", "id": "msg_a", "role": "developer", "content": [
                    {"type": "input_text", "text": " First "}]},
                  {"type": "message", "id": "msg_b", "role": "developer", "content": [
                    {"type": "input_text", "text": "Sec"}, {"type": "input_text", "text": "ond"}]},
                  {"role": "user", "content": " env "}, {"role": "user", "content": "go\n"}]
        for stream in (False, True):
            with self.subTest(stream=stream):
                payload = self.payload(input=inputs, stream=stream)
                if stream:
                    self.assertEqual(self.events(payload)[-1]["type"], "response.completed")
                else:
                    self.assertEqual(self.request(payload)[0], 200)
                self.assertEqual(self.session.backend.tokenizer.messages, [
                    {"role": "system", "content": " First \n\nSecond"},
                    {"role": "user", "content": " env "}, {"role": "user", "content": "go\n"}])

    def test_interleaved_system_roles_and_instructions_preserve_replay_order(self):
        inputs = [{"role": "system", "content": "SYS_A "},
                  {"role": "user", "content": " Inspect "},
                  {"role": "developer", "content": "DEV_B\n"},
                  {"type": "function_call", "id": "fc_original", "call_id": "call_original",
                   "name": "shell_command", "arguments": '{ "command":"ls" }'},
                  {"role": "system", "content": "SYS_C"},
                  {"type": "function_call_output", "call_id": "call_original", "output": " exact\n result "},
                  {"role": "assistant", "content": [{"type": "output_text", "text": " Earlier "}]},
                  {"role": "developer", "content": "DEV_D"},
                  {"role": "user", "content": "Finish"}]
        for stream in (False, True):
            with self.subTest(stream=stream):
                payload = self.payload(input=inputs, instructions=" Top ", tools=[FUNCTION], stream=stream)
                if stream:
                    self.assertEqual(self.events(payload)[-1]["type"], "response.completed")
                else:
                    self.assertEqual(self.request(payload)[0], 200)
                self.assertEqual(self.session.backend.tokenizer.messages, [
                    {"role": "system", "content": " Top \n\nSYS_A \n\nDEV_B\n\n\nSYS_C\n\nDEV_D"},
                    {"role": "user", "content": " Inspect "},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_original", "type": "function",
                        "function": {"name": "shell_command", "arguments": {"command": "ls"}}}]},
                    {"role": "tool", "tool_call_id": "call_original", "content": " exact\n result "},
                    {"role": "assistant", "content": " Earlier "},
                    {"role": "user", "content": "Finish"}])

    def test_chat_does_not_coalesce_roles_to_bypass_the_pinned_template_rule(self):
        status, _, body = self.request({"model": "macqwen-flashnext", "messages": [
            {"role": "system", "content": "First"}, {"role": "developer", "content": "Second"},
            {"role": "user", "content": "Hi"}], "max_tokens": 8}, path="/v1/chat/completions")
        self.assertEqual(status, 500)
        self.assertIn("System message must be at the beginning.", json.loads(body)["error"]["message"])
        self.assertEqual(self.session.calls, 0)
        self.assertEqual([message["role"] for message in self.session.backend.tokenizer.attempted_messages],
                         ["system", "system", "user"])

    def test_render_copy_decodes_nested_arguments_without_mutating_canonical_replay(self):
        arguments = ' { "command": "ls", "count": 3, "ratio": 1.25, "options": {"flags": [true, null, -2], "label": "été"}, "paths": ["a b", "文件"] }\n'
        inputs = [{"role": "developer", "content": "Policy"}, {"role": "user", "content": "Inspect"},
                  {"type": "function_call", "call_id": "call_nested", "name": "shell_command", "arguments": arguments},
                  {"type": "function_call_output", "call_id": "call_nested", "output": "  出力\nété  "},
                  {"role": "user", "content": "Finish"}]
        for stream in (False, True):
            with self.subTest(stream=stream):
                canonical_requests = []
                def observe_conversion(payload):
                    canonical = _responses_messages(payload)
                    canonical_requests.append(canonical)
                    return canonical
                with patch("macqwen.server._responses_messages", side_effect=observe_conversion):
                    payload = self.payload(input=inputs, tools=[FUNCTION], stream=stream)
                    if stream:
                        self.assertEqual(self.events(payload)[-1]["type"], "response.completed")
                    else:
                        status, _, body = self.request(payload)
                        self.assertEqual(status, 200, body)
                self.assertEqual(len(canonical_requests), 1)
                raw = canonical_requests[0]
                self.assertEqual(raw[2]["tool_calls"][0]["function"]["arguments"], arguments)
                self.assertEqual(raw[2]["tool_calls"][0]["id"], "call_nested")
                self.assertEqual(raw[3], {"role": "tool", "tool_call_id": "call_nested", "content": "  出力\nété  "})
                rendered = self.session.backend.tokenizer.messages
                self.assertEqual([message["role"] for message in rendered], ["system", "user", "assistant", "tool", "user"])
                self.assertEqual(rendered[2]["tool_calls"][0]["function"]["arguments"], {
                    "command": "ls", "count": 3, "ratio": 1.25,
                    "options": {"flags": [True, None, -2], "label": "été"}, "paths": ["a b", "文件"]})
                self.assertIsNot(rendered[2]["tool_calls"], raw[2]["tool_calls"])
                self.assertIsNot(rendered[2]["tool_calls"][0]["function"], raw[2]["tool_calls"][0]["function"])
                self.assertEqual(rendered[2]["tool_calls"][0]["id"], "call_nested")
                self.assertEqual(rendered[3], raw[3])
                self.assertEqual(inputs[2]["arguments"], arguments)

    def test_chat_replay_keeps_original_argument_representation(self):
        status, _, body = self.request({"model": "macqwen-flashnext", "messages": [
            {"role": "user", "content": "Inspect"}, {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_chat", "type": "function", "function": {"name": "shell_command", "arguments": '{ "command": "ls" }'}}]},
            {"role": "tool", "tool_call_id": "call_chat", "content": "Exact"}], "max_tokens": 8}, path="/v1/chat/completions")
        self.assertEqual(status, 500)
        self.assertIn("Can only get item pairs from a mapping.", json.loads(body)["error"]["message"])
        self.assertEqual(self.session.calls, 0)
        attempted = self.session.backend.tokenizer.attempted_messages
        self.assertEqual(attempted[-2]["tool_calls"][0]["function"]["arguments"], '{ "command": "ls" }')

    def test_models_catalog_advertises_only_supported_capabilities(self):
        status, _, body = self.request(path="/v1/models")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["data"][0]["id"], "macqwen-flashnext")
        card = data["models"][0]
        self.assertEqual(card["context_window"], 4096)
        self.assertEqual(card["default_reasoning_level"], "none")
        self.assertEqual([v["effort"] for v in card["supported_reasoning_levels"]], ["none"])
        self.assertFalse(card["supports_reasoning_summaries"])
        self.assertEqual(card["input_modalities"], ["text"])
        self.assertFalse(card["supports_parallel_tool_calls"])
        self.assertEqual(card["shell_type"], "shell_command")
        self.assertIsNone(card["apply_patch_tool_type"])
        self.assertIsNone(card["upgrade"])


if __name__ == "__main__":
    unittest.main()
