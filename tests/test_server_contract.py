"""HTTP contracts; only the expensive model calculation is replaced."""
import http.client
import json
import socket
import threading
import time
import unittest
from types import SimpleNamespace

from macqwen.sampling import Sampling
from macqwen.server import MacqwenHTTPServer, ModelService
from macqwen.tests.test_server import FakeSession


class BackendSession(FakeSession):
    def __init__(self):
        super().__init__(["hello"])
        self.preferences["thinking_enabled"] = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.delay = False
        self.calls = 0
        self.model_threads = []
        self.backend.sampling = Sampling.greedy_settings()
        self.backend.generate = self.generate

    def generate(self, max_tokens, out=None, should_cancel=None,
                 on_prefill_progress=None):
        self.model_threads.append(threading.current_thread())
        self.calls += 1
        self.started.set()
        if on_prefill_progress:
            on_prefill_progress(0, len(self.backend.pending))
        until = time.monotonic() + 3
        while self.delay and time.monotonic() < until and not self.release.wait(.01):
            if should_cancel and should_cancel():
                self.cancelled.set()
                raise RuntimeError("generation cancelled")
        setting = self.backend.sampling
        text = (f"temperature={setting.temperature};top_p={setting.top_p};"
                f"presence_penalty={setting.presence_penalty};max={max_tokens}")
        self.backend.tape.extend(self.backend.pending)
        self.backend.pending = []
        self.backend.tape.extend(self.backend.encode(text))
        if out:
            out(text)
        return text, SimpleNamespace(finish="stop", tokens=3, prompt_tokens=14)


class ServerContracts(unittest.TestCase):
    def setUp(self):
        self.session = BackendSession()
        self.service = ModelService(self.session)
        self.server = MacqwenHTTPServer(("127.0.0.1", 0), self.service, "test-key")
        self.server.heartbeat_interval = .03
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.session.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(1)

    def payload(self, **overrides):
        return dict(model="macqwen-flashnext", messages=[{"role":"user", "content":"hi"}],
                    max_tokens=8, **overrides)

    def request(self, payload=None, path="/v1/chat/completions", key="test-key", timeout=4):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port,
                                                timeout=timeout)
        body = json.dumps(payload).encode() if payload is not None else None
        connection.request("POST" if payload is not None else "GET", path, body,
                           {"Authorization": f"Bearer {key}", "Content-Type":"application/json"})
        reply = connection.getresponse()
        result = reply.status, dict(reply.getheaders()), reply.read()
        connection.close()
        return result

    def test_rejects_excessive_output_instead_of_running_it(self):
        payload = self.payload()
        payload["max_tokens"] = 1025
        self.assertEqual(self.request(payload)[0], 400)
        self.assertEqual(self.session.calls, 0)

    def test_rejects_full_template_overflow_before_starting_stream(self):
        self.session.backend.tokenizer.rendered = "p" * 4090
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.assertEqual(self.request(self.payload(stream=stream))[0], 400)
        self.assertEqual(self.session.calls, 0)

    def test_exact_context_boundary_is_accepted(self):
        self.session.backend.tokenizer.rendered = "p" * 4088
        self.assertEqual(self.request(self.payload())[0], 200)

    def test_rejects_images_instead_of_silently_discarding_them(self):
        payload = self.payload()
        payload["messages"][0]["content"] = [{"type":"image_url", "image_url":{"url":"https://example.invalid/x.png"}}]
        self.assertEqual(self.request(payload)[0], 400)
        self.assertEqual(self.session.calls, 0)

    def test_rejects_unknown_model_and_unsupported_options(self):
        for override in ({"model":"another-model"}, {"frequency_penalty":1}, {"n":2}):
            payload = self.payload()
            payload.update(override)
            self.assertEqual(self.request(payload)[0], 400)
        self.assertEqual(self.session.calls, 0)

    def test_sampling_options_reach_generation_and_do_not_leak(self):
        status, _, body = self.request(self.payload(temperature=.3, top_p=.6, presence_penalty=.5))
        self.assertEqual(status, 200)
        text = json.loads(body)["choices"][0]["message"]["content"]
        self.assertEqual(text, "temperature=0.3;top_p=0.6;presence_penalty=0.5;max=8")
        _, _, body = self.request(self.payload())
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"],
                         "temperature=0.7;top_p=0.8;presence_penalty=1.5;max=8")

    def test_busy_health_is_responsive_and_second_inference_is_rejected(self):
        self.session.delay = True
        first = threading.Thread(target=lambda: self.request(self.payload()), daemon=True)
        first.start()
        self.assertTrue(self.session.started.wait(1))
        health = busy = None
        try:
            try:
                health = self.request(path="/health", timeout=.25)
                busy = self.request(self.payload(), timeout=.25)
            except (TimeoutError, socket.timeout):
                pass
            self.assertIsNotNone(health, "health was blocked behind inference")
            self.assertEqual(health[0], 200)
            self.assertEqual(json.loads(health[2])["status"], "busy")
            self.assertEqual(busy[0], 503)
            self.assertEqual(busy[1].get("Retry-After"), "5")
        finally:
            self.session.release.set()
            first.join(1)

    def open_stream(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=.3)
        connection.request("POST", "/v1/chat/completions", json.dumps(self.payload(stream=True)),
                           {"Authorization":"Bearer test-key", "Content-Type":"application/json"})
        reply = connection.getresponse()
        self.assertEqual(reply.status, 200)
        return connection, reply

    def test_heartbeat_is_sent_during_prefill(self):
        self.session.delay = True
        connection, reply = self.open_stream()
        observed = False
        try:
            try:
                until = time.monotonic() + .25
                while time.monotonic() < until:
                    if reply.readline().startswith(b": generating"):
                        observed = True
                        break
            except (TimeoutError, socket.timeout):
                pass
            self.assertTrue(observed, "no heartbeat during blocked prefill")
        finally:
            reply.close()
            connection.close()

    def test_client_disconnect_cancels_inference_and_next_request_works(self):
        self.session.delay = True
        connection, reply = self.open_stream()
        reply.readline()
        reply.close()
        connection.close()
        self.assertTrue(self.session.cancelled.wait(1), "disconnected inference kept running")
        # Cancellation is observed before request cleanup releases the engine.
        until = time.monotonic() + 1
        ready = False
        while time.monotonic() < until:
            status, _, body = self.request(path="/health", timeout=.25)
            if status == 200 and json.loads(body)["status"] == "ready":
                ready = True
                break
            time.sleep(.005)
        self.assertTrue(ready, "cancelled inference did not release the engine within one second")
        self.session.delay = False
        self.assertEqual(self.request(self.payload())[0], 200)

    def test_deadline_stops_work_and_releases_the_engine(self):
        self.session.delay = True
        self.server.request_timeout = .05
        self.assertEqual(self.request(self.payload())[0], 408)
        self.session.delay = False
        self.assertEqual(self.request(self.payload())[0], 200)

    def test_authentication_remains_required(self):
        self.assertEqual(self.request(self.payload(), key="incorrect")[0], 401)
        self.assertEqual(self.session.calls, 0)

    def test_conflicting_output_limit_aliases_are_rejected(self):
        self.assertEqual(self.request(self.payload(max_completion_tokens=4))[0], 400)
        self.assertEqual(self.session.calls, 0)

    def test_usage_counts_the_complete_rendered_prompt(self):
        status, _, body = self.request(self.payload())
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["usage"], {
            "prompt_tokens": 15, "completion_tokens": 3, "total_tokens": 18,
        })

    def test_model_turns_use_one_persistent_worker(self):
        for _ in range(2):
            self.assertEqual(self.request(self.payload())[0], 200)
        self.assertEqual(len(self.session.model_threads), 2)
        self.assertIs(self.session.model_threads[0], self.session.model_threads[1],
                      "native MLX cache cannot move to a fresh thread each turn")


if __name__ == "__main__":
    unittest.main()
