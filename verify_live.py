"""Exercise the loaded model and retain public smoke-test measurements."""
import os
import argparse
import json
from pathlib import Path
import subprocess
import threading
import time

import requests

ROOT = Path(os.environ.get("MACQWEN_ROOT", str(Path.home() / "llm" / "macqwen")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/live-api.json")
    args = parser.parse_args()
    headers = {"Authorization": "Bearer " + (ROOT / "config/api-key").read_text().strip()}
    measurements = []
    stopping = threading.Event()

    def resources():
        while not stopping.is_set():
            rss = subprocess.run(["ps", "-axo", "pid=,rss=,args="], capture_output=True,
                                 text=True, check=True).stdout
            processes = [{"pid": int(row.split()[0]), "rss_kib": int(row.split()[1])}
                         for row in rss.splitlines()
                         if "venv/bin/python" in row and "start.py" in row
                         and row.split()[2].endswith("/bin/python")]
            swap = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True,
                                  text=True, check=True).stdout.strip()
            snapshot = {"elapsed_seconds": round(time.monotonic() - began, 3),
                        "processes": processes, "swap": swap}
            if processes and len(measurements) % 6 == 0:
                footprint = subprocess.run(["/usr/bin/footprint", "-p", str(processes[0]["pid"])],
                                          capture_output=True, text=True, timeout=20)
                snapshot["physical_footprint"] = [line.strip() for line in footprint.stdout.splitlines()
                                                  if "phys_footprint" in line]
            measurements.append(snapshot)
            stopping.wait(5)

    def get(path, *, authenticate=True):
        response = requests.get(args.base_url + path, headers=headers if authenticate else {},
                                timeout=15)
        return response.status_code, response.json()

    def post(payload):
        started = time.monotonic()
        response = requests.post(args.base_url + "/v1/chat/completions", headers=headers,
                                 json=payload, timeout=(15, 1200))
        return response.status_code, response.json(), round(time.monotonic() - started, 3)

    def payload(messages, limit=128, **options):
        return dict(model="macqwen-flashnext", messages=messages, max_tokens=limit, **options)

    began = time.monotonic()
    sampler = threading.Thread(target=resources, daemon=True)
    sampler.start()
    report = {"endpoint": args.base_url, "checks": {}, "requests": []}
    try:
        assert get("/health", authenticate=False)[0] == 401
        assert get("/health")[1]["status"] == "ready"
        status, models = get("/v1/models")
        assert status == 200 and models["data"][0]["id"] == "macqwen-flashnext"
        report["checks"]["authentication_and_models"] = True
        messages = [{"role": "user", "content":
                     "日本語で、SSDに保存したMoEモデルを16GBのMacで推論する仕組みを、3文で説明してください。"}]
        status, result, elapsed = post(payload(messages, temperature=0))
        assert status == 200 and result["choices"][0]["message"]["content"]
        report["requests"].append({"name": "japanese", "elapsed_seconds": elapsed,
                                    "response": result, "health": get("/health")[1]})
        reused_before = report["requests"][-1]["health"]["cache_reused"]
        messages += [result["choices"][0]["message"],
                     {"role": "user", "content":
                      "その構成で推論速度に最も影響する要因は何ですか。1文で答えてください。"}]
        status, result, elapsed = post(payload(messages, 96, temperature=0))
        assert status == 200
        report["requests"].append({"name": "followup", "elapsed_seconds": elapsed,
                                    "response": result, "health": get("/health")[1]})
        assert report["requests"][-1]["health"]["cache_reused"] > reused_before, "conversation cache was not reused"
        report["checks"]["conversation_cache_reuse"] = True

        for name, invalid in (
            ("output_limit", payload(messages, 1025)),
            ("context_limit", payload([{"role":"user", "content":"入力上限検証 " * 20000}], 8)),
            ("images", payload([{"role":"user", "content":[{"type":"image_url",
                "image_url":{"url":"https://example.invalid/image.png"}}]}])),
        ):
            assert post(invalid)[0] == 400, name
            report["checks"][name] = True

        excerpt = "\n".join(f"row {number:03}: Python 3.12; model 4bit; memory 16GiB; cache 6GB."
                            for number in range(100))
        prompt = ("次の構成情報を参考に、LLMサーバーの運用時に確認する項目を20個、"
                  "番号付きで日本語で説明してください。\n" + excerpt)
        started = time.monotonic()
        response = requests.post(args.base_url + "/v1/chat/completions", headers=headers,
                                 json=payload([{"role":"user", "content":prompt}], 256,
                                              stream=True, temperature=0),
                                 stream=True, timeout=(15, 1200))
        assert response.status_code == 200, response.text
        status, health = get("/health")
        assert status == 200 and health["status"] == "busy"
        busy, _, _ = post(payload(messages, 8))
        assert busy == 503
        first_token = None
        text = []
        heartbeats = 0
        done = False
        for line in response.iter_lines(chunk_size=1):
            if line.startswith(b":"):
                heartbeats += 1
            if not line.startswith(b"data: "):
                continue
            if line == b"data: [DONE]":
                done = True
                break
            event = json.loads(line[6:])
            assert "error" not in event, event
            piece = event["choices"][0]["delta"].get("content", "")
            if piece:
                if first_token is None:
                    first_token = round(time.monotonic() - started, 3)
                text.append(piece)
        response.close()
        assert done and text
        report["checks"]["sse_busy_and_done"] = True
        report["requests"].append({"name": "longer_sse", "elapsed_seconds":
                                    round(time.monotonic() - started, 3),
                                    "first_visible_token_seconds": first_token,
                                    "heartbeats": heartbeats, "text": "".join(text),
                                    "health": get("/health")[1]})

        response = requests.post(args.base_url + "/v1/chat/completions", headers=headers,
                                 json=payload([{"role":"user", "content":
                                    "1から1000までの数を順番に、空白で区切って書いてください。"}],
                                              512, stream=True),
                                 stream=True, timeout=(15, 1200))
        assert response.status_code == 200
        next(response.iter_lines(chunk_size=1))
        response.close()
        until = time.monotonic() + 120
        while get("/health")[1]["status"] != "ready":
            assert time.monotonic() < until, "cancellation did not release the engine"
            time.sleep(.5)
        status, result, elapsed = post(payload([{"role":"user", "content":"2+3の答えだけを書いてください。"}], 16,
                                               temperature=0))
        assert status == 200
        report["checks"]["disconnect_recovery"] = True
        report["requests"].append({"name":"recovery", "elapsed_seconds": elapsed,
                                    "response": result, "health": get("/health")[1]})
        report["passed"] = True
    finally:
        stopping.set()
        sampler.join(6)
        report["resources"] = measurements
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": report["checks"],
                      "requests": [{"name": row["name"], "elapsed_seconds": row["elapsed_seconds"],
                                    "stats": row["health"].get("last_request")}
                                   for row in report["requests"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
