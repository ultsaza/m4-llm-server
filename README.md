# M4 Qwen server

MACQWEN patches and tooling for serving a 4-bit Qwen MoE model on an Apple M4 Mac mini with 16 GiB RAM. Expert weights stay on SSD; MLX/Metal performs inference.

- Authenticated Chat Completions, Responses and Anthropic-compatible HTTP APIs, with SSE and bounded requests.
- Codex function calls, namespace tools, text-result replay and encrypted remote compaction v2.
- A dedicated `codex-m4` profile: 16,384-token context, automatic compaction at 10,000 tokens. Files and shell commands run on the client.
- Opt-in SSD sidecar, native I/O, SLRU and S3-FIFO experiments. These candidates were measured but **not enabled in the deployed server**.

## Reproduce

The tested environment is CPython 3.12.15, MACQWEN 0.5.1, Q4/G32 weights, exact routing and MTP disabled. Source revisions, dependency hashes and model revision are pinned. Other versions are unverified.

```sh
python3 scripts/prepare-runtime.py --variant all
python3 -m pip install cryptography==50.0.2 requests numpy pexpect
PYTHONPATH=.runtime/api python3 -m unittest discover -s tests
```

GPU/model execution requires macOS on Apple Silicon. Native Codex tests additionally require Codex CLI 0.162.1; they skip when it is unavailable.

See [setup](docs/setup.md) and [measured results](docs/results.md). This is a source archive, not a hosted service. Model weights, credentials, private transcripts and machine-specific logs are not included.

## Limits

One inference at a time. Context includes instructions, tools, history and the output budget; output defaults to 512 tokens and is capped at 1,024. Long prefill and compaction can take several minutes and use swap. Compaction is lossy; keep important requirements in project files.

Images, custom/freeform tools, server-side conversation storage and structured output are unsupported. Codex's optional automatic title request uses structured output and may fail independently of normal chat. Arbitrary long coding tasks were not qualified.

MIT licensed. MACQWEN is by [1architect](https://github.com/1architect/macqwen-releases); its license is retained in [third_party](third_party/MACQWEN-LICENSE). Each patch in [sources.json](sources.json) applies independently to the same public upstream revision.
