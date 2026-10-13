# Measured results

Historical measurements on one M4 Mac mini, 16 GiB RAM, with the pinned model/runtime. Pool: 6 decimal GB; MLX allocator cache: 512 MiB; wired limit: 0; read workers: 8; chunk: 2. These are observations, not general performance guarantees.

| Case | Input tokens | Time | Result |
|---|---:|---:|---|
| Long input | 7,918 | 103.45 s | Three unknown identifiers recovered |
| Long input | 16,128 | 247.49 s | Three unknown identifiers recovered |
| Native Codex auto-compaction task | 10,737 at compaction | 357.23 s total | Two file reads and final answer completed |
| Compaction within that task | 10,737 | 195.25 s | Subsequent inputs reduced to about 3,000 tokens |
| Resume after server restart | — | 99.57 s | Previous value retained; new file actually read |

Long-input tests used server-local HTTP; Codex cases used a verification relay to the real HTTPS endpoint. At 16k, MLX allocation peaked at 29.62 GB and sampled system swap at 18,274.88 MiB. These differ from process RSS and physical RAM usage.

Six fresh-process pairs, 128 greedy tokens and identical token IDs measured sidecar +3.67%, native dispatch −0.07%, SLRU +2.70%, and sidecar + SLRU +6.03%. Warmed code increased cache misses with the combined candidate, so it was rejected. S3-FIFO was also rejected after warmed-code misses increased. Experimental patches remain available; none were adopted.

The final namespace fix passed 166 local tests, 75 API/startup tests and 36 engine-server tests on CPython 3.12.15. Native TUI tool execution was verified with synthetic generation; a separate real-model TUI text request completed over HTTPS in 53.06 s. Tests distinguish transport, real tool execution and model behavior. Private raw records are retained locally rather than published.

The portable public snapshot passed 167 local tests. All seven rebuilt source variants matched their preserved revision bytes. Experimental API runs need a separate release with an adapter hash matching that variant; the default setup installs the qualified API variant only.
