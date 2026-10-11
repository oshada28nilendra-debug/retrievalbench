# Build status — 2026-10-11

Destination: oshada28nilendra-debug/retrievalbench, feature branch `feat/retrievalbench-mvp`.

Implemented: UTF-8 text/Markdown paragraph ingestion, validated relevance annotations and corpus fingerprints, TF-IDF and pinned pretrained MiniLM ONNX retrieval, Recall@k/MRR@k/hit rates, per-query misses, latency and provenance export, local CLI and browser explorer.

Verified locally:
- Clean Python 3.12 virtual environment installed exact dependencies from requirements-lock.txt and the editable package.
- 28 Python tests passed, including real-model paraphrase retrieval and truncation disclosure, metric calculations, malformed inputs, HTTP guards and CLI behavior.
- Headless Chromium 133 browser checks passed: both-method synthetic demo, 32 query/method rows, expanded ranked evidence, JSON download, mobile overflow, invalid upload and no JavaScript errors. Desktop and mobile screenshots inspected. Browser timing differs from committed CLI measurements because these are separate runs.
- CLI evaluated 20 synthetic passages / 16 labelled questions at k=1 and k=3. Reports include synthetic provenance. k=3 TF-IDF Recall 0.8125, MRR 0.625; MiniLM Recall and MRR 1.0. See report JSON for timing and environment details. No real-user adoption or production performance claimed.
- Read-only review found omitted synthetic provenance in CLI reports; fixed and regression-tested. No other must-fix findings reported.

GitHub Actions workflow is included. Publication and remote CI status must be confirmed through the PR; local passing tests do not imply remote CI passed. No merge or deployment performed.

Setup and interview explanation: README.md. No paid API or service is needed. First MiniLM use needs public model downloads; offline mode requires a populated cache.
