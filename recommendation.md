# Recommendations for production deployment

LocalSage is built for a single laptop. None of the design choices below are wrong for that
audience, but several would have to change to host this for multiple users in a real
environment. This document walks through each layer.

## Vector store
**Today.** Chroma persistent client on a mounted volume, single process, single collection
with a `type` metadata field.

**Production.** Move to a hosted vector DB (Qdrant, Weaviate, or pgvector if you already
run Postgres). Reasons:
- Concurrent writers — Chroma's persistent client locks the SQLite file.
- Replication and snapshots — production needs point-in-time restores.
- Filtered hybrid search — pgvector + a `tsvector` column gives you BM25 + cosine in one
  query, which improves recall on rare terms (e.g. "Pyramids of Giza" beats embeddings
  on exact-match scoring).

If volumes stay small (<1M vectors) Qdrant on a single small VM is the lowest-friction
upgrade.

## Embedder
**Today.** `all-MiniLM-L6-v2` on CPU. ~7ms/query, ~30ms/chunk batched.

**Production.**
- For mostly-English content: stay on MiniLM, but pin a specific commit hash and bake the
  weights into the image so cold-start is deterministic.
- For multilingual content (e.g. Turkish Wikipedia per the user's example URL): switch to
  `intfloat/multilingual-e5-small` or `BAAI/bge-m3`. Same dim ballpark, far better
  cross-lingual recall.
- For higher accuracy at the cost of latency: `bge-large-en-v1.5` on a small GPU.
- Add a reranker (e.g. `bge-reranker-base`) between retrieval and the LLM. The router +
  oversample pattern in `retriever.py` is exactly the place to insert it.

## LLM
**Today.** Ollama with `gemma3n:e2b` on the host's CPU. Streaming via `/api/chat`. One
model loaded at a time.

**Production.**
- Replace Ollama with vLLM or TGI behind an internal load balancer. Both keep weights in
  GPU memory across requests, and both expose an OpenAI-compatible API so the client code
  in `src/llm.py` is a 20-line swap.
- Pin model versions (e.g. `gemma-3n-e2b-2026-01`) so behavior doesn't drift after a `pull`.
- Add request-level timeouts and a circuit breaker. Today the CLI just hangs if Ollama
  stalls.
- Cache responses keyed by `(question, retrieved_chunk_ids, model_version, temperature)`.
  Highly repeated questions (e.g. "Who was Einstein?") become free.

## Ingestion
**Today.** Synchronous: `/wiki <query>` blocks until fetch + chunk + embed + upsert is
done. One page at a time.

**Production.**
- Move ingestion behind a job queue (RQ, Celery, or just SQS + a worker). The CLI/API
  enqueues and returns immediately.
- Persist raw HTML so you can re-chunk or re-embed without re-fetching when you tune
  parameters or change the embedder.
- Track provenance: page revision ID, fetch timestamp, embedder version, chunker config.
  When you change any of those, you can decide whether to re-ingest or keep the old
  vectors.
- Respect robots.txt and rate-limit Wikipedia fetches per the API guidance. The bootstrap
  script's 0.4s sleep is a placeholder, not a real policy.

## Retrieval and prompting
**Today.** Cosine over MiniLM embeddings, top-k with min-similarity floor, light layer
preference.

**Production.**
- **Hybrid retrieval.** Combine vector similarity with BM25/keyword. For named entities
  ("Pyramids of Giza"), keyword wins; for abstract questions ("Who fought against
  occupation in India"), embeddings win.
- **Cross-encoder rerank.** Take 30 candidates, rerank to 5. Quality jump is large.
- **Query rewriting.** A small LLM call to expand the user's query into 2-3 search-friendly
  variants before embedding lifts recall on terse questions.
- **Stricter grounding.** Force the model to emit citations as JSON, then verify each
  citation actually appears in the passage it points to. If verification fails, downgrade
  to "I don't know."

## Auth, multi-tenancy, and data isolation
**Today.** Single user. Anything goes.

**Production.**
- One Chroma collection per tenant, or per-row tenant ID with metadata filtering enforced
  server-side (never client-side).
- A real auth layer in front of any HTTP API (today there is none).
- PII review on ingest. Wikipedia is fine; user-provided documents are not.

## Observability
**Today.** `print` statements and a CLI that shows source passages.

**Production.**
- Structured logs (JSON) with request IDs that span ingest → retrieve → generate.
- Per-request metrics: retrieval latency, generation latency, tokens generated, top-k
  similarity scores, "I don't know" rate.
- Sample 1% of conversations for offline quality review.

## Cost and ops
**Today.** Free, on your laptop.

**Production.** The dominant cost will be GPU time for the LLM, not storage or embedding.
A single L4 or A10 GPU on a 4–7B parameter model will serve hundreds of QPS comfortably
with vLLM. The embedder can stay CPU. Chroma → Qdrant on a 2-vCPU VM handles tens of
millions of vectors before you have to think hard.

## Migration path summary
1. Bake model weights into the image; pin embedder and LLM versions.
2. Swap Chroma for Qdrant or pgvector.
3. Put ingestion behind a queue; persist raw HTML and provenance.
4. Add a cross-encoder reranker after retrieval.
5. Replace Ollama with vLLM/TGI; add response cache.
6. Add auth, tenant isolation, structured logs, metrics.
7. Front the whole thing with a thin HTTP API; keep the CLI as a power-user client.
