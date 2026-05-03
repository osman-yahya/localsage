# LocalSage — Product Requirements Document

## What it is
A local-first chat assistant that answers questions about famous people and places by
retrieving from a Wikipedia-derived knowledge base it has ingested itself. The whole stack
runs on a laptop: language model, embedder, vector store, CLI.

## Why it exists
Two reasons. First, the homework assignment: combine Project 1's index/retrieve work with
Project 2's AI workflows into a complete RAG application. Second, the scaffold should also
be a credible starting point for a "talk to your own corpus" tool — the Wikipedia source is
swappable.

## Goals
- Run with one command (`docker compose up`) on a clean laptop with no Python installed.
- Cover the homework's required 20 people + 20 places out of the box (`scripts/bootstrap.py`).
- Let a user ingest more pages on demand from the chat (`/wiki`, `/wiki-url`).
- Answer questions about the ingested corpus, citing the passages used, streaming tokens.
- Refuse cleanly with "I don't know." when the answer isn't in the corpus.
- Persist all state (vector DB, model weights, embedder cache, chat history) under one
  mounted directory so the project is portable.

## Non-goals
- Live web search. The system only knows what's been ingested.
- Multi-user / hosted deployment. Single-user, localhost only.
- Fine-tuning or training. We use off-the-shelf weights.
- Conversational memory across questions. Each question is answered independently from
  retrieval; this keeps the contract with the user simple ("the answer comes from the
  passages, not from earlier turns").

## Users and primary flows
A student / researcher who wants to query a curated knowledge base offline.

1. **Bootstrap** — `docker compose up`, then `scripts/bootstrap.py` seeds the homework's
   40 entities. ~2 minutes including the model download (after the model itself is pulled).
2. **Ad-hoc ingest** — at the prompt: `/wiki Marie Curie`. The system fetches and embeds.
3. **Question** — at the prompt: `What did Marie Curie discover?` Answer streams; the
   source passages are shown below.
4. **Tune** — `/config retrieval.top_k 7` to widen retrieval. `/sources off` to hide sources.

## Architectural decisions

### Vector store layout — Option B (one collection, metadata-filtered)
The spec gave us two options. We chose Option B (single collection with `type` metadata)
because:
- Cross-type queries ("Compare Einstein and the Eiffel Tower") work in a single shot.
- One HNSW index is cheaper than two on a small corpus.
- Chroma's `where` filter on metadata is essentially free, so person-only or place-only
  retrieval still benefits from filtering when the router is confident.

The cost: when the router gets it wrong, results can be diluted by the wrong type. We
mitigate by oversampling and by retrying without a filter when filtered retrieval comes
back empty.

### Two-layer overlapping chunks
We index at two granularities side by side:
- **Outer** (~900 chars, 200 overlap): broad context, good for "who was X" questions.
- **Inner** (~350 chars, 100 overlap): narrow facts, good for "when did X die".

Both are stored with `layer` metadata. The retriever lightly prefers outer chunks when
similarity scores are close, since they carry more semantic context per token of prompt
budget. Sentence-aware boundary snapping avoids cutting mid-word when a sentence end is
nearby.

### Embedder — sentence-transformers MiniLM
`all-MiniLM-L6-v2`. 22M parameters, 384-dim, runs on CPU in milliseconds. It's a
BERT-family encoder (the spec said "BERT or another solution"). Embeddings are normalized so
cosine similarity is exact and Chroma's distance trivially converts to a similarity score.

### Routing — keyword + known-entity match
A small rule-based router classifies each query as `person`, `place`, `both`, or `unknown`.
Two signals:
1. The query mentions a title we've actually ingested (we keep an in-memory index of
   ingested titles by type). High confidence.
2. Generic keyword cues — "who/born/wife" → person; "where/located/mountain" → place.

The router is deliberately simple. The spec accepted "keyword based or rule based"
approaches and the router's mistakes degrade gracefully (we fall back to unfiltered
retrieval).

### Generation — Ollama, streaming
Ollama via plain HTTP (`/api/chat`, NDJSON stream). Default model `gemma3n:e2b`, swappable
via `ollama.model` config. The system prompt is strict: answer only from the passages,
reply "I don't know." otherwise, cite passages by `[#]`. Temperature is 0.2.

### Storage — files, not a managed service
Chroma persistent client on a mounted volume. SQLite is implicit (Chroma uses it
internally). Model weights live under `data/ollama/`. HF cache under `data/hf_cache/`.
Move the directory, move the project.

### CLI as the primary UI
The spec accepted "Streamlit or CLI". A CLI is a better fit for this homework because:
- Single command to launch (`docker compose up`).
- Streams tokens cleanly into a terminal.
- Slash commands map naturally onto ingestion/admin actions.
- One less moving part to keep alive in Docker (no extra HTTP server).

`rich` for output formatting (panels, tables, live markdown). `prompt_toolkit` for the
input loop (history, completion).

## Risks and limits
- **Wikipedia search non-determinism.** A query like "Curie" could redirect to a
  disambiguation page. We use the homework's exact entity list in `bootstrap.py` for the
  required set, so the demo is reproducible.
- **Small model can still hallucinate.** Mitigated by tight system prompt + low temperature
  + showing sources, but not eliminated. The "I don't know" instruction is a safety net,
  not a guarantee.
- **Single-machine assumption.** Concurrency is not a concern; Chroma's persistent client
  is fine for one process.
- **Embedder language.** MiniLM is multilingual-ish but trained mostly on English. For
  Turkish Wikipedia content, consider switching to a multilingual model (config-only).
