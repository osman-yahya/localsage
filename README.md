# LocalSage

A local Wikipedia RAG assistant. Everything runs on your laptop — language model, embedder,
vector store, and CLI. No external LLM API.

```
ingest  →  chunk (two-layer overlap)  →  embed (BERT)  →  Chroma
                                                            │
question  →  embed  →  route (person/place/both)  ──────────┘
                                                            │
                                                       top-k chunks
                                                            │
                                                  prompt → Ollama (stream)
                                                            │
                                                          answer
```

## Stack

| layer        | choice                                             |
|--------------|----------------------------------------------------|
| LLM          | Ollama, default `gemma3n:e2b` (configurable)       |
| Embedder     | `sentence-transformers/all-MiniLM-L6-v2` (BERT-family, 384-dim) |
| Vector store | Chroma (single collection, metadata-filtered)      |
| CLI          | `rich` + `prompt_toolkit`                          |
| Wiki fetch   | `requests` + `BeautifulSoup` (no `wikipedia` lib)  |
| Runtime      | Docker Compose — one command brings everything up  |

## Quick start

You need Docker Desktop (or Docker Engine + the Compose plugin). Nothing else is required —
Python, Ollama, the model and the embedder all live inside containers.

```bash
git clone https://github.com/osman-yahya/localsage
cd LocalSage

# 1. build the app image (one-time, a few minutes)
docker compose build app

# 2. start ollama and pull the LLM in the background
docker compose up -d ollama model-puller

# 3. wait until the puller exits successfully (one-time, ~1-2 GB)
docker compose logs -f model-puller   # ctrl-C once you see "model ready"

# 4. (optional) seed the store with the homework's 20 people + 20 places
docker compose run --rm app python -m scripts.bootstrap

# 5. open the chat
./localsage
# or:  docker compose run --rm app (Windows users)
```

Once the image is built, `docker compose up` (or `docker compose up --build` to rebuild
on code changes) starts everything and attaches to the CLI. The app waits for the puller
to complete before opening the prompt.

To stop everything: `docker compose down`. Your ingested data lives under `./data/` and
survives container removal.

## Using the CLI

After the chat opens you'll see a help card. Slash commands:

| command | what it does |
|---|---|
| `/wiki <query>` | Search Wikipedia. If Wikipedia auto-redirects to a page (the "go" match), that page is ingested. Otherwise the top results are. |
| `/wiki-url <url>` | Ingest one specific Wikipedia URL. |
| `/type person\|place` | Tag the next `/wiki` ingest. Default is auto-guess from the page text. |
| `/list` | Show what's been ingested. |
| `/sources on\|off` | Toggle the source table after each answer. |
| `/config` | Print current configuration. |
| `/config <key> <value>` | Update a config key, e.g. `/config retrieval.top_k 7`. Most keys apply live; embedder/store changes require a restart. |
| `/reset` | Drop the entire vector store. Asks for confirmation. |
| `/help` | Show the help card again. |
| `/exit` (or Ctrl-D) | Quit. |

Anything that isn't a slash command is treated as a question. The model streams its answer
and (by default) shows the source passages it used.

### Example session

```
/wiki Ada Lovelace
  + Ada Lovelace  (person, 41 chunks)

Who was Ada Lovelace and what is she known for?
→ route: person (confident)  ·  5 chunks
Ada Lovelace was a 19th-century English mathematician known for her work on
Charles Babbage's proposed Analytical Engine [1]. She is recognised as one of
the first to recognize that the machine had applications beyond pure
calculation, and is often described as the first computer programmer [2][3].
```

## Configuration

`config/config.yaml` is mounted into the container, so edits on the host take effect on the
next start. Keys you'll likely tune from the CLI:

```
retrieval.top_k          # how many chunks the LLM sees   (default 5)
retrieval.min_similarity # cosine floor for retrieved chunks (default 0.25)
chunking.outer_size      # coarse layer chunk size in chars (default 900)
chunking.inner_size      # fine layer chunk size in chars   (default 350)
ollama.model             # any model you've pulled into ollama
ollama.temperature       # 0.2 keeps it grounded
ui.show_sources          # whether to print the source table after answers
```

### Wiki language

The fetch URL pattern works for any Wikipedia language; default is `en`. To use Turkish
Wikipedia (`tr.wikipedia.org`), `/config wiki.language tr`.

## Data on disk

Everything portable lives under `./data/`:

```
data/
├── chroma/        # vector store
├── ollama/        # downloaded model weights
├── hf_cache/      # sentence-transformers cache
└── .localsage_history   # CLI history
```

Move the project? Move the directory. Same data, same answers.

## Repo layout

```
LocalSage/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── localsage                   # wrapper: docker compose run --rm app
├── config/
│   └── config.yaml             # editable, mounted into the container
├── data/                       # persistent state, mounted into the container
├── scripts/
│   └── bootstrap.py            # one-shot: ingest 20 people + 20 places
└── src/
    ├── main.py                 # entry: python -m src.main
    ├── cli.py                  # interactive prompt + slash commands
    ├── config.py               # YAML config with live updates
    ├── wiki.py                 # search + parse Wikipedia (requests/bs4)
    ├── chunker.py              # two-layer overlapping chunker
    ├── embedder.py             # sentence-transformers BERT wrapper
    ├── vectorstore.py          # Chroma wrapper, single collection + metadata
    ├── router.py               # keyword/known-entity query router
    ├── retriever.py            # embed → route → top-k → light rerank
    ├── llm.py                  # Ollama HTTP client (streaming)
    ├── prompts.py              # system + user prompt builders
    └── ingest.py               # WikiPage → chunks → embeddings → upsert
```

## Example queries (from the spec)

- `Who was Albert Einstein and what is he known for`
- `What did Marie Curie discover`
- `Compare Lionel Messi and Cristiano Ronaldo`
- `Where is the Eiffel Tower located`
- `What was the Colosseum used for`
- `Which famous place is located in Turkey`
- `Compare Albert Einstein and Nikola Tesla`
- `Who is the president of Mars`  ← should answer "I don't know."

## Troubleshooting

- **`ollama @ ... error`** — give the puller container time to finish on first start; check
  `docker compose logs model-puller`. The CLI is usable for `/wiki` and `/list` even before
  the model is ready; only generation needs Ollama.
- **`model 'X' not pulled yet`** — set `LOCALSAGE_MODEL=<name>` before `docker compose up`,
  or pull manually inside the ollama container: `docker compose exec ollama ollama pull <name>`.
- **First answer is slow** — sentence-transformers downloads on first use into
  `data/hf_cache/`. Subsequent runs reuse it.
- **Want to start fresh?** `/reset` inside the chat, or delete `data/chroma/` on the host.

See `recommendation.md` for production deployment notes and `product_prd.md` for the design
intent that drove these choices.
