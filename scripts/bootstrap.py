"""One-shot ingestion of the homework's required entity set (and a few more to reach 20+20).

Run from the host:
    docker compose run --rm app python -m scripts.bootstrap
or once inside the container:
    python -m scripts.bootstrap

This is idempotent: the vector store keys chunks by (url, layer, index), so re-running
upserts rather than duplicates.
"""
from __future__ import annotations

import sys
import time

from src.config import Config
from src.embedder import Embedder
from src.ingest import ingest_pages
from src.vectorstore import VectorStore
from src.wiki import WikiClient


PEOPLE = [
    "Albert Einstein", "Marie Curie", "Leonardo da Vinci", "William Shakespeare",
    "Ada Lovelace", "Nikola Tesla", "Lionel Messi", "Cristiano Ronaldo",
    "Taylor Swift", "Frida Kahlo",
    # extras to reach 20
    "Isaac Newton", "Charles Darwin", "Stephen Hawking", "Mahatma Gandhi",
    "Nelson Mandela", "Vincent van Gogh", "Pablo Picasso", "Mozart",
    "Ludwig van Beethoven", "Alan Turing",
]

PLACES = [
    "Eiffel Tower", "Great Wall of China", "Taj Mahal", "Grand Canyon",
    "Machu Picchu", "Colosseum", "Hagia Sophia", "Statue of Liberty",
    "Pyramids of Giza", "Mount Everest",
    # extras to reach 20
    "Stonehenge", "Petra", "Christ the Redeemer", "Angkor Wat",
    "Mount Fuji", "Niagara Falls", "Sagrada Família", "Burj Khalifa",
    "Acropolis of Athens", "Sydney Opera House",
]


def main() -> int:
    cfg = Config()
    wiki = WikiClient(
        language=cfg.get("wiki.language", "en"),
        user_agent=cfg.get("wiki.user_agent", "LocalSage/0.1"),
        skip_sections=cfg.get("wiki.skip_sections", []) or [],
        search_results=1,  # bootstrap takes the top match only
    )
    embedder = Embedder(
        model_name=cfg.get("embedder.model"),
        device=cfg.get("embedder.device", "cpu"),
        normalize=cfg.get("embedder.normalize", True),
    )
    store = VectorStore(
        path=cfg.get("vectorstore.path"),
        collection=cfg.get("vectorstore.collection"),
    )

    chunk_args = dict(
        outer_size=cfg.get("chunking.outer_size"),
        outer_overlap=cfg.get("chunking.outer_overlap"),
        inner_size=cfg.get("chunking.inner_size"),
        inner_overlap=cfg.get("chunking.inner_overlap"),
    )

    failures: list[tuple[str, str]] = []
    total = 0

    for label, items, etype in (("people", PEOPLE, "person"),
                                 ("places", PLACES, "place")):
        print(f"\n=== ingesting {len(items)} {label} ===", flush=True)
        for name in items:
            try:
                pages = wiki.search_and_fetch(name)
                if not pages:
                    print(f"  - {name}: no result", flush=True)
                    failures.append((name, "no result"))
                    continue
                reports = ingest_pages(pages[:1], entity_type=etype, store=store,
                                       embedder=embedder, **chunk_args)
                for r in reports:
                    print(f"  + {r.title}  ({r.chunks} chunks)", flush=True)
                    total += r.chunks
                time.sleep(0.4)  # polite to Wikipedia
            except Exception as e:
                print(f"  ! {name}: {e}", flush=True)
                failures.append((name, str(e)))

    print(f"\ndone. {total} chunks across {store.count()} stored chunks total.")
    if failures:
        print(f"\nfailures ({len(failures)}):")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
