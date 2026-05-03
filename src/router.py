"""Keyword/rule-based query router. Decides whether a query is about people, places, or both.

The router has two signals:
  1) Known-entity match. We keep an in-memory index of titles that have actually been ingested
     (with their type from the metadata). If a query mentions one of those titles by substring
     match, we are very confident about the type.
  2) Generic keyword cues. Words like 'who', 'born', 'wife' lean person; 'where', 'located',
     'city', 'mountain' lean place.

The output is a category in {"person", "place", "both", "unknown"} plus a confidence flag.
The retriever turns "person"/"place" into a Chroma `where` filter; "both"/"unknown" query
unfiltered and let the embeddings rank it out.
"""
from __future__ import annotations

from dataclasses import dataclass

PERSON_CUES = {
    "who", "whom", "whose", "born", "died", "wife", "husband", "spouse", "father",
    "mother", "son", "daughter", "discovered", "invented", "wrote", "painted",
    "composer", "scientist", "physicist", "actor", "actress", "athlete",
    "footballer", "singer", "artist", "author", "philosopher", "engineer",
}

PLACE_CUES = {
    "where", "located", "city", "country", "continent", "region", "river",
    "mountain", "tower", "wall", "monument", "temple", "palace", "cathedral",
    "ruins", "landmark", "valley", "canyon", "island", "lake", "altitude",
    "elevation", "near", "in turkey", "in china", "in italy", "in egypt",
}


@dataclass
class Routed:
    category: str        # "person" | "place" | "both" | "unknown"
    confident: bool
    matched_titles: list[str]


class Router:
    def __init__(self):
        # title -> type. Refreshed by the CLI after each ingest.
        self._titles: dict[str, str] = {}

    def update_known(self, items: list[dict]) -> None:
        self._titles = {}
        for it in items:
            title = (it.get("title") or "").strip()
            t = (it.get("type") or "unknown").strip().lower()
            if title:
                self._titles[title.lower()] = t

    def classify(self, query: str) -> Routed:
        q = query.lower()
        matched_types: set[str] = set()
        matched_titles: list[str] = []
        for title_l, t in self._titles.items():
            if title_l and title_l in q:
                matched_types.add(t)
                matched_titles.append(title_l)

        if matched_types:
            if matched_types == {"person"}:
                return Routed("person", True, matched_titles)
            if matched_types == {"place"}:
                return Routed("place", True, matched_titles)
            return Routed("both", True, matched_titles)

        person_hits = sum(1 for w in PERSON_CUES if w in q)
        place_hits = sum(1 for w in PLACE_CUES if w in q)
        if person_hits and not place_hits:
            return Routed("person", False, [])
        if place_hits and not person_hits:
            return Routed("place", False, [])
        if person_hits and place_hits:
            return Routed("both", False, [])
        return Routed("unknown", False, [])
