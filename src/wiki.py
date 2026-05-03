"""Wikipedia search and fetch.

Two entry points:
  - search_and_fetch(query): hit /w/index.php?search=...&title=Special:Search. If Wikipedia
    redirects to a content page (a "go" match), we ingest that single page. Otherwise we parse
    the search results page and ingest the top N results.
  - fetch_url(url): fetch a single canonical Wikipedia article URL.

Parsing keeps it native: requests + BeautifulSoup. We strip references, infoboxes, navboxes,
and the configured boilerplate sections. The resulting plain text is what feeds the chunker.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup, Tag

# Wikipedia returns its "go" redirect via 302; requests follows it. We can detect by comparing
# the final URL against the search endpoint.
SEARCH_PATH = "/w/index.php"


@dataclass
class WikiPage:
    title: str
    url: str
    text: str
    lang: str


class WikiClient:
    def __init__(self, language: str = "en", user_agent: str = "LocalSage/0.1",
                 skip_sections: Iterable[str] = (), search_results: int = 3):
        self.language = language
        self.skip_sections = {s.strip().lower() for s in skip_sections}
        self.search_results = search_results
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    @property
    def base(self) -> str:
        return f"https://{self.language}.wikipedia.org"

    def search_and_fetch(self, query: str) -> list[WikiPage]:
        url = f"{self.base}{SEARCH_PATH}"
        params = {"search": query, "title": "Special:Search", "ns0": "1"}
        resp = self.session.get(url, params=params, timeout=20)
        resp.raise_for_status()

        # If Wikipedia matched the query to a single page it redirects there. The final URL
        # will look like /wiki/Article_Title rather than the search endpoint.
        final = urlparse(resp.url)
        if "/wiki/" in final.path and "Special:Search" not in resp.url:
            return [self._parse_page(resp.url, resp.text)]

        # Otherwise we are on the search results page. Pick the top N article links.
        soup = BeautifulSoup(resp.text, "lxml")
        results: list[WikiPage] = []
        seen: set[str] = set()
        for a in soup.select("ul.mw-search-results li.mw-search-result a, .mw-search-result-heading a"):
            href = a.get("href")
            if not href:
                continue
            full = urljoin(self.base, href)
            if full in seen:
                continue
            seen.add(full)
            try:
                page = self.fetch_url(full)
                results.append(page)
            except Exception:
                continue
            if len(results) >= self.search_results:
                break
        return results

    def fetch_url(self, url: str) -> WikiPage:
        resp = self.session.get(url, timeout=20)
        resp.raise_for_status()
        return self._parse_page(resp.url, resp.text)

    def _parse_page(self, url: str, html: str) -> WikiPage:
        soup = BeautifulSoup(html, "lxml")

        title_el = soup.select_one("h1#firstHeading, h1.firstHeading, h1")
        title = title_el.get_text(strip=True) if title_el else url

        # Modern Wikipedia ships TWO div.mw-parser-output elements: a sibling stub used by
        # the skin and the real article body (class includes mw-content-ltr). select_one
        # returns the first, which is empty — pick whichever has the most <p>.
        candidates = soup.select("div.mw-parser-output")
        if not candidates:
            content = soup.select_one("#mw-content-text") or soup
        else:
            content = max(candidates, key=lambda c: len(c.find_all("p")))

        for sel in [
            "table.infobox", "table.navbox", "table.vertical-navbox", "table.metadata",
            "div.navbox", "div.thumb", "div.hatnote", "div.reflist", "ol.references",
            "sup.reference", "sup.noprint", "span.mw-editsection", "div.mw-empty-elt",
            "div.toc", "div#toc", "div.shortdescription", "style", "script",
        ]:
            for n in content.select(sel):
                n.decompose()

        lang = self._infer_lang(url)
        text = self._extract_text(content)
        return WikiPage(title=title, url=url, text=text, lang=lang)

    def _extract_text(self, content: Tag) -> str:
        # Walk the article in document order, collecting paragraphs / lists, switching
        # `skip` state on headings. find_all is recursive so this handles modern
        # Wikipedia's heading wrappers (div.mw-heading > h2) and nested content.
        out: list[str] = []
        skip = False
        seen_ids: set[int] = set()  # avoid double-emitting nested content

        for el in content.find_all(["h2", "h3", "h4", "p", "ul", "ol", "dl"]):
            # Skip nested elements we've already emitted via their parent (e.g. a <p>
            # inside a <dl> would otherwise be counted twice).
            if any(id(p) in seen_ids for p in el.parents):
                continue
            tag = el.name
            if tag in {"h2", "h3", "h4"}:
                heading = el.get_text(" ", strip=True)
                heading = re.sub(r"\[edit\]\s*$", "", heading).strip()
                low = heading.lower()
                skip = low in self.skip_sections
                if not skip and heading:
                    out.append(f"\n## {heading}\n")
                continue
            if skip:
                continue
            seen_ids.add(id(el))
            if tag == "p":
                txt = el.get_text(" ", strip=True)
                if txt:
                    out.append(txt)
            elif tag in {"ul", "ol"}:
                for li in el.find_all("li", recursive=False):
                    txt = li.get_text(" ", strip=True)
                    if txt:
                        out.append(f"- {txt}")
            elif tag == "dl":
                for dt in el.find_all(["dt", "dd"], recursive=False):
                    txt = dt.get_text(" ", strip=True)
                    if txt:
                        out.append(txt)

        joined = "\n".join(out)
        joined = re.sub(r"\[\d+\]", "", joined)          # leftover footnote markers
        joined = re.sub(r"[ \t]+", " ", joined)
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        return joined.strip()

    @staticmethod
    def _infer_lang(url: str) -> str:
        host = urlparse(url).netloc
        return host.split(".")[0] if host.endswith("wikipedia.org") else "unknown"
