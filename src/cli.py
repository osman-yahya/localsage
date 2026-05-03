"""LocalSage CLI. Interactive prompt with slash commands."""
from __future__ import annotations

import sys
from typing import Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import Config, ConfigError
from .embedder import Embedder
from .ingest import ingest_pages
from .llm import Ollama
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .retriever import Retriever
from .router import Router
from .vectorstore import VectorStore
from .wiki import WikiClient

HELP = """\
[bold cyan]Commands[/bold cyan]
  [bold]/wiki[/bold] <query>          Search Wikipedia, ingest the matched page(s).
  [bold]/wiki-url[/bold] <url>        Ingest one specific Wikipedia URL.
  [bold]/type[/bold] person|place    Set the type for the next /wiki ingest (default: auto-guess).
  [bold]/list[/bold]                  List ingested documents.
  [bold]/sources[/bold] on|off        Toggle showing retrieved sources after each answer.
  [bold]/config[/bold]                Show configuration.
  [bold]/config[/bold] <key> <value>  Update a config key — keys must be dotted, e.g.
                          /config retrieval.top_k 7
  [bold]/config reset[/bold]          Restore configuration to defaults.
  [bold]/reset[/bold]                 Drop the entire vector store.
  [bold]/help[/bold]                  Show this help.
  [bold]/exit[/bold] (Ctrl-D)         Quit.

Anything else is treated as a question. Answers stream from the local model and cite the
passages used.
"""

PERSON_KEYWORDS = ("person", "scientist", "physicist", "actor", "singer", "athlete",
                   "footballer", "artist", "writer", "philosopher", "engineer", "painter")
PLACE_KEYWORDS = ("place", "tower", "wall", "monument", "city", "mountain", "river",
                  "temple", "palace", "ruins", "cathedral", "valley", "canyon", "island")


def _guess_type(title: str, text: str) -> str:
    """Cheap fallback guesser used when the user does not /type before /wiki."""
    head = (title + " " + text[:1500]).lower()
    p = sum(1 for k in PERSON_KEYWORDS if k in head)
    pl = sum(1 for k in PLACE_KEYWORDS if k in head)
    # Strong signal: born / died usually mean a person.
    if "born " in head or " died " in head:
        p += 3
    if "located " in head or "located in" in head:
        pl += 3
    if p == pl:
        return "unknown"
    return "person" if p > pl else "place"


class LocalSageCLI:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.console = Console()
        self.next_type: str | None = None  # set by /type for the upcoming /wiki

        self.embedder = Embedder(
            model_name=cfg.get("embedder.model"),
            device=cfg.get("embedder.device", "cpu"),
            normalize=cfg.get("embedder.normalize", True),
        )
        self.store = VectorStore(
            path=cfg.get("vectorstore.path"),
            collection=cfg.get("vectorstore.collection"),
        )
        self.router = Router()
        self.router.update_known(self.store.known_documents())
        self.retriever = Retriever(
            store=self.store, embedder=self.embedder, router=self.router,
            top_k=cfg.get("retrieval.top_k", 5),
            oversample=cfg.get("retrieval.oversample", 4),
            min_similarity=cfg.get("retrieval.min_similarity", 0.25),
        )
        self.llm = Ollama(
            host=cfg.get("ollama.host"),
            model=cfg.get("ollama.model"),
            temperature=cfg.get("ollama.temperature", 0.2),
            num_ctx=cfg.get("ollama.num_ctx", 4096),
        )
        self.wiki = WikiClient(
            language=cfg.get("wiki.language", "en"),
            user_agent=cfg.get("wiki.user_agent", "LocalSage/0.1"),
            skip_sections=cfg.get("wiki.skip_sections", []) or [],
            search_results=cfg.get("wiki.search_results", 3),
        )

        self.session = PromptSession(
            history=FileHistory("/data/.localsage_history"),
            completer=WordCompleter(
                ["/wiki", "/wiki-url", "/type", "/list", "/sources", "/config",
                 "/reset", "/help", "/exit"],
                ignore_case=True,
            ),
        )

    # ---------- top-level loop ----------

    def run(self) -> None:
        if self.cfg.get("ui.banner", True):
            self._banner()
        self._print_help()
        self._health_check()

        while True:
            try:
                line = self.session.prompt(HTML("<ansicyan><b>localsage</b></ansicyan> ❯ "))
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]bye.[/dim]")
                return
            line = (line or "").strip()
            if not line:
                continue
            try:
                if line.startswith("/"):
                    self._handle_command(line)
                else:
                    self._handle_question(line)
            except Exception as e:
                self.console.print(f"[red]error:[/red] {e}")

    # ---------- pretty bits ----------

    def _banner(self) -> None:
        title = Text("LocalSage", style="bold cyan")
        sub = Text("  local Wikipedia RAG  ·  ollama + chroma + sentence-transformers",
                   style="dim")
        self.console.print(Panel.fit(Text.assemble(title, "\n", sub), border_style="cyan"))

    def _print_help(self) -> None:
        self.console.print(Panel.fit(HELP, title="help", border_style="grey50"))

    def _health_check(self) -> None:
        ok, msg = self.llm.health()
        marker = "[green]●[/green]" if ok else "[red]●[/red]"
        self.console.print(f"{marker} ollama @ {self.llm.host} — {msg}")
        self.console.print(
            f"[dim]embedder:[/dim] {self.embedder.model_name}  "
            f"[dim]store:[/dim] {self.store.count()} chunks across "
            f"{len(self.store.known_documents())} docs"
        )

    # ---------- commands ----------

    def _handle_command(self, line: str) -> None:
        parts = line.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/help", "/?"):
            self._print_help()
        elif cmd in ("/exit", "/quit"):
            raise EOFError
        elif cmd == "/wiki":
            if not arg:
                self.console.print("[yellow]usage:[/yellow] /wiki <query>")
                return
            self._cmd_wiki(arg)
        elif cmd == "/wiki-url":
            if not arg:
                self.console.print("[yellow]usage:[/yellow] /wiki-url <url>")
                return
            self._cmd_wiki_url(arg)
        elif cmd == "/type":
            self._cmd_type(arg)
        elif cmd == "/list":
            self._cmd_list()
        elif cmd == "/sources":
            self._cmd_sources(arg)
        elif cmd == "/config":
            self._cmd_config(arg)
        elif cmd == "/reset":
            self._cmd_reset()
        else:
            self.console.print(f"[yellow]unknown command:[/yellow] {cmd}  (try /help)")

    def _cmd_wiki(self, query: str) -> None:
        self.console.print(f"[dim]searching wikipedia for[/dim] '{query}' …")
        pages = self.wiki.search_and_fetch(query)
        if not pages:
            self.console.print("[red]no results.[/red]")
            return
        self._do_ingest(pages, forced_type=self.next_type)
        self.next_type = None

    def _cmd_wiki_url(self, url: str) -> None:
        self.console.print(f"[dim]fetching[/dim] {url} …")
        page = self.wiki.fetch_url(url)
        self._do_ingest([page], forced_type=self.next_type)
        self.next_type = None

    def _cmd_type(self, arg: str) -> None:
        arg = arg.lower().strip()
        if arg in ("person", "place"):
            self.next_type = arg
            self.console.print(f"[green]ok[/green] next /wiki ingest will be tagged "
                               f"as [bold]{arg}[/bold].")
        elif arg in ("", "auto", "off", "clear"):
            self.next_type = None
            self.console.print("[green]ok[/green] type set to auto-guess.")
        else:
            self.console.print("[yellow]usage:[/yellow] /type person|place|auto")

    def _cmd_list(self) -> None:
        docs = self.store.known_documents()
        if not docs:
            self.console.print("[dim]no documents ingested yet.[/dim]")
            return
        table = Table(title=f"ingested documents  ({len(docs)})", title_style="dim")
        table.add_column("type", style="cyan", no_wrap=True)
        table.add_column("title")
        table.add_column("url", style="dim")
        for d in sorted(docs, key=lambda x: (x.get("type", ""), x.get("title", ""))):
            table.add_row(d.get("type", "?"), d.get("title", "?"), d.get("url", ""))
        self.console.print(table)

    def _cmd_sources(self, arg: str) -> None:
        arg = arg.lower().strip()
        if arg in ("on", "true", "1", "yes"):
            self.cfg.set("ui.show_sources", True)
        elif arg in ("off", "false", "0", "no"):
            self.cfg.set("ui.show_sources", False)
        else:
            self.console.print("[yellow]usage:[/yellow] /sources on|off")
            return
        self.cfg.save()
        self.console.print(f"[green]ok[/green] show_sources = {self.cfg.get('ui.show_sources')}")

    def _cmd_config(self, arg: str) -> None:
        if not arg:
            self._print_config()
            return
        if arg.strip().lower() == "reset":
            self.cfg.reset_to_defaults()
            # Re-pull every runtime knob we know about.
            for k in ("retrieval.top_k", "ollama.model", "wiki.language", "ui.show_sources"):
                self._apply_runtime_config(k)
            self.console.print("[green]ok[/green] config restored to defaults.")
            return

        tokens = arg.split()
        if len(tokens) < 2:
            self._config_usage()
            return
        if len(tokens) > 2:
            # Common mistake: /config retrieval top_k 10 instead of /config retrieval.top_k 10.
            self.console.print(
                f"[yellow]too many arguments.[/yellow]  did you mean "
                f"[bold]/config {tokens[0]}.{tokens[1]} {' '.join(tokens[2:])}[/bold]?"
            )
            return

        key, value = tokens
        if "." not in key:
            self.console.print(
                f"[yellow]config keys must be dotted[/yellow] "
                f"(e.g. [bold]retrieval.top_k[/bold], not [bold]top_k[/bold]). "
                f"see /config for the layout."
            )
            return

        try:
            self.cfg.set(key, value)
        except ConfigError as e:
            self.console.print(f"[red]✗[/red] {e}")
            return
        self.cfg.save()
        self.console.print(f"[green]ok[/green] {key} = {self.cfg.get(key)!r}  "
                           f"[dim](some changes apply on next restart)[/dim]")
        self._apply_runtime_config(key)

    def _config_usage(self) -> None:
        self.console.print(
            "[yellow]usage:[/yellow] /config                       show config\n"
            "        /config <key> <value>         set a dotted key, e.g. /config retrieval.top_k 10\n"
            "        /config reset                 restore defaults"
        )

    def _cmd_reset(self) -> None:
        confirm = self.session.prompt(
            HTML("<ansired>this drops all ingested data. type YES to confirm:</ansired> ")
        )
        if confirm.strip() != "YES":
            self.console.print("[dim]cancelled.[/dim]")
            return
        self.store.reset()
        self.router.update_known([])
        self.console.print("[green]ok[/green] vector store cleared.")

    # ---------- ingest helper ----------

    def _do_ingest(self, pages, *, forced_type: str | None) -> None:
        if not pages:
            self.console.print("  [red]✗[/red] no pages returned from Wikipedia.")
            return
        for p in pages:
            if not p.text.strip():
                self.console.print(
                    f"  [red]✗[/red] {p.title}: parsed page had no extractable text "
                    f"[dim]({p.url})[/dim]"
                )
                continue
            entity_type = forced_type or _guess_type(p.title, p.text)
            reports = ingest_pages(
                [p],
                entity_type=entity_type,
                store=self.store,
                embedder=self.embedder,
                outer_size=self.cfg.get("chunking.outer_size"),
                outer_overlap=self.cfg.get("chunking.outer_overlap"),
                inner_size=self.cfg.get("chunking.inner_size"),
                inner_overlap=self.cfg.get("chunking.inner_overlap"),
            )
            if not reports:
                self.console.print(
                    f"  [yellow]·[/yellow] {p.title}: produced no chunks "
                    f"[dim](text len: {len(p.text)})[/dim]"
                )
                continue
            for r in reports:
                tag = r.type if r.type != "unknown" else "[yellow]unknown[/yellow]"
                self.console.print(
                    f"  [green]+[/green] {r.title}  [dim]({tag}, {r.chunks} chunks)[/dim]"
                )
        self.router.update_known(self.store.known_documents())

    # ---------- question handling ----------

    def _handle_question(self, question: str) -> None:
        retrieved = self.retriever.retrieve(question)
        route = retrieved.route
        route_label = route.category + (" (confident)" if route.confident else "")
        self.console.print(f"[dim]→ route: {route_label}  "
                           f"·  {len(retrieved.chunks)} chunks[/dim]")

        if not retrieved.chunks:
            self.console.print(Panel("I don't know.", border_style="grey50"))
            return

        user_prompt = build_user_prompt(
            question=question,
            chunks=retrieved.chunks,
            max_chars=self.cfg.get("llm.max_context_chars", 6000),
        )

        self._stream_answer(user_prompt)

        if self.cfg.get("ui.show_sources", True):
            self._print_sources(retrieved.chunks)

    def _stream_answer(self, user_prompt: str) -> None:
        # Live-render markdown as tokens arrive.
        accumulated = ""
        with Live(Markdown(""), console=self.console, refresh_per_second=20) as live:
            try:
                for token in self.llm.chat_stream(SYSTEM_PROMPT, user_prompt):
                    accumulated += token
                    live.update(Markdown(accumulated))
            except Exception as e:
                live.update(Text(f"[llm error] {e}", style="red"))
                return

    def _print_sources(self, chunks) -> None:
        table = Table(title="sources", title_style="dim", show_lines=False, box=None)
        table.add_column("#", style="cyan", no_wrap=True)
        table.add_column("title")
        table.add_column("layer", style="dim")
        table.add_column("score", justify="right", style="dim")
        table.add_column("url", style="dim")
        for i, c in enumerate(chunks, start=1):
            table.add_row(
                str(i),
                str(c.metadata.get("title", "?")),
                str(c.metadata.get("layer", "?")),
                f"{c.score:.2f}",
                str(c.metadata.get("url", "")),
            )
        self.console.print(table)

    # ---------- config helpers ----------

    def _print_config(self) -> None:
        import yaml
        rendered = yaml.safe_dump(self.cfg.as_dict(), sort_keys=False, allow_unicode=True)
        self.console.print(Panel(rendered, title="config", border_style="grey50"))

    def _apply_runtime_config(self, key: str) -> None:
        """Pick up the few keys we can change without restarting."""
        if key.startswith("retrieval."):
            self.retriever.top_k = self.cfg.get("retrieval.top_k", 5)
            self.retriever.oversample = self.cfg.get("retrieval.oversample", 4)
            self.retriever.min_similarity = self.cfg.get("retrieval.min_similarity", 0.25)
        elif key.startswith("ollama."):
            self.llm.model = self.cfg.get("ollama.model", self.llm.model)
            self.llm.temperature = self.cfg.get("ollama.temperature", self.llm.temperature)
            self.llm.num_ctx = self.cfg.get("ollama.num_ctx", self.llm.num_ctx)
        elif key.startswith("wiki."):
            self.wiki.language = self.cfg.get("wiki.language", self.wiki.language)
            self.wiki.search_results = self.cfg.get("wiki.search_results",
                                                    self.wiki.search_results)
