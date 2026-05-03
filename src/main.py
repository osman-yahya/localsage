"""Module entry point: `python -m src.main`."""
from __future__ import annotations

import sys

from .cli import LocalSageCLI
from .config import Config


def main() -> int:
    cfg = Config()
    cli = LocalSageCLI(cfg)
    cli.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
