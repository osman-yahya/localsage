"""Config loader. Reads a YAML file from disk; mutations are written back to the same file."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(os.environ.get("LOCALSAGE_CONFIG", "config/config.yaml"))
# Baked into the image. Used by Config.reset_to_defaults() and to guard /config writes.
DEFAULTS_PATH = Path(__file__).resolve().parent / "default_config.yaml"


class ConfigError(Exception):
    pass


class Config:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = path
        self._data: dict[str, Any] = {}
        self._defaults: dict[str, Any] = self._read(DEFAULTS_PATH)
        self.reload()

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def reload(self) -> None:
        self._data = self._read(self.path)

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._data, f, sort_keys=False, allow_unicode=True)

    def reset_to_defaults(self) -> None:
        """Overwrite the user config with the baked-in defaults."""
        shutil.copyfile(DEFAULTS_PATH, self.path)
        self.reload()

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        """Set a dotted key. Refuses to overwrite a dict-valued slot with a scalar, and
        refuses to create a top-level scalar key that doesn't exist in the defaults — both
        symptoms of malformed CLI input."""
        parts = dotted.split(".")
        if any(not p for p in parts):
            raise ConfigError(f"invalid key: {dotted!r}")

        # Refuse to clobber a dict subtree (e.g. /config retrieval "top_k 10" would
        # turn the whole `retrieval:` section into a string).
        existing = self.get(dotted, _SENTINEL)
        if isinstance(existing, dict):
            raise ConfigError(
                f"refusing to overwrite section {dotted!r} with a scalar; "
                f"did you mean a dotted child key like {dotted}.<field>?"
            )

        # Refuse to invent unknown top-level keys (most common mistake: forgetting the dot).
        if len(parts) == 1 and parts[0] not in self._defaults:
            known = ", ".join(self._defaults.keys())
            raise ConfigError(
                f"unknown top-level key {parts[0]!r}; "
                f"use a dotted path. known sections: {known}"
            )

        node = self._data
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = _coerce(value)

    def as_dict(self) -> dict[str, Any]:
        return self._data


_SENTINEL = object()


def _coerce(value: Any) -> Any:
    """Turn '5' into 5, 'true' into True, leave real types alone. Lets /config take string args."""
    if not isinstance(value, str):
        return value
    s = value.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", "~"):
        return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s
