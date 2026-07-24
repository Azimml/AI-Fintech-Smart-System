"""Loading and parsing of the ``intents.json`` knowledge base.

The intents file is a JSON document of the shape::

    {
        "intents": [
            {
                "tag": "greeting",
                "patterns": ["Hi", "Hello", ...],
                "responses": ["Hello! How can I assist you today?", ...],
                "context_set": ""
            },
            ...
        ]
    }

This module wraps that structure in small, typed helpers so the rest of the
package does not have to reach into raw dictionaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Default location of the intents file, resolved relative to the repository
# root (two levels up from this file: src/finbot/intents.py -> repo root).
DEFAULT_INTENTS_PATH = Path(__file__).resolve().parents[2] / "intents.json"


@dataclass(frozen=True)
class Intent:
    """A single intent: its tag, training patterns and candidate responses."""

    tag: str
    patterns: list[str] = field(default_factory=list)
    responses: list[str] = field(default_factory=list)
    context_set: str = ""


@dataclass(frozen=True)
class IntentCatalog:
    """A parsed collection of :class:`Intent` objects."""

    intents: list[Intent]

    @property
    def tags(self) -> list[str]:
        """Return the intent tags in file order."""
        return [intent.tag for intent in self.intents]

    def get(self, tag: str) -> Intent:
        """Return the intent with ``tag`` or raise ``KeyError``."""
        for intent in self.intents:
            if intent.tag == tag:
                return intent
        raise KeyError(f"unknown intent tag: {tag!r}")

    def __len__(self) -> int:
        return len(self.intents)


def parse_intents(data: dict) -> IntentCatalog:
    """Build an :class:`IntentCatalog` from an already-loaded JSON ``dict``."""
    raw_intents = data.get("intents", [])
    intents = [
        Intent(
            tag=item["tag"],
            patterns=list(item.get("patterns", [])),
            responses=list(item.get("responses", [])),
            context_set=item.get("context_set", ""),
        )
        for item in raw_intents
    ]
    return IntentCatalog(intents=intents)


def load_intents(path: str | Path = DEFAULT_INTENTS_PATH) -> IntentCatalog:
    """Load and parse the intents JSON file at ``path``."""
    text = Path(path).read_text(encoding="utf-8")
    return parse_intents(json.loads(text))
