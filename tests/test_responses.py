"""Tests for response selection."""

from __future__ import annotations

from pathlib import Path

from finbot.intents import load_intents, parse_intents
from finbot.responses import FALLBACK_RESPONSE, get_response

# Deterministic chooser: always take the first candidate response.
first = lambda responses: responses[0]  # noqa: E731


def _catalog():
    return parse_intents(
        {
            "intents": [
                {"tag": "greeting", "responses": ["Hello", "Hi"]},
                {"tag": "silent", "responses": []},
            ]
        }
    )


def test_get_response_returns_a_configured_response() -> None:
    catalog = _catalog()
    assert get_response("greeting", catalog, chooser=first) == "Hello"


def test_get_response_uses_chooser() -> None:
    catalog = _catalog()
    assert get_response("greeting", catalog, chooser=lambda r: r[-1]) == "Hi"


def test_unknown_tag_falls_back() -> None:
    assert get_response("nope", _catalog()) == FALLBACK_RESPONSE


def test_none_tag_falls_back() -> None:
    assert get_response(None, _catalog()) == FALLBACK_RESPONSE


def test_tag_without_responses_falls_back() -> None:
    assert get_response("silent", _catalog()) == FALLBACK_RESPONSE


def test_every_real_intent_yields_a_response(intents_path: Path) -> None:
    catalog = load_intents(intents_path)
    for tag in catalog.tags:
        reply = get_response(tag, catalog, chooser=first)
        assert reply != FALLBACK_RESPONSE
        assert reply in catalog.get(tag).responses
