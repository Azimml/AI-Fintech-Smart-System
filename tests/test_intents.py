"""Tests for parsing the intents knowledge base."""

from __future__ import annotations

from pathlib import Path

import pytest

from finbot.intents import IntentCatalog, load_intents, parse_intents

# Intent tags expected in the shipped intents.json.
EXPECTED_TAGS = {
    "greeting",
    "goodbye",
    "hours",
    "balance",
    "transactions",
    "credit",
    "payment_date",
    "pay_bill",
    "activate_card",
    "lock_cancel_card",
    "account_number",
    "stock_portfolio",
}


def test_parse_intents_from_dict() -> None:
    data = {
        "intents": [
            {"tag": "greeting", "patterns": ["Hi"], "responses": ["Hello"]},
            {"tag": "bye", "patterns": ["Bye"], "responses": ["Goodbye"]},
        ]
    }
    catalog = parse_intents(data)
    assert isinstance(catalog, IntentCatalog)
    assert len(catalog) == 2
    assert catalog.tags == ["greeting", "bye"]


def test_parse_handles_missing_optional_fields() -> None:
    catalog = parse_intents({"intents": [{"tag": "empty"}]})
    intent = catalog.get("empty")
    assert intent.patterns == []
    assert intent.responses == []
    assert intent.context_set == ""


def test_get_unknown_tag_raises() -> None:
    catalog = parse_intents({"intents": [{"tag": "greeting"}]})
    with pytest.raises(KeyError):
        catalog.get("does_not_exist")


def test_load_real_intents_file(intents_path: Path) -> None:
    catalog = load_intents(intents_path)
    assert len(catalog) == 12
    assert set(catalog.tags) == EXPECTED_TAGS


def test_every_intent_has_patterns_and_responses(intents_path: Path) -> None:
    catalog = load_intents(intents_path)
    for intent in catalog.intents:
        assert intent.patterns, f"{intent.tag} has no patterns"
        assert intent.responses, f"{intent.tag} has no responses"


def test_balance_intent_content(intents_path: Path) -> None:
    catalog = load_intents(intents_path)
    balance = catalog.get("balance")
    assert "What's my balance?" in balance.patterns
    assert any("balance" in r.lower() for r in balance.responses)
