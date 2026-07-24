"""Tests for the command line interface.

These exercise argument parsing and the reply-formatting helper without loading
TensorFlow, so they run in a bare environment.
"""

from __future__ import annotations

import pytest

from finbot import cli
from finbot.intents import parse_intents


def test_parser_accepts_single_message() -> None:
    args = cli.build_parser().parse_args(["What is my balance?"])
    assert args.message == "What is my balance?"
    assert args.show_intent is False


def test_parser_show_intent_flag() -> None:
    args = cli.build_parser().parse_args(["hi", "--show-intent"])
    assert args.show_intent is True


def test_parser_no_message_is_none() -> None:
    args = cli.build_parser().parse_args([])
    assert args.message is None


class _StubClassifier:
    def top_intent(self, message: str) -> str:
        return "greeting"


def _catalog():
    return parse_intents(
        {"intents": [{"tag": "greeting", "responses": ["Hello there"]}]}
    )


def test_reply_plain() -> None:
    reply = cli._reply(_StubClassifier(), _catalog(), "hi", show_intent=False)
    assert reply == "Hello there"


def test_reply_with_intent() -> None:
    reply = cli._reply(_StubClassifier(), _catalog(), "hi", show_intent=True)
    assert reply == "[greeting] Hello there"


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "finbot" in out
