"""Command line interface for the AI Fintech Smart System chatbot.

Usage::

    python -m finbot "What is my account balance?"   # single message
    python -m finbot                                  # interactive REPL

The CLI loads the trained model lazily, so ``--help`` works even without
TensorFlow installed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from finbot.intents import load_intents
from finbot.preprocessing import ensure_nltk_data
from finbot.responses import get_response


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``finbot`` command."""
    parser = argparse.ArgumentParser(
        prog="finbot",
        description="Intent-classification chatbot for fintech customer support.",
    )
    parser.add_argument(
        "message",
        nargs="?",
        help="A single message to classify. If omitted, an interactive session starts.",
    )
    parser.add_argument(
        "--show-intent",
        action="store_true",
        help="Also print the predicted intent tag alongside the reply.",
    )
    return parser


def _reply(classifier, catalog, message: str, show_intent: bool) -> str:
    tag = classifier.top_intent(message)
    reply = get_response(tag, catalog)
    if show_intent:
        return f"[{tag}] {reply}"
    return reply


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``finbot`` console script."""
    args = build_parser().parse_args(argv)

    # Imported here so ``--help`` does not require TensorFlow.
    from finbot.model import IntentClassifier

    ensure_nltk_data()
    catalog = load_intents()
    classifier = IntentClassifier.load()

    if args.message is not None:
        print(_reply(classifier, catalog, args.message, args.show_intent))
        return 0

    print("Finbot is running. Type a message (Ctrl-D or 'quit' to exit).")
    try:
        while True:
            try:
                message = input("> ")
            except EOFError:
                break
            if message.strip().lower() in {"quit", "exit"}:
                break
            print(_reply(classifier, catalog, message, args.show_intent))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
