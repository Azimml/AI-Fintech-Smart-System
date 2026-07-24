"""Minimal example: classify a message and print a reply.

Run from the repository root::

    python examples/usage.py "What is my account balance?"

Requires the runtime dependencies (see requirements.txt) and the NLTK
``punkt``/``wordnet`` data, which :func:`ensure_nltk_data` downloads on first
use.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running the example directly from a source checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finbot.intents import load_intents  # noqa: E402
from finbot.model import IntentClassifier  # noqa: E402
from finbot.preprocessing import ensure_nltk_data  # noqa: E402
from finbot.responses import get_response  # noqa: E402


def main() -> None:
    message = sys.argv[1] if len(sys.argv) > 1 else "What is my account balance?"

    ensure_nltk_data()
    catalog = load_intents()
    classifier = IntentClassifier.load()

    intent = classifier.top_intent(message)
    reply = get_response(intent, catalog)

    print(f"You:    {message}")
    print(f"Intent: {intent}")
    print(f"Finbot: {reply}")


if __name__ == "__main__":
    main()
