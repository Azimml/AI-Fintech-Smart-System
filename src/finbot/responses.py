"""Selecting a reply for a predicted intent.

This mirrors ``get_response`` from ``chatbot.ipynb``: given a predicted intent
tag, pick one of the configured responses for that intent. A pluggable chooser
(defaulting to :func:`random.choice`) keeps selection deterministic in tests.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from finbot.intents import IntentCatalog

# A chooser takes the list of candidate responses and returns one of them.
Chooser = Callable[[Sequence[str]], str]

# Fallback reply used when a predicted tag has no configured responses or when
# no intent could be determined at all.
FALLBACK_RESPONSE = "Sorry, I did not understand that. Could you rephrase?"


def get_response(
    tag: str | None,
    catalog: IntentCatalog,
    chooser: Chooser = random.choice,
) -> str:
    """Return a response for ``tag`` from ``catalog``.

    Returns :data:`FALLBACK_RESPONSE` when ``tag`` is ``None``, is unknown, or
    has no configured responses.
    """
    if tag is None:
        return FALLBACK_RESPONSE
    try:
        intent = catalog.get(tag)
    except KeyError:
        return FALLBACK_RESPONSE
    if not intent.responses:
        return FALLBACK_RESPONSE
    return chooser(intent.responses)
