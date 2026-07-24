"""Text preprocessing: tokenization, lemmatization and bag-of-words vectors.

This mirrors the ``clean_up_sentence`` and ``bag_of_words`` helpers from
``chatbot.ipynb``. The bag-of-words encoding is a pure function of a token
list and the vocabulary, so it is kept independent of NLTK and can be tested
without any downloaded corpora. Tokenization/lemmatization use NLTK when
available; the required corpora (``punkt``/``punkt_tab`` and ``wordnet``) can
be fetched with :func:`ensure_nltk_data`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# NLTK resources needed for tokenization and lemmatization.
NLTK_RESOURCES = ("punkt", "punkt_tab", "wordnet", "omw-1.4")


def ensure_nltk_data(resources: Sequence[str] = NLTK_RESOURCES) -> None:
    """Download the NLTK corpora required for tokenization/lemmatization.

    Safe to call repeatedly; NLTK skips resources that are already present.
    Raises ``ImportError`` if NLTK itself is not installed.
    """
    import nltk

    for resource in resources:
        nltk.download(resource, quiet=True)


def clean_up_sentence(sentence: str) -> list[str]:
    """Tokenize ``sentence`` and lemmatize each token, as in ``chatbot.ipynb``.

    Requires NLTK and its ``punkt``/``wordnet`` data. Call
    :func:`ensure_nltk_data` first if the corpora may be missing.
    """
    import nltk
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word) for word in tokens]


def bag_of_words(tokens: Sequence[str], vocabulary: Sequence[str]) -> np.ndarray:
    """Return a binary bag-of-words vector of ``tokens`` over ``vocabulary``.

    The result has the same length as ``vocabulary``; position ``i`` is ``1``
    when ``vocabulary[i]`` appears in ``tokens`` and ``0`` otherwise. This is a
    pure function and needs no external corpora, which makes it directly
    testable.
    """
    token_set = set(tokens)
    bag = [1 if word in token_set else 0 for word in vocabulary]
    return np.array(bag)


def sentence_to_bag(sentence: str, vocabulary: Sequence[str]) -> np.ndarray:
    """Clean ``sentence`` and vectorize it against ``vocabulary``.

    Convenience wrapper combining :func:`clean_up_sentence` and
    :func:`bag_of_words`; requires NLTK data.
    """
    tokens = clean_up_sentence(sentence)
    return bag_of_words(tokens, vocabulary)
