"""Tests for tokenization and bag-of-words vectorization.

The pure bag-of-words tests need no external corpora. The tokenization /
lemmatization tests require NLTK and its ``punkt``/``wordnet`` data; they are
skipped gracefully when either is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from finbot.preprocessing import bag_of_words


def test_bag_of_words_shape_matches_vocabulary(sample_vocabulary: list[str]) -> None:
    vec = bag_of_words(["hello"], sample_vocabulary)
    assert vec.shape == (len(sample_vocabulary),)
    assert vec.dtype.kind in "iu"


def test_bag_of_words_marks_present_tokens(sample_vocabulary: list[str]) -> None:
    vec = bag_of_words(["account", "payment"], sample_vocabulary)
    # sample_vocabulary == ["account", "balance", "card", "hello", "payment"]
    assert list(vec) == [1, 0, 0, 0, 1]


def test_bag_of_words_ignores_unknown_tokens(sample_vocabulary: list[str]) -> None:
    vec = bag_of_words(["wire", "transfer"], sample_vocabulary)
    assert vec.sum() == 0


def test_bag_of_words_is_binary_not_counts(sample_vocabulary: list[str]) -> None:
    vec = bag_of_words(["card", "card", "card"], sample_vocabulary)
    assert set(np.unique(vec)).issubset({0, 1})
    assert vec[sample_vocabulary.index("card")] == 1


def test_bag_of_words_empty_tokens(sample_vocabulary: list[str]) -> None:
    vec = bag_of_words([], sample_vocabulary)
    assert vec.shape == (len(sample_vocabulary),)
    assert vec.sum() == 0


def _nltk_data_available() -> bool:
    try:
        import nltk

        nltk.data.find("corpora/wordnet")
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.data.find("tokenizers/punkt")
        return True
    except (ImportError, LookupError):
        return False


nltk_required = pytest.mark.skipif(
    not _nltk_data_available(),
    reason="NLTK and its punkt/wordnet data are required",
)


@nltk_required
def test_clean_up_sentence_tokenizes_and_lemmatizes() -> None:
    from finbot.preprocessing import clean_up_sentence

    tokens = clean_up_sentence("How much money do I have?")
    assert tokens == ["How", "much", "money", "do", "I", "have", "?"]


@nltk_required
def test_sentence_to_bag_against_real_vocabulary(words_path) -> None:
    import pickle

    from finbot.preprocessing import sentence_to_bag

    with open(words_path, "rb") as handle:
        words = pickle.load(handle)

    vec = sentence_to_bag("How much money do I have?", words)
    assert vec.shape == (len(words),)
    active = {words[i] for i, x in enumerate(vec) if x}
    assert active == {"How", "I", "do", "have", "money", "much"}
