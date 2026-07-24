"""Tests for the IntentClassifier inference wrapper.

The prediction/ranking logic is exercised with a fake model so it runs without
TensorFlow. Tests that load the real ``chatbotmodel.h5`` use
``pytest.importorskip("tensorflow")`` and additionally skip when the NLTK data
needed for vectorization is unavailable.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from finbot.model import ERROR_THRESHOLD, IntentClassifier, Prediction, load_pickle


class FakeModel:
    """A stand-in Keras model that returns a fixed probability vector."""

    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = np.array(probabilities, dtype=float)

    def predict(self, batch, *args, **kwargs):
        # Ignore the input batch and echo the configured probabilities.
        return np.array([self._probabilities])


@pytest.fixture
def fake_classifier() -> IntentClassifier:
    words = ["balance", "card", "hello"]
    classes = ["greeting", "balance", "card"]
    clf = IntentClassifier(model=FakeModel([0.1, 0.8, 0.1]), words=words, classes=classes)
    # Vectorize without NLTK by splitting on whitespace for these tests.
    clf.vectorize = lambda sentence: np.array(
        [1 if w in sentence.lower().split() else 0 for w in words]
    )
    return clf


def test_load_pickle_reads_vocabulary(words_path: Path) -> None:
    words = load_pickle(words_path)
    assert isinstance(words, list)
    assert len(words) == 82
    assert "balance" in words


def test_load_pickle_reads_classes(classes_path: Path) -> None:
    classes = load_pickle(classes_path)
    assert isinstance(classes, list)
    assert len(classes) == 12
    assert "greeting" in classes


def test_predict_returns_scored_intents(fake_classifier: IntentClassifier) -> None:
    predictions = fake_classifier.predict("balance")
    assert all(isinstance(p, Prediction) for p in predictions)
    assert predictions[0].intent == "balance"
    assert predictions[0].probability == pytest.approx(0.8)


def test_predict_applies_threshold(fake_classifier: IntentClassifier) -> None:
    # Only the 0.8 class is above the default 0.25 threshold.
    predictions = fake_classifier.predict("balance")
    assert len(predictions) == 1


def test_predict_sorted_descending() -> None:
    words = ["a"]
    classes = ["x", "y", "z"]
    clf = IntentClassifier(model=FakeModel([0.3, 0.6, 0.1]), words=words, classes=classes)
    clf.vectorize = lambda sentence: np.array([1])
    predictions = clf.predict("a")
    probs = [p.probability for p in predictions]
    assert probs == sorted(probs, reverse=True)


def test_top_intent_none_when_below_threshold() -> None:
    clf = IntentClassifier(model=FakeModel([0.1, 0.1]), words=["a"], classes=["x", "y"])
    clf.vectorize = lambda sentence: np.array([0])
    assert clf.top_intent("a") is None


def test_error_threshold_default() -> None:
    assert ERROR_THRESHOLD == 0.25


def _nltk_ready() -> bool:
    try:
        import nltk

        nltk.data.find("corpora/wordnet")
        return True
    except (ImportError, LookupError):
        return False


def test_real_model_predicts_expected_intents(
    words_path: Path,
    classes_path: Path,
    repo_root: Path,
) -> None:
    pytest.importorskip("tensorflow")
    if not _nltk_ready():
        pytest.skip("NLTK wordnet data required for vectorization")

    from finbot.preprocessing import ensure_nltk_data

    ensure_nltk_data()

    classifier = IntentClassifier.load(
        model_path=repo_root / "chatbotmodel.h5",
        words_path=words_path,
        classes_path=classes_path,
    )
    with open(classes_path, "rb") as handle:
        classes = pickle.load(handle)

    # Each message should map to its expected intent tag.
    cases = {
        "What is my account balance?": "balance",
        "How do I activate my new card?": "activate_card",
        "What are your hours?": "hours",
    }
    for message, expected in cases.items():
        top = classifier.top_intent(message)
        assert top in classes
        assert top == expected
