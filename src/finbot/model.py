"""Inference wrapper around the trained Keras intent classifier.

The original ``chatbot.ipynb`` loaded three artifacts at module scope:

* ``chatbotmodel.h5`` -- the trained Keras network (bag-of-words -> softmax
  over intent tags);
* ``words.pkl`` -- the sorted vocabulary used to build bag-of-words vectors;
* ``classes.pkl`` -- the sorted list of intent tags.

:class:`IntentClassifier` bundles those into one object with a ``predict``
method. TensorFlow/Keras is imported lazily so that importing this module (and
using the pickle/vocabulary helpers) does not require TensorFlow to be present.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from finbot.preprocessing import bag_of_words, clean_up_sentence

# Artifact locations, resolved relative to the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = _REPO_ROOT / "chatbotmodel.h5"
DEFAULT_WORDS_PATH = _REPO_ROOT / "words.pkl"
DEFAULT_CLASSES_PATH = _REPO_ROOT / "classes.pkl"

# Minimum softmax probability for a class to be reported (from chatbot.ipynb).
ERROR_THRESHOLD = 0.25


def load_pickle(path: str | Path) -> list:
    """Load a pickled list (vocabulary or class labels) from ``path``."""
    with open(path, "rb") as handle:
        return pickle.load(handle)


@dataclass
class Prediction:
    """A single scored intent."""

    intent: str
    probability: float


class IntentClassifier:
    """Load the trained model plus vocabulary/classes and predict intents."""

    def __init__(self, model, words: list[str], classes: list[str]) -> None:
        self.model = model
        self.words = words
        self.classes = classes

    @classmethod
    def load(
        cls,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        words_path: str | Path = DEFAULT_WORDS_PATH,
        classes_path: str | Path = DEFAULT_CLASSES_PATH,
    ) -> IntentClassifier:
        """Load the Keras model and pickled vocabulary/classes from disk.

        Importing Keras happens here so callers that only need the vocabulary
        helpers never pay the TensorFlow import cost.
        """
        from tensorflow.keras.models import load_model

        model = load_model(model_path)
        words = load_pickle(words_path)
        classes = load_pickle(classes_path)
        return cls(model=model, words=words, classes=classes)

    def vectorize(self, sentence: str) -> np.ndarray:
        """Turn ``sentence`` into a bag-of-words vector over the vocabulary."""
        tokens = clean_up_sentence(sentence)
        return bag_of_words(tokens, self.words)

    def predict(self, sentence: str, threshold: float = ERROR_THRESHOLD) -> list[Prediction]:
        """Predict intents for ``sentence``, sorted by descending probability.

        Mirrors ``predict_class`` from ``chatbot.ipynb``: only classes scoring
        above ``threshold`` are returned.
        """
        bow = self.vectorize(sentence)
        scores = self.model.predict(np.array([bow]))[0]
        results = [
            Prediction(intent=self.classes[i], probability=float(score))
            for i, score in enumerate(scores)
            if score > threshold
        ]
        results.sort(key=lambda p: p.probability, reverse=True)
        return results

    def top_intent(self, sentence: str) -> str | None:
        """Return the highest-scoring intent tag, or ``None`` if none passes."""
        predictions = self.predict(sentence)
        return predictions[0].intent if predictions else None
