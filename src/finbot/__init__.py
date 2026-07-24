"""AI Fintech Smart System - an intent-classification chatbot for fintech support.

This package extracts the reusable logic that originally lived inline in the
project notebooks (``train_chatbot.ipynb`` and ``chatbot.ipynb``) into an
importable, testable form:

* :mod:`finbot.intents` -- load and parse ``intents.json``.
* :mod:`finbot.preprocessing` -- tokenize, lemmatize and vectorize messages
  into bag-of-words vectors.
* :mod:`finbot.model` -- an inference wrapper that loads the trained Keras
  model together with the pickled vocabulary and class labels and predicts an
  intent for a user message.
* :mod:`finbot.responses` -- select a response for a predicted intent.
* :mod:`finbot.cli` -- a small command line entry point.

The notebooks are kept as the canonical training/exploration surface; this
package mirrors their behaviour so the same logic can be reused and tested.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
