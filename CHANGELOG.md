# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `finbot` package extracted from the notebooks, covering intents parsing,
  preprocessing (tokenize / lemmatize / bag-of-words), the `IntentClassifier`
  inference wrapper, response selection and a CLI.
- Offline test suite for intents, preprocessing, responses, the classifier and
  the CLI. TensorFlow- and NLTK-dependent tests skip when those are absent.
- Project tooling: `pyproject.toml` (packaging + ruff), `requirements.txt` /
  `requirements-dev.txt`, `.editorconfig`, `.gitignore`, `Makefile`.
- Documentation: comprehensive `README.md`, `CONTRIBUTING.md`, and a runnable
  `examples/usage.py`.

### Notes
- The original notebooks (`train_chatbot.ipynb`, `chatbot.ipynb`) and the
  trained artifacts (`chatbotmodel.h5`, `words.pkl`, `classes.pkl`,
  `intents.json`) are unchanged; the package mirrors their behaviour.
