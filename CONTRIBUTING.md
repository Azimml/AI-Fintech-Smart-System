# Contributing

Thanks for your interest in improving the AI Fintech Smart System.

## Development setup

```bash
python -m pip install -r requirements-dev.txt
```

## Project structure

The Jupyter notebooks (`train_chatbot.ipynb`, `chatbot.ipynb`) are the canonical
training/exploration surface and are kept intact. Reusable logic lives in the
`finbot` package under `src/`, with tests under `tests/`. When you change
behaviour in a notebook, mirror it in the package (and vice versa) so the two
stay in sync.

## Making changes

1. Create a branch for your change.
2. Keep the package importable **without** TensorFlow: import Keras lazily
   inside functions, never at module top level. Tests that need TensorFlow or
   NLTK data must skip gracefully when those are unavailable.
3. Add or update tests under `tests/`.
4. Run the checks below before opening a pull request.

## Checks

```bash
python -m pytest      # tests must pass (TF/NLTK-gated tests may skip)
ruff check src tests  # lint must be clean
```

Or simply:

```bash
make check
```

## Adding a new intent

1. Add the intent (`tag`, `patterns`, `responses`) to `intents.json`.
2. Re-run `train_chatbot.ipynb` to regenerate `chatbotmodel.h5`, `words.pkl`
   and `classes.pkl`.
3. Update the intent count and list in `README.md`, and any affected tests.

## Commit messages

Use short, honest [Conventional Commits](https://www.conventionalcommits.org/)
style prefixes: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `build:`,
`refactor:`.
