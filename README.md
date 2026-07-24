# AI Fintech Smart System

An intent-classification chatbot for fintech customer support. Given a customer
message such as *"What is my account balance?"* or *"How do I activate my new
card?"*, the system classifies the message into one of a fixed set of intents
and replies with a configured response for that intent.

The project began as two Jupyter notebooks — one to train the model and one to
run it interactively. Those notebooks are kept as the canonical
training/exploration surface, and the reusable logic they contain has been
extracted into an importable, testable Python package (`finbot`) with a CLI and
a test suite.

## How it works

The system is a classic retrieval-style chatbot built on a small feed-forward
neural network:

1. **Knowledge base** — `intents.json` defines each *intent* as a `tag`, a set
   of example `patterns` (training phrases) and a list of `responses`.
2. **Preprocessing** — a message is tokenized (NLTK `word_tokenize`) and each
   token is lemmatized (WordNet), then encoded as a **bag-of-words** vector: a
   binary vector the length of the vocabulary, with a `1` for every vocabulary
   word present in the message.
3. **Classification** — the bag-of-words vector is fed to a trained Keras
   network that outputs a probability over the intent tags.
4. **Response** — the highest-scoring intent above a confidence threshold is
   selected, and one of that intent's responses is returned.

### Model architecture

Defined and trained in `train_chatbot.ipynb`:

| Layer      | Details                                  |
|------------|------------------------------------------|
| Input      | bag-of-words vector, length = vocabulary |
| Dense      | 128 units, ReLU                          |
| Dropout    | 0.5                                      |
| Dense      | 64 units, ReLU                           |
| Dropout    | 0.5                                      |
| Dense      | softmax over the intent tags             |

- **Optimizer:** SGD with Nesterov momentum (0.9) and an exponential-decay
  learning-rate schedule (initial `0.01`, decay rate `0.96`).
- **Loss:** categorical cross-entropy.
- **Training:** 200 epochs, batch size 5.

### Training data

`intents.json` currently defines **12 intents** covering common fintech support
requests:

`greeting`, `goodbye`, `hours`, `balance`, `transactions`, `credit`,
`payment_date`, `pay_bill`, `activate_card`, `lock_cancel_card`,
`account_number`, `stock_portfolio`.

Training produces three artifacts consumed at inference time:

- `chatbotmodel.h5` — the trained Keras model.
- `words.pkl` — the sorted vocabulary (82 tokens) used to build bag-of-words
  vectors.
- `classes.pkl` — the sorted list of the 12 intent tags.

## Repository layout

```
.
├── chatbot.ipynb          # interactive inference notebook (original)
├── train_chatbot.ipynb    # data prep + model training notebook (original)
├── intents.json           # intents knowledge base
├── chatbotmodel.h5         # trained Keras model
├── words.pkl / classes.pkl # pickled vocabulary and class labels
├── src/finbot/            # reusable package extracted from the notebooks
│   ├── intents.py         # load/parse intents.json
│   ├── preprocessing.py   # tokenize / lemmatize / bag-of-words
│   ├── model.py           # IntentClassifier inference wrapper
│   ├── responses.py       # response selection
│   └── cli.py             # command line entry point
├── tests/                 # offline test suite
└── examples/usage.py      # minimal end-to-end example
```

## Installation

```bash
python -m pip install -r requirements.txt
```

For development (adds pytest and ruff):

```bash
python -m pip install -r requirements-dev.txt
```

The first run downloads the NLTK `punkt`/`wordnet` corpora automatically via
`finbot.preprocessing.ensure_nltk_data()`.

## Training the model

Open and run `train_chatbot.ipynb` top to bottom. It reads `intents.json`,
builds the vocabulary and bag-of-words training set, trains the network, and
saves `chatbotmodel.h5`, `words.pkl` and `classes.pkl`. Re-run it whenever you
change `intents.json`.

## Running the chatbot

**From the original notebook:** open and run `chatbot.ipynb`, which loads the
saved artifacts and starts an interactive loop.

**From the CLI (extracted package):**

```bash
# single message
python -m finbot "What is my account balance?"

# show the predicted intent tag too
python -m finbot --show-intent "How do I activate my new card?"

# interactive session
python -m finbot
```

**From Python:**

```python
from finbot.intents import load_intents
from finbot.model import IntentClassifier
from finbot.preprocessing import ensure_nltk_data
from finbot.responses import get_response

ensure_nltk_data()
catalog = load_intents()
classifier = IntentClassifier.load()

message = "What are your hours?"
intent = classifier.top_intent(message)
print(intent, "->", get_response(intent, catalog))
```

See `examples/usage.py` for a complete runnable script.

## Testing

The test suite is offline by design. Tests for intent parsing, bag-of-words
vectorization and response selection run with no model and no downloaded
corpora. Tests that need TensorFlow or the NLTK data skip automatically when
those are absent (via `pytest.importorskip` and data-presence guards).

```bash
python -m pytest
ruff check src tests
```

## Dependencies

- **TensorFlow / Keras** — model definition, training and inference.
- **NLTK** — tokenization and WordNet lemmatization.
- **NumPy** — bag-of-words vectors and model I/O.

Pinned versions are in `requirements.txt`.

## Limitations

- **Responses are static and illustrative.** Values such as balances, account
  numbers and transactions are hard-coded sample data in `intents.json`, not
  live account information. Do **not** treat responses as real financial data.
- **Small vocabulary and single-label output.** Prediction relies on exact
  (lemmatized) vocabulary matches; out-of-vocabulary phrasing degrades
  accuracy, and each message maps to a single intent.
- **No conversational context.** The `context_set` field is unused; each
  message is classified independently.
- **Not for production security.** Some sample responses request sensitive
  details (e.g. SSN digits) purely as demo text — never wire this to real
  customer channels as-is.

## License

MIT — see `pyproject.toml`.
