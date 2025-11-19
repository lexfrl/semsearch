# semsearch

Prototype semantic search for Telegram messages. The script loads a Telegram export (or the included sample), encodes each message with a Sentence-Transformers model, and returns the most relevant messages for a natural-language query.

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run a search over the sample data:

```bash
python semantic_search.py --query "Where are the sprint notes?" --top-k 3
```

Use a different Telegram JSON export:

```bash
python semantic_search.py --query "reminder about demo" --data /path/to/telegram_export.json
```

Change the embedding model:

```bash
python semantic_search.py --query "retrieval experiments" --model sentence-transformers/all-mpnet-base-v2
```

The script prints the top matches with their similarity scores so you can quickly test the prototype end-to-end.
