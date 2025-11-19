"""Prototype semantic search for Telegram messages."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class TelegramMessage:
    """Represents a single Telegram message."""

    id: int
    date: str
    sender: str
    text: str

    @classmethod
    def from_dict(cls, data: dict) -> "TelegramMessage":
        return cls(
            id=int(data["id"]),
            date=data["date"],
            sender=data["from"],
            text=data["text"].strip(),
        )


class SemanticSearcher:
    """Encodes and searches over Telegram messages using sentence embeddings."""

    def __init__(self, messages: Sequence[TelegramMessage], model_name: str) -> None:
        self.messages: List[TelegramMessage] = list(messages)
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [message.text for message in self.messages], convert_to_numpy=True
        )

    def search(self, query: str, top_k: int = 3) -> List[Tuple[TelegramMessage, float]]:
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        scores = cosine_similarity(query_embedding, self.embeddings)[0]
        ranking = scores.argsort()[::-1][:top_k]
        return [(self.messages[index], float(scores[index])) for index in ranking]


def load_messages(path: Path) -> List[TelegramMessage]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return [TelegramMessage.from_dict(item) for item in data]


def format_result(message: TelegramMessage, score: float) -> str:
    preview = message.text
    if len(preview) > 140:
        preview = preview[:137] + "..."
    return (
        f"#{message.id} | {message.date} | {message.sender}\n"
        f"Score: {score:.3f}\n"
        f"Text: {preview}\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic search prototype for Telegram logs")
    parser.add_argument(
        "--query",
        required=True,
        help="Natural language search query",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/sample_messages.json"),
        help="Path to a Telegram export JSON file",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results to display",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    messages = load_messages(args.data)
    searcher = SemanticSearcher(messages, args.model)
    results = searcher.search(args.query, top_k=args.top_k)

    print(f"Top {len(results)} results for query: {args.query!r}\n")
    for message, score in results:
        print(format_result(message, score))


if __name__ == "__main__":
    main()
