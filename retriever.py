"""Merged retriever: top-3 from each collection, queried in parallel (FR-06, FR-07)."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from langchain_core.documents import Document

import config
from vectorstore import get_collection


@dataclass
class Hit:
    doc: Document
    score: float          # relevance in [0, 1], higher is better

    @property
    def source(self) -> str:
        return self.doc.metadata.get("source", "?")

    @property
    def ref(self) -> str:
        return self.doc.metadata.get("ref", "?")


class MergedRetriever:
    def __init__(self, collections=config.COLLECTIONS, k: int = config.TOP_K):
        self.k = k
        self.stores = {name: get_collection(name) for name in collections}
        self._pool = ThreadPoolExecutor(max_workers=len(self.stores))

    def _search(self, name: str, query: str) -> list[Hit]:
        results = self.stores[name].similarity_search_with_relevance_scores(query, k=self.k)
        return [Hit(doc, float(score)) for doc, score in results]

    def retrieve(self, query: str) -> list[Hit]:
        futures = [self._pool.submit(self._search, name, query) for name in self.stores]
        hits = [h for f in futures for h in f.result()]
        return sorted(hits, key=lambda h: h.score, reverse=True)

    def is_ready(self) -> bool:
        return all(store.get(limit=1, include=[])["ids"] for store in self.stores.values())


def format_context(hits: list[Hit]) -> str:
    """Source-labelled context block for the prompt (FR-08)."""
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"[{i}] ({h.source} | {h.ref})\n{h.doc.page_content}")
    return "\n\n".join(blocks)
