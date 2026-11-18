"""Embedding model + persistent Chroma collections (shared by ingest and retrieval)."""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    # Runs locally; the model (~90 MB) downloads on first use (NFR-02).
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_collection(name: str) -> Chroma:
    return Chroma(
        collection_name=name,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def replace_documents(name: str, docs, ids) -> int:
    """Idempotent load (FR-17): wipe the collection, then add the current docs."""
    store = get_collection(name)
    existing = store.get(include=[])["ids"]
    if existing:
        store.delete(ids=existing)
    if docs:
        store.add_documents(docs, ids=ids)
    return len(docs)
