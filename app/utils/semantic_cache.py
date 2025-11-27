"""Lightweight semantic response cache using embeddings for near-duplicate queries."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from app.core.config import (
    SEMANTIC_CACHE_ENABLED,
    SEMANTIC_CACHE_MAX_ITEMS,
    SEMANTIC_CACHE_TTL,
    SEMANTIC_CACHE_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class _SemanticEntry:
    embedding: np.ndarray  # L2-normalized vector
    response: Dict[str, Any]
    model: str
    system_prompt: str
    created_at: float


class SemanticCache:
    """In-memory semantic cache keyed by approximate prompt similarity."""

    def __init__(self):
        self._entries: list[_SemanticEntry] = []
        self._lock = threading.Lock()
        self._embedder = None

    def _get_embedder(self):
        """Lazily initialize an embedder (fast local, fallback to OpenAI)."""
        if self._embedder is not None:
            return self._embedder

        # Prefer fast local embedder
        try:
            from app.services.ml.fast_embedder import get_fast_embedder

            self._embedder = get_fast_embedder()
            logger.info("Semantic cache using FastLocalEmbedder")
            return self._embedder
        except Exception as e:
            logger.debug(f"Fast embedder unavailable, falling back to OpenAI embedder: {e}")

        # Fallback to OpenAI embeddings
        try:
            from app.services.ml.embedder import QueryEmbedder

            self._embedder = QueryEmbedder()
            logger.info("Semantic cache using OpenAI QueryEmbedder fallback")
        except Exception as e:
            logger.warning(f"No embedder available for semantic cache: {e}")
            self._embedder = None

        return self._embedder

    def _embed(self, text: str) -> Optional[np.ndarray]:
        embedder = self._get_embedder()
        if embedder is None:
            return None

        try:
            vec = embedder.embed(text)
            if vec is None:
                return None
            # Ensure ndarray and normalize for cosine similarity
            arr = np.asarray(vec, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm == 0 or np.isnan(norm):
                return None
            return arr / norm
        except Exception as e:
            logger.debug(f"Semantic cache embedding failed: {e}")
            return None

    def _cleanup(self):
        """Remove expired entries to bound memory."""
        now = time.time()
        self._entries = [
            entry for entry in self._entries if (now - entry.created_at) <= SEMANTIC_CACHE_TTL
        ]

    def get(self, prompt: str, model: str, system_prompt: str = "") -> Optional[Dict[str, Any]]:
        """Return cached response if a semantically similar prompt is found."""
        if not SEMANTIC_CACHE_ENABLED:
            return None

        embedding = self._embed(prompt)
        if embedding is None:
            return None

        with self._lock:
            self._cleanup()
            best_entry: Optional[_SemanticEntry] = None
            best_score = 0.0

            for entry in self._entries:
                if entry.model != model:
                    continue
                if entry.system_prompt != (system_prompt or ""):
                    continue
                score = float(np.dot(embedding, entry.embedding))
                if score > best_score:
                    best_score = score
                    best_entry = entry

            if best_entry and best_score >= SEMANTIC_CACHE_THRESHOLD:
                logger.info(f"Semantic cache HIT (sim={best_score:.3f}) for model={model}")
                cached = dict(best_entry.response)
                cached["similarity"] = best_score
                return cached

        return None

    def set(self, prompt: str, model: str, system_prompt: str, response: Dict[str, Any]) -> None:
        """Store response embedding for future semantic lookups."""
        if not SEMANTIC_CACHE_ENABLED:
            return

        embedding = self._embed(prompt)
        if embedding is None:
            return

        entry = _SemanticEntry(
            embedding=embedding,
            response=dict(response),
            model=model,
            system_prompt=system_prompt or "",
            created_at=time.time(),
        )

        with self._lock:
            self._cleanup()
            # Evict oldest if at capacity
            if len(self._entries) >= SEMANTIC_CACHE_MAX_ITEMS:
                self._entries.pop(0)
            self._entries.append(entry)
            logger.debug(
                f"Semantic cache stored entry (size={len(self._entries)}/{SEMANTIC_CACHE_MAX_ITEMS})"
            )


_SEMANTIC_CACHE = SemanticCache()


def get_semantic_cached_response(
    prompt: str, model: str, system_prompt: str = ""
) -> Optional[Dict[str, Any]]:
    return _SEMANTIC_CACHE.get(prompt, model, system_prompt)


def store_semantic_response(prompt: str, model: str, system_prompt: str, response: Dict[str, Any]) -> None:
    _SEMANTIC_CACHE.set(prompt, model, system_prompt, response)
