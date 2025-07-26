import spacy
import numpy as np
import gc
from typing import List, Dict, Optional


class SmartEmbedderManager:
    """
    Lightweight sentence‑embedding helper that uses spaCy’s 300‑d GloVe
    vectors (package: `en_core_web_md`).  All vectors are L2‑normalised
    so you can compute cosine similarity with simple dot products.

    Caches up to `_max_cache_size` embeddings keyed by hash(text[:200]).
    """

    _nlp: Optional[spacy.language.Language] = None         # singleton model
    _cache: Dict[int, np.ndarray] = {}                     # LRU cache
    _max_cache_size: int = 1_000

    # ───────────────────── internal helpers ──────────────────────
    @classmethod
    def _load_nlp(cls) -> spacy.language.Language:
        """Lazy‑load the spaCy model only once."""
        if cls._nlp is None:
            # model size ≈120 MB; disable components we don’t need
            cls._nlp = spacy.load(
                "en_core_web_md",
                disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"],
            )
        return cls._nlp

    @staticmethod
    def _unit(vec: np.ndarray) -> np.ndarray:
        """Return L2‑normalised vector (zero‑safe)."""
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ───────────────────── public interface ─────────────────────
    @classmethod
    def encode_with_cache(cls, texts: List[str]) -> np.ndarray:
        """
        Vectorise each string in `texts` and return an array of shape
        (len(texts), 300).  Uses an in‑memory LRU cache to avoid
        recomputing embeddings for repeated inputs.
        """
        nlp = cls._load_nlp()

        results: Dict[int, np.ndarray] = {}
        to_compute: List[tuple[int, str]] = []

        for i, txt in enumerate(texts):
            key = hash(txt[:200])                # cheap fingerprint
            if key in cls._cache:
                results[i] = cls._cache[key]
            else:
                to_compute.append((i, txt))

        if to_compute:
            docs = list(
                nlp.pipe((t for _, t in to_compute), batch_size=128, n_process=1)
            )
            for (idx, orig_text), doc in zip(to_compute, docs):
                vec = cls._unit(doc.vector.astype(np.float32))
                results[idx] = vec
                cls._cache[hash(orig_text[:200])] = vec

            # rudimentary LRU eviction
            if len(cls._cache) > cls._max_cache_size:
                excess = len(cls._cache) - cls._max_cache_size
                for _ in range(excess):
                    cls._cache.pop(next(iter(cls._cache)))

        # preserve original ordering
        return np.vstack([results[i] for i in range(len(texts))])

    @classmethod
    def cleanup(cls) -> None:
        """Free RAM—call once at program exit."""
        cls._nlp = None
        cls._cache.clear()
        gc.collect()
