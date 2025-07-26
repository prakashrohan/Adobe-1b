from __future__ import annotations
import json, sys, datetime, re, string, heapq, gc
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
import pdfplumber
import spacy                     # NEW – lightweight, no torch backend
from scipy.spatial.distance import cdist  # handy for cosine similarity

import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

#!/usr/bin/env python3
"""
persona_di_intel.py  –  **torch‑free** variant

A drop‑in replacement for the original script that relied on
`sentence_transformers` / PyTorch.  All deep‑embedding logic now uses
spaCy’s 300‑d GloVe vectors (≈120 MB) and runs fully on CPU.
"""

# nlp = spacy.load("en_core_web_md")


# ───────────────────────────── stdlib ──────────────────────────────
import json, sys, datetime, re, string, heapq, gc
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict, Counter

# ─────────────────────── 3rd‑party (CPU‑only) ─────────────────────
import numpy as np
import pdfplumber
import networkx as nx
import spacy                         # NEW – replaces sentence_transformers
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ╭───────────────────  Embedding manager (spaCy)  ───────────────────╮
class SmartEmbedderManager:
    """
    Tiny wrapper around spaCy’s 300‑d GloVe vectors with an LRU cache.
    Completely torch‑free and CPU‑friendly.
    """
    _nlp: Optional[spacy.language.Language] = None
    _cache: Dict[int, np.ndarray] = {}
    _max_cache_size: int = 1_000

    # ─── helpers ───
    @classmethod
    def _load_nlp(cls) -> spacy.language.Language:
        if cls._nlp is None:
            # model size ≈120 MB – well within the ≤1 GB constraint
            cls._nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner"])
        return cls._nlp

    @staticmethod
    def _unit(vec: np.ndarray) -> np.ndarray:
        nrm = np.linalg.norm(vec)
        return vec / nrm if nrm > 0 else vec

    # ─── public API ───
    @classmethod
    def encode_with_cache(cls, texts: List[str]) -> np.ndarray:
        """
        Vectorise sentences with memoisation.  The first 200 characters
        are hashed as a cache key – good enough for typical docs.
        Returns an array of shape (len(texts), 300) with L2‑normalised vectors.
        """
        nlp = cls._load_nlp()
        results: Dict[int, np.ndarray] = {}
        to_encode: List[Tuple[int, str]] = []

        for i, txt in enumerate(texts):
            h = hash(txt[:200])
            if h in cls._cache:
                results[i] = cls._cache[h]
            else:
                to_encode.append((i, txt))

        if to_encode:
            docs = list(nlp.pipe((t for _, t in to_encode), batch_size=128))
            for (i, _), doc in zip(to_encode, docs):
                vec = cls._unit(doc.vector.astype(np.float32))
                results[i] = vec
                cls._cache[hash(texts[i][:200])] = vec

            # simple LRU eviction
            if len(cls._cache) > cls._max_cache_size:
                evict = len(cls._cache) - cls._max_cache_size
                for _ in range(evict):
                    cls._cache.pop(next(iter(cls._cache)))

        # preserve ordering
        return np.vstack([results[i] for i in range(len(texts))])

    @classmethod
    def cleanup(cls):
        cls._nlp = None
        cls._cache.clear()
        gc.collect()
# ╰──────────────────────────────────────────────────────────────────╯


# ──────────────────────── data structures ──────────────────────────
@dataclass
class TextChunk:
    filename: str
    page: int
    chunk_id: int
    text: str
    start_pos: int
    end_pos: int
    sim: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0


@dataclass
class Page:
    filename: str
    page: int
    text: str
    sim: float = 0.0
    chunks: List[TextChunk] = None

    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []


# ─────────────────── keyword / key‑phrase extraction ───────────────
def extract_smart_keywords(text: str, top_k: int = 15) -> List[str]:
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    words = clean.split()

    domain_terms = {
        "business": ["strategy", "market", "revenue", "profit", "customer", "growth", "sales"],
        "technical": ["system", "process", "method", "algorithm", "data", "analysis", "performance"],
        "legal": ["compliance", "regulation", "policy", "requirement", "standard", "audit"],
        "finance": ["cost", "budget", "investment", "return", "financial", "economic"],
    }

    word_freq = Counter(words)
    stop_words = {
        "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    }
    freq_keywords = [
        w for w, _ in word_freq.most_common(50) if len(w) > 2 and w not in stop_words
    ]

    domain_keywords = []
    for _, terms in domain_terms.items():
        domain_keywords.extend([t for t in terms if t in clean])

    all_kw = list(dict.fromkeys(freq_keywords + domain_keywords))
    return all_kw[:top_k]


def extract_keyphrases(text: str, max_phrases: int = 10) -> List[str]:
    try:
        vec = TfidfVectorizer(
            ngram_range=(2, 4),
            max_features=50,
            stop_words="english",
        )
        m = vec.fit_transform([text])
        feats = vec.get_feature_names_out()
        scores = m.toarray()[0]
        ranked = sorted(zip(feats, scores), key=lambda x: x[1], reverse=True)
        return [p for p, s in ranked[:max_phrases] if s > 0]
    except Exception:
        return []


# ────────────────────────── chunking ───────────────────────────────
def smart_chunk_page(
    text: str,
    filename: str,
    page_num: int,
    chunk_size: int = 400,
    overlap: int = 50,
) -> List[TextChunk]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[TextChunk] = []

    current = ""
    start = 0
    cid = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        potential = f"{current} {sent}" if current else sent
        if len(potential.split()) > chunk_size and current:
            chunks.append(
                TextChunk(
                    filename=filename,
                    page=page_num,
                    chunk_id=cid,
                    text=current.strip(),
                    start_pos=start,
                    end_pos=start + len(current),
                )
            )
            overlap_text = " ".join(current.split()[-overlap:])
            current = f"{overlap_text} {sent}" if overlap_text else sent
            start += len(current) - len(f"{overlap_text} {sent}")
            cid += 1
        else:
            current = potential

    if current.strip():
        chunks.append(
            TextChunk(
                filename=filename,
                page=page_num,
                chunk_id=cid,
                text=current.strip(),
                start_pos=start,
                end_pos=start + len(current),
            )
        )
    return chunks


# ───────────────── similarity + scoring ────────────────────────────
def hybrid_similarity_scoring(chunks: List[TextChunk], persona: str, task: str) -> List[TextChunk]:
    if not chunks:
        return chunks

    query_text = f"{persona}. {task}"
    query_keywords = set(extract_smart_keywords(query_text))
    query_phrases = set(extract_keyphrases(query_text))

    all_texts = [query_text] + [c.text for c in chunks]
    embeddings = SmartEmbedderManager.encode_with_cache(all_texts)

    query_vec = embeddings[0]
    chunk_vecs = embeddings[1:]
    semantic_scores = chunk_vecs @ query_vec  # dot product (all vectors are unit‑norm)

    for i, chunk in enumerate(chunks):
        text_lower = chunk.text.lower()
        words = set(text_lower.split())

        keyword_overlap = len(query_keywords & words) / max(len(query_keywords), 1)
        phrase_matches = sum(1 for p in query_phrases if p.lower() in text_lower)
        phrase_score = phrase_matches / max(len(query_phrases), 1)
        length_score = min(len(chunk.text.split()) / 200, 1.0)

        chunk.sim = float(semantic_scores[i])
        chunk.keyword_score = keyword_overlap * 0.4 + phrase_score * 0.6
        chunk.combined_score = 0.5 * semantic_scores[i] + 0.3 * chunk.keyword_score + 0.2 * length_score

    return chunks


# ──────────────── page pre‑filtering (unchanged) ───────────────────
def intelligent_page_filter(pages: List[Page], persona: str, task: str, max_pages: int = 80) -> List[Page]:
    if len(pages) <= max_pages:
        return pages

    q_keywords = set(extract_smart_keywords(f"{persona} {task}"))
    q_phrases = set(extract_keyphrases(f"{persona} {task}"))

    scored: List[Tuple[Page, float]] = []

    for pg in pages:
        text_lower = pg.text.lower()
        words = set(text_lower.split())

        kd = len(q_keywords & words) / max(len(q_keywords), 1)
        pm = sum(1 for p in q_phrases if p.lower() in text_lower)
        richness = min(len(pg.text.split()) / 500, 1.0)

        indicators = [
            "summary",
            "conclusion",
            "key",
            "important",
            "critical",
            "main",
            "primary",
            "executive",
            "overview",
            "findings",
        ]
        bonus = sum(0.1 for word in indicators if word in text_lower)
        quick = 0.4 * kd + 0.3 * (pm / max(len(q_phrases), 1)) + 0.2 * richness + 0.1 * min(bonus, 0.5)

        scored.append((pg, quick))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored[:max_pages]]


# ─────────────── PDF extraction (unchanged) ────────────────────────
def extract_pages_advanced(pdf_path: Path, min_words: int = 30) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    try:
        with pdfplumber.open(pdf_path) as doc:
            for n, page in enumerate(doc.pages, 1):
                txt = None
                try:
                    txt = page.extract_text(layout=True, x_tolerance=2, y_tolerance=2)
                except Exception:
                    pass
                if not txt:
                    try:
                        txt = page.extract_text()
                    except Exception:
                        continue
                if not txt:
                    continue

                lines = txt.split("\n")
                if len(lines) > 4:
                    if len(lines[0].split()) < 5 and any(c.isdigit() for c in lines[0]):
                        lines = lines[1:]
                    if len(lines) > 0 and len(lines[-1].split()) < 5 and any(c.isdigit() for c in lines[-1]):
                        lines = lines[:-1]

                txt = "\n".join(lines)
                txt = re.sub(r"\n\s*\n\s*\n+", "\n\n", txt)
                txt = re.sub(r"[ \t]+", " ", txt)
                txt = re.sub(r"-\n", "", txt).strip()

                words = txt.split()
                if len(words) >= min_words:
                    alpha_ratio = sum(1 for w in words[:20] if w.isalpha()) / min(len(words), 20)
                    if alpha_ratio > 0.5:
                        pages.append((n, txt))
    except Exception as e:
        print(f"Warning: error processing {pdf_path}: {e}")
    return pages


# ─────────── TextRank + headline generation (minor tweaks) ─────────
def smart_textrank(text: str, k: int = 5) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= k:
        return text

    try:
        vec = TfidfVectorizer(stop_words="english", max_features=500, ngram_range=(1, 2))
        sent_vecs = vec.fit_transform(sentences)

        if len(sentences) > k * 3:
            n_clusters = max(k, min(k * 2, len(sentences) // 3))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(sent_vecs)
            cluster_sentences: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
            for i, cid in enumerate(clusters):
                cluster_sentences[cid].append((i, sentences[i]))

            chosen: List[str] = []
            for cid, sent_list in cluster_sentences.items():
                if len(chosen) >= k:
                    break
                idxs = [i for i, _ in sent_list]
                vecs = sent_vecs[idxs]
                sim = (vecs @ vecs.T).toarray()
                centrality = np.mean(sim, axis=1)
                best = max(range(len(idxs)), key=lambda j: centrality[j])
                chosen.append(sent_list[best][1])

            if len(chosen) < k:
                remaining_idxs = set(range(len(sentences))) - {
                    i for sents in cluster_sentences.values() for i, _ in sents
                }
                residual = [(np.sum(sent_vecs[i].toarray()), sentences[i]) for i in remaining_idxs]
                residual.sort(reverse=True)
                chosen.extend([s for _, s in residual[: k - len(chosen)]])

            return " ".join(chosen[:k])

        sim = (sent_vecs @ sent_vecs.T).toarray()
        np.fill_diagonal(sim, 0)
        g = nx.from_numpy_array(sim)
        scores = nx.pagerank(g, alpha=0.85, max_iter=50, tol=1e-4)
        ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
        return " ".join(sentences[i] for i in ranked[:k])

    except Exception:
        return " ".join(sentences[:k])


def _trim(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def _fallback_head(text: str, n: int) -> str:
    first = re.split(r"[.!?]", text, 1)[0]
    return _trim(first, n).title()


def context_aware_headline(
    text: str,
    persona: str,
    task: str,
    max_ngram: int = 4,
    top_k: int = 3,
    max_words: int = 12,
) -> str:
    sample = text[:1500]
    try:
        candidates: Set[str] = set()

        vec = CountVectorizer(ngram_range=(2, max_ngram), stop_words="english", max_features=100)
        candidates.update(vec.fit([sample]).get_feature_names_out())

        candidates.update(extract_keyphrases(sample, max_phrases=20))

        for sent in re.split(r"(?<=[.!?])\s+", sample)[:3]:
            w = sent.split()
            if 4 <= len(w) <= max_words:
                candidates.add(sent.strip())

        if not candidates:
            return _fallback_head(sample, max_words)

        all_texts = [sample, f"{persona}. {task}"] + list(candidates)
        embeddings = SmartEmbedderManager.encode_with_cache(all_texts)

        page_vec, query_vec = embeddings[0], embeddings[1]
        cand_vecs = embeddings[2:]

        page_sims = cand_vecs @ page_vec
        query_sims = cand_vecs @ query_vec
        length_scores = np.array([min(len(c.split()) / 8, 1.0) for c in candidates])
        pos_scores = np.array([max(0, 1 - sample.lower().find(c.lower()) / 1000) for c in candidates])

        final = 0.4 * page_sims + 0.3 * query_sims + 0.2 * length_scores + 0.1 * pos_scores
        cand_list = list(candidates)
        ordered = np.argsort(-final)

        chosen: List[str] = []
        used: Set[str] = set()
        for idx in ordered:
            phrase = cand_list[idx]
            words = set(phrase.lower().split())
            if len(words & used) < len(words) * 0.6:
                used |= words
                chosen.append(phrase)
                if len(chosen) == top_k:
                    break

        if not chosen:
            return _fallback_head(sample, max_words)

        main = _trim(chosen[0], max_words).title()
        if len(chosen) == 1:
            return main
        subs = [_trim(p, max_words // 2).title() for p in chosen[1:]]
        if len(subs) == 1:
            return f"{main}: {subs[0]}"
        return f"{main}: {', '.join(subs[:-1])} and {subs[-1]}"

    except Exception:
        return _fallback_head(sample, max_words)


# ───────────────────────── main pipeline ───────────────────────────
# def process(cfg: Dict) -> Dict:
#     persona = cfg["persona"]["role"]
#     task = cfg["job_to_be_done"]["task"]

#     print(f"Persona: {persona}")
#     print(f"Task:    {task}")

#     # Stage 1 – extract pages
#     pages: List[Page] = []
#     files = [Path(d["filename"]) for d in cfg["documents"]]
#     print(f"Reading {len(files)} PDFs…")
#     for p in tqdm(files, desc="PDF", ncols=80):
#         for pg_num, txt in extract_pages_advanced(p):
#             pages.append(Page(str(p), pg_num, txt))
#     if not pages:
#         raise ValueError("No pages extracted.")

#     # Stage 2 – pre‑filter
#     pages = intelligent_page_filter(pages, persona, task, max_pages=80)

#     # Stage 3 – chunk
#     all_chunks: List[TextChunk] = []
#     for page in tqdm(pages, desc="Chunk", ncols=80):
#         ch = smart_chunk_page(page.text, page.filename, page.page)
#         page.chunks = ch
#         all_chunks.extend(ch)

#     # Stage 4 – scoring
#     scored = hybrid_similarity_scoring(all_chunks, persona, task)

#     # Stage 5 – aggregate by document
#     buckets: Dict[str, List[TextChunk]] = defaultdict(list)
#     for c in scored:
#         buckets[c.filename].append(c)

#     doc_scores: List[Tuple[str, float, TextChunk]] = []
#     for fname, lst in buckets.items():
#         top5 = heapq.nlargest(5, lst, key=lambda c: c.combined_score)
#         avg = float(np.mean([c.combined_score for c in top5]))
#         best = max(top5, key=lambda c: c.combined_score)
#         doc_scores.append((fname, avg, best))

#     doc_scores.sort(key=lambda x: -x[1])
#     top_docs = doc_scores[:5]

#     # Stage 6 – headline + summary
#     extracted = []
#     analysis = []

#     for rank, (fname, _, best) in enumerate(top_docs, 1):
#         page_chunks = sorted(
#             [c for c in buckets[fname] if c.page == best.page], key=lambda c: c.chunk_id
#         )
#         full_text = " ".join(c.text for c in page_chunks)

#         title = context_aware_headline(full_text, persona, task)
#         summary = smart_textrank(full_text, 7)

#         extracted.append(
#             {
#                 "document": fname,
#                 "section_title": title,
#                 "importance_rank": rank,
#                 "page_number": best.page,
#             }
#         )
#         analysis.append(
#             {
#                 "document": fname,
#                 "refined_text": summary,
#                 "page_number": best.page,
#             }
#         )

#     return {
#         "metadata": {
#             "input_documents": [d["filename"] for d in cfg["documents"]],
#             "persona": persona,
#             "job_to_be_done": task,
#             "processing_timestamp": datetime.datetime.utcnow().isoformat(),
#         },
#         "extracted_sections": extracted,
#         "subsection_analysis": analysis,
#     }

def process(cfg: Dict, cfg_path: str) -> Dict:
    # Resolve directory that contains input.json
    cfg_dir = Path(cfg_path).parent.resolve()

    persona = cfg["persona"]["role"]
    task = cfg["job_to_be_done"]["task"]

    print(f"Persona: {persona}")
    print(f"Task:    {task}")

    # Stage 1 – extract pages
    pages: List[Page] = []

    # ⬇️ Updated: resolve filenames relative to input.json directory
    doc_files = [(cfg_dir / d["filename"]).resolve() for d in cfg["documents"]]

    print(f"Reading {len(doc_files)} PDFs…")
    for p in tqdm(doc_files, desc="PDF", ncols=80):
        for pg_num, txt in extract_pages_advanced(p):
            pages.append(Page(str(p), pg_num, txt))
    if not pages:
        raise ValueError("No pages extracted.")

    # Stage 2 – pre‑filter
    pages = intelligent_page_filter(pages, persona, task, max_pages=80)

    # Stage 3 – chunk
    all_chunks: List[TextChunk] = []
    for page in tqdm(pages, desc="Chunk", ncols=80):
        ch = smart_chunk_page(page.text, page.filename, page.page)
        page.chunks = ch
        all_chunks.extend(ch)

    # Stage 4 – scoring
    scored = hybrid_similarity_scoring(all_chunks, persona, task)

    # Stage 5 – aggregate by document
    buckets: Dict[str, List[TextChunk]] = defaultdict(list)
    for c in scored:
        buckets[c.filename].append(c)

    doc_scores: List[Tuple[str, float, TextChunk]] = []
    for fname, lst in buckets.items():
        top5 = heapq.nlargest(5, lst, key=lambda c: c.combined_score)
        avg = float(np.mean([c.combined_score for c in top5]))
        best = max(top5, key=lambda c: c.combined_score)
        doc_scores.append((fname, avg, best))

    doc_scores.sort(key=lambda x: -x[1])
    top_docs = doc_scores[:5]

    # Stage 6 – headline + summary
    extracted = []
    analysis = []

    for rank, (fname, _, best) in enumerate(top_docs, 1):
        page_chunks = sorted(
            [c for c in buckets[fname] if c.page == best.page], key=lambda c: c.chunk_id
        )
        full_text = " ".join(c.text for c in page_chunks)

        title = context_aware_headline(full_text, persona, task)
        summary = smart_textrank(full_text, 7)

        extracted.append(
            {
                "document": Path(fname).name,

                "section_title": title,
                "importance_rank": rank,
                "page_number": best.page,
            }
        )
        analysis.append(
            {
                "document": Path(fname).name,

                "refined_text": summary,
                "page_number": best.page,
            }
        )

    return {
        "metadata": {
            "input_documents": [d["filename"] for d in cfg["documents"]],
            "persona": persona,
            "job_to_be_done": task,
            "processing_timestamp": datetime.datetime.utcnow().isoformat(),
        },
        "extracted_sections": extracted,
        "subsection_analysis": analysis,
    }



# ───────────────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python persona_di_intel.py input.json output.json")
        sys.exit(1)

    cfg_path, out_path = sys.argv[1:]

    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)

        result = process(cfg, cfg_path)  # ✅ correct


        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        print(f"✓ Results written to {out_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    finally:
        SmartEmbedderManager.cleanup()
