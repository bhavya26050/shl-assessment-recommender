import hashlib
import logging
import re
from typing import List, Optional, Set

import numpy as np

try:
    import faiss
except ModuleNotFoundError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None

from app.catalog import Assessment
from app.config import EMBEDDING_MODEL, TOP_K_RETRIEVAL


logger = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
FALLBACK_EMBEDDING_DIM = 384


class _HashingEmbedder:
    def __init__(self, dimension: int = FALLBACK_EMBEDDING_DIM):
        self.dimension = dimension

    def encode(self, texts, normalize_embeddings: bool = True):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row_index, text in enumerate(texts):
            tokens = TOKEN_PATTERN.findall(text.lower())
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dimension
                weight = 1.0 + (int.from_bytes(digest[4:8], "little") % 3)
                embeddings[row_index, bucket] += weight

            if normalize_embeddings:
                norm = np.linalg.norm(embeddings[row_index])
                if norm:
                    embeddings[row_index] /= norm

        return embeddings


class CatalogRetriever:
    def __init__(self, catalog: List[Assessment]):
        self.catalog = catalog
        self.model = self._load_embedding_model()
        self._build_index()

    def _load_embedding_model(self):
        if SentenceTransformer is None:
            logger.warning("sentence-transformers is unavailable; using fallback hashing embeddings")
            return _HashingEmbedder()

        try:
            return SentenceTransformer(EMBEDDING_MODEL)
        except Exception as exc:
            logger.warning("Falling back to hashing embeddings because the transformer model could not be loaded: %s", exc)
            return _HashingEmbedder()
    
    def _build_index(self):
        if not self.catalog:
            self.embeddings = np.empty((0, FALLBACK_EMBEDDING_DIM), dtype=np.float32)
            self.index = None
            return

        texts = [a.search_text for a in self.catalog]
        self.embeddings = np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)

        if self.embeddings.ndim == 1:
            self.embeddings = self.embeddings.reshape(1, -1)

        dim = self.embeddings.shape[1]
        if faiss is not None:
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
        else:
            self.index = None
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        filter_job_levels: Optional[Set[str]] = None,
        filter_test_types: Optional[Set[str]] = None,
    ) -> List[Assessment]:
        if not self.catalog or not query.strip():
            return []

        query_emb = np.asarray(self.model.encode([query], normalize_embeddings=True), dtype=np.float32)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        # Search for a larger pool to allow filtering.
        pool_size = min(max(top_k * 4, 1), len(self.catalog))

        if self.index is not None:
            _, indices = self.index.search(query_emb, pool_size)
            candidate_indices = indices[0]
        else:
            scores = query_emb @ self.embeddings.T
            candidate_indices = np.argsort(-scores[0])[:pool_size]
        
        results = []
        for idx in candidate_indices:
            if idx < 0:
                continue
            assessment = self.catalog[idx]
            
            # Filter by job levels (case-insensitive check)
            if filter_job_levels:
                filter_jl_lower = {f.lower() for f in filter_job_levels}
                match_jl = any(jl.lower() in filter_jl_lower for jl in assessment.job_levels)
                if not match_jl:
                    continue
            
            # Filter by test type codes
            if filter_test_types:
                match_tt = any(code in filter_test_types for code in assessment.test_type_codes)
                if not match_tt:
                    continue
            
            results.append(assessment)
            if len(results) >= top_k:
                break
        
        return results
    
    def find_by_names(self, names: List[str]) -> List[Assessment]:
        """Find assessments by name matching."""
        found = []
        for name in names:
            name_lower = name.lower().strip()
            for a in self.catalog:
                if name_lower == a.name.lower() or name_lower in a.name.lower() or a.name.lower() in name_lower:
                    found.append(a)
                    break
        return found
