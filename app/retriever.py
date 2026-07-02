import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Set
from app.catalog import Assessment
from app.config import EMBEDDING_MODEL, TOP_K_RETRIEVAL


class CatalogRetriever:
    def __init__(self, catalog: List[Assessment]):
        self.catalog = catalog
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self._build_index()
    
    def _build_index(self):
        texts = [a.search_text for a in self.catalog]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        filter_job_levels: Optional[Set[str]] = None,
        filter_test_types: Optional[Set[str]] = None,
    ) -> List[Assessment]:
        query_emb = self.model.encode([query], normalize_embeddings=True)
        # Search for a larger pool to allow filtering
        pool_size = min(top_k * 4, len(self.catalog))
        scores, indices = self.index.search(query_emb.astype(np.float32), pool_size)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
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
