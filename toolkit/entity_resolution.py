import numpy as np
import difflib
from typing import List, Optional

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARNING] faiss-cpu or sentence-transformers not found. Falling back to difflib only.")

class HybridEntityResolver:
    def __init__(self, entities: List[str]):
        """
        Initializes the resolver with a list of valid entity names.
        Creates a FAISS index for vector-based semantic matching.
        """
        self.entities = entities
        self.entities_lower = [e.lower() for e in entities]
        self.index = None
        self.model = None

        if not entities:
            return

        if FAISS_AVAILABLE:
            print(f"[HYBRID RESOLVER] Initializing embedding model and building FAISS index for {len(entities)} entities...")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = self.model.encode(self.entities, convert_to_numpy=True)
            
            # Normalize embeddings to use inner product for cosine similarity
            faiss.normalize_L2(embeddings)
            
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)
            print("[HYBRID RESOLVER] FAISS index built successfully.")

    def fuzzy_score(self, query: str, candidate: str) -> float:
        """Calculate a straightforward string similarity score using difflib [0, 1]."""
        return difflib.SequenceMatcher(None, query.lower(), candidate.lower()).ratio()

    def resolve(self, query: str, top_k_vectors: int = 5, alpha: float = 0.5) -> str:
        """
        Resolves a search query to the closest entity name using a hybrid
        combination of FAISS vector similarity and fuzzy string matching.
        
        alpha: Weight for vector similarity, (1-alpha) for fuzzy string similarity.
        """
        if not self.entities or not query:
            return query
            
        best_candidate = query
        best_score = -1.0
        
        if FAISS_AVAILABLE and self.index is not None:
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            distances, indices = self.index.search(query_embedding, min(top_k_vectors, len(self.entities)))
            
            for i, idx in enumerate(indices[0]):
                vector_sim = float(distances[0][i])
                candidate_text = self.entities[idx]
                
                fuzzy_sim = self.fuzzy_score(query, candidate_text)
                
                # Formula: α * Vector_Similarity + (1-α) * Fuzzy_Similarity
                hybrid_score = (alpha * vector_sim) + ((1 - alpha) * fuzzy_sim)
                
                if hybrid_score > best_score:
                    best_score = hybrid_score
                    best_candidate = candidate_text
            
            print(f"[HYBRID RESOLVER] Selected: '{query}' -> '{best_candidate}' (Hybrid Score: {best_score:.3f})")
            return best_candidate
        else:
            matches = difflib.get_close_matches(query.lower(), self.entities_lower, n=1, cutoff=0.5)
            if matches:
                 idx = self.entities_lower.index(matches[0])
                 best_candidate = self.entities[idx]
                 print(f"[HYBRID RESOLVER] (Fallback) Selected '{query}' -> '{best_candidate}'")
                 return best_candidate
            
            for c in self.entities:
                if query.lower() in c.lower() or c.lower() in query.lower():
                    print(f"[HYBRID RESOLVER] (Substring) Selected '{query}' -> '{c}'")
                    return c
                    
            return query
