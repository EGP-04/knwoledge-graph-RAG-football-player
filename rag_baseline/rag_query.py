import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

class BaselineRAGQuery:
    def __init__(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vector_store_dir = os.path.join(project_root, "rag_baseline", "vector_store")
        index_path = os.path.join(vector_store_dir, "player_index.faiss")
        docs_path = os.path.join(vector_store_dir, "player_docs.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            raise FileNotFoundError("Vector store files missing. Run 'python -m rag_baseline.build_index' first.")
            
        print("[RAG QUERY] Loading persistent FAISS index and documents...")
        self.index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)
            
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[RAG QUERY] System ready.")

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Embeds a query, searches the persistent index, and returns the top K documents concatenated."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append(self.documents[idx])
            
        return "\n".join(results)
