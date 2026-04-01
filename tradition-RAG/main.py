# tradition-RAG/main.py
import os
import sys

# Add project root to sys.path for absolute imports
if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import faiss
from typing import List
from sentence_transformers import SentenceTransformer

from src.llm_router import Router
from src.executor import execute as graph_execute

class TraditionRAG:
    """
    Implements traditional Vector-based RAG on the player database.
    """
    def __init__(self, data_path: str = 'data/player-data-cleaned.csv'):
        self.data_path = data_path
        self.router = Router()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.df = None
        self.documents = []
        
        self._initialize_index()

    def _initialize_index(self):
        """Loads CSV data and builds the FAISS index."""
        if not os.path.exists(self.data_path):
            print(f"[ERROR] Data file {self.data_path} not found.")
            return

        print(f"[Tradition-RAG] Loading and indexing {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        
        # Convert each row to a descriptive text string
        for _, row in self.df.iterrows():
            doc = (f"Player: {row['name']}, Position: {row['position']}, "
                   f"Nationality: {row['nationality']}, Club: {row['club']}, "
                   f"League: {row['league']}")
            self.documents.append(doc)
        
        # Build vector index
        embeddings = self.model.encode(self.documents, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        print(f"[Tradition-RAG] Index built successfully with {len(self.documents)} players.")

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Finds the most semantically relevant player records."""
        if self.index is None:
            return []
            
        query_vec = self.model.encode([query])
        query_vec = np.array(query_vec).astype('float32')
        faiss.normalize_L2(query_vec)
        
        D, I = self.index.search(query_vec, top_k)
        return [self.documents[idx] for idx in I[0]]

    def execute(self, query: str) -> str:
        """Full traditional RAG flow: Retrieve -> Augment -> Generate."""
        context_docs = self.retrieve(query)
        context_text = "\n".join(context_docs)
        
        prompt = f"""
        You are a football expert. Answer the following question based on the provided context from the player database.
        
        CONTEXT:
        {context_text}
        
        QUESTION:
        {query}
        
        If the information is not in the context, say so. Keep your answer concise.
        """
        return self.router.generate(prompt)

if __name__ == "__main__":
    rag = TraditionRAG()
    
    print("\n--- Tradition-RAG System Loaded ---")
    print("Type 'graph' to switch to Graph-based RAG or 'exit' to quit.")
    
    mode = "traditional"
    
    while True:
        query = input(f"\n[{mode.upper()}] Ask: ").strip()
        
        if query.lower() == 'exit':
            break
        elif query.lower() == 'graph':
             mode = "graph"
             print("Switched to Graph-based RAG.")
             continue
        elif query.lower() == 'traditional':
             mode = "traditional"
             print("Switched to Traditional RAG.")
             continue

        if mode == "traditional":
            response = rag.execute(query)
        else:
            response = graph_execute(query)
            
        print("\nAnswer:", response)