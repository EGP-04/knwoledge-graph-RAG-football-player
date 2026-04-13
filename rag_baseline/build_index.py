import os
import sys
import pickle
import faiss
from sentence_transformers import SentenceTransformer

if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from rag_baseline.csv_loader import load_and_format_documents

def build_and_save_index():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "data", "player-data-cleaned.csv")
    vector_store_dir = os.path.join(project_root, "rag_baseline", "vector_store")
    
    os.makedirs(vector_store_dir, exist_ok=True)
    
    documents = load_and_format_documents(csv_path)
    
    print(f"[BUILD INDEX] Generating vector embeddings for {len(documents)} documents...")
    print("[BUILD INDEX] This typically takes 15-30 seconds depending on CPU...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    index_path = os.path.join(vector_store_dir, "player_index.faiss")
    docs_path = os.path.join(vector_store_dir, "player_docs.pkl")
    
    faiss.write_index(index, index_path)
    with open(docs_path, "wb") as f:
        pickle.dump(documents, f)
        
    print(f"[BUILD INDEX] Success! Saved index to {index_path} and docs to {docs_path}")

if __name__ == "__main__":
    build_and_save_index()
