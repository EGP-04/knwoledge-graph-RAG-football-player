import os
import sys

if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.local_llm import LocalLLM
from rag_baseline.rag_query import BaselineRAGQuery

def main():
    print("Initializing Modular Baseline RAG...")
    try:
        baseline = BaselineRAGQuery()
        llm = LocalLLM()
    except Exception as e:
        print(f"[ERROR] Failed to load Baseline RAG components: {e}")
        return

    print("\n--- BASELINE RAG INTERFACE ---")
    print("Type your football questions below. Press Ctrl+C to exit.")
    
    while True:
        try:
            query = input("\nAsk Baseline: ")
            
            # Fetch context from FAISS
            context = baseline.retrieve_context(query, top_k=15)
            
            # Prompt Local LLM
            prompt = f"""
            You are a helpful football assistant. Answer ONLY based on the context below.

            Context Data:
            {context}

            User Question:
            {query}
            
            Final Answer:
            """
            
            response = llm.generate(prompt)
            print("\nAnswer:", response.strip())
            
        except KeyboardInterrupt:
            print("\nExiting Baseline RAG...")
            break
        except Exception as e:
            print(f"\n[System Error] An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
