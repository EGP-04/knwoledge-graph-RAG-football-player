import os
import sys
import pandas as pd
import time
from tqdm import tqdm

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import KG-RAG 
from src.executor import execute as kg_execute

# Import Tradition-RAG (Handling hyphen in directory name)
sys.path.append(os.path.join(project_root, "tradition-RAG"))
try:
    from main import TraditionRAG
except ImportError:
    # Alternative import if the first one fails due to pathing
    import importlib.util
    spec = importlib.util.spec_from_file_location("trad_main", os.path.join(project_root, "tradition-RAG", "main.py"))
    trad_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trad_main)
    TraditionRAG = trad_main.TraditionRAG

def run_evaluation():
    # Paths
    input_csv = os.path.join(project_root, "evaluation", "qa-pair.csv")
    output_csv = os.path.join(project_root, "evaluation", "responses.csv")

    if not os.path.exists(input_csv):
        print(f"[ERROR] Input file {input_csv} not found.")
        return

    print("[Evaluation] Initializing Tradition-RAG (building index)...")
    trad_rag = TraditionRAG()

    print(f"[Evaluation] Reading questions from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Initialize output list
    results = []

    print(f"[Evaluation] Starting collection for {len(df)} questions...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        qid = row['id']
        question = row['question']
        
        # 1. Get KG-RAG response
        start_kg = time.time()
        try:
            kg_res = kg_execute(question)
            kg_time = time.time() - start_kg
        except Exception as e:
            print(f"[ERROR] KG-RAG failed on Q{qid}: {e}")
            kg_res = "ERROR"
            kg_time = 0

        # 2. Get Tradition-RAG response
        start_trad = time.time()
        try:
            trad_res = trad_rag.execute(question)
            trad_time = time.time() - start_trad
        except Exception as e:
            print(f"[ERROR] Tradition-RAG failed on Q{qid}: {e}")
            trad_res = "ERROR"
            trad_time = 0

        results.append({
            "id": qid,
            "question": question,
            "KG_RAG_response": kg_res,
            "KG_RAG_time": round(kg_time, 3),
            "trad_RAG_response": trad_res,
            "trad_RAG_time": round(trad_time, 3)
        })

    # Save to CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False)
    print(f"[Evaluation] Successfully saved {len(results)} responses to {output_csv}")

if __name__ == "__main__":
    run_evaluation()
