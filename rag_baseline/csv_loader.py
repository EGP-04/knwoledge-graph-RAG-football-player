import pandas as pd

def load_and_format_documents(csv_path: str) -> list[str]:
    print("[CSV LOADER] Loading CSV data...")
    df = pd.read_csv(csv_path)
    
    # Simple data cleaning: just grab required columns
    for col in ["name", "position", "nationality", "club", "league"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    print("[CSV LOADER] Converting rows to text documents...")
    documents = []
    for _, row in df.iterrows():
        doc = (f"{row.get('name', 'Unknown')} plays as {row.get('position', 'Unknown')} "
               f"for {row.get('club', 'Unknown')} in the {row.get('league', 'Unknown')} league. "
               f"Nationality: {row.get('nationality', 'Unknown')}.")
        documents.append(doc)
    
    return documents
