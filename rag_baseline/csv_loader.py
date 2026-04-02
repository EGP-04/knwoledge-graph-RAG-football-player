import pandas as pd
from toolkit.position_utils import POSITION_MAP, POSITION_GROUP

def load_and_format_documents(csv_path: str) -> list[str]:
    print("[CSV LOADER] Loading CSV data...")
    df = pd.read_csv(csv_path)

    for col in ["name", "position", "nationality", "club", "league"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    print("[CSV LOADER] Creating semantic documents...")
    documents = []

    for _, row in df.iterrows():
        name = row.get('name', 'Unknown')
        pos_code = row.get('position', 'Unknown').upper()
        nationality = row.get('nationality', 'Unknown')
        club = row.get('club', 'Unknown')
        league = row.get('league', 'Unknown')

        full_pos = POSITION_MAP.get(pos_code, pos_code)
        pos_group = POSITION_GROUP.get(pos_code, "Player")

        # Create multiple semantic facts
        documents.append(f"{name} is a football player.")
        documents.append(f"{name} plays as a {full_pos}.")
        documents.append(f"{name} plays in the {pos_group} position category.")
        documents.append(f"{name} plays for {club}.")
        documents.append(f"{club} plays in the {league} league.")
        documents.append(f"{name} is from {nationality}.")

    return documents
