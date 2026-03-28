import pandas as pd

REQUIRED_COLUMNS = ["name", "position", "nationality", "club", "league"]


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [str(col).strip().lower() for col in df.columns]

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in CSV: {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}"
        )

    # Keep only required fields, drop incomplete rows, then trim whitespace.
    df = df[REQUIRED_COLUMNS].copy()
    df = df.dropna(subset=REQUIRED_COLUMNS)
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].astype(str).str.strip()

    # Drop rows that become empty strings after trimming.
    for col in REQUIRED_COLUMNS:
        df = df[df[col] != ""]

    return df