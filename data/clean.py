import pandas as pd

# Load your CSV
input_path = "data/FIFA-football-player-dataset.csv"   # change this
output_path = "data/player-data-cleaned.csv"

df = pd.read_csv(input_path)

# Extract first position
df["position"] = df["player_positions"].apply(
    lambda x: x.split(",")[0].strip() if isinstance(x, str) else x
)

# Rename columns to required format
df_final = df.rename(columns={
    "short_name": "name",
    "nationality_name": "nationality",
    "club_name": "club",
    "league_name": "league"
})

# Select only required columns
df_final = df_final[["name", "position", "nationality", "club", "league"]]

# Save new CSV
df_final.to_csv(output_path, index=False)

print(f"Saved processed CSV to {output_path}")