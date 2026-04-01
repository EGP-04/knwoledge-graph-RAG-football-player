from toolkit.graph_tools import run_query
import difflib

rows = run_query("MATCH (p:Player) WHERE toLower(p.name) CONTAINS 'ney' RETURN p.name AS name")
print("=== Players containing 'ney' ===")
for r in rows:
    print(" ", r["name"])

all_players = [r["name"] for r in run_query("MATCH (p:Player) RETURN p.name AS name") if r["name"]]
matches = difflib.get_close_matches("neymar", [n.lower() for n in all_players], n=5, cutoff=0.4)
print("\n=== difflib matches for 'neymar' (cutoff 0.4) ===")
print(matches)

# Also print how names starting with N look
print("\n=== Player names starting with 'N' ===")
n_players = [n for n in all_players if n.lower().startswith("n")]
for n in sorted(n_players):
    print(" ", n)
