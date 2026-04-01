from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from typing import Optional, List, Dict
from toolkit.entity_resolution import HybridEntityResolver

# -------------------------------------------------------
# In-memory cache: fetched ONCE, reused for all queries
# -------------------------------------------------------
_CACHE: dict[str, List[str]] = {}
_RESOLVERS: dict[str, HybridEntityResolver] = {}

def _load_cache():
    """Fetch all candidate name lists from Neo4j once at startup."""
    global _CACHE
    if _CACHE:
        return  # already loaded
    print("[CACHE] Loading name lists from Neo4j...")
    _CACHE["players"]   = [r["name"] for r in run_query("MATCH (p:Player) RETURN p.name AS name") if r["name"]]
    _CACHE["nations"]   = [r["name"] for r in run_query("MATCH (n:Nation) RETURN n.name AS name") if r["name"]]
    _CACHE["clubs"]     = [r["name"] for r in run_query("MATCH (c:Club) RETURN c.name AS name") if r["name"]]
    _CACHE["leagues"]   = [r["name"] for r in run_query("MATCH (l:League) RETURN l.name AS name") if r["name"]]
    _CACHE["positions"] = [r["name"] for r in run_query("MATCH (pos:Position) RETURN pos.name AS name") if r["name"]]
    print(f"[CACHE] Loaded: {len(_CACHE['players'])} players, {len(_CACHE['nations'])} nations, "
          f"{len(_CACHE['clubs'])} clubs, {len(_CACHE['leagues'])} leagues, {len(_CACHE['positions'])} positions")
          
    # Initialize Hybrid Resolvers
    _RESOLVERS["players"] = HybridEntityResolver(_CACHE["players"])
    _RESOLVERS["nations"] = HybridEntityResolver(_CACHE["nations"])
    _RESOLVERS["clubs"] = HybridEntityResolver(_CACHE["clubs"])
    _RESOLVERS["leagues"] = HybridEntityResolver(_CACHE["leagues"])
    _RESOLVERS["positions"] = HybridEntityResolver(_CACHE["positions"])

def refresh_cache():
    """Call this if Neo4j data changes and you want to reload the name lists."""
    global _CACHE, _RESOLVERS
    _CACHE = {}
    _RESOLVERS = {}
    _load_cache()

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)


# -----------------------------
# Generic query runner
# -----------------------------
def run_query(query: str, params: Optional[dict] = None) -> List[Dict]:
    with driver.session() as session:
        result = session.run(query, params or {})
        return [dict(record) for record in result]


# -----------------------------
# Hybrid resolver: finds closest DB value via Vector + Fuzzy Match
# e.g. "cristino" -> "Cristiano Ronaldo"
# -----------------------------
def _fuzzy_resolve(user_input: str, entity_type: str) -> str:
    """
    Given a user's fuzzy input, find the closest match from the database
    using a Hybrid Strategy: FAISS Vector Embeddings + FuzzyWuzzy/DiffLib.
    """
    _load_cache()
    if not user_input or entity_type not in _RESOLVERS:
        return user_input
        
    resolved = _RESOLVERS[entity_type].resolve(user_input, top_k_vectors=5, alpha=0.5)
    
    if resolved.lower() != user_input.lower():
        print(f"[RESOLVER] Resolved '{user_input}' -> '{resolved}'")
        
    return resolved


def _all_player_names() -> List[str]:
    _load_cache()
    return _CACHE["players"]

def _all_nation_names() -> List[str]:
    _load_cache()
    return _CACHE["nations"]

def _all_club_names() -> List[str]:
    _load_cache()
    return _CACHE["clubs"]

def _all_league_names() -> List[str]:
    _load_cache()
    return _CACHE["leagues"]

def _all_position_names() -> List[str]:
    _load_cache()
    return _CACHE["positions"]


# -----------------------------
# Name normalization for fuzzy matching
# "L.Messi" == "L Messi" == "l. messi"
# -----------------------------
_NAME_MATCH = """
replace(replace(toLower(p.name), '.', ''), ' ', '') =
replace(replace(toLower($player_name), '.', ''), ' ', '')
"""

_NAME_CONTAINS = """
toLower(p.name) CONTAINS toLower($player_name)
"""


# -----------------------------
# Player Info
# -----------------------------
def get_player_info(player_name: str):
    """Returns Club, League, Positions, and Nationality for a specific player."""
    player_name = _fuzzy_resolve(player_name, "players")

    query = f"""
    MATCH (p:Player)
    WHERE {_NAME_MATCH} OR {_NAME_CONTAINS}
    OPTIONAL MATCH (p)-[:PLAYS_FOR]->(c:Club)
    OPTIONAL MATCH (c)-[:PART_OF]->(l:League)
    OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
    OPTIONAL MATCH (p)-[:REPRESENTS]->(n:Nation)
    RETURN
        p.name AS player,
        c.name AS club,
        l.name AS league,
        collect(DISTINCT pos.name) AS positions,
        n.name AS nationality
    """
    return run_query(query, {"player_name": player_name})


# -----------------------------
# Players by Position
# -----------------------------
def get_players_by_position(position: str, limit: int = 20):
    position = _fuzzy_resolve(position, "positions")
    query = """
    MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
    WHERE toLower(pos.name) CONTAINS toLower($position)
       OR toLower($position) CONTAINS toLower(pos.name)
    RETURN p.name AS name
    LIMIT $limit
    """
    return run_query(query, {"position": position, "limit": int(limit)})


# -----------------------------
# Players by Nationality
# -----------------------------
def get_players_by_nationality(nationality: str, limit: int = 20):
    nationality = _fuzzy_resolve(nationality, "nations")
    query = """
    MATCH (p:Player)-[:REPRESENTS]->(n:Nation)
    WHERE toLower(n.name) CONTAINS toLower($nationality)
       OR toLower($nationality) CONTAINS toLower(n.name)
    RETURN p.name AS name
    LIMIT $limit
    """
    return run_query(query, {"nationality": nationality, "limit": int(limit)})


# -----------------------------
# Players by League
# -----------------------------
def get_players_by_league(league: str, limit: int = 20):
    league = _fuzzy_resolve(league, "leagues")
    query = """
    MATCH (l:League)<-[:PART_OF]-(c:Club)<-[:PLAYS_FOR]-(p:Player)
    WHERE toLower(l.name) CONTAINS toLower($league)
       OR toLower($league) CONTAINS toLower(l.name)
    RETURN p.name AS name
    LIMIT $limit
    """
    return run_query(query, {"league": league, "limit": int(limit)})


# -----------------------------
# Players by Club
# -----------------------------
def get_players_by_club(club: str, limit: int = 20):
    club = _fuzzy_resolve(club, "clubs")
    query = """
    MATCH (c:Club)<-[:PLAYS_FOR]-(p:Player)
    WHERE toLower(c.name) CONTAINS toLower($club)
       OR toLower($club) CONTAINS toLower(c.name)
    RETURN p.name AS name
    LIMIT $limit
    """
    return run_query(query, {"club": club, "limit": int(limit)})


# -----------------------------
# League of a Club
# -----------------------------
def get_league_of_club(club: str):
    club = _fuzzy_resolve(club, "clubs")
    query = """
    MATCH (c:Club)-[:PART_OF]->(l:League)
    WHERE toLower(c.name) CONTAINS toLower($club)
       OR toLower($club) CONTAINS toLower(c.name)
    RETURN DISTINCT l.name AS league
    """
    return run_query(query, {"club": club})


# -----------------------------
# League Info (clubs in league)
# -----------------------------
def get_league_info(league: str):
    league = _fuzzy_resolve(league, "leagues")
    query = """
    MATCH (l:League)
    WHERE toLower(l.name) CONTAINS toLower($league)
       OR toLower($league) CONTAINS toLower(l.name)
    OPTIONAL MATCH (c:Club)-[:PART_OF]->(l)
    RETURN l.name AS league, collect(c.name) AS clubs
    """
    return run_query(query, {"league": league})


# -----------------------------
# Close driver
# -----------------------------
def close():
    driver.close()