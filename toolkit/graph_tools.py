from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

from typing import Optional, List, Any


def run_query(query: str, params: Optional[dict] = None):
    with driver.session() as session:
        return [dict(r) for r in session.run(query, params or {})]


def get_player_info(player_name: str):
    """Returns Club, League, Position, and Nationality for a specific player."""
    query = """
    MATCH (p:Player)
    WHERE toLower(p.name) = toLower($player_name)
    OPTIONAL MATCH (p)-[:PLAYS_FOR]->(c:Club)
    OPTIONAL MATCH (c)-[:PART_OF]->(l:League)
    OPTIONAL MATCH (p)-[:HAS_POSITION]->(pos:Position)
    OPTIONAL MATCH (p)-[:HAS_NATIONALITY]->(n:Nationality)
    RETURN
        c.name AS club,
        l.name AS league,   
        pos.name AS position,
        n.name AS nationality
    """
    return run_query(query, {"player_name": player_name})


def get_players_by_position(position: str, limit: int = 20):
    """Returns a list of players playing at a specific position."""
    query = """
    MATCH (p:Player)-[:HAS_POSITION]->(pos:Position)
    WHERE toLower(pos.name) = toLower($position)
    RETURN p.name AS name
    """
    return run_query(query, {"position": position, "limit": int(limit)})


def get_players_by_nationality(nationality: str, limit: int = 20):
    """Returns a list of players with a specific nationality."""
    query = """
    MATCH (p:Player)-[:HAS_NATIONALITY]->(n:Nationality)
    WHERE toLower(n.name) = toLower($nationality)
    RETURN p.name AS name
    """
    return run_query(query, {"nationality": nationality, "limit": int(limit)})


def get_players_by_league(league: str, limit: int = 20):
    """Returns a list of players playing in a specific league."""
    query = """
    MATCH (l:League)<-[:PART_OF]-(c:Club)<-[:PLAYS_FOR]-(p:Player)
    WHERE toLower(l.name) = toLower($league)
    RETURN p.name AS name
    """
    return run_query(query, {"league": league, "limit": int(limit)})


def get_players_by_club(club: str, limit: int = 20):
    """Returns a list of players playing for a specific club."""
    query = """
    MATCH (c:Club)<-[:PLAYS_FOR]-(p:Player)
    WHERE toLower(c.name) = toLower($club)
    RETURN p.name AS name
    """
    return run_query(query, {"club": club, "limit": int(limit)})


def get_league_of_club(club: str):
    """Returns the league of a particular club."""
    query = """
    MATCH (c:Club)-[:PART_OF]->(l:League)
    WHERE toLower(c.name) = toLower($club)
    RETURN DISTINCT l.name AS league
    """
    return run_query(query, {"club": club})


def get_clubs_of_league(league: str):
    """Returns the name of a league and the clubs in it if it exists."""
    query = """
    MATCH (l:League)
    WHERE toLower(l.name) = toLower($league)
    OPTIONAL MATCH (c:Club)-[:PART_OF]->(l)
    RETURN l.name AS league, collect(c.name) AS clubs
    """
    return run_query(query, {"league": league})


def filter_players(player_names: List[Any], nation: Optional[str] = None, position: Optional[str] = None, club: Optional[str] = None):
    """
    Filters a list of player names based on optional nation, position, and club.
    Can handle a list of strings or a list of dictionaries with a 'name' key.
    """
    # Decoding: extract names if input is a list of dicts
    if player_names and isinstance(player_names, list):
        if len(player_names) > 0 and isinstance(player_names[0], dict):
            player_names = [p.get("name") for p in player_names if isinstance(p, dict) and p.get("name")]

    query = """
    MATCH (p:Player)
    WHERE p.name IN $player_names
    OPTIONAL MATCH (p)-[:HAS_NATIONALITY]->(n:Nationality)
    OPTIONAL MATCH (p)-[:HAS_POSITION]->(pos:Position)
    OPTIONAL MATCH (p)-[:PLAYS_FOR]->(c:Club)
    WITH p, n, pos, c
    WHERE ($nation IS NULL OR toLower(n.name) = toLower($nation))
      AND ($position IS NULL OR toLower(pos.name) = toLower($position))
      AND ($club IS NULL OR toLower(c.name) = toLower($club))
    RETURN DISTINCT p.name AS name
    """
    return run_query(query, {
        "player_names": player_names,
        "nation": nation,
        "position": position,
        "club": club
    })