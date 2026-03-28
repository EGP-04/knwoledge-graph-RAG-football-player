from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

from typing import Optional


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
    LIMIT $limit
    """
    return run_query(query, {"position": position, "limit": int(limit)})


def get_players_by_nationality(nationality: str, limit: int = 20):
    """Returns a list of players with a specific nationality."""
    query = """
    MATCH (p:Player)-[:HAS_NATIONALITY]->(n:Nationality)
    WHERE toLower(n.name) = toLower($nationality)
    RETURN p.name AS name
    LIMIT $limit
    """
    return run_query(query, {"nationality": nationality, "limit": int(limit)})


def get_players_by_league(league: str, limit: int = 20):
    """Returns a list of players playing in a specific league."""
    query = """
    MATCH (l:League)<-[:PART_OF]-(c:Club)<-[:PLAYS_FOR]-(p:Player)
    WHERE toLower(l.name) = toLower($league)
    RETURN p.name AS name
    LIMIT $limit
    """
    return run_query(query, {"league": league, "limit": int(limit)})


def get_players_by_club(club: str, limit: int = 20):
    """Returns a list of players playing for a specific club."""
    query = """
    MATCH (c:Club)<-[:PLAYS_FOR]-(p:Player)
    WHERE toLower(c.name) = toLower($club)
    RETURN p.name AS name
    LIMIT $limit
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


def get_league_info(league: str):
    """Returns the name of a league and the clubs in it if it exists."""
    query = """
    MATCH (l:League)
    WHERE toLower(l.name) = toLower($league)
    OPTIONAL MATCH (c:Club)-[:PART_OF]->(l)
    RETURN l.name AS league, collect(c.name) AS clubs
    """
    return run_query(query, {"league": league})