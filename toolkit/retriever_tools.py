from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from toolkit.graph_tools import (
    get_player_info as _get_player_info,
    get_players_by_position as _get_players_by_position,
    get_players_by_nationality as _get_players_by_nationality,
    get_players_by_league as _get_players_by_league,
    get_players_by_club as _get_players_by_club,
    get_league_of_club as _get_league_of_club,
    get_league_info as _get_league_info,
)


@tool
def get_player_info(player_name: str):
    """
    Get full info about a player (Club, League, Position, Nationality).
    Use this when the user asks for a comprehensive summary of a player.
    """
    return _get_player_info(player_name)


@tool
def get_players_by_position(position: str, limit: int = 20):
    """
    Get a list of player names who play at a specific position (e.g., 'Forward', 'Midfielder').
    """
    return _get_players_by_position(position, limit)


@tool
def get_players_by_nationality(nationality: str, limit: int = 20):
    """
    Get a list of player names with a specific nationality (e.g., 'Argentina', 'Brazil').
    """
    return _get_players_by_nationality(nationality, limit)


@tool
def get_players_by_league(league: str, limit: int = 20):
    """
    Get a list of player names playing in a specific league (e.g., 'Premier League').
    """
    return _get_players_by_league(league, limit)


@tool
def get_players_by_club(club: str, limit: int = 20):
    """
    Get a list of player names playing for a specific club (e.g., 'Real Madrid', 'FC Barcelona').
    """
    return _get_players_by_club(club, limit)


@tool
def get_league_of_club(club: str):
    """
    Get the league that a particular club belongs to.
    """
    return _get_league_of_club(club)


@tool
def get_league_info(league: str):
    """
    Get information about a particular league (confirms its name and existence).
    """
    return _get_league_info(league)


RETRIEVER_TOOLS = [
    get_player_info,
    get_players_by_position,
    get_players_by_nationality,
    get_players_by_league,
    get_players_by_club,
    get_league_of_club,
    get_league_info,
]


def get_retriever_tool_context() -> str:
    """
    Compact tool documentation to inject into the Retriever.
    """

    lines = ["Tooling available to query the football knowledge graph:"]
    for t in RETRIEVER_TOOLS:
        # langchain tool has .name and .description
        desc = t.description or ""

        # Best-effort: include argument names for the model.
        param_names = []
        args_schema = getattr(t, "args_schema", None)
        if args_schema is not None:
            try:
                schema = args_schema.model_json_schema()  # pydantic v2
            except Exception:
                try:
                    schema = args_schema.schema()  # pydantic v1
                except Exception:
                    schema = {}
            props = schema.get("properties") or {}
            param_names = list(props.keys())

        if param_names:
            lines.append(f"- {t.name}: {desc} (params: {', '.join(param_names)})")
        else:
            lines.append(f"- {t.name}: {desc}")
    return "\n".join(lines)

