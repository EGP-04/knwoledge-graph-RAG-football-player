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
    get_clubs_of_league as _get_clubs_of_league,
)


@tool
def get_player_info(player_name: str):
    """
    Get full info about a player (Club, League, Position, Nationality).
    Use this when the user asks for a comprehensive summary of a player.
    """
    return _get_player_info(player_name)


get_player_info.response_format = "List[Dict(club, league, position, nationality)]"


@tool
def get_players_by_position(position: str, limit: int = 20):
    """
    Get a list of player names who play at a specific position (e.g., 'Forward', 'Midfielder').
    """
    return _get_players_by_position(position, limit)


get_players_by_position.response_format = "List[Dict(name)]"


@tool
def get_players_by_nationality(nationality: str, limit: int = 20):
    """
    Get a list of player names with a specific nationality (e.g., 'Argentina', 'Brazil').
    """
    return _get_players_by_nationality(nationality, limit)


get_players_by_nationality.response_format = "List[Dict(name)]"


@tool
def get_players_by_league(league: str, limit: int = 20):
    """
    Get a list of player names playing in a specific league (e.g., 'Premier League').
    """
    return _get_players_by_league(league, limit)


get_players_by_league.response_format = "List[Dict(name)]"


@tool
def get_players_by_club(club: str, limit: int = 20):
    """
    Get a list of player names playing for a specific club (e.g., 'Real Madrid', 'FC Barcelona').
    """
    return _get_players_by_club(club, limit)


get_players_by_club.response_format = "List[Dict(name)]"


@tool
def get_league_of_club(club: str):
    """
    Get the league that a particular club belongs to.
    """
    return _get_league_of_club(club)


get_league_of_club.response_format = "List[Dict(league)]"


@tool
def get_clubs_of_league(league: str):
    """
    Get the clubs of a particular league.
    """
    return _get_clubs_of_league(league)


get_clubs_of_league.response_format = "List[Dict(league, clubs)]"


RETRIEVER_TOOLS = [
    get_player_info,
    get_players_by_position,
    get_players_by_nationality,
    get_players_by_league,
    get_players_by_club,
    get_league_of_club,
    get_clubs_of_league,
]


def get_retriever_tool_context() -> str:
    """
    Compact tool documentation to inject into the Retriever.
    """

    lines = ["Tooling available to query the football knowledge graph:"]
    for t in RETRIEVER_TOOLS:
        desc = t.description or ""
        
        # 1. Inputs: name, type, and description if available
        input_info = []
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
            required = schema.get("required") or []
            
            for p_name, p_info in props.items():
                p_type = p_info.get("type", "any")
                p_desc = p_info.get("description", "")
                p_req = "required" if p_name in required else "optional"
                
                if p_desc:
                    input_info.append(f"{p_name}: {p_type} ({p_req}, {p_desc})")
                else:
                    input_info.append(f"{p_name}: {p_type} ({p_req})")

        # 2. Output: response_format attribute if present
        output_info = getattr(t, "response_format", "Unknown")

        tool_line = f"- {t.name}: {desc}"
        if input_info:
            tool_line += f"\n  - Inputs: {', '.join(input_info)}"
        tool_line += f"\n  - Output: {output_info}"
        
        lines.append(tool_line)
        
    return "\n".join(lines)

