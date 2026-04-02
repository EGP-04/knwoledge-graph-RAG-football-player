from __future__ import annotations

from typing import Optional, List, Any

from langchain_core.tools import tool

from toolkit.graph_tools import (
    get_player_info as _get_player_info,
    get_players_by_position as _get_players_by_position,
    get_players_by_nationality as _get_players_by_nationality,
    get_players_by_league as _get_players_by_league,
    get_players_by_club as _get_players_by_club,
    get_league_of_club as _get_league_of_club,
    get_clubs_of_league as _get_clubs_of_league,
    filter_players as _filter_players,
)


@tool
def get_player_info(player_name: str):
    """
    Get full info about a player (Club, League, Position, Nationality).
    Explanation: Use this when the user asks for a comprehensive summary of a player.
    Input Format: player_name (str) - The name of the player (e.g., 'L. Messi' or 'Vini Jr.').
    Output Format: List[Dict(club, league, position, nationality)]
    Example: get_player_info(player_name='L. Messi') -> [{'club': 'Inter Miami', 'league': 'MLS', 'position': 'RW', 'nationality': 'Argentina'}]
    """
    return _get_player_info(player_name)


@tool
def get_players_by_position(position: str, limit: int = 20):
    """
    Get a list of player names who play at a specific position.
    Explanation: Use this to find all players in a role (e.g., 'ST', 'GK').
    Input Format: position (str) - The short position code (e.g., 'ST', 'GK', 'CAM', 'CB').
    Output Format: List[Dict(name)]
    Example: get_players_by_position(position='ST') -> [{'name': 'E. Haaland'}, {'name': 'R. Lewandowski'}]
    """
    return _get_players_by_position(position, limit)


@tool
def get_players_by_nationality(nationality: str, limit: int = 20):
    """
    Get a list of player names with a specific nationality.
    Explanation: Use this to find players by country (e.g., 'Argentina', 'Brazil').
    Input Format: nationality (str) - The country name (e.g., 'France', 'Spain').
    Output Format: List[Dict(name)]
    Example: get_players_by_nationality(nationality='Brazil') -> [{'name': 'Neymar Jr'}, {'name': 'Rodrygo'}]
    """
    return _get_players_by_nationality(nationality, limit)


@tool
def get_players_by_league(league: str, limit: int = 20):
    """
    Get a list of player names playing in a specific league.
    Explanation: Returns names of players playing in a domestic league.
    Input Format: league (str) - The league name (e.g., 'Premier League', 'La Liga').
    Output Format: List[Dict(name)]
    Example: get_players_by_league(league='Premier League') -> [{'name': 'K. De Bruyne'}, {'name': 'M. Salah'}]
    """
    return _get_players_by_league(league, limit)


@tool
def get_players_by_club(club: str, limit: int = 20):
    """
    Get a list of player names playing for a specific club.
    Explanation: Returns players who are currently signed to this club.
    Input Format: club (str) - The club name (e.g., 'Real Madrid', 'Liverpool').
    Output Format: List[Dict(name)]
    Example: get_players_by_club(club='Real Madrid') -> [{'name': 'Vini Jr.'}, {'name': 'J. Bellingham'}]
    """
    return _get_players_by_club(club, limit)


@tool
def get_league_of_club(club: str):
    """
    Get the league that a particular club belongs to.
    Explanation: Use this to map a club to its primary domestic competition.
    Input Format: club (str) - The club name (e.g., 'FC Bayern München').
    Output Format: List[Dict(league)]
    Example: get_league_of_club(club='FC Barcelona') -> [{'league': 'La Liga'}]
    """
    return _get_league_of_club(club)


@tool
def get_clubs_of_league(league: str):
    """
    Get the clubs of a particular league.
    Explanation: Lists all football clubs currently in the specified league.
    Input Format: league (str) - The league name (e.g., 'Serie A').
    Output Format: List[Dict(league, clubs)]
    Example: get_clubs_of_league(league='Premier League') -> [{'league': 'Premier League', 'clubs': ['Arsenal', 'Chelsea', ...]}]
    """
    return _get_clubs_of_league(league)


@tool
def filter_players(player_names: List[Any], nation: Optional[str] = None, position: Optional[str] = None, club: Optional[str] = None):
    """
    Filter a given list of player names by nation, position, and/or club.
    Explanation: Use this to narrow down a list of candidates from a previous step (multi-hop).
    Input Format: player_names (List[str] or reference), nation/position/club (optional filters).
    Output Format: List[Dict(name)]
    Example: filter_players(player_names='$step1.output', nation='Spain') -> [{'name': 'Pedri'}, ... ]
    """
    return _filter_players(player_names, nation, position, club)


RETRIEVER_TOOLS = [
    get_player_info,
    get_players_by_position,
    get_players_by_nationality,
    get_players_by_league,
    get_players_by_club,
    get_league_of_club,
    get_clubs_of_league,
    filter_players,
]


def get_retriever_tool_context() -> str:
    """
    Detailed tool documentation to inject into the Retriever.
    """

    lines = ["## Football Knowledge Graph Tools Settings"]
    for t in RETRIEVER_TOOLS:
        desc = t.description or ""
        
        # 1. Inputs: name and type
        input_info = []
        args_schema = getattr(t, "args_schema", None)
        if args_schema is not None:
            try:
                schema = args_schema.model_json_schema()
            except Exception:
                try:
                    schema = args_schema.schema()
                except Exception:
                    schema = {}
            
            props = schema.get("properties") or {}
            required = schema.get("required") or []
            
            for p_name, p_info in props.items():
                p_type = p_info.get("type", "any")
                p_req = "required" if p_name in required else "optional"
                input_info.append(f"{p_name} ({p_type}, {p_req})")

        # 2. Extract specific metadata from docstring
        explanation = desc.split('Explanation: ')[1].split('\n')[0] if 'Explanation: ' in desc else desc
        output_info = desc.split('Output Format: ')[1].split('\n')[0] if 'Output Format: ' in desc else "Unknown"
        example_info = desc.split('Example: ')[1].split('\n')[0] if 'Example: ' in desc else "No example provided"

        tool_doc = f"### {t.name}\n"
        tool_doc += f"- **Explanation**: {explanation}\n"
        if input_info:
            tool_doc += f"- **Input Format**: {', '.join(input_info)}\n"
        tool_doc += f"- **Output Format**: {output_info}\n"
        tool_doc += f"- **Example**: `{example_info}`"
        
        lines.append(tool_doc)
        
    return "\n\n".join(lines)

