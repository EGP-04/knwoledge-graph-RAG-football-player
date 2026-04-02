POSITION_MAP = {
    "GK": "Goalkeeper",
    "CB": "Center Back",
    "LB": "Left Back",
    "RB": "Right Back",
    "CM": "Center Midfielder",
    "CDM": "Defensive Midfielder",
    "CAM": "Attacking Midfielder",
    "LW": "Left Wing",
    "RW": "Right Wing",
    "ST": "Striker",
    "CF": "Center Forward"
}

POSITION_GROUP = {
    "GK": "Goalkeeper",
    "CB": "Defender",
    "LB": "Defender",
    "RB": "Defender",
    "CM": "Midfielder",
    "CDM": "Midfielder",
    "CAM": "Midfielder",
    "LW": "Forward",
    "RW": "Forward",
    "ST": "Forward",
    "CF": "Forward"
}

POSITION_SYNONYMS = {
    "goalkeeper": "GK",
    "keeper": "GK",
    "defender": "CB",
    "center back": "CB",
    "midfielder": "CM",
    "attacking midfielder": "CAM",
    "defensive midfielder": "CDM",
    "forward": "ST",
    "striker": "ST",
    "winger": "RW",
    "right wing": "RW",
    "left wing": "LW"
}

def normalize_position(text):
    text = str(text).lower().strip()
    if text in POSITION_SYNONYMS:
        return POSITION_SYNONYMS[text]
    return text.upper()
