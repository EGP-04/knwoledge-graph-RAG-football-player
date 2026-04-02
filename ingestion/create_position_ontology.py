from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

POSITION_MAP = {
    "GK": "Goalkeeper",
    "CB": "Center Back",
    "LB": "Left Back",
    "RB": "Right Back",
    "LWB": "Left Wing Back",
    "RWB": "Right Wing Back",
    "CM": "Center Midfielder",
    "CDM": "Defensive Midfielder",
    "CAM": "Attacking Midfielder",
    "AM": "Attacking Midfielder",
    "LW": "Left Wing",
    "RW": "Right Wing",
    "ST": "Striker",
    "CF": "Center Forward",
    "SW": "Sweeper"
}

POSITION_GROUP = {
    "GK": "Goalkeeper",
    "CB": "Defender",
    "LB": "Defender",
    "RB": "Defender",
    "LWB": "Defender",
    "RWB": "Defender",
    "SW": "Defender",
    "CM": "Midfielder",
    "CDM": "Midfielder",
    "CAM": "Midfielder",
    "AM": "Midfielder",
    "LW": "Forward",
    "RW": "Forward",
    "ST": "Forward",
    "CF": "Forward"
}

def create_ontology():
    with driver.session() as session:
        for code, name in POSITION_MAP.items():
            group = POSITION_GROUP[code]

            session.run("""
                MERGE (g:PositionGroup {name:$group})
                MERGE (p:Position {code:$code})
                SET p.name = $name
                MERGE (p)-[:BELONGS_TO]->(g)
            """, code=code, name=name, group=group)

    print("Position ontology created.")

if __name__ == "__main__":
    create_ontology()