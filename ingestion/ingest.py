import os
import sys

# Allow running as either:
# - python -m ingestion.ingest ...
# - python ingestion/ingest.py ...
if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from neo4j import GraphDatabase

from ingestion.loader import load_csv
from ingestion.schema import create_schema
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def ingest_data(csv_path):
    df = load_csv(csv_path)
    create_schema()

    with driver.session() as session:
        for row in df.to_dict("records"):
            query = """
            MERGE (p:Player {name: $name})
            MERGE (c:Club {name: $club})
            MERGE (l:League {name: $league})
            MERGE (pos:Position {name: $position})
            MERGE (n:Nationality {name: $nationality})

            MERGE (p)-[:PLAYS_FOR]->(c)
            MERGE (c)-[:PART_OF]->(l)
            MERGE (p)-[:HAS_POSITION]->(pos)
            MERGE (p)-[:HAS_NATIONALITY]->(n)
            """
            session.run(query, row)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage:\n"
            "  python -m ingestion.ingest <path_to_csv>\n"
            "  python ingestion/ingest.py <path_to_csv>"
        )

    ingest_data(sys.argv[1])