from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def create_schema():
    queries = [
        "CREATE CONSTRAINT player IF NOT EXISTS FOR (p:Player) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT club IF NOT EXISTS FOR (c:Club) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT league IF NOT EXISTS FOR (l:League) REQUIRE l.name IS UNIQUE",
        "CREATE CONSTRAINT position IF NOT EXISTS FOR (pos:Position) REQUIRE pos.name IS UNIQUE",
        "CREATE CONSTRAINT nationality IF NOT EXISTS FOR (n:Nationality) REQUIRE n.name IS UNIQUE",
    ]

    with driver.session() as session:
        for q in queries:
            session.run(q)