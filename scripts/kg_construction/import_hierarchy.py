#!/usr/bin/env python3
"""
Import the 3-level cell type hierarchy into Neo4j.

Reads the cell hierarchy data from CSV files and creates Level1, Level2, Level3
nodes with BELONGS_TO and PART_OF relationships.

Usage:
    python import_hierarchy.py --uri bolt://localhost:7687 --user neo4j --password YOUR_PASSWORD
"""

import argparse
import csv
from pathlib import Path

from neo4j import GraphDatabase


def parse_args():
    parser = argparse.ArgumentParser(description="Import cell hierarchy into Neo4j")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", required=True, help="Neo4j password")
    parser.add_argument("--database", default="neural_cell_kg", help="Neo4j database name")
    parser.add_argument("--cells", default="data/cell_hierarchy/neural_cells.csv", help="Cell nodes CSV")
    parser.add_argument("--relations", default="data/cell_hierarchy/hierarchy_relations.csv", help="Hierarchy relations CSV")
    return parser.parse_args()


def load_cells(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_relations(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    cells = load_cells(args.cells)
    relations = load_relations(args.relations)

    with driver.session(database=args.database) as session:
        # Clear existing data
        session.run("MATCH (n) DETACH DELETE n")

        # Create constraints
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Level1) REQUIRE n.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Level2) REQUIRE n.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Level3) REQUIRE n.name IS UNIQUE")

        # Create cell nodes
        for cell in cells:
            level = cell["level"]
            label = f"Level{level}"
            name = cell["cell_name"]
            session.run(
                f"MERGE (n:{label} {{name: $name}}) "
                "SET n.description = $desc, n.description_source = $src, n.level = $level",
                name=name, desc=cell.get("description", ""),
                src=cell.get("description_source", ""), level=int(level),
            )

        # Create hierarchy relationships
        for rel in relations:
            child = rel["child"]
            parent = rel["parent"]
            rel_type = rel["relation_type"]

            # Determine child level
            child_cell = next((c for c in cells if c["cell_name"] == child), None)
            if not child_cell:
                continue
            child_label = f"Level{child_cell['level']}"

            # Determine parent level
            parent_cell = next((c for c in cells if c["cell_name"] == parent), None)
            if not parent_cell:
                continue
            parent_label = f"Level{parent_cell['level']}"

            session.run(
                f"MATCH (c:{child_label} {{name: $child}}) "
                f"MATCH (p:{parent_label} {{name: $parent}}) "
                f"MERGE (c)-[:{rel_type}]->(p)",
                child=child, parent=parent,
            )

        # Verify
        result = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count")
        for record in result:
            print(f"  {record['label']}: {record['count']} nodes")

    driver.close()
    print("Cell hierarchy import completed.")


if __name__ == "__main__":
    main()
