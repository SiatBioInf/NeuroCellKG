#!/usr/bin/env python3
"""
Export knowledge graph triples from Neo4j for KGE training.

Exports all entity-to-entity triples (excluding Paper nodes) in
head/relation/tail TSV format suitable for knowledge graph embedding.

Usage:
    python export_triples.py --uri bolt://localhost:7687 --user neo4j --password YOUR_PASSWORD
"""

import argparse
from pathlib import Path

from neo4j import GraphDatabase


def parse_args():
    parser = argparse.ArgumentParser(description="Export KG triples from Neo4j")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", required=True, help="Neo4j password")
    parser.add_argument("--database", default="neural_cell_kg", help="Neo4j database name")
    parser.add_argument("--output", default="data/kge_splits/all_triples.tsv", help="Output TSV file")
    return parser.parse_args()


LABEL_PREFIX = {
    "Level1": "L1",
    "Level2": "L2",
    "Level3": "L3",
    "Gene": "GENE",
    "Factor": "FACTOR",
}


def main():
    args = parse_args()
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with driver.session(database=args.database) as session:
        # Get all relationships where neither endpoint is a Paper node
        result = session.run(
            "MATCH (h)-[r]->(t) "
            "WHERE NOT 'Paper' IN labels(h) AND NOT 'Paper' IN labels(t) "
            "RETURN labels(h)[0] AS h_label, h.name AS h_name, "
            "       type(r) AS rel_type, "
            "       labels(t)[0] AS t_label, t.name AS t_name"
        )

        records = list(result)

    driver.close()

    # Write triples
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("head\trelation\ttail\n")
        for rec in records:
            h_prefix = LABEL_PREFIX.get(rec["h_label"], "")
            t_prefix = LABEL_PREFIX.get(rec["t_label"], "")
            h_name = f"{h_prefix}:{rec['h_name']}" if h_prefix else rec["h_name"]
            t_name = f"{t_prefix}:{rec['t_name']}" if t_prefix else rec["t_name"]
            f.write(f"{h_name}\t{rec['rel_type']}\t{t_name}\n")

    # Print statistics
    from collections import Counter
    rel_counts = Counter(rec["rel_type"] for rec in records)
    print(f"Exported {len(records)} triples to {args.output}")
    for rel, count in sorted(rel_counts.items()):
        print(f"  {rel}: {count}")


if __name__ == "__main__":
    main()
