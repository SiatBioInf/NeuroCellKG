#!/usr/bin/env python3
"""
Import marker gene data into Neo4j.

Reads marker gene data from TSV and creates Gene nodes with MARKER_OF
relationships to Level2/Level3 cell nodes.

Usage:
    python import_marker_genes.py --uri bolt://localhost:7687 --user neo4j --password YOUR_PASSWORD
"""

import argparse
import csv
from pathlib import Path

from neo4j import GraphDatabase


def parse_args():
    parser = argparse.ArgumentParser(description="Import marker genes into Neo4j")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", required=True, help="Neo4j password")
    parser.add_argument("--database", default="neural_cell_kg", help="Neo4j database name")
    parser.add_argument("--markers", default="data/molecular_fingerprints/marker_genes.tsv",
                        help="Marker genes TSV file")
    return parser.parse_args()


# Manual mappings for cell types that don't match Level3 nodes exactly
MANUAL_MAPPING = {
    "Hippocampal CA1, CA2, and CA3 pyramidal neuron": "Level2:Pyramidal neuron",
    "Hippocampal CA4 pyramidal neuron": None,  # Skip - no corresponding node
}


def main():
    args = parse_args()
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    markers = []
    with open(args.markers, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            markers.append(row)

    print(f"Loaded {len(markers)} marker gene records")

    with driver.session(database=args.database) as session:
        # Create constraints
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gene) REQUIRE g.ensembl_id IS UNIQUE")

        created_genes = set()
        created_rels = 0

        for row in markers:
            gene_symbol = row["gene_symbol"]
            ensembl_id = row["ensembl_id"]
            cell_type = row["cell_type"]

            # Check manual mapping
            if cell_type in MANUAL_MAPPING:
                target = MANUAL_MAPPING[cell_type]
                if target is None:
                    continue
                target_label, target_name = target.split(":")
            else:
                # Try Level3 first, then Level2
                result = session.run(
                    "MATCH (n:Level3 {name: $name}) RETURN 'Level3' AS label, n.name AS name "
                    "UNION "
                    "MATCH (n:Level2 {name: $name}) RETURN 'Level2' AS label, n.name AS name "
                    "LIMIT 1",
                    name=cell_type,
                )
                record = result.single()
                if record is None:
                    continue
                target_label = record["label"]
                target_name = record["name"]

            # Create Gene node
            if ensembl_id not in created_genes:
                session.run(
                    "MERGE (g:Gene {ensembl_id: $eid}) "
                    "SET g.symbol = $sym, g.biotype = $bio",
                    eid=ensembl_id, sym=gene_symbol,
                    bio=row.get("gene_biotype", "protein_coding"),
                )
                created_genes.add(ensembl_id)

            # Create MARKER_OF relationship
            session.run(
                f"MATCH (g:Gene {{ensembl_id: $eid}}) "
                f"MATCH (c:{target_label} {{name: $cname}}) "
                "MERGE (g)-[:MARKER_OF {"
                "  rank: $rank, score: $score, "
                "  pct_expressed: $pct_in, pct_others: $pct_out, "
                "  p_val_adj: $pval"
                "}]->(c)",
                eid=ensembl_id, cname=target_name,
                rank=to_int(row.get("rank")),
                score=to_float(row.get("score")),
                pct_in=to_float(row.get("pct_expressed")),
                pct_out=to_float(row.get("pct_others")),
                pval=to_float(row.get("p_val_adj")),
            )
            created_rels += 1

    driver.close()
    print(f"Created {len(created_genes)} Gene nodes and {created_rels} MARKER_OF relationships")


def to_int(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def to_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    main()
