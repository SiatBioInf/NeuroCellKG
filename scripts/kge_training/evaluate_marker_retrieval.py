#!/usr/bin/env python3
"""
Marker-gene retrieval evaluation for train_v5.

For each L3 cell type with at least one held-out ground-truth marker gene,
all GENE entities in the model vocabulary are ranked by the score of the
triple (GENE, MARKER_OF, CELL).  Retrieval metrics are then computed against
the set of known true markers for that cell.

Scoring procedure
-----------------
We call model.score_hrt() directly on all (gene, MARKER_OF, cell) combinations
rather than using pykeen.predict.predict_target, which returns incorrect scores
for models trained under the sLCWA objective (TransE, RotatE).  Scoring all
candidate (gene, MARKER_OF, cell) triples via score_hrt() is model-agnostic
and produces consistent results across all four architectures.

Gene pool (filtered setting)
-----------------------------
For each cell, the candidate gene pool is the full GENE:* vocabulary **minus**
the marker genes that appear as training-set positives for that cell.  This
mirrors the standard filtered evaluation used in link prediction (FB15k-237,
etc.) and prevents training positives from crowding out held-out test positives
in the ranking.  The pool size therefore varies slightly per cell (≈40 genes
instead of 80 when the 80/20 split is exact).

Pass --no-filter to disable this behaviour and rank over all 80 genes (useful
for reproducing the old, biased results for comparison).

Metrics (macro-averaged across evaluated cells)
-----------------------------------------------
  Precision@k  =  |top-k ∩ true markers| / k
  Recall@k     =  |top-k ∩ true markers| / |true markers|
  Overlap@k    =  |top-k ∩ true markers|           (integer count)
  MPRR         =  mean over true markers of 1 / rank(marker)
                  (Mean Positive Reciprocal Rank; summarises the full ranking)

k values reported: 5, 10, 20, 30

Outputs
-------
  <outdir>/per_cell_metrics.tsv
  <outdir>/summary_metrics.json   <- read by aggregate_seed_results.py

Usage
-----
  python scripts/kge_training/evaluate_marker_retrieval.py \\
      --model-dir       outputs/rotate_seed42 \\
      --triples         data/kge_splits/test_marker.txt \\
      --train-triples   data/kge_splits/train.txt \\
      --ks              5 10 20 30 \\
      --score-batch-size 512

  # Reproduce old (unfiltered) results for comparison:
  python scripts/kge_training/evaluate_marker_retrieval.py \\
      --model-dir outputs/rotate_seed42 --no-filter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from pykeen.triples import TriplesFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Marker-gene retrieval evaluation for train_v5."
    )
    parser.add_argument("--model-dir",        type=Path, default=Path("outputs/transe_run"))
    parser.add_argument(
        "--triples",
        type=Path,
        default=Path("data/kge_splits/test_marker.txt"),
        help="Held-out MARKER_OF triples produced by split_triples.py.",
    )
    parser.add_argument(
        "--train-triples",
        type=Path,
        default=Path("data/kge_splits/train.txt"),
        help="Training triples used to build the filtered gene pool (MARKER_OF positives excluded).",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        default=False,
        help="Disable filtered evaluation: rank over all GENE entities (reproduces old biased results).",
    )
    parser.add_argument("--cell-prefix",       type=str, default="L3:")
    parser.add_argument("--ks",                type=int, nargs="+", default=[5, 10, 20, 30])
    parser.add_argument("--min-markers",       type=int, default=1,
                        help="Minimum true markers per cell to include in evaluation.")
    parser.add_argument("--max-cells",         type=int, default=0,
                        help="Limit to first N cells (0 = no limit; for debugging).")
    parser.add_argument("--score-batch-size",  type=int, default=512,
                        help="Gene entities scored per forward pass (tune for GPU memory).")
    parser.add_argument("--outdir",            type=Path, default=None)
    return parser.parse_args()


def _sorted_unique_ints(values: Iterable[int]) -> list[int]:
    return sorted({int(v) for v in values if int(v) > 0})


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _rank_genes_for_cell(
    model: torch.nn.Module,
    gene_ids: torch.Tensor,
    gene_labels: list[str],
    rel_id: int,
    cell_id: int,
    device: torch.device,
    batch_size: int,
) -> list[str]:
    """
    Score every (gene, MARKER_OF, cell) triple and return gene labels
    sorted by descending score.
    """
    scores: list[torch.Tensor] = []
    n = len(gene_ids)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        h_b = gene_ids[start:end].to(device)
        bs  = h_b.size(0)
        r_b = torch.full((bs,), rel_id,  dtype=torch.long, device=device)
        t_b = torch.full((bs,), cell_id, dtype=torch.long, device=device)
        hrt = torch.stack([h_b, r_b, t_b], dim=1)
        with torch.no_grad():
            s = model.score_hrt(hrt).squeeze(-1).cpu()
        scores.append(s)

    scores_np = torch.cat(scores).numpy()
    order = np.argsort(-scores_np)  # descending
    return [gene_labels[i] for i in order]


def main() -> None:
    args = parse_args()
    if args.outdir is None:
        args.outdir = args.model_dir / "marker_eval"
    args.outdir.mkdir(parents=True, exist_ok=True)

    ks = _sorted_unique_ints(args.ks)
    if not ks:
        raise ValueError("No valid k values provided via --ks.")

    # --- Ground truth (test set) ---
    triples = pd.read_csv(args.triples, sep="\t")[["head", "relation", "tail"]].dropna()
    marker_df = triples[
        (triples["relation"] == "MARKER_OF")
        & (triples["head"].str.startswith("GENE:"))
        & (triples["tail"].str.startswith(args.cell_prefix))
    ].drop_duplicates()

    if marker_df.empty:
        raise ValueError("No MARKER_OF triples found under the current filters.")

    gt: dict[str, set[str]] = {}
    for _, row in marker_df.iterrows():
        gt.setdefault(row["tail"], set()).add(row["head"])

    # --- Training-set positives (for filtered gene pool) ---
    train_gt: dict[str, set[str]] = {}
    if not args.no_filter:
        if not args.train_triples.exists():
            raise FileNotFoundError(
                f"Train triples file not found: {args.train_triples}\n"
                "Pass --no-filter to skip filtered evaluation."
            )
        train_triples_df = pd.read_csv(
            args.train_triples, sep="\t"
        )[["head", "relation", "tail"]].dropna()
        train_marker_df = train_triples_df[
            (train_triples_df["relation"] == "MARKER_OF")
            & (train_triples_df["head"].str.startswith("GENE:"))
            & (train_triples_df["tail"].str.startswith(args.cell_prefix))
        ].drop_duplicates()
        for _, row in train_marker_df.iterrows():
            train_gt.setdefault(row["tail"], set()).add(row["head"])

    cells = sorted([c for c, genes in gt.items() if len(genes) >= args.min_markers])
    if args.max_cells > 0:
        cells = cells[: args.max_cells]
    if not cells:
        raise ValueError("No cells remain after applying --min-markers filter.")

    # --- Load model ---
    # Compatibility patch: pykeen >=1.10 renamed TuckERInteraction -> TuckerInteraction
    import pykeen.nn.modules as _pykeen_modules
    if not hasattr(_pykeen_modules, "TuckERInteraction") and hasattr(_pykeen_modules, "TuckerInteraction"):
        _pykeen_modules.TuckERInteraction = _pykeen_modules.TuckerInteraction
    model_path = args.model_dir / "trained_model.pkl"
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()
    tf = TriplesFactory.from_path_binary(args.model_dir / "training_triples")

    if "MARKER_OF" not in tf.relation_to_id:
        raise ValueError(
            "MARKER_OF relation not found in model vocabulary.  "
            "Ensure MARKER_OF triples were included in the training set."
        )
    rel_id = tf.relation_to_id["MARKER_OF"]
    device = next(model.parameters()).device

    # Full gene pool: all GENE:* entities known to the model
    all_labels = list(tf.entity_to_id.keys())
    all_gene_labels = sorted(e for e in all_labels if e.startswith("GENE:"))

    filter_mode = not args.no_filter
    print(
        f"Model        : {args.model_dir.name}\n"
        f"Gene pool    : {len(all_gene_labels)} entities (full)\n"
        f"Filtered eval: {'YES — training positives excluded per cell' if filter_mode else 'NO  — all genes ranked (unfiltered)'}\n"
        f"Cells        : {len(cells)}\n"
        f"GT triples   : {len(marker_df)}\n"
        f"k values     : {ks}"
    )

    # --- Per-cell evaluation ---
    per_cell_rows: list[dict] = []

    for idx, cell in enumerate(cells, start=1):
        if cell not in tf.entity_to_id:
            print(f"  WARNING: '{cell}' not in model vocabulary — skipped.")
            continue

        cell_id = tf.entity_to_id[cell]

        # Build (optionally filtered) gene pool for this cell
        if filter_mode:
            train_positives_for_cell = train_gt.get(cell, set())
            gene_labels = [g for g in all_gene_labels if g not in train_positives_for_cell]
        else:
            gene_labels = all_gene_labels
        gene_ids = torch.tensor(
            [tf.entity_to_id[g] for g in gene_labels], dtype=torch.long
        )

        ranked  = _rank_genes_for_cell(
            model, gene_ids, gene_labels, rel_id, cell_id, device, args.score_batch_size,
        )
        rank_map  = {gene: i + 1 for i, gene in enumerate(ranked)}
        positives = gt[cell]

        row: dict = {
            "cell":             cell,
            "num_true_markers": len(positives),
            "num_ranked_genes": len(ranked),
        }

        # MPRR: average reciprocal rank over all true markers
        rr_values = [_safe_div(1.0, rank_map[g]) for g in positives if g in rank_map]
        row["mean_positive_rr"] = float(sum(rr_values) / len(rr_values)) if rr_values else 0.0

        for k in ks:
            topk    = set(ranked[:k])
            overlap = len(topk & positives)
            row[f"overlap@{k}"]   = overlap
            row[f"precision@{k}"] = _safe_div(overlap, k)
            row[f"recall@{k}"]    = _safe_div(overlap, len(positives))

        per_cell_rows.append(row)

        if idx % 10 == 0 or idx == len(cells):
            print(f"  Evaluated {idx}/{len(cells)} cells")

    per_cell_df = pd.DataFrame(per_cell_rows)
    per_cell_df.to_csv(args.outdir / "per_cell_metrics.tsv", sep="\t", index=False)

    # --- Macro-average summary ---
    summary: dict = {
        "model":                              args.model_dir.name,
        "filtered_eval":                      filter_mode,
        "num_cells_evaluated":                int(len(per_cell_df)),
        "num_marker_triples_in_ground_truth": int(len(marker_df)),
        "full_gene_pool_size":                len(all_gene_labels),
        "mean_gene_pool_size_per_cell":       float(per_cell_df["num_ranked_genes"].mean()),
        "mean_num_true_markers_per_cell":     float(per_cell_df["num_true_markers"].mean()),
        "mean_positive_rr_macro":             float(per_cell_df["mean_positive_rr"].mean()),
    }
    for k in ks:
        summary[f"precision@{k}_macro"] = float(per_cell_df[f"precision@{k}"].mean())
        summary[f"recall@{k}_macro"]    = float(per_cell_df[f"recall@{k}"].mean())
        summary[f"overlap@{k}_macro"]   = float(per_cell_df[f"overlap@{k}"].mean())

    (args.outdir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nSummary:\n{json.dumps(summary, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
