#!/usr/bin/env python3
"""
INTERACTS_WITH (PPI) link-prediction evaluation for train_v6.

Modelled on evaluate_regulates_lp.py, adapted for PPI:
  - Relation: INTERACTS_WITH instead of REGULATES
  - Candidate pool: only GENE:* entities (not all ~10,660 entities),
    since PPI is Gene-Gene.  This is semantically correct and avoids
    inflated metrics from ranking against structurally impossible candidates.
  - Symmetric handling: both head and tail predictions are computed.

Outputs
-------
  <outdir>/ppi_eval/per_triple.tsv     head/tail rank per triple
  <outdir>/ppi_eval/summary.json       MRR, Hits@1/3/10 (both/head/tail)
  <outdir>/ppi_eval/per_gene_rank.tsv  per-gene breakdown

Usage
-----
  python scripts/evaluate_ppi_lp.py \\
      --model-dir  outputs/rotate_seed42 \\
      --test       data/splits/test_ppi.tsv \\
      --train      data/splits/train.tsv
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pykeen.triples import TriplesFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPI link-prediction evaluation for train_v6."
    )
    parser.add_argument("--model-dir",  type=Path, default=Path("outputs/rotate_run"))
    parser.add_argument("--test",       type=Path, default=Path("data/splits/test_ppi.tsv"))
    parser.add_argument("--train",      type=Path, default=Path("data/splits/train.tsv"),
                        help="Used to build the filtered triple set.")
    parser.add_argument("--valid",      type=Path, default=Path("data/splits/valid.tsv"),
                        help="Also included in the filter set.")
    parser.add_argument("--outdir",     type=Path, default=None)
    parser.add_argument("--score-batch", type=int, default=1024,
                        help="Entity batch size per forward pass (tune for GPU memory).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["head", "relation", "tail"])
    return pd.read_csv(path, sep="\t")[["head", "relation", "tail"]].dropna()


def _hits_at_k(ranks: list[int], k: int) -> float:
    return float(sum(1 for r in ranks if r <= k) / len(ranks)) if ranks else 0.0


def _mrr(ranks: list[int]) -> float:
    return float(np.mean([1.0 / r for r in ranks])) if ranks else 0.0


def _score_batch(
    model: torch.nn.Module,
    heads: torch.Tensor,
    rel_id: int,
    tails: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Score a list of (head, rel, tail) triples in batches. Returns 1-D scores."""
    n = heads.size(0)
    scores = []
    rel_t = torch.tensor([rel_id], dtype=torch.long, device=device)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        h_b = heads[start:end].to(device)
        r_b = rel_t.expand(end - start)
        t_b = tails[start:end].to(device)
        hrt = torch.stack([h_b, r_b, t_b], dim=1)
        with torch.no_grad():
            s = model.score_hrt(hrt).squeeze(-1).cpu()
        scores.append(s)
    return torch.cat(scores)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.outdir is None:
        args.outdir = args.model_dir / "ppi_eval"
    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    import pykeen.nn.modules as _pykeen_modules
    if not hasattr(_pykeen_modules, "TuckERInteraction") and hasattr(_pykeen_modules, "TuckerInteraction"):
        _pykeen_modules.TuckERInteraction = _pykeen_modules.TuckerInteraction
    model = torch.load(
        args.model_dir / "trained_model.pkl", map_location="cpu", weights_only=False
    )
    model.eval()
    tf = TriplesFactory.from_path_binary(args.model_dir / "training_triples")
    device = next(model.parameters()).device

    if "INTERACTS_WITH" not in tf.relation_to_id:
        raise ValueError("INTERACTS_WITH not found in model vocabulary.")
    rel_id = tf.relation_to_id["INTERACTS_WITH"]

    # --- Build GENE-only candidate pool ---
    all_entity_labels = list(tf.entity_to_id.keys())
    gene_labels = sorted(e for e in all_entity_labels if e.startswith("GENE:"))
    gene_ids = torch.tensor(
        [tf.entity_to_id[e] for e in gene_labels], dtype=torch.long
    )
    # Build fast lookups: entity_id -> index in gene_labels list
    gene_label_to_idx = {label: i for i, label in enumerate(gene_labels)}
    id_to_gene_idx: dict[int, int] = {}
    for label in gene_labels:
        id_to_gene_idx[tf.entity_to_id[label]] = gene_label_to_idx[label]

    print(f"GENE candidate pool : {len(gene_labels)} entities")

    # --- Load test triples ---
    test_df = _load_df(args.test)
    test_ppi = test_df[test_df["relation"] == "INTERACTS_WITH"].copy()
    if test_ppi.empty:
        raise ValueError("No INTERACTS_WITH triples found in test file.")

    # --- Build filter sets (all known true triples for INTERACTS_WITH) ---
    # We filter within the GENE candidate pool only
    tail_filter: dict[int, set[int]] = defaultdict(set)  # h_id -> set of true t_ids
    head_filter: dict[int, set[int]] = defaultdict(set)  # t_id -> set of true h_ids

    for src_path in [args.train, args.valid, args.test]:
        src_df = _load_df(src_path)
        for _, row in src_df[src_df["relation"] == "INTERACTS_WITH"].iterrows():
            h, t = row["head"], row["tail"]
            if h in tf.entity_to_id and t in tf.entity_to_id:
                h_id = tf.entity_to_id[h]
                t_id = tf.entity_to_id[t]
                tail_filter[h_id].add(t_id)
                head_filter[t_id].add(h_id)

    # --- Per-triple evaluation ---
    rows: list[dict] = []
    skipped = 0

    for idx, (_, triple) in enumerate(test_ppi.iterrows()):
        h_label, t_label = triple["head"], triple["tail"]
        if h_label not in tf.entity_to_id or t_label not in tf.entity_to_id:
            skipped += 1
            continue

        h_id = tf.entity_to_id[h_label]
        t_id = tf.entity_to_id[t_label]

        # ---- Tail ranking: (h, INTERACTS_WITH, ?) among GENE entities ----
        h_tens = torch.full((len(gene_ids),), h_id, dtype=torch.long)
        all_tail_scores = _score_batch(model, h_tens, rel_id, gene_ids, device, args.score_batch)

        if t_label in gene_label_to_idx:
            t_gene_idx = gene_label_to_idx[t_label]
            true_score_t = all_tail_scores[t_gene_idx].item()
            filtered_scores_t = all_tail_scores.clone()
            for fid in tail_filter.get(h_id, set()) - {t_id}:
                if fid in id_to_gene_idx:
                    filtered_scores_t[id_to_gene_idx[fid]] = float("-inf")
            tail_rank = int((filtered_scores_t > true_score_t).sum().item()) + 1
        else:
            tail_rank = len(gene_labels)

        # ---- Head ranking: (?, INTERACTS_WITH, t) among GENE entities ----
        t_tens = torch.full((len(gene_ids),), t_id, dtype=torch.long)
        all_head_scores = _score_batch(model, gene_ids, rel_id, t_tens, device, args.score_batch)

        if h_label in gene_label_to_idx:
            h_gene_idx = gene_label_to_idx[h_label]
            true_score_h = all_head_scores[h_gene_idx].item()
            filtered_scores_h = all_head_scores.clone()
            for fid in head_filter.get(t_id, set()) - {h_id}:
                if fid in id_to_gene_idx:
                    filtered_scores_h[id_to_gene_idx[fid]] = float("-inf")
            head_rank = int((filtered_scores_h > true_score_h).sum().item()) + 1
        else:
            head_rank = len(gene_labels)

        rows.append({
            "head":      h_label,
            "tail":      t_label,
            "head_rank": head_rank,
            "tail_rank": tail_rank,
            "head_rr":   round(1.0 / head_rank, 6),
            "tail_rr":   round(1.0 / tail_rank, 6),
        })

        if (idx + 1) % 200 == 0:
            print(f"  Evaluated {idx + 1}/{len(test_ppi)} triples...")

    if skipped:
        print(f"WARNING: {skipped} test triples skipped (entity not in vocabulary).")

    per_triple_df = pd.DataFrame(rows)
    per_triple_df.to_csv(args.outdir / "per_triple.tsv", sep="\t", index=False)

    head_ranks = per_triple_df["head_rank"].tolist()
    tail_ranks = per_triple_df["tail_rank"].tolist()
    both_ranks = head_ranks + tail_ranks

    # --- Global summary ---
    summary: dict = {
        "model":            args.model_dir.name,
        "n_test_triples":   len(per_triple_df),
        "n_skipped":        skipped,
        "gene_pool_size":   len(gene_labels),
        # Both (standard reporting)
        "mrr_both":         round(_mrr(both_ranks), 6),
        "hits_at_1_both":   round(_hits_at_k(both_ranks, 1), 6),
        "hits_at_3_both":   round(_hits_at_k(both_ranks, 3), 6),
        "hits_at_10_both":  round(_hits_at_k(both_ranks, 10), 6),
        # Head (Gene ranking as head)
        "mrr_head":         round(_mrr(head_ranks), 6),
        "hits_at_1_head":   round(_hits_at_k(head_ranks, 1), 6),
        "hits_at_10_head":  round(_hits_at_k(head_ranks, 10), 6),
        # Tail (Gene ranking as tail)
        "mrr_tail":         round(_mrr(tail_ranks), 6),
        "hits_at_1_tail":   round(_hits_at_k(tail_ranks, 1), 6),
        "hits_at_10_tail":  round(_hits_at_k(tail_ranks, 10), 6),
    }

    # --- Per-gene breakdown ---
    gene_rows = []
    for gene, grp in per_triple_df.groupby("head"):
        g_head_ranks = grp["head_rank"].tolist()
        g_tail_ranks = grp["tail_rank"].tolist()
        g_both_ranks = g_head_ranks + g_tail_ranks
        gene_rows.append({
            "gene":        gene,
            "n_test":      len(grp),
            "mrr_both":    round(_mrr(g_both_ranks), 6),
            "hits_at_1":   round(_hits_at_k(g_both_ranks, 1), 6),
            "hits_at_10":  round(_hits_at_k(g_both_ranks, 10), 6),
            "median_rank": int(np.median(g_both_ranks)),
        })
    gene_df = pd.DataFrame(gene_rows).sort_values("mrr_both", ascending=False)
    gene_df.to_csv(args.outdir / "per_gene_rank.tsv", sep="\t", index=False)

    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n=== PPI Link Prediction: {args.model_dir.name} ===")
    print(json.dumps(summary, indent=2))
    print(f"\nPer-gene breakdown : {args.outdir / 'per_gene_rank.tsv'}")


if __name__ == "__main__":
    main()
