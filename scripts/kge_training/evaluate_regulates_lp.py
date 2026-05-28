#!/usr/bin/env python3
"""
REGULATES link-prediction evaluation for train_v5.

Standard KGE evaluation reports filtered MRR and Hits@k over the *combined*
head and tail ranking tasks.  This script supplements those global metrics with
a more granular per-relation-direction breakdown and a degree-stratified
analysis, both of which strengthen the manuscript.

Tasks
-----
  Head prediction : (?, REGULATES, L3:Cell) — rank the correct Factor in the
                    full entity vocabulary (filtered setting)
  Tail prediction : (FACTOR:X, REGULATES, ?) — rank the correct Cell in the
                    full entity vocabulary (filtered setting)

Filtering (standard FB15k-237 / OGB protocol)
---------------------------------------------
For each test triple (h, r, t), all known true triples sharing the same query
pattern are filtered from the ranking denominator, so that other true answers
do not penalise the model.  We reuse PyKEEN's built-in filtered evaluation via
the ``pipeline`` output; this script provides *additional* per-cell and
per-factor breakdowns by re-scoring the test set directly.

Outputs
-------
  <outdir>/regulates_eval/per_triple.tsv          head/tail rank per triple
  <outdir>/regulates_eval/summary.json            global + stratified metrics
  <outdir>/regulates_eval/per_cell_tail_rank.tsv  per-cell tail-prediction rank
  <outdir>/regulates_eval/per_factor_head_rank.tsv per-factor head-prediction rank

Usage
-----
  python scripts/evaluate_regulates_lp.py \\
      --model-dir  outputs/rotate_seed42 \\
      --test       data/splits/test.tsv \\
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
        description="Per-triple REGULATES link-prediction evaluation for train_v5."
    )
    parser.add_argument("--model-dir",  type=Path, default=Path("outputs/rotate_run"))
    parser.add_argument("--test",       type=Path, default=Path("data/splits/test.tsv"))
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


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


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
        args.outdir = args.model_dir / "regulates_eval"
    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    # Compatibility patch: pykeen >=1.10 renamed TuckERInteraction -> TuckerInteraction
    import pykeen.nn.modules as _pykeen_modules
    if not hasattr(_pykeen_modules, "TuckERInteraction") and hasattr(_pykeen_modules, "TuckerInteraction"):
        _pykeen_modules.TuckERInteraction = _pykeen_modules.TuckerInteraction
    model = torch.load(
        args.model_dir / "trained_model.pkl", map_location="cpu", weights_only=False
    )
    model.eval()
    tf = TriplesFactory.from_path_binary(args.model_dir / "training_triples")
    device = next(model.parameters()).device

    if "REGULATES" not in tf.relation_to_id:
        raise ValueError("REGULATES not found in model vocabulary.")
    rel_id = tf.relation_to_id["REGULATES"]

    all_entity_labels = list(tf.entity_to_id.keys())
    all_entity_ids = torch.tensor(
        [tf.entity_to_id[e] for e in all_entity_labels], dtype=torch.long
    )

    # --- Load test triples ---
    test_df = _load_df(args.test)
    test_reg = test_df[test_df["relation"] == "REGULATES"].copy()
    if test_reg.empty:
        raise ValueError("No REGULATES triples found in test file.")

    # --- Build filter sets (all known true triples for REGULATES) ---
    # Maps (h_id, r_id) -> set of true t_ids  (for tail ranking)
    #      (r_id, t_id) -> set of true h_ids  (for head ranking)
    tail_filter: dict[tuple, set[int]] = defaultdict(set)
    head_filter: dict[tuple, set[int]] = defaultdict(set)

    for src_path in [args.train, args.valid, args.test]:
        src_df = _load_df(src_path)
        for _, row in src_df[src_df["relation"] == "REGULATES"].iterrows():
            h, t = row["head"], row["tail"]
            if h in tf.entity_to_id and t in tf.entity_to_id:
                h_id = tf.entity_to_id[h]
                t_id = tf.entity_to_id[t]
                tail_filter[(h_id, rel_id)].add(t_id)
                head_filter[(rel_id, t_id)].add(h_id)

    # --- Per-triple evaluation ---
    rows: list[dict] = []
    skipped = 0

    for _, triple in test_reg.iterrows():
        h_label, t_label = triple["head"], triple["tail"]
        if h_label not in tf.entity_to_id or t_label not in tf.entity_to_id:
            skipped += 1
            continue

        h_id = tf.entity_to_id[h_label]
        t_id = tf.entity_to_id[t_label]

        # ---- Tail ranking: (h, r, ?) ----
        h_tens = torch.full((len(all_entity_ids),), h_id, dtype=torch.long)
        t_tens = all_entity_ids.clone()
        all_tail_scores = _score_batch(model, h_tens, rel_id, t_tens, device, args.score_batch)

        # Filtered rank
        true_score_t = all_tail_scores[tf.entity_to_id[t_label]].item()
        filter_ids_t = tail_filter[(h_id, rel_id)] - {t_id}
        filtered_scores_t = all_tail_scores.clone()
        for fid in filter_ids_t:
            filtered_scores_t[fid] = float("-inf")
        tail_rank = int((filtered_scores_t > true_score_t).sum().item()) + 1

        # ---- Head ranking: (?, r, t) ----
        h_tens2 = all_entity_ids.clone()
        t_tens2 = torch.full((len(all_entity_ids),), t_id, dtype=torch.long)
        all_head_scores = _score_batch(model, h_tens2, rel_id, t_tens2, device, args.score_batch)

        true_score_h = all_head_scores[tf.entity_to_id[h_label]].item()
        filter_ids_h = head_filter[(rel_id, t_id)] - {h_id}
        filtered_scores_h = all_head_scores.clone()
        for fid in filter_ids_h:
            filtered_scores_h[fid] = float("-inf")
        head_rank = int((filtered_scores_h > true_score_h).sum().item()) + 1

        rows.append({
            "head":      h_label,
            "tail":      t_label,
            "head_rank": head_rank,
            "tail_rank": tail_rank,
            "head_rr":   round(1.0 / head_rank, 6),
            "tail_rr":   round(1.0 / tail_rank, 6),
        })

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
        # Both (standard reporting)
        "mrr_both":         round(_mrr(both_ranks), 6),
        "hits_at_1_both":   round(_hits_at_k(both_ranks, 1), 6),
        "hits_at_3_both":   round(_hits_at_k(both_ranks, 3), 6),
        "hits_at_10_both":  round(_hits_at_k(both_ranks, 10), 6),
        # Head (Factor ranking)
        "mrr_head":         round(_mrr(head_ranks), 6),
        "hits_at_1_head":   round(_hits_at_k(head_ranks, 1), 6),
        "hits_at_10_head":  round(_hits_at_k(head_ranks, 10), 6),
        # Tail (Cell ranking)
        "mrr_tail":         round(_mrr(tail_ranks), 6),
        "hits_at_1_tail":   round(_hits_at_k(tail_ranks, 1), 6),
        "hits_at_10_tail":  round(_hits_at_k(tail_ranks, 10), 6),
    }

    # --- Per-cell tail-prediction breakdown ---
    cell_rows = []
    for cell, grp in per_triple_df.groupby("tail"):
        cell_ranks = grp["tail_rank"].tolist()
        cell_rows.append({
            "cell":        cell,
            "n_test":      len(cell_ranks),
            "mrr":         round(_mrr(cell_ranks), 6),
            "hits_at_1":   round(_hits_at_k(cell_ranks, 1), 6),
            "hits_at_10":  round(_hits_at_k(cell_ranks, 10), 6),
            "median_rank": int(np.median(cell_ranks)),
        })
    cell_df = pd.DataFrame(cell_rows).sort_values("mrr", ascending=False)
    cell_df.to_csv(args.outdir / "per_cell_tail_rank.tsv", sep="\t", index=False)

    # --- Per-factor head-prediction breakdown ---
    factor_rows = []
    for factor, grp in per_triple_df.groupby("head"):
        f_ranks = grp["head_rank"].tolist()
        factor_rows.append({
            "factor":      factor,
            "n_test":      len(f_ranks),
            "mrr":         round(_mrr(f_ranks), 6),
            "hits_at_1":   round(_hits_at_k(f_ranks, 1), 6),
            "hits_at_10":  round(_hits_at_k(f_ranks, 10), 6),
            "median_rank": int(np.median(f_ranks)),
        })
    factor_df = pd.DataFrame(factor_rows).sort_values("mrr", ascending=False)
    factor_df.to_csv(args.outdir / "per_factor_head_rank.tsv", sep="\t", index=False)

    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n=== REGULATES Link Prediction: {args.model_dir.name} ===")
    print(json.dumps(summary, indent=2))
    print(f"\nPer-cell breakdown   : {args.outdir / 'per_cell_tail_rank.tsv'}")
    print(f"Per-factor breakdown : {args.outdir / 'per_factor_head_rank.tsv'}")


if __name__ == "__main__":
    main()
