#!/usr/bin/env python3
"""
Collect and print comparison tables across all four models (train_v6).

Reads (for each model run directory):
  metrics_summary.json                 KGE link-prediction on REGULATES test set
  regulates_eval/summary.json          Detailed REGULATES LP breakdown
  marker_eval/summary_metrics.json     MARKER_OF retrieval
  ppi_eval/summary.json                PPI link prediction

All numeric values are reported as mean +/- std (populated by aggregate_seed_results.py).

Also loads the random-marker baseline from outputs/random_baseline/marker_eval/.

Tables printed
--------------
  Table 1: KGE Link Prediction (REGULATES test set, filtered MRR)
  Table 2: MARKER_OF Retrieval (held-out test_marker.tsv, macro-averaged)
  Table 3: PPI Link Prediction (INTERACTS_WITH test set, GENE-only pool)

Usage
-----
  python scripts/compare_models.py
  python scripts/compare_models.py --outdirs outputs/transe_run outputs/rotate_run ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate comparison table for train_v6."
    )
    parser.add_argument(
        "--outdirs",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/transe_run"),
            Path("outputs/rotate_run"),
            Path("outputs/complex_run"),
            Path("outputs/tucker_run"),
        ],
    )
    parser.add_argument("--marker-eval-subdir",    type=str, default="marker_eval")
    parser.add_argument("--ppi-eval-subdir",       type=str, default="ppi_eval")
    parser.add_argument(
        "--random-baseline-dir",
        type=Path,
        default=Path("outputs/random_baseline/marker_eval"),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _fmt(mean, std=None) -> str:
    if mean is None:
        return "—"
    if isinstance(mean, (int, float)):
        if std is not None and isinstance(std, float) and std > 0:
            return f"{mean:.4f}±{std:.4f}"
        return f"{mean:.4f}"
    return str(mean)


def main() -> None:
    args = parse_args()

    rows = []
    for outdir in args.outdirs:
        kge = _load_json(outdir / "metrics_summary.json")
        if not kge:
            continue
        model = kge.get("model") or outdir.name

        def v(key: str) -> tuple:
            return kge.get(key), kge.get(f"{key}_std")

        row: dict = {
            "Model":   model,
            # KGE link-prediction on REGULATES test set
            "MRR":     _fmt(*v("both.realistic.mrr")),
            "Hits@1":  _fmt(*v("both.realistic.hits_at_1")),
            "Hits@3":  _fmt(*v("both.realistic.hits_at_3")),
            "Hits@10": _fmt(*v("both.realistic.hits_at_10")),
        }

        # Marker retrieval
        marker = _load_json(outdir / args.marker_eval_subdir / "summary_metrics.json")
        if marker:
            def mv(k):
                return marker.get(k), marker.get(f"{k}_std")
            row["P@10"] = _fmt(*mv("precision@10_macro"))
            row["R@10"] = _fmt(*mv("recall@10_macro"))
            row["P@20"] = _fmt(*mv("precision@20_macro"))
            row["R@20"] = _fmt(*mv("recall@20_macro"))
            row["P@30"] = _fmt(*mv("precision@30_macro"))
            row["R@30"] = _fmt(*mv("recall@30_macro"))
            row["MPRR"] = _fmt(*mv("mean_positive_rr_macro"))

        # PPI link prediction
        ppi = _load_json(outdir / args.ppi_eval_subdir / "summary.json")
        if ppi:
            def pv(k):
                return ppi.get(k), ppi.get(f"{k}_std")
            row["PPI_MRR"]     = _fmt(*pv("mrr_both"))
            row["PPI_H@1"]     = _fmt(*pv("hits_at_1_both"))
            row["PPI_H@3"]     = _fmt(*pv("hits_at_3_both"))
            row["PPI_H@10"]    = _fmt(*pv("hits_at_10_both"))

        rows.append(row)

    # Random marker baseline row
    rand = _load_json(args.random_baseline_dir / "summary_metrics.json")
    if rand:
        def rv(k):
            return rand.get(k), None
        rows.append({
            "Model":   "Random",
            "MRR":     "—",
            "Hits@1":  "—",
            "Hits@3":  "—",
            "Hits@10": "—",
            "P@10":    _fmt(*rv("precision@10_macro")),
            "R@10":    _fmt(*rv("recall@10_macro")),
            "P@20":    _fmt(*rv("precision@20_macro")),
            "R@20":    _fmt(*rv("recall@20_macro")),
            "P@30":    _fmt(*rv("precision@30_macro")),
            "R@30":    _fmt(*rv("recall@30_macro")),
            "MPRR":    _fmt(*rv("mean_positive_rr_macro")),
            "PPI_MRR": "—",
            "PPI_H@1": "—",
            "PPI_H@3": "—",
            "PPI_H@10": "—",
        })

    if not rows:
        print("No metrics found.  Run run_all.sh first.")
        return

    df = pd.DataFrame(rows)

    print("\n=== Table 1: KGE Link Prediction (REGULATES test set, filtered MRR) ===")
    kge_cols = [c for c in ["Model", "MRR", "Hits@1", "Hits@3", "Hits@10"] if c in df.columns]
    print(df[kge_cols].to_string(index=False))

    marker_cols = [
        c for c in ["Model", "P@10", "R@10", "P@20", "R@20", "P@30", "R@30", "MPRR"]
        if c in df.columns
    ]
    if len(marker_cols) > 1:
        print("\n=== Table 2: MARKER_OF Retrieval (held-out test_marker.tsv, macro-avg) ===")
        print(df[marker_cols].to_string(index=False))

    ppi_cols = [
        c for c in ["Model", "PPI_MRR", "PPI_H@1", "PPI_H@3", "PPI_H@10"]
        if c in df.columns
    ]
    if len(ppi_cols) > 1:
        print("\n=== Table 3: PPI Link Prediction (INTERACTS_WITH test set, GENE-only pool) ===")
        print(df[ppi_cols].to_string(index=False))

    out_path = Path("outputs/comparison_table.tsv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\nFull table saved: {out_path}")

    print("\n=== Best Model Selection ===")
    for _, r in df.iterrows():
        if r["Model"] == "Random":
            continue
        print(
            f"  {r['Model']:10s}  MRR={r.get('MRR', '—')}  "
            f"R@30={r.get('R@30', '—')}  PPI_MRR={r.get('PPI_MRR', '—')}"
        )


if __name__ == "__main__":
    main()
