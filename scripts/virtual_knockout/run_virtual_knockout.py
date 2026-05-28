#!/usr/bin/env python3
"""run_virtual_knockout.py

In silico knockout validation for KGE-prioritized Microglia regulator candidates
using the scTenifoldKnk framework (Cabezas-Bratesco et al., Patterns, 2022).

Usage:
    python run_virtual_knockout.py
"""

import io
import time
import warnings
import numpy as np
import pandas as pd
from scipy.io import mmread
from pathlib import Path

from scTenifold import scTenifoldKnk

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
GENES_TO_KO = ["FMR1", "PTEN", "FKBP5"]
N_SAMP_CELLS = 600  # cells sampled per network
MAX_GENES = 3000    # gene cap after preprocessing

KGE_INFO = {
    "FMR1":   {"novel_rank": 6,  "global_rank": 763, "score": -5.755,
               "cells": "Astrocyte; Bergmann glial cell; Excitatory neuron; "
                        "Hippocampal CA1 PN; LAMP5+ interneuron; NSC; Radial glial cell"},
    "PTEN":   {"novel_rank": 26, "global_rank": 787, "score": -6.113,
               "cells": "Cerebellar inhibitory neuron; Hippocampal DG granule cell; "
                        "LAMP5+ interneuron; NSC; Pericyte"},
    "FKBP5":  {"novel_rank": 38, "global_rank": 801, "score": -6.197,
               "cells": "Astrocyte; Fibroblast"},
}


# ──────────────────────────────────────────────
# Data loading and preprocessing
# ──────────────────────────────────────────────

def load_expression_matrix():
    """Load Microglia expression matrix and downsample to 500 cells."""
    mtx_file = BASE_DIR / "data" / "microglia_matrix.mtx"
    gene_file = BASE_DIR / "data" / "microglia_gene_names.txt"

    print("\nLoading expression matrix...")
    with open(mtx_file, "rb") as f:
        mat = mmread(io.BytesIO(f.read())).tocsr()
    gene_names = open(gene_file, encoding="utf-8").read().strip().split("\n")
    if mat.shape[0] != len(gene_names):
        mat = mat.T.tocsr()

    # Downsample to 500 cells for input loading
    n_total = mat.shape[1]
    n_qc = min(500, n_total)
    np.random.seed(42)
    sel_cols = np.random.choice(n_total, n_qc, replace=False)
    mat_sub = mat[:, sel_cols]

    df = pd.DataFrame(mat_sub.toarray(), index=gene_names)
    print(f"  Loaded: {df.shape[0]} genes x {df.shape[1]} cells (downsampled from {n_total})")
    return df


def preprocess_genes(df, max_genes=3000):
    """Filter low-expression genes and cap gene count for memory."""
    gene_mask = (df.mean(axis=1) >= 0.05) & (df.sum(axis=1) >= 25)
    df = df.loc[gene_mask]
    if df.shape[0] > max_genes:
        top_genes = df.sum(axis=1).nlargest(max_genes).index
        ko_genes = [g for g in GENES_TO_KO if g in df.index]
        keep = set(top_genes) | set(ko_genes)
        df = df.loc[sorted(keep, key=lambda x: df.index.get_loc(x))]
    print(f"  After gene filtering: {df.shape[0]} genes")
    return df


# ──────────────────────────────────────────────
# Main pipeline using scTenifoldKnk
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Virtual Knockout - scTenifoldKnk")
    print(f"Genes: {', '.join(GENES_TO_KO)}")
    print(f"Sampled cells per network: {N_SAMP_CELLS}")
    print("=" * 60)

    df = load_expression_matrix()
    df = preprocess_genes(df, max_genes=MAX_GENES)

    # Check genes
    for gene in GENES_TO_KO:
        if gene in df.index:
            pct = (df.loc[gene] > 0).sum() / df.shape[1] * 100
            print(f"  {gene}: OK ({pct:.0f}% cells)")
        else:
            print(f"  {gene}: MISSING!")

    output_dir = BASE_DIR / "results"
    output_dir.mkdir(exist_ok=True)
    summary = []
    total_t0 = time.time()

    for gene in GENES_TO_KO:
        print(f"\n{'='*60}")
        print(f"[KO] {gene}  (KGE Novel Rank: {KGE_INFO[gene]['novel_rank']})")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            knk = scTenifoldKnk(
                data=df,
                ko_genes=gene,
                ko_method="default",
                qc_kws={"min_exp_avg": 0.05, "min_exp_sum": 25},
                nc_kws={"n_nets": 10, "n_samp_cells": N_SAMP_CELLS,
                        "n_comp": 3, "q": 0.95},
                ma_kws={"d": 30},
                dr_kws={"n_ko_genes": 1},
            )
            dr_df = knk.build()
            elapsed = time.time() - t0

            # Save results
            dr_df.to_csv(output_dir / f"{gene}_vk_results.csv", index=False)
            sig = dr_df[dr_df["adjusted p-value"] < 0.05]
            sig.to_csv(output_dir / f"{gene}_dr_genes.csv", index=False)
            n_sig = len(sig)

            summary.append({
                "Gene": gene,
                "KGE_Novel_Rank": KGE_INFO[gene]["novel_rank"],
                "KGE_Global_Rank": KGE_INFO[gene]["global_rank"],
                "KGE_Score": KGE_INFO[gene]["score"],
                "Known_Regulated_Cells": KGE_INFO[gene]["cells"],
                "Total_DR_genes": len(dr_df),
                "Significant_DR_genes": n_sig,
                "Top_DR_gene": dr_df.iloc[0]["Gene"] if len(dr_df) > 0 else "N/A",
                "Status": "OK",
            })

            print(f"  Done in {elapsed:.1f}s | Total: {len(dr_df)} genes, "
                  f"Significant (adj.p<0.05): {n_sig}")
            if len(dr_df) > 0:
                print("  Top 5 DR genes:")
                for _, row in dr_df.head(5).iterrows():
                    print(f"    {row['Gene']}: dist={row['Distance']:.3f}, "
                          f"adj.p={row['adjusted p-value']:.4e}")

        except Exception as e:
            print(f"  FAILED: {e}")
            summary.append({
                "Gene": gene, "Status": "FAILED",
                "KGE_Novel_Rank": KGE_INFO[gene]["novel_rank"],
                "KGE_Global_Rank": KGE_INFO[gene]["global_rank"],
                "Total_DR_genes": 0, "Significant_DR_genes": 0,
            })

    # Summary
    total_elapsed = time.time() - total_t0
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / "knockout_summary.csv", index=False)

    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE ({total_elapsed:.1f}s total)")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
