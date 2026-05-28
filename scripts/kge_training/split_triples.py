#!/usr/bin/env python3
"""
Split triples for train_v6 KGE training.

Relation handling
-----------------
  BELONGS_TO, PART_OF  → train only  (cell-type hierarchy; structural context)
  REGULATES            → 8:1:1 transductive-safe train / valid / test split
  MARKER_OF            → 80:20 transductive-safe train / test_marker holdout
  INTERACTS_WITH       → 80:20 transductive-safe train / test_ppi holdout
                          (pair-level split: both directions stay together)

Key design decisions
--------------------
  - UP_REGULATES and DOWN_REGULATES have been merged into REGULATES by the
    preprocessing step; this script therefore expects only REGULATES in the
    regulatory slot (raises an error if UP/DOWN still present, to catch pipeline
    mistakes early).

  - Transductive constraint: any triple moved to valid, test, test_marker, or
    test_ppi must have both its head and tail entities already present in the
    final training entity set.  Triples that violate this are moved back into
    train (backfill).  The final transductive check is verified twice (double-
    pass) to handle the rare case where backfill introduces new tail entities
    visible to remaining eval triples.

  - INTERACTS_WITH pair-level split: since PPI is undirected and stored as
    both (A→B) and (B→A), we group by canonical pair (min(h,t), max(h,t))
    and split at the pair level.  Both directed triples in a pair always
    land in the same split.

  - No overlap: all pairwise intersections between splits are empty.

  - Reproducibility: split is seeded; default seed=42.

  - Early stopping validation stays REGULATES-only (valid.tsv).

Output files
------------
  data/splits/train.tsv           (REGULATES train + MARKER_OF train + PPI train + STRUCTURAL)
  data/splits/valid.tsv           (REGULATES valid — used for KGE early stopping)
  data/splits/test.tsv            (REGULATES test  — used for KGE evaluation)
  data/splits/test_marker.tsv     (MARKER_OF holdout — used for retrieval eval)
  data/splits/test_ppi.tsv        (INTERACTS_WITH holdout — used for PPI LP eval)
  data/splits/split_stats.txt     (machine-readable provenance)

Usage
-----
  python scripts/split_triples.py \\
      --input  data/raw/triples_v6.tsv \\
      --outdir data/splits \\
      --seed   42 \\
      --train-ratio          0.8 \\
      --valid-ratio          0.1 \\
      --marker-holdout-ratio 0.2 \\
      --ppi-holdout-ratio    0.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

STRUCTURAL_RELS = {"BELONGS_TO", "PART_OF"}
FORBIDDEN_DIRECTION_RELS = {"UP_REGULATES", "DOWN_REGULATES"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transductive-safe train/valid/test split for train_v6."
    )
    parser.add_argument("--input",  type=Path, default=Path("data/raw/triples_v6.tsv"))
    parser.add_argument("--outdir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed",   type=int,  default=42)
    parser.add_argument("--train-ratio",          type=float, default=0.8)
    parser.add_argument("--valid-ratio",          type=float, default=0.1)
    parser.add_argument("--marker-holdout-ratio", type=float, default=0.2)
    parser.add_argument("--ppi-holdout-ratio",    type=float, default=0.2)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entities(df: pd.DataFrame) -> set[str]:
    if df.empty:
        return set()
    return set(df["head"].astype(str)) | set(df["tail"].astype(str))


def _count_unseen(df: pd.DataFrame, train_entities: set[str]) -> int:
    if df.empty:
        return 0
    bad = (~df["head"].isin(train_entities)) | (~df["tail"].isin(train_entities))
    return int(bad.sum())


def _move_unseen_to_train(
    candidate: pd.DataFrame,
    train_entities: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (keep, backfill) where backfill has at least one unseen entity."""
    if candidate.empty:
        empty = candidate.iloc[0:0].copy()
        return empty, empty
    ok = candidate["head"].isin(train_entities) & candidate["tail"].isin(train_entities)
    return candidate[ok].copy(), candidate[~ok].copy()


def _check_no_overlap(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_marker_df: pd.DataFrame,
    test_ppi_df: pd.DataFrame,
) -> None:
    def _to_set(df: pd.DataFrame) -> set:
        return set(map(tuple, df[["head", "relation", "tail"]].values.tolist()))

    sets = {
        "train":  _to_set(train_df),
        "valid":  _to_set(valid_df),
        "test":   _to_set(test_df),
        "marker": _to_set(test_marker_df),
        "ppi":    _to_set(test_ppi_df),
    }
    names = list(sets.keys())
    bad = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = f"{names[i]}∩{names[j]}"
            overlap = sets[names[i]] & sets[names[j]]
            if overlap:
                bad[key] = len(overlap)
    if bad:
        raise RuntimeError(f"Split integrity violation — overlapping triples: {bad}")


# ---------------------------------------------------------------------------
# Split functions
# ---------------------------------------------------------------------------

def _split_regulates_transductive(
    df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """8:1:1 random split for REGULATES triples with transductive safety."""
    test_ratio = 1.0 - train_ratio - valid_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + valid_ratio must be < 1.")
    if len(df) < 30:
        raise ValueError(f"Too few REGULATES triples to split robustly: {len(df)}")

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    df_train_raw, df_rest = train_test_split(
        df, test_size=(1.0 - train_ratio), random_state=seed, shuffle=True,
    )
    valid_frac = valid_ratio / (valid_ratio + test_ratio)
    df_valid_raw, df_test_raw = train_test_split(
        df_rest, test_size=(1.0 - valid_frac), random_state=seed, shuffle=True,
    )

    df_train_raw = df_train_raw.drop_duplicates()
    df_valid_raw = df_valid_raw.drop_duplicates()
    df_test_raw  = df_test_raw.drop_duplicates()

    # --- Pass 1: backfill ---
    train_ents = _entities(df_train_raw)
    df_valid_keep, df_valid_bf1 = _move_unseen_to_train(df_valid_raw, train_ents)
    df_test_keep,  df_test_bf1  = _move_unseen_to_train(df_test_raw,  train_ents)

    df_train = pd.concat(
        [df_train_raw, df_valid_bf1, df_test_bf1], axis=0, ignore_index=True
    ).drop_duplicates()

    # --- Pass 2: re-check after backfill (rare edge case) ---
    train_ents2 = _entities(df_train)
    df_valid_final, df_valid_bf2 = _move_unseen_to_train(df_valid_keep, train_ents2)
    df_test_final,  df_test_bf2  = _move_unseen_to_train(df_test_keep,  train_ents2)

    if len(df_valid_bf2) or len(df_test_bf2):
        df_train = pd.concat(
            [df_train, df_valid_bf2, df_test_bf2], axis=0, ignore_index=True
        ).drop_duplicates()

    final_train_ents = _entities(df_train)
    stats = {
        "raw_total":            len(df),
        "initial_train":        len(df_train_raw),
        "initial_valid":        len(df_valid_raw),
        "initial_test":         len(df_test_raw),
        "backfill_from_valid":  len(df_valid_bf1) + len(df_valid_bf2),
        "backfill_from_test":   len(df_test_bf1)  + len(df_test_bf2),
        "final_train":          len(df_train),
        "final_valid":          len(df_valid_final),
        "final_test":           len(df_test_final),
        "final_valid_unseen":   _count_unseen(df_valid_final, final_train_ents),
        "final_test_unseen":    _count_unseen(df_test_final,  final_train_ents),
    }
    return df_train, df_valid_final, df_test_final, stats


def _split_marker_transductive(
    df: pd.DataFrame,
    holdout_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """80:20 holdout for MARKER_OF triples with transductive safety."""
    if len(df) < 10:
        raise ValueError(f"Too few MARKER_OF triples to split: {len(df)}")

    df_train_raw, df_test_raw = train_test_split(
        df, test_size=holdout_ratio, random_state=seed, shuffle=True,
    )
    df_train_raw = df_train_raw.drop_duplicates()
    df_test_raw  = df_test_raw.drop_duplicates()

    # --- Pass 1 ---
    train_ents = _entities(df_train_raw)
    df_test_keep, df_test_bf1 = _move_unseen_to_train(df_test_raw, train_ents)
    df_train = pd.concat(
        [df_train_raw, df_test_bf1], axis=0, ignore_index=True
    ).drop_duplicates()

    # --- Pass 2 ---
    train_ents2 = _entities(df_train)
    df_test_final, df_test_bf2 = _move_unseen_to_train(df_test_keep, train_ents2)
    if len(df_test_bf2):
        df_train = pd.concat(
            [df_train, df_test_bf2], axis=0, ignore_index=True
        ).drop_duplicates()

    final_train_ents = _entities(df_train)
    stats = {
        "raw_total":               len(df),
        "initial_train":           len(df_train_raw),
        "initial_test_marker":     len(df_test_raw),
        "backfill_from_test":      len(df_test_bf1) + len(df_test_bf2),
        "final_train":             len(df_train),
        "final_test_marker":       len(df_test_final),
        "final_test_marker_unseen": _count_unseen(df_test_final, final_train_ents),
    }
    return df_train, df_test_final, stats


def _split_interacts_transductive(
    df: pd.DataFrame,
    holdout_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Pair-level 80:20 holdout for INTERACTS_WITH triples with transductive safety.

    Since PPI is undirected (stored as both A→B and B→A), we:
      1. Group triples by canonical pair: (min(h,t), max(h,t))
      2. Split at the pair level
      3. Expand pairs back to both directed triples
      4. Apply transductive backfill (same double-pass as REGULATES)
    """
    if len(df) < 10:
        raise ValueError(f"Too few INTERACTS_WITH triples to split: {len(df)}")

    # Assign canonical pair ID to each triple
    df = df.copy()
    df["_pair"] = df.apply(
        lambda r: (min(r["head"], r["tail"]), max(r["head"], r["tail"])),
        axis=1,
    )

    # Get unique pairs
    unique_pairs = df["_pair"].drop_duplicates().reset_index(drop=True)
    pairs_df = pd.DataFrame({"_pair": unique_pairs})
    n_pairs = len(pairs_df)
    print(f"  INTERACTS_WITH: {len(df)} triples, {n_pairs} canonical pairs")

    # Split at pair level
    pairs_train, pairs_test = train_test_split(
        pairs_df, test_size=holdout_ratio, random_state=seed, shuffle=True,
    )
    train_pair_set = set(pairs_train["_pair"])
    test_pair_set  = set(pairs_test["_pair"])

    df_train_raw = df[df["_pair"].isin(train_pair_set)].drop(columns=["_pair"]).drop_duplicates()
    df_test_raw  = df[df["_pair"].isin(test_pair_set)].drop(columns=["_pair"]).drop_duplicates()

    # --- Pass 1: backfill ---
    train_ents = _entities(df_train_raw)
    df_test_keep, df_test_bf1 = _move_unseen_to_train(df_test_raw, train_ents)
    df_train = pd.concat(
        [df_train_raw, df_test_bf1], axis=0, ignore_index=True
    ).drop_duplicates()

    # --- Pass 2 ---
    train_ents2 = _entities(df_train)
    df_test_final, df_test_bf2 = _move_unseen_to_train(df_test_keep, train_ents2)
    if len(df_test_bf2):
        df_train = pd.concat(
            [df_train, df_test_bf2], axis=0, ignore_index=True
        ).drop_duplicates()

    final_train_ents = _entities(df_train)
    stats = {
        "raw_total":             len(df),
        "raw_pairs":             n_pairs,
        "initial_train_triples": len(df_train_raw),
        "initial_test_triples":  len(df_test_raw),
        "initial_train_pairs":   len(pairs_train),
        "initial_test_pairs":    len(pairs_test),
        "backfill_from_test":    len(df_test_bf1) + len(df_test_bf2),
        "final_train":           len(df_train),
        "final_test_ppi":        len(df_test_final),
        "final_test_ppi_unseen": _count_unseen(df_test_final, final_train_ents),
    }
    return df_train, df_test_final, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, sep="\t")
    df = df[["head", "relation", "tail"]].dropna().drop_duplicates()

    # Guard: catch accidental use of pre-merge data
    forbidden_present = set(df["relation"].unique()) & FORBIDDEN_DIRECTION_RELS
    if forbidden_present:
        raise ValueError(
            f"Input still contains directional relation types: {forbidden_present}.\n"
            "Run preprocess_merge_relations.py first to collapse UP/DOWN → REGULATES."
        )

    structural      = df[df["relation"].isin(STRUCTURAL_RELS)].copy()
    regulates       = df[df["relation"] == "REGULATES"].copy()
    marker_of       = df[df["relation"] == "MARKER_OF"].copy()
    interacts_with  = df[df["relation"] == "INTERACTS_WITH"].copy()

    other = df[~df["relation"].isin(
        STRUCTURAL_RELS | {"REGULATES", "MARKER_OF", "INTERACTS_WITH"}
    )]
    if not other.empty:
        print(f"WARNING: {len(other)} triples with unrecognised relations will be "
              "included in train only.")

    print(f"Input triples        : {len(df)}")
    print(f"  REGULATES          : {len(regulates)}")
    print(f"  MARKER_OF          : {len(marker_of)}")
    print(f"  INTERACTS_WITH     : {len(interacts_with)}")
    print(f"  Structural         : {len(structural)}")

    # --- Split REGULATES ---
    reg_train, reg_valid, reg_test, reg_stats = _split_regulates_transductive(
        regulates, args.train_ratio, args.valid_ratio, args.seed
    )

    # --- Split MARKER_OF ---
    marker_train, marker_test, marker_stats = _split_marker_transductive(
        marker_of, args.marker_holdout_ratio, args.seed
    )

    # --- Split INTERACTS_WITH ---
    if not interacts_with.empty:
        ppi_train, ppi_test, ppi_stats = _split_interacts_transductive(
            interacts_with, args.ppi_holdout_ratio, args.seed
        )
    else:
        ppi_train = pd.DataFrame(columns=["head", "relation", "tail"])
        ppi_test  = pd.DataFrame(columns=["head", "relation", "tail"])
        ppi_stats = {"raw_total": 0}

    # --- Assemble ---
    train_df = pd.concat(
        [structural, other, reg_train, marker_train, ppi_train],
        axis=0, ignore_index=True,
    ).drop_duplicates()
    valid_df       = reg_valid.drop_duplicates().reset_index(drop=True)
    test_df        = reg_test.drop_duplicates().reset_index(drop=True)
    test_marker_df = marker_test.drop_duplicates().reset_index(drop=True)
    test_ppi_df    = ppi_test.drop_duplicates().reset_index(drop=True)

    # --- Final cross-split transductive check ---
    train_ents = _entities(train_df)
    for name, ev_df in [
        ("valid", valid_df),
        ("test", test_df),
        ("test_marker", test_marker_df),
        ("test_ppi", test_ppi_df),
    ]:
        n_unseen = _count_unseen(ev_df, train_ents)
        if n_unseen:
            keep, backfill = _move_unseen_to_train(ev_df, train_ents)
            train_df = pd.concat([train_df, backfill], axis=0, ignore_index=True).drop_duplicates()
            if name == "valid":
                valid_df = keep
            elif name == "test":
                test_df = keep
            elif name == "test_marker":
                test_marker_df = keep
            else:
                test_ppi_df = keep
            print(f"  Final backfill ({name}): {len(backfill)} triples moved to train.")
        train_ents = _entities(train_df)

    train_df       = train_df.drop_duplicates().reset_index(drop=True)
    valid_df       = valid_df.drop_duplicates().reset_index(drop=True)
    test_df        = test_df.drop_duplicates().reset_index(drop=True)
    test_marker_df = test_marker_df.drop_duplicates().reset_index(drop=True)
    test_ppi_df    = test_ppi_df.drop_duplicates().reset_index(drop=True)

    final_train_ents    = _entities(train_df)
    final_valid_unseen  = _count_unseen(valid_df,       final_train_ents)
    final_test_unseen   = _count_unseen(test_df,        final_train_ents)
    final_marker_unseen = _count_unseen(test_marker_df, final_train_ents)
    final_ppi_unseen    = _count_unseen(test_ppi_df,    final_train_ents)

    # --- Integrity check ---
    _check_no_overlap(train_df, valid_df, test_df, test_marker_df, test_ppi_df)

    # --- Save ---
    train_df.to_csv(       args.outdir / "train.tsv",       sep="\t", index=False)
    valid_df.to_csv(       args.outdir / "valid.tsv",       sep="\t", index=False)
    test_df.to_csv(        args.outdir / "test.tsv",        sep="\t", index=False)
    test_marker_df.to_csv( args.outdir / "test_marker.tsv", sep="\t", index=False)
    test_ppi_df.to_csv(    args.outdir / "test_ppi.tsv",    sep="\t", index=False)

    # --- Write stats ---
    test_ratio = 1.0 - args.train_ratio - args.valid_ratio
    lines = [
        f"input_total={len(df)}",
        f"input_regulates={len(regulates)}",
        f"input_marker_of={len(marker_of)}",
        f"input_interacts_with={len(interacts_with)}",
        f"input_structural={len(structural)}",
        "",
        "[REGULATES]",
        f"raw_total={reg_stats['raw_total']}",
        f"initial_train={reg_stats['initial_train']}",
        f"initial_valid={reg_stats['initial_valid']}",
        f"initial_test={reg_stats['initial_test']}",
        f"backfill_from_valid={reg_stats['backfill_from_valid']}",
        f"backfill_from_test={reg_stats['backfill_from_test']}",
        f"final_train={reg_stats['final_train']}",
        f"final_valid={reg_stats['final_valid']}",
        f"final_test={reg_stats['final_test']}",
        f"final_valid_unseen={reg_stats['final_valid_unseen']}",
        f"final_test_unseen={reg_stats['final_test_unseen']}",
        "",
        "[MARKER_OF]",
        f"raw_total={marker_stats['raw_total']}",
        f"initial_train={marker_stats['initial_train']}",
        f"initial_test_marker={marker_stats['initial_test_marker']}",
        f"backfill_from_test={marker_stats['backfill_from_test']}",
        f"final_train={marker_stats['final_train']}",
        f"final_test_marker={marker_stats['final_test_marker']}",
        f"final_test_marker_unseen={marker_stats['final_test_marker_unseen']}",
        "",
        "[INTERACTS_WITH]",
        f"raw_total={ppi_stats.get('raw_total', 0)}",
        f"raw_pairs={ppi_stats.get('raw_pairs', 0)}",
        f"initial_train_triples={ppi_stats.get('initial_train_triples', 0)}",
        f"initial_test_triples={ppi_stats.get('initial_test_triples', 0)}",
        f"initial_train_pairs={ppi_stats.get('initial_train_pairs', 0)}",
        f"initial_test_pairs={ppi_stats.get('initial_test_pairs', 0)}",
        f"backfill_from_test={ppi_stats.get('backfill_from_test', 0)}",
        f"final_train={ppi_stats.get('final_train', 0)}",
        f"final_test_ppi={ppi_stats.get('final_test_ppi', 0)}",
        f"final_test_ppi_unseen={ppi_stats.get('final_test_ppi_unseen', 0)}",
        "",
        "[FINAL_MERGED]",
        f"train_total={len(train_df)}",
        f"valid_total={len(valid_df)}",
        f"test_total={len(test_df)}",
        f"test_marker_total={len(test_marker_df)}",
        f"test_ppi_total={len(test_ppi_df)}",
        f"final_valid_unseen={final_valid_unseen}",
        f"final_test_unseen={final_test_unseen}",
        f"final_test_marker_unseen={final_marker_unseen}",
        f"final_test_ppi_unseen={final_ppi_unseen}",
        f"final_train_entities={len(final_train_ents)}",
        "",
        f"seed={args.seed}",
        f"train_ratio={args.train_ratio}",
        f"valid_ratio={args.valid_ratio}",
        f"test_ratio={test_ratio:.2f}",
        f"marker_holdout_ratio={args.marker_holdout_ratio}",
        f"ppi_holdout_ratio={args.ppi_holdout_ratio}",
    ]
    (args.outdir / "split_stats.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nSplit completed.")
    print(f"  train      : {len(train_df):>6}  -> data/splits/train.tsv")
    print(f"  valid      : {len(valid_df):>6}  -> data/splits/valid.tsv  (REGULATES)")
    print(f"  test       : {len(test_df):>6}  -> data/splits/test.tsv   (REGULATES)")
    print(f"  test_marker: {len(test_marker_df):>6}  -> data/splits/test_marker.tsv")
    print(f"  test_ppi   : {len(test_ppi_df):>6}  -> data/splits/test_ppi.tsv")
    print(f"\nTransductive sanity (must all be 0):")
    print(f"  valid unseen       : {final_valid_unseen}")
    print(f"  test unseen        : {final_test_unseen}")
    print(f"  test_marker unseen : {final_marker_unseen}")
    print(f"  test_ppi unseen    : {final_ppi_unseen}")


if __name__ == "__main__":
    main()
