#!/usr/bin/env python3
"""
KGE training script for train_v6 (TransE / RotatE / ComplEx / TuckER).

Changes from train_v4
---------------------
  - The validation and test sets now contain REGULATES triples only (the
    directional UP/DOWN split has been removed).  Early stopping and link-
    prediction evaluation are therefore performed on the REGULATES relation.
  - DistMult remains absent (replaced by ComplEx in v4); ComplEx is retained
    for its general bilinear expressivity, not for asymmetry per se.
  - TuckER relation_dim is kept at 128: the MARKER_OF sub-graph (~1,210 train
    triples) is still small relative to the embedding space, and the reduced
    core tensor (128³ ≈ 2M vs 256³ ≈ 16M parameters) provides necessary
    regularisation.

Model      Loss            Training loop   Neg. sampler          Notes
-------    -----------     -------------   -----------------     -----
TransE     default (MRL)   sLCWA           Bernoulli (128 negs)  L2-norm scoring
RotatE     NSSALoss        sLCWA           Bernoulli (128 negs)  self-adversarial
ComplEx    BCE             LCWA            all negatives         bilinear, complex
TuckER     BCE             LCWA            all negatives         bilinear, dropout

Early stopping
--------------
Validation MRR is checked every `stopper_freq` epochs.  Training halts when
MRR has not improved by at least 0.2 % (relative_delta=0.002) for `patience`
consecutive checks.  Default: patience=20 checks × freq=10 epochs = 200 epochs
without improvement before halting.

Usage
-----
  python scripts/train_pykeen_model.py --model RotatE --seed 42
  python scripts/train_pykeen_model.py --model TuckER --seed 43 --outdir outputs/tucker_seed43
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

SUPPORTED_MODELS = ["TransE", "RotatE", "ComplEx", "TuckER"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a KGE model for train_v6.")
    parser.add_argument("--model",         type=str,   default="TransE", choices=SUPPORTED_MODELS)
    parser.add_argument("--train",         type=Path,  default=Path("data/kge_splits/train.txt"))
    parser.add_argument("--valid",         type=Path,  default=Path("data/kge_splits/valid.txt"))
    parser.add_argument("--test",          type=Path,  default=Path("data/kge_splits/test.txt"))
    parser.add_argument("--outdir",        type=Path,  default=None,
                        help="Output directory. Defaults to outputs/<model_lower>_run.")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--embedding-dim", type=int,   default=256)
    parser.add_argument("--epochs",        type=int,   default=500)
    parser.add_argument("--batch-size",    type=int,   default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-negs",      type=int,   default=128,
                        help="Negative samples per positive triple (sLCWA only).")
    parser.add_argument("--patience",      type=int,   default=20,
                        help="Early stopping patience (number of evaluation checks).")
    parser.add_argument("--stopper-freq",  type=int,   default=10,
                        help="Evaluate on validation set every N epochs.")
    parser.add_argument("--device",        type=str,   default="auto",
                        choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path, sep="\t")[["head", "relation", "tail"]].dropna()


def _model_kwargs(model: str, embedding_dim: int) -> dict:
    """Return model-specific constructor keyword arguments."""
    if model == "TransE":
        # L2-norm scoring function (p=2) per original paper (Bordes et al., 2013)
        return dict(embedding_dim=embedding_dim, scoring_fct_norm=2)

    if model == "RotatE":
        # Rotation in complex space; embedding_dim here is the real dimension;
        # RotatE internally halves it for complex representation (Sun et al., 2019)
        return dict(embedding_dim=embedding_dim)

    if model == "ComplEx":
        # Complex-valued embeddings (Trouillon et al., 2016); handles
        # both symmetric and anti-symmetric patterns
        return dict(embedding_dim=embedding_dim)

    if model == "TuckER":
        # Tucker decomposition (Balazevic et al., 2019)
        # relation_dim=128: keeps core tensor 128³≈2M params vs 256³≈16M;
        # important given only ~1,210 MARKER_OF training triples.
        # Dropout schedule follows the original paper recommendations.
        return dict(
            embedding_dim=embedding_dim,
            relation_dim=128,
            dropout_0=0.3,
            dropout_1=0.4,
            dropout_2=0.5,
        )

    raise ValueError(f"Unknown model: {model}")


def main() -> None:
    args = parse_args()

    if args.outdir is None:
        args.outdir = Path(f"outputs/{args.model.lower()}_run")
    args.outdir.mkdir(parents=True, exist_ok=True)

    train_df = _load(args.train)
    valid_df = _load(args.valid)
    test_df  = _load(args.test)

    # Build TriplesFactory; valid and test share the entity/relation vocabulary
    # defined by the training set (transductive assumption).
    train_tf = TriplesFactory.from_labeled_triples(train_df.values)
    valid_tf = TriplesFactory.from_labeled_triples(
        valid_df.values,
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
    )
    test_tf = TriplesFactory.from_labeled_triples(
        test_df.values,
        entity_to_id=train_tf.entity_to_id,
        relation_to_id=train_tf.relation_to_id,
    )

    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )

    print(f"Model      : {args.model}")
    print(f"Device     : {device}")
    print(f"Emb dim    : {args.embedding_dim}   Epochs max: {args.epochs}   Batch: {args.batch_size}")
    print(f"LR         : {args.learning_rate}   Patience: {args.patience} × every {args.stopper_freq} epochs")
    print(f"Train triples : {train_tf.num_triples}")
    print(f"Valid triples : {valid_tf.num_triples}  (REGULATES)")
    print(f"Test  triples : {test_tf.num_triples}   (REGULATES)")
    print(f"Entities      : {train_tf.num_entities}")
    print(f"Relations     : {train_tf.num_relations}")

    # Early stopping on filtered validation MRR
    stopper_kwargs = dict(
        frequency=args.stopper_freq,
        patience=args.patience,
        relative_delta=0.002,
        metric="mean_reciprocal_rank",
        larger_is_better=True,
    )

    common = dict(
        model=args.model,
        training=train_tf,
        validation=valid_tf,
        testing=test_tf,
        random_seed=args.seed,
        device=device,
        model_kwargs=_model_kwargs(args.model, args.embedding_dim),
        optimizer="Adam",
        optimizer_kwargs=dict(lr=args.learning_rate),
        evaluator="rankbased",
        stopper="early",
        stopper_kwargs=stopper_kwargs,
    )

    if args.model == "RotatE":
        result = pipeline(
            **common,
            loss="NSSALoss",
            loss_kwargs=dict(margin=6.0, adversarial_temperature=1.0),
            training_loop="sLCWA",
            negative_sampler="bernoulli",
            negative_sampler_kwargs=dict(num_negs_per_pos=args.num_negs),
            training_kwargs=dict(num_epochs=args.epochs, batch_size=args.batch_size),
        )
    elif args.model in ("ComplEx", "TuckER"):
        result = pipeline(
            **common,
            loss="BCEWithLogitsLoss",
            training_loop="LCWA",
            training_kwargs=dict(num_epochs=args.epochs, batch_size=args.batch_size),
        )
    else:
        # TransE: Bernoulli sampler corrects for 1-to-N / N-to-1 relation cardinality
        result = pipeline(
            **common,
            training_loop="sLCWA",
            negative_sampler="bernoulli",
            negative_sampler_kwargs=dict(num_negs_per_pos=args.num_negs),
            training_kwargs=dict(num_epochs=args.epochs, batch_size=args.batch_size),
        )

    result.save_to_directory(args.outdir)

    # --- Save metrics ---
    metrics = result.metric_results.to_dict()
    (args.outdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    both_real = metrics.get("both", {}).get("realistic", {})
    summary = {
        "model":                      args.model,
        "seed":                       args.seed,
        "both.realistic.mrr":         both_real.get("inverse_harmonic_mean_rank"),
        "both.realistic.hits_at_1":   both_real.get("hits_at_1"),
        "both.realistic.hits_at_3":   both_real.get("hits_at_3"),
        "both.realistic.hits_at_10":  both_real.get("hits_at_10"),
    }
    (args.outdir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Reproducibility metadata
    metadata = {
        "model":          args.model,
        "seed":           args.seed,
        "embedding_dim":  args.embedding_dim,
        "epochs_max":     args.epochs,
        "batch_size":     args.batch_size,
        "learning_rate":  args.learning_rate,
        "num_negs":       args.num_negs,
        "patience":       args.patience,
        "stopper_freq":   args.stopper_freq,
        "device":         device,
        "train_v":        "v6",
    }
    (args.outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nTraining finished → {args.outdir}")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("model", "seed")}, indent=2))


if __name__ == "__main__":
    main()
