# Neural Cell Knowledge Graph

A hierarchical multimodal knowledge graph for neural cell-type-specific regulation integrating single-cell transcriptomics and literature evidence.

## Overview

This repository contains the data and code for constructing and analyzing a neural-cell-centric knowledge graph that integrates:

1. **Hierarchical cell type taxonomy** anchored to the Cell Ontology
2. **Molecular fingerprints** from single-cell transcriptomics
3. **Literature-derived regulatory evidence** extracted from PubMed
4. **Protein-protein interactions** from STRING database

## Repository Structure

```
neural-cell-knowledge-graph/
├── README.md
├── DATA_AVAILABILITY.md
├── DATA_DOCUMENTATION.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
│
├── data/
│   ├── cell_hierarchy/
│   │   ├── neural_cells.csv           # Cell types with descriptions
│   │   └── hierarchy_relations.csv    # BELONGS_TO and PART_OF edges
│   ├── molecular_fingerprints/
│   │   └── marker_genes.tsv           # MARKER_OF relationships
│   ├── regulatory_triples/
│   │   ├── regulatory_triples.tsv     # LLM-extracted regulatory triples
│   │   └── pubmed_sources.txt         # Source PMIDs
│   ├── ppi/
│   │   └── string_ppi.tsv             # PPI pairs from STRING
│   ├── kge_splits/
│   │   ├── train.txt                  # Training triples
│   │   ├── valid.txt                  # Validation triples
│   │   └── test.txt                   # Test triples
│   ├── kge_results/
│   │   ├── model_comparison.csv       # Cross-model REGULATES link prediction results
│   │   ├── rotate_summary.json        # RotatE REGULATES evaluation details
│   │   ├── rotate_summary_metrics.json # RotatE marker retrieval metrics
│   │   ├── microglia_all_predictions.tsv # RotatE Microglia combined predictions
│   │   ├── microglia_regulating_factors.tsv # RotatE Microglia regulator predictions
│   │   └── microglia_marker_genes.tsv # RotatE Microglia marker predictions
│   └── virtual_knockout/
│       ├── knockout_summary_v2.csv    # Candidate summary with KGE ranks
│       ├── *_vk_results.csv           # Gene-level virtual knockout results
│       └── *_dr_genes.csv             # Significant differentially regulated genes
│
├── scripts/
│   ├── kg_construction/
│   │   ├── import_hierarchy.py        # Import cell hierarchy to Neo4j
│   │   ├── import_marker_genes.py     # Import marker genes to Neo4j
│   │   └── export_triples.py          # Export triples for KGE training
│   ├── kge_training/
│   │   ├── train_pykeen_model.py      # Train KGE models (TransE/RotatE/ComplEx/TuckER)
│   │   ├── split_triples.py           # Transductive-safe triple splitting
│   │   ├── evaluate_regulates_lp.py   # REGULATES link prediction evaluation
│   │   ├── evaluate_marker_retrieval.py # Marker gene retrieval evaluation
│   │   ├── evaluate_ppi_lp.py         # PPI link prediction evaluation
│   │   └── compare_models.py          # Cross-model comparison
│   └── virtual_knockout/
│       ├── run_virtual_knockout.py    # Virtual knockout analysis
│       └── run_virtual_knockout_v2.py # Virtual knockout analysis (v2)
│
└── supplementary/
    ├── llm_prompts.txt                # LLM extraction prompt templates
    └── validation_samples.csv         # Expert-validated extraction samples
```

## Installation

### Prerequisites

- Python 3.8+
- Neo4j 4.4+ (for graph database)
- CUDA-capable GPU (optional, for faster KGE training)

### Setup

```bash
pip install -r requirements.txt
```

For Neo4j functionality:
```bash
pip install neo4j
```

## Quick Start

### 1. Explore the Data

```python
import pandas as pd

# Load cell hierarchy
cells = pd.read_csv('data/cell_hierarchy/neural_cells.csv')

# Load regulatory triples
triples = pd.read_csv('data/regulatory_triples/regulatory_triples.tsv', sep='\t')

# Load marker genes
markers = pd.read_csv('data/molecular_fingerprints/marker_genes.tsv', sep='\t')

# Load PPI
ppi = pd.read_csv('data/ppi/string_ppi.tsv', sep='\t')
```

### 2. Train KGE Models

```bash
# Train RotatE model (best performing)
python scripts/kge_training/train_pykeen_model.py \
    --model RotatE \
    --train data/kge_splits/train.txt \
    --valid data/kge_splits/valid.txt \
    --test data/kge_splits/test.txt \
    --seed 42 \
    --outdir outputs/rotate_seed42

# Evaluate REGULATES link prediction
python scripts/kge_training/evaluate_regulates_lp.py \
    --model-dir outputs/rotate_seed42
```

### 3. Import to Neo4j

```bash
# Import cell hierarchy
python scripts/kg_construction/import_hierarchy.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password YOUR_PASSWORD

# Import marker genes
python scripts/kg_construction/import_marker_genes.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password YOUR_PASSWORD

# Export triples for KGE training
python scripts/kg_construction/export_triples.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password YOUR_PASSWORD
```

## Data Sources

### Single-Cell Transcriptomics

1. **First-trimester developing human brain**
   - Braun et al., Science 382, eadf1226 (2023)
   - DOI: 10.1126/science.adf1226

2. **Adult human brain**
   - Siletti et al., Science 382, eadd7046 (2023)
   - DOI: 10.1126/science.add7046

### Cell Ontology

- Tan et al., arXiv:2506.10037 (2025)
- DOI: 10.48550/arXiv.2506.10037

### Literature Evidence

- PubMed abstracts extracted using large language models (S1-Base-Ultra)

### Protein-Protein Interactions

- STRING database v12.0
- Szklarczyk et al., Nucleic Acids Research 51, D638-D646 (2023)
- DOI: 10.1093/nar/gkac1000

See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for complete data source information.

## Citation

If you use this knowledge graph or code in your research, please cite:

```bibtex
@article{chen2025neural,
  title={A hierarchical multimodal knowledge graph for neural cell-type-specific regulation integrating single-cell transcriptomics and literature evidence},
  author={Chen, Chuangyu and Ni, Xiaomin and Min, Yang and Wang, Zhen and Xu, Zhilan and Zhang, Yang and Yu, Hao},
  journal={[Journal Name]},
  year={2025},
  note={In preparation}
}
```

## License

- **Code**: MIT License
- **Data**: CC BY 4.0 License

## Contact

- **Hao Yu**: hao.yu@siat.ac.cn
- **Yang Zhang**: zhangyang@szbl.ac.cn

## Acknowledgments

This project was supported by:
- Shenzhen Fundamental Research Program (No. JCYJ20240813155824032)
- GuangDong Basic and Applied Basic Research Foundation (2514050002006)
- Science and Technology special fund of Hainan Province (No. ZDYF2024SHFZ045)
- National Natural Science Foundation of China (NSFC82303769)
