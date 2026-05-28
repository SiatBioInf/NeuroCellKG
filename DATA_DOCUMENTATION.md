# Data Documentation

This document describes the format and content of all data files in the knowledge graph.

## Table of Contents

1. [Cell Hierarchy Data](#cell-hierarchy-data)
2. [Molecular Fingerprints](#molecular-fingerprints)
3. [Regulatory Triples](#regulatory-triples)
4. [Protein-Protein Interactions](#protein-protein-interactions)
5. [KGE Training Splits](#kge-training-splits)

---

## Cell Hierarchy Data

### File: `data/cell_hierarchy/neural_cells.csv`

**Description**: Complete list of 79 neural cell types organized in a 3-level hierarchy.

**Format**: CSV with header

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| `cell_name` | string | Standardized cell type name |
| `level` | integer | Hierarchy level (1, 2, or 3) |
| `parent` | string | Parent cell type name (empty for Level 1) |
| `description` | string | Cell type definition from Cell Ontology or PubMed |
| `description_source` | string | URI or DOI of the definition source |

**Statistics**:
- Total nodes: 79
- Level 1 (major classes): 4 (Neuron, Glial cell, Non-neural cell of nervous system, Progenitor)
- Level 2 (functional types): 30
- Level 3 (fine-grained subtypes): 45

**Example**:
```csv
cell_name,level,parent,description,description_source
Neuron,1,,"The basic cellular unit of nervous tissue...",http://purl.obolibrary.org/obo/CL:0000540
Astrocyte,2,Glial cell,"A class of large neuroglial cells...",http://purl.obolibrary.org/obo/CL_0000127
Microglia,3,Macrophage,"A tissue-resident macrophage of the central nervous system...",http://purl.obolibrary.org/obo/CL_0000129
```

### File: `data/cell_hierarchy/hierarchy_relations.csv`

**Description**: Parent-child relationships in the cell type hierarchy.

**Format**: CSV with header

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| `child` | string | Child cell type name |
| `parent` | string | Parent cell type name |
| `relation_type` | string | `BELONGS_TO` (L3→L2) or `PART_OF` (L2→L1) |

**Statistics**:
- BELONGS_TO edges: 45 (Level 3 → Level 2)
- PART_OF edges: 30 (Level 2 → Level 1)

---

## Molecular Fingerprints

### File: `data/molecular_fingerprints/marker_genes.tsv`

**Description**: Marker genes for 31 cell subtypes derived from single-cell transcriptomics (4.16 million cells).

**Format**: Tab-separated values (TSV) with header

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| `cell_type` | string | Cell type name (matches Level 2 or Level 3 in hierarchy) |
| `gene_symbol` | string | HGNC gene symbol |
| `ensembl_id` | string | Ensembl gene ID |
| `rank` | integer | Rank within cell type (1 = most specific) |
| `score` | float | Average log2 fold change |
| `pct_expressed` | float | Percentage of cells expressing in target type |
| `pct_others` | float | Percentage of cells expressing in other types |
| `p_val_adj` | float | Adjusted p-value |
| `gene_biotype` | string | Gene biotype (all "protein_coding") |

**Statistics**:
- Total relationships: 1,550
- Cell types with markers: 31
- Top 50 protein-coding genes per cell type

**Selection criteria**: Top 50 protein-coding genes ranked by differential expression (Seurat FindAllMarkers, Wilcoxon rank-sum test, only.pos=TRUE, min.pct=0.25, logfc.threshold=0.25, p_val_adj < 0.05).

---

## Regulatory Triples

### File: `data/regulatory_triples/regulatory_triples.tsv`

**Description**: Literature-derived regulatory relationships extracted from PubMed abstracts using a large language model (S1-Base-Ultra).

**Format**: Tab-separated values (TSV) with header

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| `pmid` | string | PubMed ID |
| `factor_name` | string | Name of the perturbing factor |
| `factor_type_level1` | string | Broad category: Biological, Chemical, or Physical |
| `factor_type_level2` | string | Specific category (controlled vocabulary, see below) |
| `perturbation_action` | string | Action type (treatment, knockout, knockdown, etc.) |
| `cell_type_mentioned` | string | Cell type as mentioned in the abstract |
| `cell_type_mapping` | string | Mapped cell type name (L1/L2 extraction only; empty for L3) |
| `cell_type_granularity` | string | Granularity of mapping (L1/L2 extraction only; empty for L3) |
| `effect` | string | Downstream molecular or cellular change |
| `effect_direction` | integer | 1 (increase/activation), -1 (decrease/inhibition), or empty |
| `evidence_sentence` | string | Supporting evidence quote from the abstract |
| `confidence_score` | integer | Extraction confidence (1–10) |
| `extraction_level` | string | Extraction hierarchy level: L1, L2, or L3 |

**factor_type_level2 controlled vocabulary**:

| Category | Subcategories |
|----------|--------------|
| Biological | Biological Signaling Molecules, Cellular Interactions, Microbiome, Genetic and Molecular Perturbations |
| Chemical | Pharmacological and Small-Molecule Perturbations, Metabolic and Nutrient Stress, Metal Ions |
| Physical | Environment, Mechanical Factors, Radiation and Energy Stress |

**Statistics**:
- Total extracted records: 25,812
- L1 records: 544
- L2 records: 6,549
- L3 records: 18,721
- Unique REGULATES relationships (after deduplication): 12,494
- Unique PMIDs: 7,461
- Extraction accuracy: 93.5% (200-sample expert validation)

### File: `data/regulatory_triples/pubmed_sources.txt`

**Description**: Complete list of PubMed IDs used for triple extraction.

**Format**: Plain text, one PMID per line

---

## Protein-Protein Interactions

### File: `data/ppi/string_ppi.tsv`

**Description**: Protein-protein interaction data from STRING database (v12.0). Only interactions where both proteins are marker genes in the knowledge graph are included.

**Format**: Tab-separated values (TSV) with header

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| `protein1` | string | First protein (HGNC gene symbol) |
| `protein2` | string | Second protein (HGNC gene symbol) |
| `relation` | string | Always "INTERACTS_WITH" |

**Statistics**:
- Total unique pairs: 2,470
- Source: STRING v12.0 (Homo sapiens, combined_score >= 700)
- Only pairs where both genes are in the marker gene set

---

## KGE Training Splits

### Files:
- `data/kge_splits/train.txt`
- `data/kge_splits/valid.txt`
- `data/kge_splits/test.txt`

**Description**: Train/validation/test splits for knowledge graph embedding models.

**Format**: Tab-separated values (TSV) without header

**Columns**:
| Position | Type | Description |
|----------|------|-------------|
| 1 | string | Head entity (with prefix: L1:, L2:, L3:, GENE:, FACTOR:) |
| 2 | string | Relation type |
| 3 | string | Tail entity (with prefix) |

**Relation types**: BELONGS_TO, PART_OF, MARKER_OF, REGULATES, INTERACTS_WITH

**Split strategy**:
- BELONGS_TO / PART_OF: all in training set (structural edges)
- REGULATES: ~8:1:1 train/valid/test
- MARKER_OF: 80/20 train/holdout (in additional test files)
- INTERACTS_WITH: 80/20 train/holdout (in additional test files)
- Transductive-safe: all entities in valid/test also appear in train

**Statistics**:
| Split | Count |
|-------|-------|
| Train | 17,822 |
| Valid | 383 (REGULATES only) |
| Test | 378 (REGULATES only) |

**Entity statistics**:
| Prefix | Count |
|--------|-------|
| FACTOR: | 9,782 |
| GENE: | 799 |
| L1: | 4 |
| L2: | 30 |
| L3: | 45 |
| **Total** | **10,660** |

---

## KGE Results

### File: `data/kge_results/model_comparison.csv`

**Description**: REGULATES link prediction results for all four KGE models (3 seeds averaged).

**Format**: CSV with header

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| `model` | string | Model name (TransE/RotatE/ComplEx/TuckER) |
| `mrr` | float | Mean Reciprocal Rank |
| `mrr_std` | float | Standard deviation across seeds |
| `hits_at_1` | float | Hits@1 |
| `hits_at_1_std` | float | Standard deviation |
| `hits_at_3` | float | Hits@3 |
| `hits_at_3_std` | float | Standard deviation |
| `hits_at_10` | float | Hits@10 |
| `hits_at_10_std` | float | Standard deviation |

### File: `data/kge_results/microglia_regulating_factors.tsv`

**Description**: Top-30 upstream regulator predictions for L3:Microglia by the RotatE model (REGULATES head prediction).

**Format**: TSV with header

**Columns**: rank, factor, factor_name, cell_type, prediction_score, prediction_type

### File: `data/kge_results/microglia_marker_genes.tsv`

**Description**: Top-30 marker gene predictions for L3:Microglia by the RotatE model (MARKER_OF head prediction).

**Format**: TSV with header

**Columns**: rank, gene, gene_symbol, cell_type, prediction_score, prediction_type

---

## Data Processing Pipeline

### 1. Cell Hierarchy Construction
```
Cell Ontology → Expert curation → 3-level hierarchy (79 nodes)
```

### 2. Molecular Fingerprint Extraction
```
scRNA-seq data (4.16M cells) → Seurat FindAllMarkers →
Protein-coding filter → Top 50 per cell type →
1,550 MARKER_OF relationships
```

### 3. Literature Triple Extraction
```
PubMed abstracts → LLM screening (S1-Base-Ultra) →
LLM extraction (S1-Base-Ultra) → Post-processing →
Expert validation (93.5% accuracy) → 25,812 records
```

### 4. PPI Integration
```
STRING database → Filter by confidence (score >= 700) →
Intersect with marker gene set → 2,470 unique pairs
```

### 5. Knowledge Graph Assembly
```
All components → Neo4j import → Unified knowledge graph →
Export for KGE training → Transductive-safe split
```

---

## Usage Notes

1. **Cell type names**: Standardized according to Cell Ontology; Level 3 names match those used in the KGE entity vocabulary
2. **Gene symbols**: HGNC official symbols (Homo sapiens)
3. **Entity prefixes in KGE**: L1:, L2:, L3: for cell nodes; GENE: for gene nodes; FACTOR: for regulatory factor nodes
4. **Missing values**: Represented as empty strings
5. **Encoding**: UTF-8

---

## Version

- **Version**: 1.0
- **Last updated**: 2026-05-16


---

## Virtual Knockout Source Data

### Directory: `data/virtual_knockout/`

**Description**: Source data for in silico knockout analysis of KGE-prioritized Microglia regulator candidates (FMR1, PTEN, FKBP5).

**Key files**:

| File pattern | Description |
|--------------|-------------|
| `knockout_summary_v2.csv` | KGE rank, score, known regulated cells, number of tested genes and number of significant differentially regulated genes for each candidate |
| `*_vk_results.csv` | Complete gene-level virtual knockout results with distance, transformed distance, Z score, fold-change statistic, nominal P value and adjusted P value |
| `*_dr_genes.csv` | Significant differentially regulated genes for each virtual knockout |
| `run_log_300cells_20260521_110543.txt` | Run log for the 300-cell optimized analysis |
| `INDEX.md` and `SIGNIFICANT_DR_GENES_COMPLETE_LIST.md` | Human-readable result summaries |

**Analysis summary**: FMR1, PTEN and FKBP5 were selected as KGE-prioritized Microglia regulator candidates without direct REGULATES edges to Microglia in the training graph. The analysis tested 3,001 genes after quality control and identified 27, 20 and 18 significant differentially regulated genes for FMR1, PTEN and FKBP5 knockout, respectively.

### Directory: `scripts/virtual_knockout/`

Scripts used to run the virtual knockout analysis and generate the source-data outputs.
