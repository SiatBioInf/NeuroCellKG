# Data Availability Statement

## Source Code

All source code for knowledge graph construction and knowledge graph embedding (KGE) model training is publicly available at:

**GitHub Repository**: https://github.com/SiatBioInf/NeuroCellKG

The repository includes:
- Knowledge graph construction pipeline (cell hierarchy, marker gene, and regulatory triple import to Neo4j)
- KGE model training and evaluation code (TransE, RotatE, ComplEx, TuckER)
- LLM extraction prompt templates
- Expert validation samples

## Data Sources

### 1. Single-Cell Transcriptomic Datasets

The molecular fingerprints (1,550 MARKER_OF relationships for 31 cell subtypes) were derived from two large-scale human brain single-cell RNA-seq datasets:

- **First-trimester developing human brain** (1.78 million cells):
  - Braun et al., "Comprehensive cell atlas of the first-trimester developing human brain." *Science* 382, eadf1226 (2023).
  - DOI: [10.1126/science.adf1226](https://doi.org/10.1126/science.adf1226)

- **Adult human brain** (2.37 million cells):
  - Siletti et al., "Transcriptomic diversity of cell types across the adult human brain." *Science* 382, eadd7046 (2023).
  - DOI: [10.1126/science.add7046](https://doi.org/10.1126/science.add7046)

### 2. Cell Type Ontology

The hierarchical cell type taxonomy (79 nodes: 4 Level-1 classes, 30 Level-2 functional types, 45 Level-3 fine-grained subtypes) was constructed based on:

- **Cell Ontology**:
  - Tan et al., "The Cell Ontology in the age of single-cell omics." *arXiv preprint* arXiv:2506.10037 (2025).
  - DOI: [10.48550/arXiv.2506.10037](https://doi.org/10.48550/arXiv.2506.10037)
  - Cell Ontology database: http://www.obofoundry.org/ontology/cl.html

### 3. Literature-Derived Regulatory Evidence

The regulatory triples were extracted from PubMed abstracts using large language models (S1-Base-Ultra, provided by China Science and Technology Cloud):

- Total extracted records across all hierarchy levels: 25,812
- Unique regulatory relationships after deduplication (REGULATES): 12,494
- Unique source PMIDs: 7,461
- Extraction accuracy: 93.5% (200-sample expert validation)
- The complete list of PubMed IDs (PMIDs) is provided in the repository

### 4. Protein-Protein Interaction Data

Protein-protein interaction data were integrated from:

- **STRING database** (v12.0):
  - Szklarczyk et al., "The STRING database in 2023." *Nucleic Acids Research* 51, D638-D646 (2023).
  - DOI: [10.1093/nar/gkac1000](https://doi.org/10.1093/nar/gkac1000)
  - Data available at: https://string-db.org/

## Knowledge Graph Data

The complete neural cell knowledge graph is available in the following formats:

1. **Triple files**: Tab-separated values (TSV) format
   - Cell hierarchy relationships (75 edges: 45 BELONGS_TO + 30 PART_OF)
   - Molecular fingerprints (1,550 MARKER_OF relations)
   - Literature-derived regulatory triples (25,812 extracted records; 12,494 unique REGULATES edges)
   - Protein-protein interactions (2,470 unique pairs)
2. **KGE training splits**: Train/validation/test sets for knowledge graph embedding models
   - Train: 17,822 triples
   - Validation: 383 triples (REGULATES only)
   - Test: 378 triples (REGULATES only)

All data files are available in the GitHub repository under the `data/` directory.

## License

The code is released under the MIT License. The data are released under CC BY 4.0 License.

## Contact

For questions regarding data access or usage, please contact:
- Hao Yu: hao.yu@siat.ac.cn
- Yang Zhang: zhangyang@szbl.ac.cn
