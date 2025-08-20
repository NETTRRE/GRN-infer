# LogSR: Prior-guided Monte Carlo tree search for symbolic inference of gene regulatory network from single-cell transcriptomics data

This repository does not bundle datasets. LogSR consumes three inputs that can be obtained from well-established public resources

1) Single-cell expression matrices (rows = cells, columns = genes)

2) PPI interaction matrix (weighted)
Construct a square TF×TF interaction matrix from a curated protein–protein interaction (PPI) resource; LogSR symmetrizes it internally.  
STRING (recommended): download the protein.links / combined-score table for your species and map proteins to TF symbols (https://string-db.org/).
Alternatives: ENCODE(https://www.encodeproject.org/),BioGRID(https://thebiogrid.org/), depending on your field’s conventions.

3) TF list
Provide the set of transcription factors for the organism:
AnimalTFDB (for vertebrates/invertebrates, https://guolab.wchscu.cn/AnimalTFDB4/#/) or PlantTFDB (plants, https://planttfdb.gao-lab.org/).