# LogSR: Prior-Guided, Multi-Objective Symbolic Regression for GRN Inference

This repository does not bundle datasets. LogSR consumes three inputs that can be obtained from well-established public resources

1) Single-cell expression matrices (rows = cells, columns = genes)

2) PPI interaction matrix (weighted)
Construct a square TF×TF interaction matrix from a curated protein–protein interaction (PPI) resource; LogSR symmetrizes it internally.  
STRING (recommended): download the protein.links / combined-score table for your species and map proteins to TF symbols.
Alternatives: ENCODE,BioGRID, depending on your field’s conventions.

3) TF list
Provide the set of transcription factors for the organism:
AnimalTFDB (for vertebrates/invertebrates) or PlantTFDB (plants).