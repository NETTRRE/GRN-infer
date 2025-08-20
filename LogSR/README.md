# LogSR: Prior-Guided, Multi-Objective Symbolic Regression for GRN Inference

LogSR is a compact, research-friendly implementation for inferring **Boolean regulatory rules** and projecting them into **gene regulatory networks (GRNs)** from single-cell expression. It couples **symbolic regression over Boolean expressions** with a **prior-guided, multi-objective Monte Carlo Tree Search (MCTS)** to balance predictive accuracy, biological plausibility, and model parsimony.

## Key Features
- Boolean logic representation (AND / OR / NOT) with symbolic parse trees
- **Multi-objective** scoring: predictive accuracy, prior consistency, parsimony
- **Prior guidance** (e.g., PPI) as soft constraints
- MCTS search with UCB1 (selection/expansion/simulation/backpropagation)


## How to run

counts_df = ...          # pd.DataFrame
tf_names = [...]         # list[str]
prior_edges = [...]      # list[tuple[str,str]]


Inputs:

a preprocessed gene expression matrix (cells × genes; numeric; we binarize as expr > 0),

a PPI interaction matrix (square, TF×TF; weighted; symmetric or not—handled),

a TF list file (one TF per line).

Output: inferred GRN over all non-TF target genes (TF→target edges), plus the learned rule (Boolean expression) per target.


python -m logsr.run \
  --expr ./expr_matrix.csv \
  --ppi  ./ppi_matrix.csv \
  --tfs  ./tf_list.txt \
  --out_edges ./grn_edges.tsv \
  --iter 800 --cuct 1.4 --w1 0.30 --w2 0.50 --gamma 1e-3 --eta 0.95 --seed 7

