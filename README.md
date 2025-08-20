# [Prior-guided Monte Carlo tree search for symbolic inference of gene regulatory network from single-cell transcriptomics data](https://github.com/DZ-Z/LogicGep)

![Screenshot](Figure/MCTS1-2.png)

  In this work, we introduce LogicSR, a computational framework that reconstructs gene regulatory networks from single-cell gene expression data with high accuracy by integrating the mechanistic interpretability of logical models with the equation-discovery capabilities of symbolic regression. It incorporates prior knowledge into a multi-objective Monte Carlo tree search framework, leveraging it to ensure biological plausibility and accelerate the search for optimal governing equations.

## Dependencies
requirements.txt
  
## Usage

1. Preparing  for  time-series gene expression profiles
   
   For the input data, ensure that it is structured with rows representing different time points and columns 
   corresponding to various genes
   
3. Command to run LogicGep
 
   ``cd LogicGep ``  
   ``python main.py``
