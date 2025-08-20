# [LogicGep: Boolean networks inference using symbolic regression from time-series transcriptomic profiling data](https://github.com/DZ-Z/LogicGep)

![Screenshot](Figure/MCTS1-2.png)

 In this work, we present LogSR, a computational framework that reconstructs gene regulatory networks from single-cell gene expression data with high fidelity by integrating the mechanistic interpretability of logical models with the equation discovery capabilities of symbolic regression. It embeds prior knowledge into a multi-objective Monte Carlo tree search framework, using it both to ensure biological plausibility and to accelerate the search for optimal governing equations.

## Dependencies
requirements.txt
  
## Usage

1. Preparing  for  time-series gene expression profiles
   
   For the input data, ensure that it is structured with rows representing different time points and columns 
   corresponding to various genes
   
3. Command to run LogicGep
 
   ``cd LogicGep ``  
   ``python main.py``
