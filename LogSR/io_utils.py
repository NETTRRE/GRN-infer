
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np

def load_expression_matrix(path: str) -> pd.DataFrame:
    """
    Expression matrix: CSV/TSV with header row of gene names, rows=cells.
    Auto-detects separator by pandas.
    """
    df = pd.read_csv(path, sep=None, engine="python")
    # Ensure columns are strings
    df.columns = [str(c) for c in df.columns]
    return df

def load_ppi_to_W_and_degrees(path: str) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    PPI interaction matrix: square table (rows=TFs, cols=TFs), values are weights.
    We produce:
      - W: dict of pairwise weights with symmetric keys (i,j)
      - degrees: degree per TF (sum of absolute weights across row)
    If matrix is not symmetric, we symmetrize by averaging.
    """
    ppi = pd.read_csv(path, sep=None, engine="python", index_col=0)
    ppi.index = [str(i) for i in ppi.index]
    ppi.columns = [str(c) for c in ppi.columns]

    # Align to common set (square submatrix if needed)
    common = sorted(set(ppi.index).intersection(set(ppi.columns)))
    ppi = ppi.loc[common, common].astype(float)

    # Symmetrize
    ppi_sym = (ppi + ppi.T) / 2.0

    # Build W dict
    W = {}
    for i in common:
        for j in common:
            if i == j:
                continue
            w = float(ppi_sym.loc[i, j])
            if w != 0.0:
                W[(i, j)] = w  # keep both directions for flexible lookup

    # Degrees (sum of absolute weights)
    degrees = {g: float(np.sum(np.abs(ppi_sym.loc[g, :]))) for g in common}
    return W, degrees

def load_tf_list(path: str) -> List[str]:
    """One TF symbol per line; ignores empty lines and leading/trailing spaces."""
    tfs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tfs.append(s)
    return tfs
