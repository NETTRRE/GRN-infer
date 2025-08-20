
import argparse
import pandas as pd
from .core import LogSR
from .io_utils import load_expression_matrix, load_ppi_to_W_and_degrees, load_tf_list

def main():
    ap = argparse.ArgumentParser(description="LogSR: GRN inference from expression + PPI + TF list")
    ap.add_argument("--expr", required=True, help="Expression matrix (CSV/TSV): rows=cells, cols=genes")
    ap.add_argument("--ppi", required=True, help="PPI interaction matrix (CSV/TSV): rows=TFs, cols=TFs")
    ap.add_argument("--tfs", required=True, help="TF list file (one TF per line)")
    ap.add_argument("--out_edges", required=True, help="Output GRN edges TSV")
    ap.add_argument("--w1", type=float, default=0.3, help="Sparsity weight")
    ap.add_argument("--w2", type=float, default=0.5, help="Prior weight")
    ap.add_argument("--gamma", type=float, default=1e-3, help="Default weight for missing PPI edges")
    ap.add_argument("--eta", type=float, default=0.95, help="Length penalty base in reward")
    ap.add_argument("--iter", type=int, default=800, help="MCTS iterations per target")
    ap.add_argument("--cuct", type=float, default=1.4, help="UCB exploration constant")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    args = ap.parse_args()

    # Load inputs
    X = load_expression_matrix(args.expr)
    W, degrees = load_ppi_to_W_and_degrees(args.ppi)
    tf_list = load_tf_list(args.tfs)

    # Build model
    model = LogSR(W=W, degrees=degrees,
                  w1=args.w1, w2=args.w2, gamma=args.gamma, eta=args.eta,
                  max_iter=args.iter, exploration_c=args.cuct, seed=args.seed)

    # Infer GRN over non-TF targets
    rules, edges_df = model.infer_grn(X, tf_list)

    # Save edges
    edges_df.to_csv(args.out_edges, sep="\t", index=False)
    print(f"[LogSR] Done. Edges written to: {args.out_edges}")
    print(f"[LogSR] #targets inferred: {len(rules)} ; #edges: {len(edges_df)}")

if __name__ == "__main__":
    main()
