
import numpy as np
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd

# =========================================================
# 1) Boolean Expression Tree
# =========================================================
class Node:
    def __init__(self, operator: Optional[str] = None, left: Optional["Node"] = None,
                 right: Optional["Node"] = None, var: Optional[str] = None):
        self.operator = operator  # "AND", "OR", "NOT"
        self.left = left
        self.right = right
        self.var = var            # TF name if leaf

    def is_leaf(self) -> bool:
        return self.var is not None

    def evaluate(self, assignment: Dict[str, int]) -> int:
        if self.is_leaf():
            return int(assignment[self.var])
        if self.operator == "NOT":
            return 1 - self.left.evaluate(assignment)
        if self.operator == "AND":
            return self.left.evaluate(assignment) & self.right.evaluate(assignment)
        if self.operator == "OR":
            return self.left.evaluate(assignment) | self.right.evaluate(assignment)
        raise ValueError(f"Unknown operator: {self.operator}")

    def preorder(self) -> List[str]:
        if self.is_leaf():
            return [self.var]
        if self.operator == "NOT":
            return [self.operator] + self.left.preorder()
        return [self.operator] + self.left.preorder() + self.right.preorder()

    def used_vars(self) -> List[str]:
        if self.is_leaf():
            return [self.var]
        if self.operator == "NOT":
            return self.left.used_vars()
        return self.left.used_vars() + self.right.used_vars()

    def pretty(self) -> str:
        if self.is_leaf():
            return self.var
        if self.operator == "NOT":
            return f"(NOT {self.left.pretty()})"
        return f"({self.left.pretty()} {self.operator} {self.right.pretty()})"


# =========================================================
# 2) Multi-objective Evaluator (IE, FC, PIC), Loss & Reward
# =========================================================
class MultiObjectiveEvaluator:
    def __init__(self, w1: float = 0.3, w2: float = 0.5, gamma: float = 1e-3, eta: float = 0.95):
        self.w1, self.w2, self.gamma, self.eta = w1, w2, gamma, eta

    @staticmethod
    def inconsistency_error(y_true: np.ndarray, y_pred: np.ndarray) -> int:
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")
        return int(np.sum(y_true != y_pred))

    @staticmethod
    def function_complexity(vars_used: List[str]) -> int:
        return len(set(vars_used))

    def prior_information_consistency(self, tfs: List[str],
                                      W: Dict[Tuple[str, str], float],
                                      degrees: Dict[str, float]) -> float:
        uniq = list(dict.fromkeys(tfs))
        k = len(uniq)
        if k <= 1:
            return 0.0

        if degrees:
            max_d = float(np.max(list(degrees.values())))
            if max_d <= 0:
                max_d = 1.0
        else:
            max_d = 1.0

        pic_value = 0.0
        for i in range(k):
            for j in range(i + 1, k):
                a, b = uniq[i], uniq[j]
                w = W.get((a, b), W.get((b, a), self.gamma))
                d_i = float(degrees.get(a, 1.0))
                d_j = float(degrees.get(b, 1.0))
                pic_value += (1.0 - w * (d_i * d_j) / (max_d ** 2))
        return float(pic_value)

    def aggregate_loss(self, IE: float, FC: float, PIC: float) -> float:
        return float(IE + self.w1 * FC + self.w2 * PIC)

    def reward(self, loss: float, tree_size: int) -> float:
        if loss >= 1.0:
            loss = 0.999999
        return float((self.eta ** int(tree_size)) / (1.0 - loss))


# =========================================================
# 3) Monte Carlo Tree Search (MCTS)
# =========================================================
class MCTS:
    def __init__(self, evaluator: MultiObjectiveEvaluator,
                 W: Dict[Tuple[str, str], float],
                 degrees: Dict[str, float],
                 max_iter: int = 800,
                 exploration_c: float = 1.4,
                 seed: Optional[int] = 7):
        self.evaluator = evaluator
        self.W, self.degrees = W, degrees
        self.max_iter, self.exploration_c = max_iter, exploration_c
        self.Q = defaultdict(float)  # cumulative rewards per node
        self.N = defaultdict(int)    # visits per node
        self.children = dict()       # node -> list[Node]
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def ucb_score(self, parent: Node, child: Node) -> float:
        if self.N[child] == 0:
            return float("inf")
        return (self.Q[child] / self.N[child] +
                self.exploration_c * np.sqrt(np.log(self.N[parent] + 1) / self.N[child]))

    def select(self, root: Node):
        node = root
        path = [node]
        while node in self.children and self.children[node]:
            node = max(self.children[node], key=lambda n: self.ucb_score(path[-1], n))
            path.append(node)
        return node, path

    def expand(self, node: Node, var_list: List[str], possible_ops: List[str]):
        if node not in self.children:
            self.children[node] = []
            for op in possible_ops:
                left = Node(var=random.choice(var_list))
                right = Node(var=random.choice(var_list)) if op != "NOT" else None
                self.children[node].append(Node(operator=op, left=left, right=right))
        return self.children[node]

    def simulate(self, expr_tree: Node, X_tf: pd.DataFrame, y: np.ndarray) -> float:
        y_pred = np.array([expr_tree.evaluate(dict(zip(X_tf.columns, row)))
                           for row in X_tf.values], dtype=int)
        IE = self.evaluator.inconsistency_error(y, y_pred)
        vars_used = expr_tree.used_vars()
        FC = self.evaluator.function_complexity(vars_used)
        PIC = self.evaluator.prior_information_consistency(vars_used, self.W, self.degrees)
        loss = self.evaluator.aggregate_loss(IE, FC, PIC)
        return self.evaluator.reward(loss, len(expr_tree.preorder()))

    def backpropagate(self, path: List[Node], reward: float):
        for n in reversed(path):
            self.N[n] += 1
            self.Q[n] += reward

    def search(self, X_tf: pd.DataFrame, y: np.ndarray,
               var_list: List[str], possible_ops: Optional[List[str]] = None) -> Node:
        if possible_ops is None:
            possible_ops = ["AND", "OR", "NOT"]
        # root is a single TF leaf
        root = Node(var=random.choice(var_list))
        self.N[root] += 0; self.Q[root] += 0.0

        for _ in range(self.max_iter):
            leaf_to_expand, path = self.select(root)
            children = self.expand(leaf_to_expand, var_list, possible_ops)
            leaf = random.choice(children) if children else leaf_to_expand
            reward = self.simulate(leaf, X_tf, y)
            self.backpropagate(path + [leaf], reward)

        if root in self.children and self.children[root]:
            return max(self.children[root], key=lambda n: self.Q[n] / (self.N[n] + 1e-6))
        return root


# =========================================================
# 4) High-level API: per-target rule and full-GRN inference
# =========================================================
class LogSR:
    def __init__(self, W: Dict[Tuple[str, str], float],
                 degrees: Dict[str, float],
                 w1: float = 0.3, w2: float = 0.5,
                 gamma: float = 1e-3, eta: float = 0.95,
                 max_iter: int = 800, exploration_c: float = 1.4, seed: Optional[int] = 7):
        evaluator = MultiObjectiveEvaluator(w1=w1, w2=w2, gamma=gamma, eta=eta)
        self.mcts = MCTS(evaluator, W, degrees, max_iter=max_iter,
                         exploration_c=exploration_c, seed=seed)

    def infer_rule(self, X_tf: pd.DataFrame, y: np.ndarray, var_list: List[str]) -> Node:
        return self.mcts.search(X_tf, y, var_list)

    def infer_grn(self, X: pd.DataFrame, tf_list: List[str]) -> Tuple[Dict[str, Node], pd.DataFrame]:
        """
        X: cells × genes (numeric). Internally binarized as (X>0).
        tf_list: list of TF names to use as regulators (must intersect X.columns).
        Returns: (rules, edges_df) where edges_df has columns ["TF","TG","score","rule"]
        """
        X_bin = (X > 0).astype(int)
        genes = list(X_bin.columns)
        tf_vars = [t for t in tf_list if t in genes]
        targets = [g for g in genes if g not in tf_vars]  # non-TF targets only

        rules: Dict[str, Node] = {}
        edges = []

        if len(tf_vars) == 0:
            raise ValueError("No TFs overlap with expression matrix columns.")

        X_tf_all = X_bin[tf_vars]

        for tg in targets:
            y = X_bin[tg].values.astype(int)
            # Skip degenerate targets (all 0 or all 1)
            if len(np.unique(y)) < 2:
                continue
            expr = self.infer_rule(X_tf_all, y, tf_vars)
            rules[tg] = expr
            for tf in sorted(set(expr.used_vars())):
                edges.append((tf, tg, 1.0, expr.pretty()))

        edges_df = pd.DataFrame(edges, columns=["TF", "TG", "score", "rule"])
        return rules, edges_df
