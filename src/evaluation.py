"""
Evaluation.

Three principles, each of which the prior literature routinely violates:

1. ACCURACY IS BANNED. With a 10-20% positive rate, predicting "never
   helpful" scores 80-90%. Any paper reporting accuracy on this task is
   reporting the base rate. We report PR-AUC as primary.

2. THE TEST SET IS NEVER RESAMPLED. Class weighting and resampling apply to
   training only. A test set rebalanced to 50/50 produces a PR-AUC that
   cannot be compared to anything, including the deployment setting.

3. RANKING IS THE REAL TASK. A platform does not classify reviews in
   isolation; it orders the reviews on one product page. We therefore
   report within-product NDCG alongside the classification metrics.
"""

import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# ----------------------------------------------------------------------
# Classification metrics
# ----------------------------------------------------------------------

def _extract_scores(predictions, label_col="label"):
    """Pull (label, positive-class probability) to the driver.

    Only two columns cross the JVM boundary, so this is safe even for a
    multi-million-row test set.
    """
    pdf = (
        predictions
        .select(
            F.col(label_col).cast("int").alias("y"),
            vector_to_array("probability").getItem(1).alias("p"),
        )
        .toPandas()
    )
    return pdf["y"].values, pdf["p"].values


def precision_recall_auc(y, p):
    """Average precision. Computed directly rather than by trapezoid on a
    coarse grid, which is what makes Spark's areaUnderPR unstable at low
    base rates."""
    order = np.argsort(-p)
    y = y[order]

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(y.sum(), 1)

    # Average precision: sum of precision at each positive, weighted by the
    # recall increment it produced.
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def roc_auc(y, p):
    """Rank-based AUC. Ties handled by average rank."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").values
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def best_f1(y, p, n_thresholds=200):
    """Sweep the threshold rather than assuming 0.5.

    With an imbalanced target, 0.5 is an arbitrary and usually terrible
    operating point. Reporting F1 at 0.5 understates every model equally,
    which hides genuine differences between them.
    """
    thresholds = np.quantile(p, np.linspace(0.01, 0.99, n_thresholds))
    best = {"f1": 0.0, "threshold": 0.5, "precision": 0.0, "recall": 0.0}

    for t in np.unique(thresholds):
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())

        if tp == 0:
            continue

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        if f1 > best["f1"]:
            best = {"f1": f1, "threshold": float(t),
                    "precision": precision, "recall": recall}

    return best


def lift_at_k(y, p, k=0.10):
    """Precision in the top k fraction, divided by the base rate.

    This is the number a platform actually cares about: if we surface the
    top 10% of reviews, how much better than random is that selection?
    """
    n = max(int(len(y) * k), 1)
    top = np.argsort(-p)[:n]
    base = y.mean()
    if base == 0:
        return float("nan")
    return float(y[top].mean() / base)


# ----------------------------------------------------------------------
# Ranking metric - the deployment-realistic one
# ----------------------------------------------------------------------

def within_product_ndcg(predictions, k=5, label_col="helpful_vote",
                        group_col="parent_asin", min_group=5):
    """
    NDCG@k computed within each product's review set.

    This mirrors what a review-ranking system does: given all reviews on one
    product page, order them so the most helpful appear first. Relevance is
    the actual helpful-vote count.

    Products with fewer than min_group reviews are excluded, since ranking
    three items is not a meaningful test.
    """
    pdf = (
        predictions
        .select(
            F.col(group_col).alias("g"),
            F.col(label_col).cast("double").alias("rel"),
            vector_to_array("probability").getItem(1).alias("score"),
        )
        .toPandas()
    )

    def _ndcg(group):
        if len(group) < min_group:
            return np.nan

        # DCG of our ordering
        ordered = group.sort_values("score", ascending=False)["rel"].values[:k]
        discounts = 1.0 / np.log2(np.arange(2, len(ordered) + 2))
        dcg = float(np.sum((2 ** ordered - 1) * discounts))

        # DCG of the perfect ordering
        ideal = np.sort(group["rel"].values)[::-1][:k]
        idiscounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
        idcg = float(np.sum((2 ** ideal - 1) * idiscounts))

        if idcg == 0:
            return np.nan
        return dcg / idcg

    scores = pdf.groupby("g", group_keys=False).apply(_ndcg)
    scores = scores.dropna()

    return {
        f"ndcg@{k}": float(scores.mean()) if len(scores) else float("nan"),
        "n_products_ranked": int(len(scores)),
    }


# ----------------------------------------------------------------------
# One-call evaluation
# ----------------------------------------------------------------------

def evaluate(predictions, name, label_col="label", compute_ndcg=True, k=5):
    """Full metric suite for one model on one split."""
    y, p = _extract_scores(predictions, label_col)

    base_rate = float(y.mean())
    pr = precision_recall_auc(y, p)

    row = {
        "model": name,
        "n_test": int(len(y)),
        "base_rate": round(base_rate, 4),
        "pr_auc": round(pr, 4),
        # PR-AUC lift over the trivial classifier, whose PR-AUC equals the
        # base rate. A model with PR-AUC 0.30 at a 25% base rate has learned
        # almost nothing; this column makes that visible.
        "pr_auc_lift": round(pr / base_rate, 3) if base_rate > 0 else float("nan"),
        "roc_auc": round(roc_auc(y, p), 4),
        "lift@10pct": round(lift_at_k(y, p, 0.10), 3),
    }

    f1 = best_f1(y, p)
    row.update({
        "f1": round(f1["f1"], 4),
        "precision": round(f1["precision"], 4),
        "recall": round(f1["recall"], 4),
        "threshold": round(f1["threshold"], 4),
    })

    if compute_ndcg:
        try:
            row.update({kk: (round(vv, 4) if isinstance(vv, float) else vv)
                        for kk, vv in within_product_ndcg(predictions, k=k).items()})
        except Exception as e:
            print(f"  [ndcg] skipped: {type(e).__name__}: {e}")

    return row


def results_table(rows, sort_by="pr_auc"):
    """Assemble and sort a comparison table."""
    df = pd.DataFrame(rows)
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    return df.reset_index(drop=True)
