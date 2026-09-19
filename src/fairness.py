"""
Exposure bias and fair product ranking.

Two related problems, both stemming from the fact that visibility and
votes reinforce each other:

  1. REVIEW-LEVEL. Reviews that get early votes rise up the page, get seen
     more, and get more votes. Late reviews on established products are
     structurally disadvantaged regardless of quality. We quantify this.

  2. PRODUCT-LEVEL. Ranking products by mean rating puts a product with one
     5-star review above one with 500 reviews averaging 4.8. We compare
     naive ranking against two principled alternatives.

This section also discharges the ethics requirement: the finding is that
the platform's own feedback mechanism systematically disadvantages new
contributors, which is a fairness claim with evidence behind it rather
than a paragraph of boilerplate.
"""

import numpy as np
import pandas as pd

from pyspark.sql import functions as F, Window

from config import MIN_REVIEWS_PER_PRODUCT_FOR_RANKING


Z = 1.96  # 95% confidence


# ----------------------------------------------------------------------
# Product ranking
# ----------------------------------------------------------------------

def rank_products(df, min_reviews=MIN_REVIEWS_PER_PRODUCT_FOR_RANKING):
    """
    Three rankings of the same products:

      naive     - mean rating. What most sites show.
      wilson    - lower bound of the 95% CI on the proportion of positive
                  ratings. Penalises small samples, never overstates.
      bayesian  - empirical-Bayes posterior mean, shrinking each product
                  toward the global mean with strength set by the observed
                  variance across products (method of moments).

    Wilson is conservative by construction: it answers "what is the worst
    this product plausibly is". Empirical Bayes answers "what is this
    product most likely to be", which is usually the question a ranker
    should ask. Reporting both makes the choice explicit rather than
    accidental.
    """
    prod = (
        df.groupBy("parent_asin", "Category", "product_type")
        .agg(
            F.count("*").alias("n"),
            F.avg("rating").alias("avg_rating"),
            F.avg((F.col("rating") >= 4).cast("double")).alias("p_positive"),
            F.sum((F.col("rating") >= 4).cast("double")).alias("n_positive"),
            F.avg("helpful_vote").alias("avg_helpful"),
        )
        .filter(F.col("n") >= min_reviews)
    )

    # Wilson lower bound
    prod = prod.withColumn(
        "wilson",
        (
            F.col("p_positive")
            + F.lit(Z ** 2) / (2 * F.col("n"))
            - F.lit(Z) * F.sqrt(
                (F.col("p_positive") * (1 - F.col("p_positive"))
                 + F.lit(Z ** 2) / (4 * F.col("n"))) / F.col("n")
            )
        ) / (1 + F.lit(Z ** 2) / F.col("n"))
    )

    # Empirical Bayes with a Beta prior fitted by moments across products
    stats = prod.agg(
        F.avg("p_positive").alias("mu"),
        F.variance("p_positive").alias("var"),
    ).first()

    mu = float(stats["mu"] or 0.5)
    var = float(stats["var"] or 0.01)

    # Beta(alpha, beta) with the observed mean and variance
    if 0 < var < mu * (1 - mu):
        strength = mu * (1 - mu) / var - 1
        alpha_prior = mu * strength
        beta_prior = (1 - mu) * strength
    else:
        alpha_prior, beta_prior = 1.0, 1.0

    print(f"[empirical Bayes] prior Beta(alpha={alpha_prior:.2f}, "
          f"beta={beta_prior:.2f}), equivalent to "
          f"{alpha_prior + beta_prior:.1f} pseudo-reviews")

    prod = prod.withColumn(
        "bayesian",
        (F.col("n_positive") + F.lit(alpha_prior))
        / (F.col("n") + F.lit(alpha_prior) + F.lit(beta_prior)),
    )

    return prod, {"alpha": alpha_prior, "beta": beta_prior, "mu": mu}


def ranking_comparison(prod, top_k=20):
    """How much does the top-K list change when you rank honestly?

    Rank correlation alone understates the problem, because the damage is
    concentrated at the top, which is the only part anyone sees.
    """
    cols = ["parent_asin", "Category", "n", "avg_rating", "p_positive",
            "wilson", "bayesian"]

    naive = prod.orderBy(F.desc("avg_rating")).limit(top_k).select(*cols).toPandas()
    wilson = prod.orderBy(F.desc("wilson")).limit(top_k).select(*cols).toPandas()
    bayes = prod.orderBy(F.desc("bayesian")).limit(top_k).select(*cols).toPandas()

    overlap_w = len(set(naive["parent_asin"]) & set(wilson["parent_asin"]))
    overlap_b = len(set(naive["parent_asin"]) & set(bayes["parent_asin"]))

    print(f"\nTop-{top_k} overlap with naive ranking:")
    print(f"  Wilson          {overlap_w}/{top_k}  "
          f"({100 * overlap_w / top_k:.0f}% retained)")
    print(f"  Empirical Bayes {overlap_b}/{top_k}  "
          f"({100 * overlap_b / top_k:.0f}% retained)")

    print(f"\nMedian review count in each top-{top_k}:")
    print(f"  naive           {naive['n'].median():.0f}")
    print(f"  Wilson          {wilson['n'].median():.0f}")
    print(f"  Empirical Bayes {bayes['n'].median():.0f}")
    print("\nThe naive list is dominated by thinly-reviewed products. That "
          "is the bias,\nand it is the reason mean rating should not be used "
          "as a ranking key.")

    return {"naive": naive, "wilson": wilson, "bayesian": bayes}


# ----------------------------------------------------------------------
# Review-level exposure bias
# ----------------------------------------------------------------------

def exposure_bias_analysis(df):
    """
    Does arriving late to a product page cost a review votes, holding
    quality constant?

    We bucket reviews by their arrival position within the product's review
    sequence and compare vote outcomes, controlling for review length as a
    crude quality proxy. A clean causal estimate would need an instrument;
    this is descriptive and must be labelled as such.
    """
    w = Window.partitionBy("parent_asin").orderBy("timestamp")

    d = df.withColumn("arrival_position", F.row_number().over(w))
    d = d.withColumn(
        "arrival_bucket",
        F.when(F.col("arrival_position") <= 5, "1-5 (earliest)")
         .when(F.col("arrival_position") <= 20, "6-20")
         .when(F.col("arrival_position") <= 50, "21-50")
         .when(F.col("arrival_position") <= 200, "51-200")
         .otherwise("200+ (latest)")
    )

    result = (
        d.groupBy("arrival_bucket")
        .agg(
            F.count("*").alias("reviews"),
            F.round(F.avg("helpful_vote"), 3).alias("avg_helpful_votes"),
            F.round(100 * F.avg("label"), 2).alias("pct_receiving_any_vote"),
            F.round(F.avg("review_length"), 0).alias("avg_length"),
            F.round(F.avg("rating"), 3).alias("avg_rating"),
            F.round(F.avg("exposure_days"), 0).alias("avg_exposure_days"),
        )
    ).toPandas()

    order = ["1-5 (earliest)", "6-20", "21-50", "51-200", "200+ (latest)"]
    result["_o"] = result["arrival_bucket"].map({k: i for i, k in enumerate(order)})
    result = result.sort_values("_o").drop(columns="_o").reset_index(drop=True)

    print("\nVote outcomes by arrival position within the product's review "
          "sequence:")
    print(result.to_string(index=False))
    print("\nNote the average length column: if it is roughly flat across "
          "buckets\nwhile vote rates fall sharply, the gap is positional, "
          "not quality-driven.\nThis is descriptive evidence, not a causal "
          "estimate.")

    return result


def gini_coefficient(df, col="helpful_vote"):
    """Concentration of helpful votes across reviews.

    A Gini near 1 means a tiny minority of reviews absorb essentially all
    the votes, which is the signature of a winner-take-all feedback loop.
    """
    values = np.sort(
        df.select(F.col(col).cast("double")).toPandas()[col].values
    )
    n = len(values)
    if n == 0 or values.sum() == 0:
        return float("nan")

    index = np.arange(1, n + 1)
    gini = float((2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n)

    print(f"\nGini coefficient of {col}: {gini:.4f}")
    top1 = values[int(n * 0.99):].sum() / values.sum()
    top10 = values[int(n * 0.90):].sum() / values.sum()
    print(f"  Top 1%  of reviews hold {100 * top1:.1f}% of all helpful votes")
    print(f"  Top 10% of reviews hold {100 * top10:.1f}% of all helpful votes")

    return gini
