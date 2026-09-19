"""
Data provenance diagnostics.

WHY THIS MODULE EXISTS
----------------------
A dataset's sampling design determines which questions it can answer, and
that design is not always stated. This extract of Amazon Reviews 2023 turns
out to be a review-level random sample: 1.029 reviews per reviewer, with
97.4% of reviewers appearing exactly once.

That single fact silently invalidates any analysis that depends on user-level
structure — reviewer history, reviewer segmentation, the reviewer-product
graph. Not because the analysis is wrong, but because the variable it needs
is nearly constant by construction.

We found this only after running those analyses and getting suspicious
results. This module runs the same checks FIRST, so the constraint is
declared before any conclusion is drawn from data that cannot support it.

THE UNDERLYING STATISTICS
-------------------------
If a corpus is sampled at rate p and a user has k reviews in it, the expected
number appearing in the sample is p*k, and

    P(user appears at least twice) = 1 - (1-p)^k - k*p*(1-p)^(k-1)

For p = 0.02 and k = 10 this is about 1.6%. So even a genuinely prolific
reviewer will almost always appear once in a small random sample. Singleton
dominance is what the sampling procedure produces; it is not evidence about
how people review.

The practical consequence: **never infer user-level behaviour from a
review-level sample.** Use complete per-category dumps instead, which
preserve every reviewer's full history.
"""

import numpy as np
import pandas as pd

from pyspark.sql import functions as F


# Below this, user-level analyses are not supportable.
MIN_REVIEWS_PER_REVIEWER_FOR_USER_ANALYSIS = 1.15


def diagnose(df):
    """Characterise the sampling design and report what it permits."""
    print("\n" + "=" * 70)
    print("DATA PROVENANCE DIAGNOSTIC")
    print("=" * 70)

    n_reviews = df.count()
    n_users = df.select("user_id").distinct().count()
    n_products = df.select("parent_asin").distinct().count()

    rpu = n_reviews / max(n_users, 1)
    rpp = n_reviews / max(n_products, 1)

    user_freq = df.groupBy("user_id").count()
    singletons = user_freq.filter(F.col("count") == 1).count()
    pct_singleton = 100 * singletons / max(n_users, 1)

    prod_freq = df.groupBy("parent_asin").count()
    prod_singletons = prod_freq.filter(F.col("count") == 1).count()
    pct_prod_singleton = 100 * prod_singletons / max(n_products, 1)

    print(f"reviews              {n_reviews:>12,}")
    print(f"unique reviewers     {n_users:>12,}")
    print(f"unique products      {n_products:>12,}")
    print(f"reviews per reviewer {rpu:>12.3f}")
    print(f"reviews per product  {rpp:>12.3f}")
    print(f"single-review users  {pct_singleton:>11.2f}%")
    print(f"single-review items  {pct_prod_singleton:>11.2f}%")

    user_level_ok = rpu >= MIN_REVIEWS_PER_REVIEWER_FOR_USER_ANALYSIS

    print("\nWhat this sampling design supports:")
    checks = [
        ("Review-level prediction", True,
         "each review carries its own text and metadata"),
        ("Temporal validation", True,
         "timestamps are intact regardless of sampling"),
        ("Exposure / arrival analysis", rpp >= 1.5,
         f"needs products with multiple reviews ({rpp:.2f} per product)"),
        ("Product ranking (Wilson / Bayes)", rpp >= 1.5,
         "needs products with enough reviews to rank"),
        ("Reviewer history features", user_level_ok,
         f"needs reviewers with prior reviews ({rpu:.3f} per reviewer)"),
        ("Reviewer segmentation", user_level_ok,
         "needs reviewers with enough history to profile"),
        ("Reviewer-product graph", user_level_ok,
         "needs shared reviewers to connect products"),
    ]

    rows = []
    for name, ok, why in checks:
        mark = "YES" if ok else "NO "
        print(f"  [{mark}] {name:<34} {why}")
        rows.append({"analysis": name, "supported": bool(ok), "reason": why})

    if not user_level_ok:
        print("\n" + "!" * 70)
        print("THIS EXTRACT IS A REVIEW-LEVEL RANDOM SAMPLE")
        print("!" * 70)
        print(f"At {rpu:.3f} reviews per reviewer, user-level structure has been")
        print("destroyed by the sampling, not by the marketplace. Sampling")
        print("reviews at random from a large corpus leaves nearly every user")
        print("with exactly one review, whatever their true activity.")
        print()
        print("Consequences enforced elsewhere in this pipeline:")
        print("  * Reviewer-history findings are reported NOT TESTABLE rather")
        print("    than FAILS, because a near-constant predictor cannot")
        print("    produce evidence either way (see replication.py).")
        print("  * Segmentation and graph results are labelled as properties")
        print("    of this extract, not of Amazon.")
        print()
        print("The fix for future work is the complete per-category dumps from")
        print("the Amazon Reviews 2023 release, which preserve full reviewer")
        print("histories. Every method here transfers to them unchanged.")
        print("!" * 70)

    summary = pd.DataFrame([{
        "reviews": n_reviews,
        "unique_reviewers": n_users,
        "unique_products": n_products,
        "reviews_per_reviewer": round(rpu, 4),
        "reviews_per_product": round(rpp, 4),
        "pct_single_review_users": round(pct_singleton, 2),
        "pct_single_review_products": round(pct_prod_singleton, 2),
        "is_review_level_sample": not user_level_ok,
        "supports_user_level_analysis": user_level_ok,
    }])

    return summary, pd.DataFrame(rows)


def reviewer_frequency_table(df, max_k=10):
    """Distribution of reviews per reviewer — the evidence for the diagnosis."""
    freq = (
        df.groupBy("user_id").count()
        .groupBy("count").agg(F.count("*").alias("n_users"))
        .orderBy("count")
    ).toPandas()

    total = freq["n_users"].sum()
    freq["pct_of_reviewers"] = (100 * freq["n_users"] / max(total, 1)).round(3)
    freq = freq.rename(columns={"count": "reviews_written"})

    return freq.head(max_k)
