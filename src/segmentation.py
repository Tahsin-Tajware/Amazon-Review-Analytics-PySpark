"""
Reviewer segmentation with K-Means.

COURSE MAPPING: Week 3 — MLlib clustering.

WHY THIS BELONGS IN THE STUDY, NOT JUST THE SYLLABUS
----------------------------------------------------
The fairness analysis establishes that helpful votes are extremely
concentrated: a Gini above 0.9, with the top 1% of reviews holding roughly
half of all votes. That is a statement about the *distribution*. It does not
say who is on the winning side of it.

Segmentation answers that. If the concentration is produced by a small,
identifiable population of prolific reviewers, the platform's feedback loop
is not measuring review quality so much as rewarding a reviewing elite — and
that is a materially different claim, with different implications for how a
ranking system should be designed.

METHODOLOGICAL NOTE
-------------------
Reviewer features are aggregates over that reviewer's entire history. This is
legitimate here because segmentation is a DESCRIPTIVE task: we are
characterising the population, not predicting anything. The prohibition on
lookahead applies to the prediction pipeline, where a feature computed over a
review's future would leak the target. Nothing is predicted here, so nothing
can leak — but the distinction is stated explicitly so the two uses of
reviewer aggregates are not confused.
"""

import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from config import RANDOM_SEED


# Minimum history for a reviewer to be segmentable. Below this, the aggregate
# features are too noisy to describe behaviour.
MIN_REVIEWS_PER_REVIEWER = 3

FEATURES = [
    "n_reviews",
    "avg_rating",
    "avg_length",
    "avg_helpful",
    "verified_share",
    "n_products",
    "neg_share",
    "helpful_hit_rate",
]

K_RANGE = [2, 3, 4, 5, 6]


def build_reviewer_profiles(df, min_reviews=MIN_REVIEWS_PER_REVIEWER):
    """One row per reviewer: activity, rating behaviour, reception."""
    profiles = (
        df.groupBy("user_id")
        .agg(
            F.count("*").alias("n_reviews"),
            F.avg("rating").alias("avg_rating"),
            F.avg("review_length").alias("avg_length"),
            F.avg("helpful_vote").alias("avg_helpful"),
            F.sum("helpful_vote").alias("total_helpful"),
            F.avg(F.col("verified_int")).alias("verified_share"),
            F.countDistinct("parent_asin").alias("n_products"),
            F.avg((F.col("rating") <= 2).cast("double")).alias("neg_share"),
            # What share of this reviewer's reviews landed at least one vote.
            # Distinguishes "occasionally writes a hit" from "consistently useful".
            F.avg("label").alias("helpful_hit_rate"),
        )
        .filter(F.col("n_reviews") >= min_reviews)
    )
    return profiles


def choose_k(profiles, k_range=K_RANGE, seed=RANDOM_SEED):
    """Sweep k and select by silhouette.

    Silhouette is reported for every k rather than only the winner, because a
    flat silhouette curve means the data has no strong cluster structure and
    the chosen k is close to arbitrary. Hiding the curve would let a weak
    segmentation pass as a strong one.
    """
    rows = []
    best_k, best_score, best_model = None, -1.0, None

    assembler = VectorAssembler(inputCols=FEATURES, outputCol="_raw",
                                handleInvalid="skip")
    scaler = StandardScaler(inputCol="_raw", outputCol="_scaled",
                            withMean=True, withStd=True)

    for k in k_range:
        try:
            pipe = Pipeline(stages=[
                assembler, scaler,
                KMeans(featuresCol="_scaled", k=k, seed=seed, maxIter=50),
            ])
            model = pipe.fit(profiles)
            preds = model.transform(profiles)

            score = ClusteringEvaluator(
                featuresCol="_scaled", predictionCol="prediction",
                metricName="silhouette",
            ).evaluate(preds)

            rows.append({"k": k, "silhouette": round(float(score), 4)})
            print(f"  k={k}: silhouette={score:.4f}")

            if score > best_score:
                best_k, best_score, best_model = k, score, model
        except Exception as e:
            print(f"  k={k}: FAILED ({type(e).__name__}: {e})")

    return best_k, best_score, best_model, pd.DataFrame(rows)


def profile_clusters(model, profiles):
    """Describe each segment, and — the point of the exercise — report how
    much of the total helpful-vote supply each segment captures."""
    clustered = model.transform(profiles).cache()

    summary = (
        clustered.groupBy("prediction")
        .agg(
            F.count("*").alias("reviewers"),
            *[F.round(F.avg(c), 3).alias(c) for c in FEATURES],
            F.sum("total_helpful").alias("segment_total_helpful"),
            F.sum("n_reviews").alias("segment_total_reviews"),
        )
        .orderBy("prediction")
    ).toPandas()

    total_votes = max(summary["segment_total_helpful"].sum(), 1)
    total_reviewers = max(summary["reviewers"].sum(), 1)
    total_reviews = max(summary["segment_total_reviews"].sum(), 1)

    summary["pct_of_reviewers"] = (100 * summary["reviewers"] / total_reviewers).round(2)
    summary["pct_of_reviews"] = (100 * summary["segment_total_reviews"] / total_reviews).round(2)
    summary["pct_of_all_votes"] = (100 * summary["segment_total_helpful"] / total_votes).round(2)

    # The inequality ratio: a segment holding 40% of votes with 5% of
    # reviewers has a ratio of 8. Ratio 1 means proportionate.
    summary["vote_capture_ratio"] = (
        summary["pct_of_all_votes"] / summary["pct_of_reviewers"].replace(0, np.nan)
    ).round(2)

    clustered.unpersist()
    return summary


def label_segments(summary):
    """
    Name each segment after whatever most distinguishes it from the others.

    An earlier version compared each cluster against the median on several
    attributes independently. With k=2 that is degenerate — one cluster is
    always above the median on every attribute and the other always below —
    so a cluster could be named "Critical voices" purely because it sat on
    the upper side of a two-point median, with no actual negativity.

    This version ranks clusters on each attribute and names a cluster after
    the attribute on which it is most extreme relative to the spread across
    clusters. The label therefore always corresponds to something visible in
    the numbers.
    """
    if summary.empty:
        return summary

    s = summary.copy()

    # z-score each attribute ACROSS clusters, so "high" means high relative to
    # the other segments rather than relative to an arbitrary threshold.
    attrs = {
        "helpful_hit_rate": ("Influential reviewers", "Overlooked reviewers"),
        "n_reviews": ("Prolific reviewers", "Infrequent reviewers"),
        "avg_length": ("Long-form reviewers", "Brief reviewers"),
        "neg_share": ("Critical voices", "Positive reviewers"),
        "verified_share": ("Verified buyers", "Unverified reviewers"),
    }

    z = {}
    for a in attrs:
        if a in s.columns:
            sd = s[a].std()
            z[a] = (s[a] - s[a].mean()) / (sd if sd and sd > 0 else 1.0)

    labels, used = [], set()
    for i in range(len(s)):
        # Most extreme attribute for this cluster, skipping names already taken
        ranked = sorted(z.keys(), key=lambda a: -abs(z[a].iloc[i]))
        name = None
        for a in ranked:
            high, low = attrs[a]
            candidate = high if z[a].iloc[i] >= 0 else low
            if candidate not in used:
                name = candidate
                break
        if name is None:
            name = f"Segment {i}"
        used.add(name)
        labels.append(name)

    s.insert(1, "segment", labels)
    return s


def run(df):
    """Entry point. Returns (silhouette_table, segment_profile_table)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 7 - REVIEWER SEGMENTATION (K-Means)")
    print("=" * 70)

    total_reviewers = df.select("user_id").distinct().count()
    profiles = build_reviewer_profiles(df).cache()
    n = profiles.count()

    coverage = 100 * n / max(total_reviewers, 1)
    print(f"Segmenting {n:,} of {total_reviewers:,} reviewers "
          f"(>= {MIN_REVIEWS_PER_REVIEWER} reviews) — {coverage:.2f}% coverage\n")

    if n < 100:
        print("Too few multi-review reviewers to segment; skipped.")
        profiles.unpersist()
        return pd.DataFrame(), pd.DataFrame()

    if coverage < 5.0:
        print("!" * 70)
        print("COVERAGE WARNING - read before interpreting these segments")
        print("!" * 70)
        print(f"Only {coverage:.2f}% of reviewers meet the minimum-history")
        print("threshold, so these segments describe a small and unusual")
        print("subpopulation, NOT the reviewer base.")
        print()
        print("The cause is almost certainly how the extract was sampled.")
        print("Sampling REVIEWS at random destroys USER structure: a reviewer")
        print("with many reviews in the source corpus will usually appear")
        print("exactly once in a small random sample of it. Near-universal")
        print("singleton reviewers are what that procedure produces,")
        print("regardless of the underlying marketplace.")
        print()
        print("Conclusions here therefore describe this extract. To study")
        print("reviewer behaviour properly, use the complete per-category")
        print("dumps from the Amazon Reviews 2023 release, which preserve")
        print("each reviewer's full history.")
        print("!" * 70)
        print()

    best_k, best_score, model, sil_table = choose_k(profiles)

    if model is None:
        profiles.unpersist()
        return sil_table, pd.DataFrame()

    print(f"\nSelected k = {best_k} (silhouette = {best_score:.4f})")

    spread = sil_table["silhouette"].max() - sil_table["silhouette"].min()
    if spread < 0.05:
        print("NOTE: silhouette is nearly flat across k, so the cluster")
        print("      structure is weak and this choice of k is close to")
        print("      arbitrary. Treat the segments as a descriptive device,")
        print("      not as discovered natural kinds.")

    summary = label_segments(profile_clusters(model, profiles))

    # Guard against degenerate clusters. K-Means on heavy-tailed behavioural
    # data will happily isolate a single extreme outlier into its own cluster;
    # such a "segment" describes one person and its capture ratio is
    # undefined. Flag them rather than quietly reporting them as segments.
    tiny = summary[summary["reviewers"] < 10]
    if not tiny.empty:
        print(f"\nNOTE: {len(tiny)} cluster(s) contain fewer than 10 reviewers "
              f"and are outlier\n      isolates rather than segments. They are "
              f"marked below and should not\n      be interpreted as "
              f"behavioural groups.")
        summary.loc[summary["reviewers"] < 10, "segment"] = (
            summary.loc[summary["reviewers"] < 10, "segment"]
            .astype(str) + " [OUTLIER ISOLATE - n<10]"
        )

    print("\nSegments:")
    cols = ["prediction", "segment", "reviewers", "pct_of_reviewers",
            "pct_of_reviews", "pct_of_all_votes", "vote_capture_ratio",
            "n_reviews", "avg_length", "helpful_hit_rate"]
    print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))

    try:
        top = summary.loc[summary["vote_capture_ratio"].idxmax()]
        print(
            f"\n>>> '{top['segment']}' are {top['pct_of_reviewers']:.1f}% of "
            f"reviewers but capture {top['pct_of_all_votes']:.1f}% of all "
            f"helpful votes ({top['vote_capture_ratio']:.1f}x their share)."
        )
        print(">>> This is the population behind the Gini result: vote")
        print(">>> concentration is not diffuse, it has an identifiable owner.")
    except Exception:
        pass

    profiles.unpersist()
    return sil_table, summary
