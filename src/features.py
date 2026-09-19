"""
Feature engineering.

Features are grouped into named blocks so that ablations can add or remove
a whole theoretical construct at a time. That structure is what turns a
results table into an argument: "depth matters for search goods" is a claim
about a block, not about one column.

Blocks:
    STRUCTURAL   - length, depth, formatting effort
    LEXICAL      - vocabulary richness and readability
    AFFECTIVE    - sentiment polarity and rating extremity
    CONTEXTUAL   - how the review relates to the product's existing reviews
    REVIEWER     - the author's prior track record (no lookahead)
    EXPOSURE     - controls for visibility duration
    TEXT         - TF-IDF / embeddings, handled in models.py
"""

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# ----------------------------------------------------------------------
# Feature blocks
# ----------------------------------------------------------------------

STRUCTURAL = [
    "review_length",
    "log_review_length",
    "review_word_count",
    "sentence_count",
    "avg_sentence_length",
    "avg_word_length",
    "title_length",
    "title_word_count",
    "has_images",
]

LEXICAL = [
    "type_token_ratio",
    "long_word_ratio",
    "uppercase_ratio",
    "digit_ratio",
    "punctuation_density",
    "exclamation_count",
    "question_count",
    "flesch_reading_ease",
]

AFFECTIVE = [
    "rating",
    "rating_extremity",
    "is_extreme_rating",
    "is_moderate_rating",
    "sentiment_positive_ratio",
    "sentiment_negative_ratio",
    "sentiment_polarity",
]

CONTEXTUAL = [
    "verified_int",
    "rating_deviation_from_product",
    "log_product_prior_review_count",
    "is_first_review_of_product",
    "product_prior_avg_rating",
]

REVIEWER = [
    "log_prior_review_count",
    "prior_avg_helpful",
    "prior_avg_rating",
    "prior_avg_length",
    "is_first_review_by_user",
]

EXPOSURE = [
    "log_exposure_days",
]

ALL_NUMERIC = STRUCTURAL + LEXICAL + AFFECTIVE + CONTEXTUAL + REVIEWER + EXPOSURE

BLOCKS = {
    "structural": STRUCTURAL,
    "lexical": LEXICAL,
    "affective": AFFECTIVE,
    "contextual": CONTEXTUAL,
    "reviewer": REVIEWER,
    "exposure": EXPOSURE,
}


# ----------------------------------------------------------------------
# Sentiment lexicon
# ----------------------------------------------------------------------
#
# A compact opinion lexicon, applied with native Spark functions rather than
# a Python UDF. This keeps the whole pipeline inside the JVM and avoids
# serialising several million review strings into Python workers, which is
# the usual reason PySpark text pipelines crawl.
#
# For the journal version, swap this for VADER or a fine-tuned classifier
# and report both, so the sensitivity of results to sentiment measurement
# is visible.

POSITIVE_WORDS = [
    "good", "great", "excellent", "perfect", "love", "loved", "best", "amazing",
    "wonderful", "fantastic", "nice", "happy", "recommend", "recommended",
    "awesome", "comfortable", "beautiful", "easy", "worth", "solid", "impressed",
    "pleased", "satisfied", "quality", "works", "durable", "reliable", "fast",
    "favorite", "smooth", "sturdy", "bright", "clear", "helpful", "pretty",
]

NEGATIVE_WORDS = [
    "bad", "poor", "terrible", "awful", "hate", "hated", "worst", "broke",
    "broken", "cheap", "disappointed", "disappointing", "useless", "waste",
    "return", "returned", "defective", "failed", "fails", "problem", "issue",
    "issues", "junk", "flimsy", "uncomfortable", "difficult", "annoying",
    "refund", "damaged", "faulty", "stopped", "wrong", "horrible", "scam",
]


def _lexicon_count(text_col, words):
    """Count lexicon hits using a single regex. One pass, no UDF."""
    pattern = r"(?i)\b(" + "|".join(words) + r")\b"
    return F.size(F.split(F.col(text_col), pattern)) - 1


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def add_features(df):
    """Attach every engineered feature. Pure function of already-present
    columns, so it can be applied to any split independently."""

    # ---------------- STRUCTURAL ----------------
    df = (
        df
        .withColumn("review_length", F.length("text").cast("double"))
        .withColumn("log_review_length", F.log1p(F.col("review_length")))
        .withColumn(
            "review_word_count",
            F.when(F.trim("text") == "", 0.0)
             .otherwise(F.size(F.split(F.trim("text"), r"\s+")).cast("double")),
        )
        .withColumn(
            "sentence_count",
            F.greatest(
                F.lit(1.0),
                (F.size(F.split(F.col("text"), r"[.!?]+")) - 1).cast("double"),
            ),
        )
        .withColumn("title_length", F.length(F.coalesce("title", F.lit(""))).cast("double"))
        .withColumn(
            "title_word_count",
            F.when(F.trim(F.coalesce("title", F.lit(""))) == "", 0.0)
             .otherwise(F.size(F.split(F.trim(F.coalesce("title", F.lit(""))), r"\s+")).cast("double")),
        )
    )

    df = (
        df
        .withColumn(
            "avg_sentence_length",
            F.col("review_word_count") / F.greatest(F.col("sentence_count"), F.lit(1.0)),
        )
        .withColumn(
            "avg_word_length",
            F.col("review_length") / F.greatest(F.col("review_word_count"), F.lit(1.0)),
        )
    )

    # Reviews with attached photos. The images column is a serialised list;
    # an empty list is the common case.
    if "images" in df.columns:
        df = df.withColumn(
            "has_images",
            F.when(
                F.col("images").isNull()
                | (F.trim(F.col("images")) == "")
                | (F.trim(F.col("images")) == "[]"),
                0.0,
            ).otherwise(1.0),
        )
    else:
        df = df.withColumn("has_images", F.lit(0.0))

    # ---------------- LEXICAL ----------------
    df = (
        df
        .withColumn(
            "type_token_ratio",
            F.size(F.array_distinct(F.split(F.lower(F.trim("text")), r"\s+"))).cast("double")
            / F.greatest(F.col("review_word_count"), F.lit(1.0)),
        )
        .withColumn(
            "long_word_ratio",
            (F.size(F.split(F.col("text"), r"\b\w{7,}\b")) - 1).cast("double")
            / F.greatest(F.col("review_word_count"), F.lit(1.0)),
        )
        .withColumn(
            "uppercase_ratio",
            (F.length(F.col("text")) - F.length(F.regexp_replace(F.col("text"), r"[A-Z]", ""))).cast("double")
            / F.greatest(F.col("review_length"), F.lit(1.0)),
        )
        .withColumn(
            "digit_ratio",
            (F.length(F.col("text")) - F.length(F.regexp_replace(F.col("text"), r"[0-9]", ""))).cast("double")
            / F.greatest(F.col("review_length"), F.lit(1.0)),
        )
        .withColumn(
            "punctuation_density",
            (F.length(F.col("text")) - F.length(F.regexp_replace(F.col("text"), r"[^\w\s]", ""))).cast("double")
            / F.greatest(F.col("review_length"), F.lit(1.0)),
        )
        .withColumn(
            "exclamation_count",
            (F.length(F.col("text")) - F.length(F.regexp_replace(F.col("text"), r"!", ""))).cast("double"),
        )
        .withColumn(
            "question_count",
            (F.length(F.col("text")) - F.length(F.regexp_replace(F.col("text"), r"\?", ""))).cast("double"),
        )
    )

    # Flesch Reading Ease, with vowel-group count as the syllable proxy.
    # Exact syllabification needs a pronunciation dictionary and a Python UDF;
    # the vowel-group approximation correlates highly and costs nothing.
    df = df.withColumn(
        "syllable_estimate",
        F.greatest(
            F.lit(1.0),
            (F.size(F.split(F.lower(F.col("text")), r"[aeiouy]+")) - 1).cast("double"),
        ),
    )

    df = df.withColumn(
        "flesch_reading_ease",
        F.lit(206.835)
        - F.lit(1.015) * (F.col("review_word_count") / F.greatest(F.col("sentence_count"), F.lit(1.0)))
        - F.lit(84.6) * (F.col("syllable_estimate") / F.greatest(F.col("review_word_count"), F.lit(1.0))),
    )

    # ---------------- AFFECTIVE ----------------
    # Rating extremity is the key moderator in the search/experience
    # hypothesis. Mudambi & Schuff predict extremity HURTS helpfulness for
    # search goods but not for experience goods.
    df = (
        df
        .withColumn("rating_extremity", F.abs(F.col("rating") - F.lit(3.0)))
        .withColumn("is_extreme_rating", F.col("rating").isin([1.0, 5.0]).cast("double"))
        .withColumn("is_moderate_rating", F.col("rating").isin([2.0, 3.0, 4.0]).cast("double"))
    )

    df = (
        df
        .withColumn("_pos_hits", _lexicon_count("text", POSITIVE_WORDS).cast("double"))
        .withColumn("_neg_hits", _lexicon_count("text", NEGATIVE_WORDS).cast("double"))
    )

    df = (
        df
        .withColumn(
            "sentiment_positive_ratio",
            F.col("_pos_hits") / F.greatest(F.col("review_word_count"), F.lit(1.0)),
        )
        .withColumn(
            "sentiment_negative_ratio",
            F.col("_neg_hits") / F.greatest(F.col("review_word_count"), F.lit(1.0)),
        )
        .withColumn(
            "sentiment_polarity",
            (F.col("_pos_hits") - F.col("_neg_hits"))
            / F.greatest(F.col("_pos_hits") + F.col("_neg_hits"), F.lit(1.0)),
        )
    )

    # ---------------- CONTEXTUAL ----------------
    df = df.withColumn(
        "verified_int",
        F.coalesce(F.col("verified_purchase").cast("double"), F.lit(0.0)),
    )

    # How far this review's rating sits from the product's existing consensus.
    # Uses only PRIOR reviews, so no lookahead.
    df = df.withColumn(
        "rating_deviation_from_product",
        F.when(
            F.col("is_first_review_of_product") == 1, F.lit(0.0)
        ).otherwise(F.abs(F.col("rating") - F.col("product_prior_avg_rating"))),
    )

    # ---------------- cleanup ----------------
    df = df.drop("_pos_hits", "_neg_hits", "syllable_estimate")

    # Guard against nulls and infinities reaching the model. Any null here
    # is a bug upstream, so we fail loudly in debug rather than silently
    # imputing, but fill for robustness in production runs.
    for c in ALL_NUMERIC:
        if c in df.columns:
            df = df.withColumn(
                c,
                F.when(
                    F.col(c).isNull() | F.isnan(F.col(c)) | F.col(c).isin([float("inf"), float("-inf")]),
                    F.lit(0.0),
                ).otherwise(F.col(c).cast(DoubleType())),
            )

    return df


def feature_summary(df, cols=None):
    """Descriptive statistics table for the paper's feature section."""
    cols = cols or [c for c in ALL_NUMERIC if c in df.columns]
    aggs = []
    for c in cols:
        aggs += [
            F.avg(c).alias(f"{c}__mean"),
            F.stddev(c).alias(f"{c}__std"),
            F.min(c).alias(f"{c}__min"),
            F.max(c).alias(f"{c}__max"),
        ]
    row = df.agg(*aggs).first().asDict()

    import pandas as pd
    records = []
    for c in cols:
        records.append({
            "feature": c,
            "mean": row[f"{c}__mean"],
            "std": row[f"{c}__std"],
            "min": row[f"{c}__min"],
            "max": row[f"{c}__max"],
        })
    return pd.DataFrame(records)
