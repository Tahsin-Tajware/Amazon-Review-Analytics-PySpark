"""
Leakage-free data pipeline for Amazon review helpfulness prediction.

Design principle: at prediction time the model may use ONLY information
that existed at the moment the review was posted. Anything computed from
the review's own future, or from other reviews posted later, is leakage.

This module enforces that structurally rather than by convention, so the
guarantee survives later edits.
"""

import os

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    LongType, IntegerType, BooleanType,
)

from config import (
    DATA_PATH, COLLECTION_DATE, MIN_EXPOSURE_DAYS,
    TRAIN_QUANTILE, VAL_QUANTILE, HELPFUL_THRESHOLD,
    PRODUCT_TYPE, TEXT_MIN_CHARS, RANDOM_SEED,
    SPARK_DRIVER_MEMORY, SPARK_SHUFFLE_PARTITIONS,
    MIN_REVIEWS_PER_PRODUCT_FOR_RANKING,
)


# ----------------------------------------------------------------------
# Spark
# ----------------------------------------------------------------------

def get_spark(app_name="Review Helpfulness - Leakage-Free Pipeline"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.sql.session.timeZone", "UTC")
        # The console progress bar writes carriage-return spam into notebook
        # output and makes saved logs unreadable.
        .config("spark.ui.showConsoleProgress", "false")
        # Kryo is substantially faster than Java serialisation for the
        # wide feature vectors this pipeline shuffles.
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


SCHEMA = StructType([
    StructField("asin", StringType(), True),
    StructField("helpful_vote", IntegerType(), True),
    StructField("images", StringType(), True),
    StructField("parent_asin", StringType(), True),
    StructField("rating", DoubleType(), True),
    StructField("text", StringType(), True),
    StructField("timestamp", LongType(), True),
    StructField("title", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("verified_purchase", BooleanType(), True),
    StructField("Category", StringType(), True),
])

NA_TOKENS = [
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None",
    "n/a", "nan", "null",
]


def _nullify(colname):
    return F.when(F.col(colname).isin(NA_TOKENS), None).otherwise(F.col(colname))


# ----------------------------------------------------------------------
# Stage 1 - load and clean
# ----------------------------------------------------------------------

def load_raw(spark, path=DATA_PATH):
    """
    Load the corpus, accepting either CSV or Parquet.

    Parquet is preferred for corpora built by fetch_full.py: it is columnar,
    typed, splittable and roughly an order of magnitude faster to scan than
    quoted multiline CSV. The format is detected from the path so callers do
    not have to care.

    For CSV, the schema is declared rather than inferred - inferSchema on
    quoted multiline review text is both slow and unreliable.
    """
    is_parquet = (
        path.rstrip("/").endswith(".parquet")
        or os.path.isdir(path) and any(
            f.endswith(".parquet") for f in os.listdir(path)
        )
    )

    if is_parquet:
        raw = spark.read.parquet(path)
        # Align types with the declared schema so downstream code is identical
        # whichever format was loaded.
        for field in SCHEMA.fields:
            if field.name in raw.columns:
                raw = raw.withColumn(field.name,
                                     F.col(field.name).cast(field.dataType))
        missing = [f.name for f in SCHEMA.fields if f.name not in raw.columns]
        for m in missing:
            raw = raw.withColumn(m, F.lit(None).cast(
                dict((f.name, f.dataType) for f in SCHEMA.fields)[m]))
        return raw.select(*[f.name for f in SCHEMA.fields])

    return (
        spark.read
        .options(header=True, quote='"', escape='"', multiLine=True, mode="PERMISSIVE")
        .schema(SCHEMA)
        .csv(path)
    )


def clean(df):
    """Deterministic cleaning. Every filter here is reported in the paper's
    data-preparation table, so each one returns a named count."""

    steps = []

    def record(frame, label):
        steps.append((label, frame.count()))
        return frame

    df = record(df, "raw")

    df = df.withColumn("text", _nullify("text")).withColumn("title", _nullify("title"))

    # Valid rating on the 1-5 scale
    df = df.filter((F.col("rating") >= 1) & (F.col("rating") <= 5))
    df = record(df, "valid_rating")

    # Valid timestamp
    df = df.filter(F.col("timestamp").isNotNull() & (F.col("timestamp") > 0))
    df = df.withColumn("event_time", (F.col("timestamp") / 1000).cast("timestamp"))
    df = df.filter(F.col("event_time").isNotNull())
    df = record(df, "valid_timestamp")

    # Exact duplicates. Same user, same product, same instant.
    df = df.dropDuplicates(["user_id", "asin", "timestamp"])
    df = record(df, "deduplicated")

    # Reviews need text to be modelled from text
    df = df.filter(F.col("text").isNotNull() & (F.trim(F.col("text")) != ""))
    df = df.withColumn("title", F.coalesce(F.col("title"), F.lit("")))
    df = record(df, "has_text")

    df = df.filter(F.length(F.col("text")) >= TEXT_MIN_CHARS)
    df = record(df, f"text_at_least_{TEXT_MIN_CHARS}_chars")

    # helpful_vote must be present and non-negative to serve as a target
    df = df.filter(F.col("helpful_vote").isNotNull() & (F.col("helpful_vote") >= 0))
    df = record(df, "valid_target")

    return df, steps


# ----------------------------------------------------------------------
# Stage 2 - exposure control
# ----------------------------------------------------------------------

def apply_exposure_control(df, collection_date=COLLECTION_DATE,
                           min_exposure_days=MIN_EXPOSURE_DAYS):
    """
    Drop reviews that had insufficient time to accumulate votes, and retain
    exposure duration as an explicit control.

    Without this step, a review posted one week before collection is
    labelled "not helpful" simply because nobody has seen it yet. Training
    on such labels teaches the model to detect recency, not quality.
    """
    collection = F.to_timestamp(F.lit(collection_date))

    df = df.withColumn(
        "exposure_days",
        F.datediff(collection, F.col("event_time")).cast("double"),
    )

    # Reviews dated after collection are data errors
    df = df.filter(F.col("exposure_days") >= 0)

    n_before = df.count()
    df = df.filter(F.col("exposure_days") >= float(min_exposure_days))
    n_after = df.count()

    print(
        f"[exposure control] kept {n_after:,} of {n_before:,} reviews "
        f"({100 * n_after / max(n_before, 1):.1f}%) with >= {min_exposure_days} "
        f"days of visibility before {collection_date}"
    )

    # log-exposure is the natural scale: the marginal effect of an extra day
    # of visibility shrinks as a review ages off the first page
    df = df.withColumn("log_exposure_days", F.log1p(F.col("exposure_days")))

    return df


# ----------------------------------------------------------------------
# Stage 3 - targets
# ----------------------------------------------------------------------

def add_targets(df, threshold=HELPFUL_THRESHOLD,
                min_product_reviews=MIN_REVIEWS_PER_PRODUCT_FOR_RANKING):
    """
    Primary target: binary, did the review receive any helpful vote.
    Secondary target: within-product percentile rank of helpful_vote.

    The secondary target matters because it is exposure-normalised by
    construction. Reviews competing on the same product page face broadly
    the same visibility conditions, so their relative vote counts reflect
    relative quality far more cleanly than absolute counts do.
    """
    df = df.withColumn("label", (F.col("helpful_vote") >= threshold).cast("int"))
    df = df.withColumn("log_helpful_vote", F.log1p(F.col("helpful_vote")))

    # Within-product percentile rank.
    # NOTE: this uses all reviews of the product, including later ones. It is
    # therefore valid as an EVALUATION target (we are ranking a fixed set of
    # reviews, exactly as a platform would) but must never be used as an
    # input feature.
    w_product = Window.partitionBy("parent_asin").orderBy("helpful_vote")
    df = df.withColumn("product_review_count", F.count("*").over(Window.partitionBy("parent_asin")))
    df = df.withColumn(
        "within_product_pct_rank",
        F.when(
            F.col("product_review_count") >= min_product_reviews,
            F.percent_rank().over(w_product),
        ).otherwise(F.lit(None).cast("double")),
    )

    return df


# ----------------------------------------------------------------------
# Stage 4 - product type
# ----------------------------------------------------------------------

def add_product_type(df, mapping=PRODUCT_TYPE):
    """Map category to the search/experience taxonomy. Unmapped categories
    are flagged rather than silently bucketed."""
    expr = F.lit(None).cast("string")
    for category, ptype in mapping.items():
        expr = F.when(F.col("Category") == category, F.lit(ptype)).otherwise(expr)

    df = df.withColumn("product_type", F.coalesce(expr, F.lit("unmapped")))

    unmapped = df.filter(F.col("product_type") == "unmapped").select("Category").distinct()
    rows = unmapped.collect()
    if rows:
        print("[product type] WARNING - unmapped categories:",
              [r["Category"] for r in rows])

    return df


# ----------------------------------------------------------------------
# Stage 5 - reviewer and product history, computed WITHOUT lookahead
# ----------------------------------------------------------------------

def add_prior_history_features(df):
    """
    Reviewer and product track record, using strictly earlier reviews only.

    This is the exact step that invalidated arXiv:2412.02884. That paper
    computed each reviewer's average helpful votes across ALL of their
    reviews, including the one being predicted. For a reviewer with a single
    review in the dataset, that feature IS the label, rescaled. The reported
    96.91% accuracy is a restatement of the target.

    The fix is an expanding window bounded at the row before the current one.
    A reviewer's first review correctly gets a null history, which we fill
    with an explicit "no history" indicator rather than an imputed mean.
    """

    # Strictly prior reviews by the same user
    w_user = (
        Window.partitionBy("user_id")
        .orderBy("timestamp")
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    df = (
        df
        .withColumn("prior_review_count", F.count("*").over(w_user))
        .withColumn("prior_avg_helpful", F.avg("helpful_vote").over(w_user))
        .withColumn("prior_avg_rating", F.avg("rating").over(w_user))
        .withColumn("prior_avg_length", F.avg(F.length("text")).over(w_user))
    )

    # Strictly prior reviews of the same product
    w_product = (
        Window.partitionBy("parent_asin")
        .orderBy("timestamp")
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    df = (
        df
        .withColumn("product_prior_review_count", F.count("*").over(w_product))
        .withColumn("product_prior_avg_rating", F.avg("rating").over(w_product))
    )

    # Explicit no-history indicators. Imputing a mean here would smuggle
    # population-level information into first-time reviewers.
    df = (
        df
        .withColumn("is_first_review_by_user", (F.col("prior_review_count") == 0).cast("int"))
        .withColumn("is_first_review_of_product", (F.col("product_prior_review_count") == 0).cast("int"))
        .withColumn("prior_avg_helpful", F.coalesce(F.col("prior_avg_helpful"), F.lit(0.0)))
        .withColumn("prior_avg_rating", F.coalesce(F.col("prior_avg_rating"), F.lit(0.0)))
        .withColumn("prior_avg_length", F.coalesce(F.col("prior_avg_length"), F.lit(0.0)))
        .withColumn("product_prior_avg_rating", F.coalesce(F.col("product_prior_avg_rating"), F.lit(0.0)))
        .withColumn("log_prior_review_count", F.log1p(F.col("prior_review_count")))
        .withColumn("log_product_prior_review_count", F.log1p(F.col("product_prior_review_count")))
    )

    return df


def add_leaky_history_features(df):
    """
    DELIBERATELY LEAKY. Used only to reproduce the inflated result and
    quantify how much of it is artefact.

    Computes reviewer average helpful votes over the full window, including
    the current review, exactly as the flawed prior work did. Never call
    this for the main models.
    """
    w_all = Window.partitionBy("user_id")
    df = df.withColumn("LEAKY_user_avg_helpful", F.avg("helpful_vote").over(w_all))
    df = df.withColumn("LEAKY_user_review_count", F.count("*").over(w_all))
    return df


# ----------------------------------------------------------------------
# Stage 6 - temporal split
# ----------------------------------------------------------------------

def temporal_split(df, train_q=TRAIN_QUANTILE, val_q=VAL_QUANTILE):
    """
    Split strictly by time. The model is trained on the past and evaluated
    on the future, which is the only setting matching deployment.

    Cut points are data-driven quantiles rather than fixed dates, because
    review volume grows sharply over time and fixed dates would leave the
    early split nearly empty.
    """
    cuts = df.approxQuantile("timestamp", [train_q, val_q], 0.001)
    train_cut, val_cut = cuts[0], cuts[1]

    train = df.filter(F.col("timestamp") < train_cut)
    val = df.filter((F.col("timestamp") >= train_cut) & (F.col("timestamp") < val_cut))
    test = df.filter(F.col("timestamp") >= val_cut)

    def _describe(frame, name):
        row = frame.agg(
            F.count("*").alias("n"),
            F.min("event_time").alias("from"),
            F.max("event_time").alias("to"),
            F.avg("label").alias("base_rate"),
        ).first()
        print(
            f"[split] {name:<6} n={row['n']:>10,}  "
            f"{str(row['from'])[:10]} -> {str(row['to'])[:10]}  "
            f"base rate={100 * (row['base_rate'] or 0):.2f}%"
        )
        return row

    print("\nTemporal split (no future information enters training):")
    _describe(train, "train")
    _describe(val, "val")
    _describe(test, "test")
    print(
        "\nNOTE: base rates differ across splits. This is real drift, not a bug.\n"
        "      Test-set base rate is preserved - we never resample the test set.\n"
    )

    return train, val, test


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def build_dataset(spark, path=DATA_PATH, verbose=True):
    """Run the full pipeline and return (df, train, val, test, cleaning_log)."""

    raw = load_raw(spark, path)
    df, steps = clean(raw)

    if verbose:
        print("\nCleaning log:")
        prev = None
        for label, n in steps:
            delta = "" if prev is None else f"  ({n - prev:+,})"
            print(f"  {label:<32} {n:>12,}{delta}")
            prev = n
        print()

    df = apply_exposure_control(df)
    df = add_targets(df)
    df = add_product_type(df)
    df = add_prior_history_features(df)

    df = df.cache()
    df.count()

    train, val, test = temporal_split(df)

    return df, train, val, test, steps
