"""
Build a structure-preserving corpus from the complete Amazon Reviews 2023
category dumps.

THE PROBLEM THIS SOLVES
-----------------------
The Kaggle extract used previously is a REVIEW-LEVEL random sample: reviews
were drawn independently, so 97.4% of reviewers appear exactly once and the
mean is 1.029 reviews per reviewer. That is not a fact about Amazon; it is
arithmetic. Sampling reviews at rate p leaves a user with k reviews appearing
twice with probability 1-(1-p)^k-kp(1-p)^(k-1), which for p=0.02, k=10 is
about 1.6%.

Consequence: reviewer history, reviewer segmentation and the reviewer-product
graph are all untestable on such a file, because the variable they need is
near-constant by construction.

THE FIX: SAMPLE USERS, NOT REVIEWS
----------------------------------
If a category is too large to process whole, we select a random set of USERS
and keep EVERY review those users wrote. Each retained reviewer's history is
then complete, and reviews-per-reviewer matches the source category rather
than collapsing toward 1.

This is the difference between a sample that is smaller and one that is
broken. Review-level sampling changes the structure of the data; user-level
sampling changes only its size.

Where a category fits whole, no sampling is applied at all.

USAGE (Kaggle / Colab, internet enabled)
----------------------------------------
    import fetch_full
    path = fetch_full.build(
        categories={"Cell_Phones_and_Accessories": "Cell Phones",
                    "Video_Games": "Video Games",
                    "All_Beauty": "Beauty"},
        max_reviews_per_category=6_000_000,
    )
    os.environ["AMAZON_DATA_PATH"] = path
"""

import os
import sys
import gzip
import json
import shutil
import time
import urllib.request

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    LongType, IntegerType, BooleanType,
)

from config import RANDOM_SEED


# Primary source: UCSD McAuley Lab. Mirrored on HuggingFace.
UCSD_BASE = ("https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
             "raw/review_categories")
HF_BASE = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
           "resolve/main/raw/review_categories")

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("./output")
RAW_DIR = os.path.join(WORK, "raw_categories")
OUT_DIR = os.path.join(WORK, "corpus")

# Known review counts, for planning and for reporting coverage.
CATEGORY_SIZES = {
    "Cell_Phones_and_Accessories": 20_800_000,
    "Beauty_and_Personal_Care": 23_900_000,
    "Video_Games": 4_600_000,
    "All_Beauty": 701_500,
    "Electronics": 43_900_000,
    "Office_Products": 5_200_000,
    "Movies_and_TV": 17_300_000,
    "Books": 29_500_000,
}

# The schema the rest of the pipeline expects.
RAW_SCHEMA = StructType([
    StructField("rating", DoubleType(), True),
    StructField("title", StringType(), True),
    StructField("text", StringType(), True),
    StructField("images", StringType(), True),
    StructField("asin", StringType(), True),
    StructField("parent_asin", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("timestamp", LongType(), True),
    StructField("helpful_vote", IntegerType(), True),
    StructField("verified_purchase", BooleanType(), True),
])


# ----------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------

def _download(url, dest, chunk=1 << 20):
    """Stream to disk with progress. Returns bytes written, or None."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "research-pipeline"})
        with urllib.request.urlopen(req, timeout=60) as r:
            total = int(r.headers.get("Content-Length") or 0)
            written = 0
            t0 = time.time()
            with open(dest, "wb") as fh:
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    fh.write(buf)
                    written += len(buf)
                    if total and written % (32 << 20) < chunk:
                        pct = 100 * written / total
                        rate = written / max(time.time() - t0, 1e-9) / 1e6
                        print(f"\r    {pct:5.1f}%  {written/1e9:.2f} GB  "
                              f"{rate:.0f} MB/s", end="", flush=True)
            print(f"\r    done: {written/1e9:.2f} GB"
                  f"{' ' * 30}")
            return written
    except Exception as e:
        print(f"\r    failed: {type(e).__name__}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return None


def fetch_category(category, force=False):
    """Download one category's review dump. Tries UCSD, then HuggingFace."""
    os.makedirs(RAW_DIR, exist_ok=True)
    dest = os.path.join(RAW_DIR, f"{category}.jsonl.gz")

    if os.path.exists(dest) and not force and os.path.getsize(dest) > 1000:
        print(f"  {category}: already present "
              f"({os.path.getsize(dest)/1e9:.2f} GB)")
        return dest

    expected = CATEGORY_SIZES.get(category)
    hint = f"  (~{expected/1e6:.1f}M reviews)" if expected else ""
    print(f"  {category}{hint}")

    for base, name in [(UCSD_BASE, "UCSD"), (HF_BASE, "HuggingFace")]:
        print(f"    trying {name}...")
        if _download(f"{base}/{category}.jsonl.gz", dest) is not None:
            return dest

    raise RuntimeError(
        f"Could not download {category} from either source. Check that "
        f"internet is enabled in the notebook settings."
    )


# ----------------------------------------------------------------------
# Load and structure-preserving subsample
# ----------------------------------------------------------------------

def load_category(spark, path, category_label, max_reviews=None,
                  seed=RANDOM_SEED):
    """
    Read one category and, if it exceeds max_reviews, subsample BY USER.

    Spark reads .jsonl.gz natively. gzip is not splittable, so the initial
    read is single-threaded; we repartition immediately so everything after
    it parallelises.
    """
    df = spark.read.schema(RAW_SCHEMA).json(path)

    # images arrives as an array; the pipeline expects a string column and
    # only ever asks whether it is empty.
    if "images" in df.columns:
        df = df.withColumn("images", F.col("images").cast("string"))

    df = df.withColumn("Category", F.lit(category_label))
    df = df.repartition(64)

    n = df.count()
    print(f"    {n:,} reviews in source")

    if max_reviews is None or n <= max_reviews:
        print(f"    using the complete category (no sampling)")
        return df, {"sampled": False, "source_reviews": n, "kept_reviews": n}

    # --- USER-LEVEL sampling ---
    #
    # Draw a random set of users, then keep every review they wrote. The
    # retained reviewers have complete histories, which is the whole point.
    frac = max_reviews / n
    users = df.select("user_id").distinct()
    n_users = users.count()

    keep = users.sample(False, min(1.0, frac * 1.05), seed=seed)
    out = df.join(F.broadcast(keep) if keep.count() < 1_000_000 else keep,
                  "user_id", "inner")

    kept = out.count()
    kept_users = keep.count()
    print(f"    USER-LEVEL sample: kept {kept_users:,} of {n_users:,} users "
          f"({kept:,} reviews)")
    print(f"    every retained reviewer keeps their FULL history")

    return out, {"sampled": True, "sample_unit": "user",
                 "source_reviews": n, "kept_reviews": kept,
                 "source_users": n_users, "kept_users": kept_users}


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------

DEFAULT_CATEGORIES = {
    "Cell_Phones_and_Accessories": "Cell Phones",
    "Video_Games": "Video Games",
    "All_Beauty": "Beauty",
}


def build(spark=None, categories=None, max_reviews_per_category=6_000_000,
          out_format="parquet", seed=RANDOM_SEED):
    """
    Download, combine and write a structure-preserving corpus.

    Returns the path to set as AMAZON_DATA_PATH.
    """
    categories = categories or DEFAULT_CATEGORIES

    if spark is None:
        import data_pipeline as dp
        spark = dp.get_spark("Corpus builder")

    print("=" * 70)
    print("BUILDING STRUCTURE-PRESERVING CORPUS")
    print("=" * 70)
    print("Downloading complete category dumps. Where a category exceeds the")
    print("size budget it is subsampled BY USER, never by review, so every")
    print("retained reviewer keeps their entire history.\n")

    print("Downloads:")
    paths = {}
    for cat in categories:
        paths[cat] = fetch_category(cat)

    print("\nLoading:")
    frames, stats = [], []
    for cat, label in categories.items():
        print(f"  {cat} -> '{label}'")
        df, meta = load_category(spark, paths[cat], label,
                                 max_reviews_per_category, seed)
        meta["category"] = cat
        meta["label"] = label
        stats.append(meta)
        frames.append(df)

    combined = frames[0]
    for f in frames[1:]:
        combined = combined.unionByName(f, allowMissingColumns=True)

    combined = combined.select(
        "asin", "helpful_vote", "images", "parent_asin", "rating",
        "text", "timestamp", "title", "user_id", "verified_purchase",
        "Category",
    ).cache()

    n = combined.count()
    n_users = combined.select("user_id").distinct().count()
    rpu = n / max(n_users, 1)

    print(f"\nCombined corpus: {n:,} reviews, {n_users:,} reviewers")
    print(f"reviews per reviewer: {rpu:.3f}")

    if rpu >= 1.15:
        print("\n>>> User structure is INTACT. Reviewer history, segmentation")
        print(">>> and the reviewer-product graph are now testable.")
    else:
        print("\n>>> WARNING: reviews per reviewer is still near 1. Either the")
        print(">>> categories genuinely have little repeat reviewing, or the")
        print(">>> size budget forced too aggressive a user sample. Raise")
        print(">>> max_reviews_per_category and rebuild.")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "reviews.parquet" if out_format == "parquet"
                       else "reviews.csv")
    shutil.rmtree(out, ignore_errors=True)

    print(f"\nWriting {out} ...")
    if out_format == "parquet":
        combined.write.mode("overwrite").parquet(out)
    else:
        combined.write.mode("overwrite").option("header", True).csv(out)

    import pandas as pd
    stats_df = pd.DataFrame(stats)
    stats_path = os.path.join(OUT_DIR, "corpus_manifest.csv")
    stats_df.to_csv(stats_path, index=False)

    print(f"Wrote manifest -> {stats_path}")
    print(stats_df.to_string(index=False))
    print(f"\nSet AMAZON_DATA_PATH = {out}")

    combined.unpersist()
    return out
