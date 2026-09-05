"""
Shared pipeline utilities for the CSE 4262 Data Analytics Lab project.

Large-Scale Amazon Customer Review Analytics using PySpark
Ahsanullah University of Science and Technology, Department of CSE
Lab Group Gr-03, Group Gr-06

Every notebook in this repository imports from this module, so the schema,
the preprocessing rules, the plot style and the output paths are defined
exactly once. Import it with:

    from da_common import *
"""

from __future__ import annotations

import glob
import os
import time

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (StructType, StructField, StringType, DoubleType,
                               LongType, IntegerType, BooleanType)

__all__ = [
    "F", "Window", "SparkSession", "StringType", "np", "pd", "os", "time", "glob",
    "plt", "mticker",
    "SCHEMA", "CSV_OPTS", "NA_TOKENS", "CATS", "CAT_COLORS", "STAR_COLORS", "ACCENT",
    "MIN_REVIEWS", "HIGHLY_HELPFUL_THRESHOLD", "ROLLING_WINDOW",
    "DATA_PATH", "OUT_DIR", "FIG_DIR", "TBL_DIR", "PROC_DIR", "CLEAN_PARQUET",
    "RESULTS", "FIGURES",
    "get_spark", "read_raw", "ccol", "savefig", "save_table", "load_table",
    "preprocess", "engineer", "build_analytical", "load_analytical", "banner",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SEARCH_ROOTS = ["/kaggle/input", "/kaggle/working/data", "../data", "./data",
                 "/mnt/user-data/uploads", ".."]


def _find_dataset() -> str:
    hits = []
    for base in _SEARCH_ROOTS:
        if os.path.isdir(base):
            hits += glob.glob(os.path.join(base, "**", "*.csv"), recursive=True)
    hits = [h for h in hits if "/results" not in h and "/figures" not in h]
    if not hits:
        raise FileNotFoundError(
            "Amazon_Reviews.csv was not found. Attach the dataset on Kaggle, or place "
            "the file in a 'data/' folder next to this repository."
        )
    preferred = [h for h in hits
                 if any(k in os.path.basename(h).lower() for k in ("amazon", "review"))]
    return sorted(preferred or hits, key=os.path.getsize, reverse=True)[0]


DATA_PATH = _find_dataset()
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("./output")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TBL_DIR = os.path.join(OUT_DIR, "results")
PROC_DIR = os.path.join(OUT_DIR, "processed")
CLEAN_PARQUET = os.path.join(PROC_DIR, "reviews_analytical.parquet")

for _d in (FIG_DIR, TBL_DIR, PROC_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset contract
# ---------------------------------------------------------------------------

SCHEMA = StructType([
    StructField("asin",              StringType(),  True),
    StructField("helpful_vote",      IntegerType(), True),
    StructField("images",            StringType(),  True),
    StructField("parent_asin",       StringType(),  True),
    StructField("rating",            DoubleType(),  True),
    StructField("text",              StringType(),  True),
    StructField("timestamp",         LongType(),    True),
    StructField("title",             StringType(),  True),
    StructField("user_id",           StringType(),  True),
    StructField("verified_purchase", BooleanType(), True),
    StructField("Category",          StringType(),  True),
])

CSV_OPTS = dict(header=True, quote='"', escape='"', multiLine=True, mode="PERMISSIVE")

NA_TOKENS = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
             '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None',
             'n/a', 'nan', 'null']

CATS = ["Cell Phones", "Video Games", "Beauty"]

MIN_REVIEWS = 50
HIGHLY_HELPFUL_THRESHOLD = 10
ROLLING_WINDOW = 5

# ---------------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------------


def get_spark(app_name: str = "Amazon Review Analytics") -> SparkSession:
    spark = (SparkSession.builder
             .appName(app_name)
             .master("local[*]")
             .config("spark.driver.memory", "8g")
             .config("spark.driver.maxResultSize", "2g")
             .config("spark.sql.shuffle.partitions", "16")
             .config("spark.sql.session.timeZone", "UTC")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def read_raw(spark: SparkSession, tag: str | None = None):
    r = spark.read.options(**CSV_OPTS).schema(SCHEMA).csv(DATA_PATH)
    return r.withColumn("_src", F.lit(tag)) if tag else r


# ---------------------------------------------------------------------------
# Plot style and output savers
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.ticker as mticker      # noqa: E402

plt.rcParams.update({
    "figure.figsize": (9, 5),
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

CAT_COLORS = {"Cell Phones": "#2563eb", "Video Games": "#16a34a", "Beauty": "#db2777"}
STAR_COLORS = ["#ef4444", "#f97316", "#eab308", "#84cc16", "#16a34a"]
ACCENT = "#2563eb"

RESULTS: dict[str, pd.DataFrame] = {}
FIGURES: dict[str, str] = {}


def ccol(category: str) -> str:
    return CAT_COLORS.get(category, "#64748b")


def savefig(fig, name: str, caption: str = "") -> str:
    path = os.path.join(FIG_DIR, name + ".png")
    fig.savefig(path)
    FIGURES[name] = caption or name
    print("figure saved ->", path)
    return path


def save_table(obj, name: str) -> pd.DataFrame:
    pdf = obj.toPandas() if hasattr(obj, "toPandas") else pd.DataFrame(obj).copy()
    pdf.to_csv(os.path.join(TBL_DIR, name + ".csv"), index=False)
    try:
        pdf.to_parquet(os.path.join(TBL_DIR, name + ".parquet"), index=False)
    except Exception as e:
        print("   parquet skipped:", type(e).__name__)
    RESULTS[name] = pdf
    print(f"table saved  -> {name}  ({len(pdf)} rows)")
    return pdf


def load_table(name: str) -> pd.DataFrame | None:
    """Read a table written by an earlier notebook. Returns None if absent."""
    path = os.path.join(TBL_DIR, name + ".csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def banner(title: str) -> None:
    print("=" * 74)
    print(title)
    print("=" * 74)
    print(f"dataset : {DATA_PATH}")
    print(f"outputs : {OUT_DIR}")


# ---------------------------------------------------------------------------
# Preprocessing and feature engineering
# ---------------------------------------------------------------------------


def preprocess(raw, verbose: bool = True):
    """Four cleaning steps with a row count logged after each one."""
    log = []

    def step(name, frame):
        n = frame.count()
        log.append((name, n))
        if verbose:
            print(f"{name:<38} rows = {n:,}")
        return frame

    def norm(col):
        c = F.col(col)
        return F.when(c.isin(NA_TOKENS), None).otherwise(c)

    df = raw
    step("0. raw as loaded", df)

    df = df.withColumn("text", norm("text")).withColumn("title", norm("title"))
    step("1. sentinels converted to null", df)

    df = (df.withColumn("event_time", (F.col("timestamp") / 1000).cast("timestamp"))
            .withColumn("review_year", F.year("event_time"))
            .withColumn("review_month", F.month("event_time")))
    step("2. timestamp parsed", df)

    df = df.filter((F.col("rating") >= 1) & (F.col("rating") <= 5))
    step("3. rating range-checked", df)

    df = df.dropDuplicates(["user_id", "asin", "timestamp"])
    step("4a. duplicates removed", df)

    df = (df.filter(F.col("text").isNotNull())
            .withColumn("title", F.coalesce(F.col("title"), F.lit(""))))
    step("4b. empty review bodies dropped", df)

    audit = pd.DataFrame(log, columns=["step", "rows"])
    audit["removed"] = (audit["rows"].shift(1) - audit["rows"]).fillna(0).astype(int)
    audit["retained_pct"] = (100 * audit["rows"] / audit["rows"].iloc[0]).round(3)
    return df, audit


def engineer(df):
    """Derived columns plus product-level aggregates joined back for reuse."""
    df = (df
          .withColumn("review_length", F.length("text"))
          .withColumn("review_word_count",
                      F.when(F.trim("text") == "", 0)
                       .otherwise(F.size(F.split(F.trim("text"), r"\s+"))))
          .withColumn("log_helpful_vote", F.round(F.log1p(F.col("helpful_vote")), 4))
          .withColumn("is_highly_helpful",
                      (F.col("helpful_vote") >= HIGHLY_HELPFUL_THRESHOLD).cast("int")))

    prod = (df.groupBy("parent_asin")
              .agg(F.round(F.avg("rating"), 3).alias("product_avg_rating"),
                   F.count("*").alias("product_review_count")))
    return df.join(prod, on="parent_asin", how="left")


def build_analytical(spark, write: bool = True):
    """Run the full cleaning and feature pipeline from the raw CSV."""
    df, audit = preprocess(read_raw(spark))
    df = engineer(df)
    if write:
        (df.drop("images").write.mode("overwrite").parquet(CLEAN_PARQUET))
        print("analytical dataset written ->", CLEAN_PARQUET)
    return df, audit


def load_analytical(spark, rebuild: bool = False):
    """
    Read the cleaned dataset produced by notebook 02. If it is not present
    (for example when a notebook is run on its own), rebuild it from the raw
    CSV so every notebook stays independently runnable.
    """
    candidates = [CLEAN_PARQUET] + sorted(
        glob.glob("/kaggle/input/**/reviews_analytical.parquet", recursive=True))
    if not rebuild:
        for path in candidates:
            if os.path.exists(path):
                print("loaded cleaned dataset <-", path)
                return spark.read.parquet(path)
    print("cleaned dataset not found, rebuilding from the raw CSV...")
    df, _ = build_analytical(spark, write=True)
    return df
