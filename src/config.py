"""
Central configuration.

Every methodological choice that a reviewer might question lives here,
in one place, with a stated justification. Do not scatter magic numbers
through the pipeline.
"""

import os

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

DATA_PATH = os.environ.get(
    "AMAZON_DATA_PATH",
    "/root/.cache/kagglehub/datasets/tahsintajware/"
    "amazon-reviews-2023-three-category-extract/versions/1/Amazon_Reviews.csv",
)

OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("./output")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TBL_DIR = os.path.join(OUT_DIR, "results")
MODEL_DIR = os.path.join(OUT_DIR, "models")

for _d in (FIG_DIR, TBL_DIR, MODEL_DIR):
    os.makedirs(_d, exist_ok=True)


# ----------------------------------------------------------------------
# Exposure control  --  THE CENTRAL METHODOLOGICAL CHOICE
# ----------------------------------------------------------------------
#
# helpful_vote is a CUMULATIVE count with no upper bound and no denominator.
# Amazon does not publish how many people saw a review, only how many
# clicked "helpful". A review therefore accumulates votes for as long as
# it remains visible.
#
# Consequence: review age is mechanically correlated with helpful_vote.
# Any model given a time feature will learn "older => more votes" and
# report an inflated score while having learned nothing about review
# quality. This is the failure mode in arXiv:2412.02884 on this dataset.
#
# We control exposure two ways:
#   1. CENSORING  - drop reviews that had less than MIN_EXPOSURE_DAYS of
#      visibility before data collection, so every retained review has had
#      a comparable minimum opportunity to accumulate votes.
#   2. CONDITIONING - retain review age as an explicit control variable so
#      residual exposure effects are absorbed rather than attributed to
#      text quality.
#
# The dataset was collected in September 2023.

COLLECTION_DATE = "2023-09-01"
MIN_EXPOSURE_DAYS = 365   # every retained review had >= 1 year of visibility


# ----------------------------------------------------------------------
# Temporal splitting
# ----------------------------------------------------------------------
#
# Random splitting leaks: a model can see a user's later reviews while
# predicting their earlier ones, and can see the future state of a product
# page. We split strictly by time. Quantile-based rather than hard-coded
# dates, because review volume is heavily skewed toward recent years and
# fixed dates would produce degenerate splits.

TRAIN_QUANTILE = 0.70   # earliest 70% of reviews by timestamp
VAL_QUANTILE = 0.85     # next 15%
                        # final 15% is the test set


# ----------------------------------------------------------------------
# Target definition
# ----------------------------------------------------------------------
#
# PRIMARY: did the review receive any helpful vote at all.
# This is the standard binarisation in the literature and keeps the
# positive class large enough to model. Base rate is reported per split
# and per category, never assumed.
#
# SECONDARY: within-product percentile rank of helpful_vote. This is the
# deployment-realistic target - the platform's actual decision is how to
# ORDER reviews on one product page, not to classify them in isolation.
# It is also naturally exposure-normalised, since competing reviews on the
# same page share the same visibility conditions.

HELPFUL_THRESHOLD = 1

# Products need a minimum number of reviews for within-product ranking
# to be meaningful.
MIN_REVIEWS_PER_PRODUCT_FOR_RANKING = 5


# ----------------------------------------------------------------------
# Product type taxonomy  --  the moderation hypothesis
# ----------------------------------------------------------------------
#
# Nelson's (1970) search/experience distinction, as applied to review
# helpfulness by Mudambi & Schuff (2010, MIS Quarterly):
#
#   SEARCH goods    - quality is verifiable before purchase from published
#                     attributes (specifications, dimensions, compatibility).
#                     Prediction: depth and technical detail drive helpfulness;
#                     moderate ratings are more helpful than extreme ones.
#
#   EXPERIENCE goods - quality cannot be assessed until consumed. Fit,
#                     scent, feel, enjoyment.
#                     Prediction: personal narrative and extremity drive
#                     helpfulness; the depth effect is weaker.
#
# Assigning a whole category to one pole is a simplification and must be
# acknowledged as a limitation. Cell Phones and Accessories is
# predominantly utilitarian/search; Beauty and Video Games are
# predominantly hedonic/experience.

PRODUCT_TYPE = {
    "Cell Phones": "search",
    "Cell_Phones_and_Accessories": "search",
    "Electronics": "search",
    "Office Products": "search",
    "Tools and Home Improvement": "search",
    "Beauty": "experience",
    "All_Beauty": "experience",
    "Video Games": "experience",
    "Video_Games": "experience",
    "Movies and TV": "experience",
    "Books": "experience",
}


# ----------------------------------------------------------------------
# Modelling
# ----------------------------------------------------------------------

RANDOM_SEED = 42

TEXT_MIN_CHARS = 20          # below this, text features are noise
HASHING_FEATURES = 2 ** 15   # TF-IDF dimensionality

# Sample fraction for the ML stage. 1.0 uses everything.
# Reduce only if memory-constrained; record whatever you used.
ML_SAMPLE_FRACTION = 1.0


# ----------------------------------------------------------------------
# Spark
# ----------------------------------------------------------------------

SPARK_DRIVER_MEMORY = "8g"
SPARK_SHUFFLE_PARTITIONS = "32"
