"""
Regression on helpful-vote counts.

COURSE MAPPING: Week 3 — MLlib regression.

WHY THIS IS REPORTED WITH A CAVEAT RATHER THAN AS A HEADLINE
------------------------------------------------------------
Classification asks whether a review gets *any* vote. Regression asks *how
many*, which is the more demanding question and the more useful one if it
works.

It mostly does not work, for a reason worth stating plainly: the target is
severely zero-inflated. Roughly 83% of reviews receive no votes at all, so
the conditional mean is dominated by zeros and R-squared is correspondingly
low. A model can minimise squared error very effectively by predicting
something close to zero for everything, which is useless.

We therefore:
  * model log1p(helpful_vote) rather than the raw count, which compresses a
    distribution whose maximum is in the thousands;
  * report R-squared honestly, including when it is poor;
  * additionally report metrics on the NON-ZERO subset, which separates "the
    model cannot tell who gets votes" from "the model cannot tell how many
    votes someone gets, given that they get some".

That second split is the informative one, and reporting only the pooled
figure would hide it.
"""

import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.ml.regression import (
    LinearRegression, RandomForestRegressor, GBTRegressor,
)
from pyspark.ml.evaluation import RegressionEvaluator

from config import RANDOM_SEED
from features import ALL_NUMERIC
from models import build_feature_pipeline


TARGET = "log_helpful_vote"


def get_regressors(label_col=TARGET):
    return [
        ("Linear Regression", LinearRegression(
            featuresCol="features", labelCol=label_col,
            maxIter=50, regParam=0.01, elasticNetParam=0.0,
        )),
        ("Ridge Regression", LinearRegression(
            featuresCol="features", labelCol=label_col,
            maxIter=50, regParam=0.1, elasticNetParam=0.0,
        )),
        ("Random Forest Regressor", RandomForestRegressor(
            featuresCol="features", labelCol=label_col,
            numTrees=100, maxDepth=10, minInstancesPerNode=20,
            subsamplingRate=0.8, maxBins=32, seed=RANDOM_SEED,
        )),
        ("GBT Regressor", GBTRegressor(
            featuresCol="features", labelCol=label_col,
            maxIter=60, maxDepth=6, stepSize=0.1,
            subsamplingRate=0.8, maxBins=32, seed=RANDOM_SEED,
        )),
    ]


def _metrics(preds, label_col=TARGET, suffix=""):
    out = {}
    for name, metric in [("rmse", "rmse"), ("mae", "mae"), ("r2", "r2")]:
        try:
            out[name + suffix] = round(float(
                RegressionEvaluator(labelCol=label_col, predictionCol="prediction",
                                    metricName=metric).evaluate(preds)
            ), 4)
        except Exception:
            out[name + suffix] = None
    return out


def baseline_rows(test, label_col=TARGET):
    """Predicting the training mean for every review. Any model that cannot
    beat this has learned nothing."""
    stats = test.agg(F.avg(label_col).alias("mu")).first()
    mu = float(stats["mu"] or 0.0)

    preds = test.withColumn("prediction", F.lit(mu))
    row = {"model": "Baseline: predict mean", "feature_space": "none"}
    row.update(_metrics(preds, label_col))
    return [row]


def run(train, test, label_col=TARGET, use_text=False):
    """Entry point. Returns the regression comparison table."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 9 - REGRESSION ON HELPFUL-VOTE COUNTS")
    print("=" * 70)

    zero_share = test.filter(F.col("helpful_vote") == 0).count() / max(test.count(), 1)
    print(f"Target: log1p(helpful_vote).  {100 * zero_share:.1f}% of test "
          f"reviews have zero votes.")
    print("Zero-inflation caps achievable R-squared; metrics on the non-zero")
    print("subset are reported alongside, and are the more informative figure.\n")

    numeric = [c for c in ALL_NUMERIC if c in train.columns]

    pipe, _ = build_feature_pipeline(numeric, use_text=use_text, use_category=True)
    fitted = pipe.fit(train)

    tr = fitted.transform(train).select("features", label_col).cache()
    te = fitted.transform(test).select(
        "features", label_col, "helpful_vote").cache()
    tr.count(); te.count()

    te_nonzero = te.filter(F.col("helpful_vote") > 0).cache()
    n_nonzero = te_nonzero.count()

    rows = baseline_rows(te, label_col)
    for r in rows:
        print(f"   {r['model']:<28} RMSE={r['rmse']}  R2={r['r2']}")

    for name, reg in get_regressors(label_col):
        print(f"\n-> {name}")
        try:
            model = reg.fit(tr)

            row = {"model": name,
                   "feature_space": "numeric + category" + (" + TF-IDF" if use_text else "")}
            row.update(_metrics(model.transform(te), label_col))

            if n_nonzero > 100:
                row.update(_metrics(model.transform(te_nonzero), label_col,
                                    suffix="_nonzero"))

            rows.append(row)
            print(f"   RMSE={row['rmse']}  MAE={row['mae']}  R2={row['r2']}"
                  + (f"   |  non-zero subset: RMSE={row.get('rmse_nonzero')}  "
                     f"R2={row.get('r2_nonzero')}" if n_nonzero > 100 else ""))
        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {e}")

    tr.unpersist(); te.unpersist(); te_nonzero.unpersist()

    table = pd.DataFrame(rows)

    try:
        best = table[table["model"] != "Baseline: predict mean"].sort_values(
            "r2", ascending=False).iloc[0]
        base_r2 = table[table["model"] == "Baseline: predict mean"].iloc[0]["r2"]
        print(f"\n>>> Best: {best['model']}, R2 = {best['r2']} "
              f"(baseline {base_r2}).")
        if best["r2"] is not None and best["r2"] < 0.2:
            print(">>> Low R-squared is the expected result for a zero-inflated")
            print(">>> count target and is reported as such. The classification")
            print(">>> framing in Experiment 3 is the better-posed question;")
            print(">>> this experiment documents why.")
    except Exception:
        pass

    return table
