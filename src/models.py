"""
Models, ablations, and the leakage demonstration.

The headline experiment is experiment_leakage(), which reproduces the
inflated result from prior work and then removes each source of inflation
one at a time. That decomposition is the paper's first contribution: it
shows how much of the published performance was artefact and which specific
design choice produced it.
"""

import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler,
    RegexTokenizer, StopWordsRemover, HashingTF, IDF,
)
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier,
)

from config import RANDOM_SEED, HASHING_FEATURES
from features import BLOCKS, ALL_NUMERIC
from evaluation import evaluate, results_table


# ----------------------------------------------------------------------
# Feature pipeline construction
# ----------------------------------------------------------------------

def build_feature_pipeline(numeric_cols, use_text=False, use_category=True,
                           scale=True, text_col="text"):
    """Assemble a feature pipeline from the requested components.

    Returns (Pipeline, output_column_name).
    """
    stages = []
    assembler_inputs = []

    if numeric_cols:
        stages.append(VectorAssembler(
            inputCols=numeric_cols,
            outputCol="_numeric_raw",
            handleInvalid="keep",
        ))
        if scale:
            stages.append(StandardScaler(
                inputCol="_numeric_raw",
                outputCol="_numeric",
                withMean=True,
                withStd=True,
            ))
            assembler_inputs.append("_numeric")
        else:
            assembler_inputs.append("_numeric_raw")

    if use_category:
        stages.append(StringIndexer(
            inputCol="Category",
            outputCol="_cat_idx",
            handleInvalid="keep",
        ))
        stages.append(OneHotEncoder(inputCol="_cat_idx", outputCol="_cat_vec"))
        assembler_inputs.append("_cat_vec")

    if use_text:
        stages += [
            RegexTokenizer(inputCol=text_col, outputCol="_words",
                           pattern=r"\W+", minTokenLength=2, toLowercase=True),
            StopWordsRemover(inputCol="_words", outputCol="_clean_words"),
            HashingTF(inputCol="_clean_words", outputCol="_tf",
                      numFeatures=HASHING_FEATURES),
            IDF(inputCol="_tf", outputCol="_tfidf", minDocFreq=10),
        ]
        assembler_inputs.append("_tfidf")

    stages.append(VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features",
        handleInvalid="keep",
    ))

    return Pipeline(stages=stages), "features"


def add_class_weights(train_df, label_col="label"):
    """Inverse-frequency weights, computed on TRAINING data only.

    Weighting is preferred to resampling because it leaves the data
    distribution intact; nothing is discarded and nothing is duplicated.
    """
    counts = train_df.groupBy(label_col).count().collect()
    counts = {int(r[label_col]): int(r["count"]) for r in counts}
    n_pos = counts.get(1, 0)
    n_neg = counts.get(0, 0)

    if n_pos == 0:
        weight = 1.0
    else:
        weight = n_neg / n_pos

    print(f"[weights] negatives={n_neg:,} positives={n_pos:,} "
          f"positive weight={weight:.2f}")

    return train_df.withColumn(
        "weight",
        F.when(F.col(label_col) == 1, float(weight)).otherwise(1.0),
    )


# ----------------------------------------------------------------------
# Model zoo
# ----------------------------------------------------------------------

def get_linear_classifiers(label_col="label", use_weights=True):
    """Linear models. These handle high-dimensional sparse TF-IDF natively,
    so they are trained on the full feature space including text."""

    weight_kw = {"weightCol": "weight"} if use_weights else {}

    return [
        ("Logistic Regression (L2) + text", LogisticRegression(
            featuresCol="features", labelCol=label_col,
            maxIter=50, regParam=0.01, elasticNetParam=0.0,
            **weight_kw,
        )),
        ("Logistic Regression (L1) + text", LogisticRegression(
            featuresCol="features", labelCol=label_col,
            maxIter=50, regParam=0.01, elasticNetParam=1.0,
            **weight_kw,
        )),
    ]


def get_tree_classifiers(label_col="label", use_weights=True):
    """
    Tree ensembles, trained on DENSE NUMERIC FEATURES ONLY.

    This is not a shortcut, it is a requirement of the algorithm. Spark's
    tree learners discretise every feature into bins and build per-feature
    statistics at every node. On a 32,768-dimensional hashed TF-IDF vector
    that allocates on the order of features x bins x nodes doubles per
    level, which exhausts driver memory and kills the JVM outright (we hit
    exactly this). Sparse high-dimensional text is a job for linear models;
    trees are for the low-dimensional engineered features, where they can
    capture the interactions a linear model cannot.

    Reporting the two families on different feature spaces is therefore the
    correct comparison, and the results table records which space each model
    used so no reader is misled.
    """
    weight_kw = {"weightCol": "weight"} if use_weights else {}

    return [
        ("Random Forest (numeric)", RandomForestClassifier(
            featuresCol="features", labelCol=label_col,
            numTrees=100, maxDepth=10, minInstancesPerNode=20,
            subsamplingRate=0.8, featureSubsetStrategy="sqrt",
            maxBins=32, seed=RANDOM_SEED,
            **weight_kw,
        )),
        # GBTClassifier in Spark ML does not accept weightCol, so it trains
        # unweighted. The threshold sweep in evaluation.best_f1 compensates,
        # and PR-AUC is threshold-free so the comparison stays fair.
        ("Gradient Boosted Trees (numeric)", GBTClassifier(
            featuresCol="features", labelCol=label_col,
            maxIter=60, maxDepth=6, stepSize=0.1,
            subsamplingRate=0.8, maxBins=32, seed=RANDOM_SEED,
        )),
    ]


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------

def baseline_results(test_df, label_col="label"):
    """Trivial baselines every model must beat to be worth reporting.

    Omitting these is how "our model achieved 0.87 AUC" gets published
    without anyone noticing that review length alone achieves 0.85.
    """
    from evaluation import precision_recall_auc, roc_auc, lift_at_k

    pdf = test_df.select(
        F.col(label_col).cast("int").alias("y"),
        F.col("review_length").alias("length"),
        F.col("rating").alias("rating"),
        F.col("log_exposure_days").alias("exposure"),
    ).toPandas()

    y = pdf["y"].values
    base = float(y.mean())

    rows = [{
        "model": "Trivial (predict majority)",
        "n_test": len(y), "base_rate": round(base, 4),
        "pr_auc": round(base, 4), "pr_auc_lift": 1.0,
        "roc_auc": 0.5, "lift@10pct": 1.0,
    }]

    for name, col in [
        ("Baseline: review length only", "length"),
        ("Baseline: exposure days only", "exposure"),
        ("Baseline: rating only", "rating"),
    ]:
        p = pdf[col].values.astype(float)
        pr = precision_recall_auc(y, p)
        rows.append({
            "model": name,
            "n_test": len(y),
            "base_rate": round(base, 4),
            "pr_auc": round(pr, 4),
            "pr_auc_lift": round(pr / base, 3) if base > 0 else np.nan,
            "roc_auc": round(roc_auc(y, p), 4),
            "lift@10pct": round(lift_at_k(y, p, 0.10), 3),
        })

    return rows


# ----------------------------------------------------------------------
# Experiment 1 - the leakage decomposition  (HEADLINE RESULT)
# ----------------------------------------------------------------------

def experiment_leakage(spark, full_df, train, val, test, label_col="label"):
    """
    Reproduce the inflated result, then remove each source of inflation.

    Four conditions, each removing one flaw:

      A. REPLICATION      random split + leaky reviewer feature + no exposure
                          control. This is prior work's setup.
      B. FIX SPLIT        temporal split, leaky feature retained.
      C. FIX FEATURE      random split, honest prior-only reviewer feature.
      D. FIX BOTH         temporal split + honest features. Our protocol.

    The gap between A and D is the size of the artefact. Attributing it
    across B and C shows which flaw did the damage.
    """
    from data_pipeline import add_leaky_history_features

    print("\n" + "=" * 70)
    print("EXPERIMENT 1 - LEAKAGE DECOMPOSITION")
    print("=" * 70)

    leaky_cols = ["LEAKY_user_avg_helpful", "LEAKY_user_review_count"]
    honest_cols = ["prior_avg_helpful", "log_prior_review_count"]
    shared_cols = ["review_length", "rating", "verified_int", "has_images"]

    # CRITICAL: the leaky feature is computed ONCE over the whole corpus,
    # before any split, which is exactly what prior work did. Computing it
    # separately inside each split would make the test-set feature an average
    # over that user's test reviews alone - usually a single review, so the
    # feature would BE the label. That would inflate the temporal conditions
    # for a reason that has nothing to do with temporal splitting, and would
    # misattribute the artefact. One shared frame keeps the four conditions
    # differing only in the factors under study.
    df_leaky = add_leaky_history_features(full_df).cache()
    df_leaky.count()

    # Random split, for conditions A and C
    rnd_train, rnd_test = df_leaky.randomSplit([0.8, 0.2], seed=RANDOM_SEED)

    # Temporal split, derived from the SAME frame so the leaky column is
    # identical in construction to the random-split conditions.
    cuts = df_leaky.approxQuantile("timestamp", [0.85], 0.001)
    tmp_train = df_leaky.filter(F.col("timestamp") < cuts[0])
    tmp_test = df_leaky.filter(F.col("timestamp") >= cuts[0])

    conditions = [
        ("A. Replication (random split + leaky feature)",
         rnd_train, rnd_test, shared_cols + leaky_cols),
        ("B. Temporal split, leaky feature retained",
         tmp_train, tmp_test, shared_cols + leaky_cols),
        ("C. Random split, honest features",
         rnd_train, rnd_test, shared_cols + honest_cols),
        ("D. Temporal split, honest features (ours)",
         tmp_train, tmp_test, shared_cols + honest_cols),
    ]

    rows = []
    for name, tr, te, cols in conditions:
        print(f"\n-> {name}")
        try:
            tr_w = add_class_weights(tr, label_col)
            pipe, _ = build_feature_pipeline(cols, use_text=False, use_category=True)
            fitted = pipe.fit(tr_w)

            clf = LogisticRegression(
                featuresCol="features", labelCol=label_col,
                weightCol="weight", maxIter=50, regParam=0.01,
            )
            model = clf.fit(fitted.transform(tr_w))
            preds = model.transform(fitted.transform(te))

            row = evaluate(preds, name, label_col, compute_ndcg=False)
            rows.append(row)
            print(f"   PR-AUC={row['pr_auc']:.4f}  ROC-AUC={row['roc_auc']:.4f}  "
                  f"base rate={row['base_rate']:.4f}  lift={row['pr_auc_lift']:.2f}x")
        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {e}")

    df_leaky.unpersist()

    table = results_table(rows, sort_by=None)

    if len(table) == 4:
        inflation = table.loc[0, "pr_auc"] - table.loc[3, "pr_auc"]
        print(f"\n>>> Total inflation from methodological artefact: "
              f"{inflation:+.4f} PR-AUC")
        print(f">>> Attributable to random splitting:  "
              f"{table.loc[0, 'pr_auc'] - table.loc[1, 'pr_auc']:+.4f}")
        print(f">>> Attributable to the leaky feature: "
              f"{table.loc[0, 'pr_auc'] - table.loc[2, 'pr_auc']:+.4f}")

    return table


# ----------------------------------------------------------------------
# Experiment 2 - feature block ablation
# ----------------------------------------------------------------------

def experiment_ablation(train, test, label_col="label", use_text=True):
    """
    Add feature blocks cumulatively and measure the marginal contribution
    of each theoretical construct.

    Cumulative rather than leave-one-out, because these blocks are
    correlated; leave-one-out understates every block when substitutes
    are present.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 - FEATURE BLOCK ABLATION")
    print("=" * 70)

    order = ["exposure", "structural", "lexical", "affective", "contextual", "reviewer"]

    train_w = add_class_weights(train, label_col).cache()
    test_c = test.cache()

    rows = []
    accumulated = []

    for block in order:
        accumulated = accumulated + [c for c in BLOCKS[block] if c in train.columns]
        name = f"+ {block}"
        print(f"\n-> {name}  ({len(accumulated)} features)")

        try:
            pipe, _ = build_feature_pipeline(accumulated, use_text=False, use_category=True)
            fitted = pipe.fit(train_w)
            clf = LogisticRegression(
                featuresCol="features", labelCol=label_col,
                weightCol="weight", maxIter=50, regParam=0.01,
            )
            model = clf.fit(fitted.transform(train_w))
            preds = model.transform(fitted.transform(test_c))

            row = evaluate(preds, name, label_col, compute_ndcg=False)
            row["n_features"] = len(accumulated)
            rows.append(row)
            print(f"   PR-AUC={row['pr_auc']:.4f}  (lift {row['pr_auc_lift']:.2f}x)")
        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {e}")

    if use_text:
        print(f"\n-> + text (TF-IDF)")
        try:
            pipe, _ = build_feature_pipeline(accumulated, use_text=True, use_category=True)
            fitted = pipe.fit(train_w)
            clf = LogisticRegression(
                featuresCol="features", labelCol=label_col,
                weightCol="weight", maxIter=50, regParam=0.01,
            )
            model = clf.fit(fitted.transform(train_w))
            preds = model.transform(fitted.transform(test_c))
            row = evaluate(preds, "+ text (TF-IDF)", label_col, compute_ndcg=False)
            row["n_features"] = len(accumulated) + HASHING_FEATURES
            rows.append(row)
            print(f"   PR-AUC={row['pr_auc']:.4f}  (lift {row['pr_auc_lift']:.2f}x)")
        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {e}")

    train_w.unpersist()

    # Marginal contributions
    table = results_table(rows, sort_by=None)
    table["marginal_pr_auc"] = table["pr_auc"].diff().fillna(table["pr_auc"])

    return table


# ----------------------------------------------------------------------
# Experiment 3 - model comparison
# ----------------------------------------------------------------------

def experiment_models(train, test, label_col="label", use_text=True):
    """
    Compare classifiers on the honest feature set.

    Two feature spaces, by necessity (see get_tree_classifiers):
      * linear models  -> engineered features + category + TF-IDF text
      * tree ensembles -> engineered features + category only

    Each row of the results table records which space was used.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 - MODEL COMPARISON")
    print("=" * 70)

    numeric = [c for c in ALL_NUMERIC if c in train.columns]
    carry = ["helpful_vote", "parent_asin", "Category", "product_type"]
    carry = [c for c in carry if c in test.columns]

    train_w = add_class_weights(train, label_col).cache()
    test_c = test.cache()

    rows = baseline_results(test_c, label_col)
    for r in rows:
        r["feature_space"] = "single feature"
        print(f"   {r['model']:<34} PR-AUC={r['pr_auc']:.4f}")

    trained = {}

    def _run(classifiers, use_text_here, space_label):
        print(f"\nFitting feature pipeline [{space_label}]...")
        pipe, _ = build_feature_pipeline(
            numeric, use_text=use_text_here, use_category=True)
        fitted = pipe.fit(train_w)

        tr_feat = fitted.transform(train_w).select(
            "features", label_col, "weight").cache()
        te_feat = fitted.transform(test_c).select(
            "features", label_col, *carry).cache()
        tr_feat.count()
        te_feat.count()

        for name, clf in classifiers:
            print(f"\n-> {name}")
            try:
                model = clf.fit(tr_feat)
                preds = model.transform(te_feat)
                row = evaluate(preds, name, label_col, compute_ndcg=True)
                row["feature_space"] = space_label
                rows.append(row)
                trained[name] = (model, fitted)
                print(f"   PR-AUC={row['pr_auc']:.4f}  "
                      f"lift={row['pr_auc_lift']:.2f}x  "
                      f"NDCG@5={row.get('ndcg@5', float('nan')):.4f}")
            except Exception as e:
                print(f"   FAILED: {type(e).__name__}: {e}")

        tr_feat.unpersist()
        return te_feat

    # Trees first, on the cheap dense space, so a failure in the expensive
    # text stage does not cost us the tree results.
    te_feat = _run(get_tree_classifiers(label_col, use_weights=True),
                   False, "numeric + category")

    if use_text:
        te_feat = _run(get_linear_classifiers(label_col, use_weights=True),
                       True, "numeric + category + TF-IDF")
    else:
        te_feat = _run(get_linear_classifiers(label_col, use_weights=True),
                       False, "numeric + category")

    train_w.unpersist()

    return results_table(rows), trained, te_feat
