"""
Product-type moderation analysis.

The theoretical claim (Mudambi & Schuff 2010, MIS Quarterly, building on
Nelson 1970) is that review helpfulness is not a property of the review
alone but of the review-product pair:

  H1  Review depth increases helpfulness MORE for search goods than for
      experience goods. Search-good quality is verifiable from attributes,
      so detailed attribute coverage is directly useful. Experience-good
      quality is not, so extra length adds less.

  H2  Rating extremity DECREASES helpfulness for search goods but not for
      experience goods. For a search good, an extreme rating signals an
      idiosyncratic reviewer rather than product quality. For an experience
      good, strong reactions are the informative signal.

Mudambi & Schuff tested this on 1,587 reviews of six products. We test it
on the 2023 corpus across categories, which is the empirical contribution.

Two estimation strategies, deliberately:

  * Spark, full data, per-category models. Scales, but Spark ML gives no
    reliable standard errors under regularisation.
  * statsmodels, stratified subsample, pooled model with interactions.
    Gives proper inference, which is what a moderation claim requires.

Reporting only the first would be a methods failure; reporting only the
second would waste the dataset.
"""

import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression

from config import RANDOM_SEED, TBL_DIR
from features import ALL_NUMERIC
from models import build_feature_pipeline, add_class_weights
from evaluation import evaluate, results_table


# Features carrying the theoretical constructs under test
DEPTH_FEATURES = ["log_review_length", "review_word_count", "avg_sentence_length"]
EXTREMITY_FEATURES = ["rating_extremity", "is_extreme_rating"]


# ----------------------------------------------------------------------
# Per-category models
# ----------------------------------------------------------------------

def per_category_models(train, test, label_col="label", use_text=False):
    """Fit one model per category and compare both performance and the
    standardised coefficients on the theoretically relevant features.

    Features are standardised inside the pipeline, so coefficients are
    directly comparable across categories despite different feature scales.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4a - PER-CATEGORY MODELS")
    print("=" * 70)

    categories = [r["Category"] for r in
                  train.select("Category").distinct().collect()]

    numeric = [c for c in ALL_NUMERIC if c in train.columns]

    perf_rows = []
    coef_rows = []

    for cat in sorted(categories):
        tr = train.filter(F.col("Category") == cat)
        te = test.filter(F.col("Category") == cat)

        n_tr, n_te = tr.count(), te.count()
        if n_tr < 1000 or n_te < 200:
            print(f"\n-> {cat}: too few rows (train={n_tr}, test={n_te}), skipped")
            continue

        ptype = tr.select("product_type").first()["product_type"]
        print(f"\n-> {cat}  [{ptype}]  train={n_tr:,} test={n_te:,}")

        try:
            tr_w = add_class_weights(tr, label_col)
            # No category one-hot here: within a category it is constant.
            pipe, _ = build_feature_pipeline(
                numeric, use_text=use_text, use_category=False, scale=True)
            fitted = pipe.fit(tr_w)

            clf = LogisticRegression(
                featuresCol="features", labelCol=label_col,
                weightCol="weight", maxIter=50, regParam=0.001,
            )
            model = clf.fit(fitted.transform(tr_w))
            preds = model.transform(fitted.transform(te))

            row = evaluate(preds, cat, label_col, compute_ndcg=True)
            row["product_type"] = ptype
            row["n_train"] = n_tr
            perf_rows.append(row)
            print(f"   PR-AUC={row['pr_auc']:.4f}  lift={row['pr_auc_lift']:.2f}x")

            coefs = model.coefficients.toArray()
            for i, feat in enumerate(numeric):
                if i < len(coefs):
                    coef_rows.append({
                        "Category": cat,
                        "product_type": ptype,
                        "feature": feat,
                        "coefficient": float(coefs[i]),
                    })

        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {e}")

    return results_table(perf_rows, sort_by=None), pd.DataFrame(coef_rows)


def compare_constructs(coef_df):
    """Contrast the depth and extremity coefficients between product types.

    This is the direct test of H1 and H2 at the coefficient level.
    """
    if coef_df.empty:
        return pd.DataFrame()

    rows = []
    for construct, feats in [("depth", DEPTH_FEATURES),
                             ("extremity", EXTREMITY_FEATURES)]:
        sub = coef_df[coef_df["feature"].isin(feats)]
        if sub.empty:
            continue
        grouped = sub.groupby("product_type")["coefficient"].mean()
        search = grouped.get("search", np.nan)
        experience = grouped.get("experience", np.nan)
        rows.append({
            "construct": construct,
            "search_goods_coef": round(float(search), 5),
            "experience_goods_coef": round(float(experience), 5),
            "difference": round(float(search - experience), 5),
            "hypothesis": ("H1: depth effect larger for search goods"
                           if construct == "depth"
                           else "H2: extremity penalised for search goods"),
        })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Pooled interaction model with proper inference
# ----------------------------------------------------------------------

def interaction_model(train, sample_n=200_000, label_col="label",
                      seed=RANDOM_SEED):
    """
    Pooled logistic regression with product-type interactions, estimated on
    a stratified subsample using statsmodels.

    Why subsample: a moderation claim needs standard errors, and Spark ML
    does not provide trustworthy ones under regularisation. 200k rows gives
    ample power to detect the effect sizes at issue; the constraint is
    inferential validity, not sample size.

    Why this matters more than the per-category comparison: comparing two
    separately-fitted coefficients is not a test of whether they differ.
    The interaction term is.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4b - POOLED INTERACTION MODEL (statsmodels)")
    print("=" * 70)

    try:
        import statsmodels.api as sm
    except ImportError:
        print("statsmodels not installed. Run: pip install statsmodels")
        return None, None

    keep = [
        "log_review_length", "rating_extremity", "verified_int",
        "log_exposure_days", "type_token_ratio", "sentiment_polarity",
        "log_prior_review_count", "has_images",
        "product_type", label_col,
    ]
    keep = [c for c in keep if c in train.columns]

    total = train.count()
    frac = min(1.0, sample_n / max(total, 1))
    pdf = train.select(*keep).sample(False, frac, seed=seed).toPandas()

    pdf = pdf[pdf["product_type"].isin(["search", "experience"])].dropna()
    if len(pdf) < 1000:
        print("Insufficient rows after filtering; skipped.")
        return None, None

    print(f"Estimating on {len(pdf):,} reviews "
          f"({(pdf['product_type'] == 'search').mean() * 100:.1f}% search goods)")

    # Search goods are the reference category
    pdf["is_experience"] = (pdf["product_type"] == "experience").astype(float)

    # Standardise continuous predictors so interaction terms are interpretable
    continuous = ["log_review_length", "rating_extremity", "log_exposure_days",
                  "type_token_ratio", "sentiment_polarity", "log_prior_review_count"]
    continuous = [c for c in continuous if c in pdf.columns]

    for c in continuous:
        sd = pdf[c].std()
        pdf[c] = (pdf[c] - pdf[c].mean()) / (sd if sd > 0 else 1.0)

    # Main effects
    X_cols = continuous + ["verified_int", "has_images", "is_experience"]
    X_cols = [c for c in X_cols if c in pdf.columns]

    # The interactions that operationalise H1 and H2
    pdf["depth_x_experience"] = pdf["log_review_length"] * pdf["is_experience"]
    pdf["extremity_x_experience"] = pdf["rating_extremity"] * pdf["is_experience"]
    X_cols += ["depth_x_experience", "extremity_x_experience"]

    X = sm.add_constant(pdf[X_cols].astype(float))
    y = pdf[label_col].astype(int)

    model = sm.Logit(y, X).fit(disp=0, maxiter=200)

    summary = pd.DataFrame({
        "term": model.params.index,
        "coefficient": model.params.values.round(5),
        "std_error": model.bse.values.round(5),
        "z": (model.params / model.bse).values.round(3),
        "p_value": model.pvalues.values.round(6),
        "odds_ratio": np.exp(model.params.values).round(4),
    })

    print("\n" + summary.to_string(index=False))

    # Interpret the hypothesis tests explicitly rather than leaving it to
    # the reader to squint at a coefficient table.
    print("\n--- Hypothesis tests ---")
    for term, hypothesis in [
        ("depth_x_experience",
         "H1: depth matters LESS for experience goods (expect negative)"),
        ("extremity_x_experience",
         "H2: extremity penalised LESS for experience goods (expect positive)"),
    ]:
        if term in model.params.index:
            coef = model.params[term]
            p = model.pvalues[term]
            verdict = "SUPPORTED" if p < 0.05 else "not significant"
            direction = "negative" if coef < 0 else "positive"
            print(f"  {hypothesis}")
            print(f"    coefficient = {coef:+.5f} ({direction}), "
                  f"p = {p:.2e}  -> {verdict}")

    print("\nNOTE: with samples this large, almost any effect reaches "
          "significance.\n      Judge the odds ratios and marginal effects, "
          "not the p-values alone.")

    return summary, model


# ----------------------------------------------------------------------
# Marginal effects - what the coefficients mean in practice
# ----------------------------------------------------------------------

def marginal_effects(model, pdf_template=None):
    """Average marginal effects, which are what a reader can actually
    interpret. A logit coefficient of 0.3 means nothing to a practitioner;
    'a one-SD longer review raises helpfulness probability by 4 points'
    does."""
    if model is None:
        return None
    try:
        me = model.get_margeff(at="mean")
        out = pd.DataFrame({
            "term": me.margeff_names if hasattr(me, "margeff_names") else model.params.index[1:],
            "marginal_effect": np.round(me.margeff, 5),
            "std_error": np.round(me.margeff_se, 5),
            "p_value": np.round(me.pvalues, 6),
        })
        print("\nAverage marginal effects (probability points per 1 SD):")
        print(out.to_string(index=False))
        return out
    except Exception as e:
        print(f"Marginal effects failed: {type(e).__name__}: {e}")
        return None
