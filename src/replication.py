"""
Replication registry: do the canonical findings of the online review
literature still hold on Amazon Reviews 2023?

THE ARGUMENT
------------
Nearly every established finding about what makes an online review helpful
was produced between 2007 and 2018, on the 2014 or 2018 Amazon releases, and
frequently on samples of a few thousand reviews. Mudambi & Schuff's canonical
extremity result used 1,587 reviews of six products.

Amazon Reviews 2023 is a different corpus: 571M reviews, coverage through
September 2023, a marketplace that has since absorbed mobile-first writing,
incentivised review programmes, and post-2022 generative text. Nobody has
re-tested the field's foundational claims on it.

This module does that. Each finding is registered with its citation, the
original claim, the scale it was established at, and an explicit
operationalisation. Every finding is then tested by the same pipeline, under
temporal validation, and assigned one of four verdicts:

    REPLICATES   same direction, statistically distinguishable from zero
    WEAKENS      same direction, but effect size is trivially small
    FAILS        not distinguishable from zero
    REVERSES     opposite direction, distinguishable from zero

WHY THIS IS A CONTRIBUTION AND NOT A CRITIQUE
---------------------------------------------
Nothing here accuses prior authors of error. Their results were correct for
their data. The question is whether they describe the marketplace as it now
exists, and that question can only be answered by re-running them.

Registering the operationalisation BEFORE looking at the estimate is what
separates replication from fishing. Each Finding declares its expected
direction as a field; the verdict function compares against that declaration
rather than against whatever the data happened to produce.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
It does not claim to reproduce any original study's exact specification.
Different corpora, different available covariates, different eras. These are
CONCEPTUAL replications: same construct, same predicted direction, modern
data, honest validation. That distinction is stated in every output table so
no reader can mistake one for the other.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, List

import numpy as np
import pandas as pd

from pyspark.sql import functions as F

from config import RANDOM_SEED


# ----------------------------------------------------------------------
# Verdict thresholds
# ----------------------------------------------------------------------
#
# At n in the hundreds of thousands, p-values are close to meaningless -
# essentially any non-zero effect reaches significance. Verdicts therefore
# depend on EFFECT SIZE, with significance as a necessary but not sufficient
# condition.
#
# The threshold below is expressed in average marginal effect: the change in
# probability of receiving a helpful vote per one standard deviation of the
# predictor. One percentage point is the smallest effect we are willing to
# call practically meaningful, and that choice is declared here rather than
# chosen after seeing results.

MIN_MEANINGFUL_AME = 0.01     # 1 percentage point per 1 SD
ALPHA = 0.001                 # tightened from .05 given the sample size


@dataclass
class Finding:
    """One registered claim from the literature."""

    id: str
    short: str
    citation: str
    original_claim: str
    original_scale: str
    construct: str
    term: str                      # the model term carrying the construct
    expected_direction: str        # "positive" | "negative"
    notes: str = ""
    moderated_by: Optional[str] = None
    # Name of a precondition that the DATA must satisfy for this finding to be
    # testable at all. See TESTABILITY_CHECKS. A finding whose precondition
    # fails is reported NOT TESTABLE - never FAILS.
    requires: Optional[str] = None


# ----------------------------------------------------------------------
# Testability preconditions
# ----------------------------------------------------------------------
#
# A null result means one of two very different things:
#
#   (a) the effect is absent in this corpus  -> FAILS, an informative result
#   (b) the corpus cannot express the effect -> NOT TESTABLE, no result at all
#
# Conflating them is a serious reporting error. If 97% of reviewers have no
# prior reviews, a null coefficient on reviewer history says nothing about
# whether reviewer history matters; it says the variable is almost constant.
#
# Each check returns (ok, explanation). Checks run against the estimation
# frame before any coefficient is inspected, so a precondition can never be
# rationalised after seeing the result.

def _check_reviewer_history(pdf):
    """Reviewer-history findings need reviewers who actually have history."""
    if "log_prior_review_count" not in pdf.columns:
        return False, "prior-history features absent"

    with_history = float((pdf["log_prior_review_count"] > 0).mean())
    if with_history < 0.10:
        return False, (
            f"only {100 * with_history:.1f}% of reviews have a reviewer with "
            f"any prior review in this extract. The predictor is near-constant, "
            f"so a null coefficient carries no information about the claim. "
            f"This is a property of how the extract was sampled - review-level "
            f"random sampling destroys user-level structure - and not a "
            f"property of the marketplace."
        )
    return True, f"{100 * with_history:.1f}% of reviews have reviewer history"


def _check_product_history(pdf):
    """Consensus-deviation findings need products with prior reviews."""
    if "rating_deviation_from_product" not in pdf.columns:
        return False, "product-history features absent"
    nonzero = float((pdf["rating_deviation_from_product"] != 0).mean())
    if nonzero < 0.10:
        return False, (
            f"only {100 * nonzero:.1f}% of reviews follow a prior review of the "
            f"same product, so consensus deviation is undefined for almost all "
            f"rows"
        )
    return True, f"{100 * nonzero:.1f}% of reviews have a product consensus to deviate from"


TESTABILITY_CHECKS = {
    "reviewer_history": _check_reviewer_history,
    "product_history": _check_product_history,
}


# ----------------------------------------------------------------------
# THE REGISTRY
# ----------------------------------------------------------------------
#
# Ordered roughly by how foundational the claim is. Each `term` must be
# constructible from the feature set in features.py, or the finding is not
# testable here and is marked as such rather than silently dropped.

REGISTRY: List[Finding] = [

    Finding(
        id="F01",
        short="Review depth increases helpfulness",
        citation="Mudambi & Schuff (2010), MIS Quarterly 34(1)",
        original_claim=(
            "Review depth, operationalised as word count, has a positive "
            "effect on review helpfulness."
        ),
        original_scale="1,587 reviews across 6 products",
        construct="depth",
        term="log_review_length",
        expected_direction="positive",
        notes=(
            "The single most-cited finding in this literature. If anything "
            "replicates, it should be this."
        ),
    ),

    Finding(
        id="F02",
        short="Rating extremity reduces helpfulness",
        citation="Mudambi & Schuff (2010), MIS Quarterly 34(1)",
        original_claim=(
            "Reviews with extreme ratings are less helpful than reviews with "
            "moderate ratings, for search goods."
        ),
        original_scale="1,587 reviews across 6 products",
        construct="extremity",
        term="rating_extremity",
        expected_direction="negative",
        moderated_by="product_type",
        notes=(
            "The original predicted this for SEARCH goods specifically. We "
            "test the main effect and the product-type interaction separately."
        ),
    ),

    Finding(
        id="F03",
        short="Depth matters more for search goods",
        citation="Mudambi & Schuff (2010), MIS Quarterly 34(1)",
        original_claim=(
            "The positive effect of review depth on helpfulness is greater "
            "for search goods than for experience goods."
        ),
        original_scale="1,587 reviews across 6 products",
        construct="depth x product type",
        term="depth_x_experience",
        expected_direction="negative",   # negative on the experience interaction
        notes=(
            "Operationalised with search goods as the reference category, so "
            "the ORIGINAL claim predicts a NEGATIVE interaction coefficient."
        ),
    ),

    Finding(
        id="F04",
        short="Readability increases helpfulness",
        citation="Korfiatis, Garcia-Bariocanal & Sanchez-Alonso (2012), ECRA 11(3)",
        original_claim=(
            "More readable reviews, measured by standard readability indices, "
            "are rated as more helpful."
        ),
        original_scale="~37,000 Amazon UK reviews",
        construct="readability",
        term="flesch_reading_ease",
        expected_direction="positive",
        notes=(
            "Flesch Reading Ease rises as text gets easier. A positive "
            "coefficient therefore supports the original claim."
        ),
    ),

    Finding(
        id="F05",
        short="Longer reviews are not necessarily better",
        citation="Fink, Rosenfeld & Ravid (2018), Int. J. Information Management 39",
        original_claim=(
            "The relationship between review length and helpfulness is not "
            "monotonic; beyond a point, additional length does not help."
        ),
        original_scale="Amazon reviews, multiple categories",
        construct="depth (nonlinearity)",
        term="review_length_sq",
        expected_direction="negative",
        notes=(
            "Directly contradicts the naive reading of F01. Tested as a "
            "quadratic term: a negative coefficient on the square indicates "
            "diminishing and eventually negative returns to length. F01 and "
            "F05 can both hold simultaneously."
        ),
    ),

    Finding(
        id="F06",
        short="Verified purchase increases helpfulness",
        citation="Common assumption; e.g. Filieri et al. (2018), J. Business Research",
        original_claim=(
            "Reviews marked as verified purchases are perceived as more "
            "credible and therefore more helpful."
        ),
        original_scale="Varies; typically survey or small-sample studies",
        construct="credibility signal",
        term="verified_int",
        expected_direction="positive",
        notes=(
            "Widely assumed, less often tested at scale on observational "
            "vote data."
        ),
    ),

    Finding(
        id="F07",
        short="Reviewer track record predicts helpfulness",
        citation="Ghose & Ipeirotis (2011), IEEE TKDE 23(10)",
        original_claim=(
            "Reviewer characteristics, including past reviewing history, "
            "predict the helpfulness of their reviews."
        ),
        original_scale="~400 products, Amazon",
        construct="reviewer history",
        term="log_prior_review_count",
        expected_direction="positive",
        requires="reviewer_history",
        notes=(
            "CRITICAL: tested here with STRICTLY PRIOR reviews only. Much "
            "prior work computes reviewer history over the full corpus, which "
            "leaks the target. See Experiment 1 for what that does to the "
            "estimate. NOTE: in a review-level random sample almost every "
            "reviewer appears once, leaving no history to test - hence the "
            "testability precondition."
        ),
    ),

    Finding(
        id="F08",
        short="Lexical diversity increases helpfulness",
        citation="Krishnamoorthy (2015), Expert Systems with Applications 42(7)",
        original_claim=(
            "Linguistic richness features improve helpfulness prediction over "
            "metadata alone."
        ),
        original_scale="~1,500 reviews, 2 product categories",
        construct="lexical richness",
        term="type_token_ratio",
        expected_direction="positive",
    ),

    Finding(
        id="F09",
        short="Deviating from consensus attracts attention",
        citation="Adapted from Moe & Trusov (2011), J. Marketing Research 48(3)",
        original_claim=(
            "Reviews that diverge from the existing rating consensus receive "
            "disproportionate reader attention."
        ),
        original_scale="Bath & body retailer panel data",
        construct="consensus deviation",
        term="rating_deviation_from_product",
        expected_direction="positive",
        requires="product_history",
        notes=(
            "Computed against PRIOR reviews of the same product only, so the "
            "measure is available at posting time."
        ),
    ),

    Finding(
        id="F10",
        short="Images increase helpfulness",
        citation="Ceylan, Diehl & Proserpio (2024), J. Marketing 88(2), and related",
        original_claim=(
            "Reviews containing user-generated images are perceived as more "
            "helpful than text-only reviews."
        ),
        original_scale="Experimental plus field data",
        construct="multimedia",
        term="has_images",
        expected_direction="positive",
        notes=(
            "The 2023 release exposes review images; the 2018 release did "
            "not. This is testable at scale here for the first time."
        ),
    ),

    Finding(
        id="F11",
        short="Negative sentiment increases helpfulness",
        citation="Sen & Lerman (2007), J. Interactive Marketing 21(4)",
        original_claim=(
            "Negative reviews are perceived as more useful than positive ones, "
            "particularly for utilitarian products."
        ),
        original_scale="Laboratory experiments",
        construct="negativity bias",
        term="sentiment_polarity",
        expected_direction="negative",   # negative polarity -> more helpful
        moderated_by="product_type",
        notes=(
            "sentiment_polarity is positive for positive text, so the original "
            "claim predicts a NEGATIVE coefficient."
        ),
    ),

    Finding(
        id="F12",
        short="Early reviews accumulate disproportionate votes",
        citation="Adapted from Godes & Silva (2012), Marketing Science 31(3)",
        original_claim=(
            "Review position in a product's sequence affects its reception "
            "independently of content."
        ),
        original_scale="Book reviews, Amazon",
        construct="arrival advantage",
        term="log_arrival_position",
        expected_direction="negative",
        notes=(
            "Tested descriptively rather than causally. Establishing causation "
            "would require an instrument for visibility, which this dataset "
            "does not provide."
        ),
    ),
]


# ----------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------

def _build_frame(train, sample_n=300_000, seed=RANDOM_SEED):
    """Pull a stratified sample to the driver and construct the terms that
    the registry references but that features.py does not produce."""

    base_cols = [
        "log_review_length", "review_length", "rating_extremity",
        "flesch_reading_ease", "verified_int", "log_prior_review_count",
        "type_token_ratio", "rating_deviation_from_product", "has_images",
        "sentiment_polarity", "log_exposure_days", "product_type",
        "parent_asin", "timestamp", "label",
    ]
    cols = [c for c in base_cols if c in train.columns]

    total = train.count()
    frac = min(1.0, sample_n / max(total, 1))
    pdf = train.select(*cols).sample(False, frac, seed=seed).toPandas()

    pdf = pdf[pdf["product_type"].isin(["search", "experience"])].dropna()
    if pdf.empty:
        return pdf

    # --- derived terms the registry needs ---

    # F03: interaction, search goods as reference
    pdf["is_experience"] = (pdf["product_type"] == "experience").astype(float)

    # F05: quadratic length. Standardise BEFORE squaring so the linear and
    # quadratic terms are not near-collinear.
    z_len = (pdf["log_review_length"] - pdf["log_review_length"].mean()) / \
            max(pdf["log_review_length"].std(), 1e-9)
    pdf["review_length_sq"] = z_len ** 2

    # F12: arrival position within the product's review sequence
    pdf = pdf.sort_values(["parent_asin", "timestamp"])
    pdf["arrival_position"] = pdf.groupby("parent_asin").cumcount() + 1
    pdf["log_arrival_position"] = np.log1p(pdf["arrival_position"])

    return pdf


def _standardise(pdf, cols):
    out = pdf.copy()
    for c in cols:
        if c in out.columns:
            sd = out[c].std()
            out[c] = (out[c] - out[c].mean()) / (sd if sd > 0 else 1.0)
    return out


def estimate(train, sample_n=300_000, seed=RANDOM_SEED):
    """
    Fit ONE pooled logistic model containing every registered term, and read
    each finding's verdict off the same fit.

    One model rather than twelve, deliberately. Testing each construct in
    isolation would credit each with variance that is actually shared - the
    classic way a correlated feature set produces twelve "significant"
    findings that collectively explain very little. A single specification
    with all terms present gives each construct its unique contribution.

    Exposure duration is included as a control in every specification, so no
    finding can be an artefact of review age.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        print("statsmodels not installed. Run: pip install statsmodels")
        return None, None

    print("\n" + "=" * 70)
    print("EXPERIMENT 6 - REPLICATION OF CANONICAL FINDINGS")
    print("=" * 70)

    pdf = _build_frame(train, sample_n, seed)
    if pdf is None or len(pdf) < 5000:
        print("Insufficient rows after filtering; replication skipped.")
        return None, None

    print(f"Estimating on {len(pdf):,} reviews "
          f"({(pdf['product_type'] == 'search').mean() * 100:.1f}% search goods)")

    # Preconditions are evaluated here, before estimation, so no verdict can
    # be rationalised after the fact.
    print("\nTestability preconditions:")
    testability = run_testability_checks(pdf)

    continuous = [
        "log_review_length", "rating_extremity", "flesch_reading_ease",
        "log_prior_review_count", "type_token_ratio",
        "rating_deviation_from_product", "sentiment_polarity",
        "log_exposure_days", "log_arrival_position",
    ]
    continuous = [c for c in continuous if c in pdf.columns]
    pdf = _standardise(pdf, continuous)

    terms = continuous + ["verified_int", "has_images", "is_experience",
                          "review_length_sq"]
    terms = [t for t in terms if t in pdf.columns]

    # Interactions referenced by the registry
    if "log_review_length" in pdf.columns:
        pdf["depth_x_experience"] = pdf["log_review_length"] * pdf["is_experience"]
        terms.append("depth_x_experience")
    if "rating_extremity" in pdf.columns:
        pdf["extremity_x_experience"] = pdf["rating_extremity"] * pdf["is_experience"]
        terms.append("extremity_x_experience")
    if "sentiment_polarity" in pdf.columns:
        pdf["sentiment_x_experience"] = pdf["sentiment_polarity"] * pdf["is_experience"]
        terms.append("sentiment_x_experience")

    X = sm.add_constant(pdf[terms].astype(float))
    y = pdf["label"].astype(int)

    model = sm.Logit(y, X).fit(disp=0, maxiter=300)

    # Average marginal effects give the practically interpretable magnitude
    try:
        me = model.get_margeff(at="mean")
        ame = dict(zip(model.params.index[1:], me.margeff))
        ame_se = dict(zip(model.params.index[1:], me.margeff_se))
    except Exception:
        ame, ame_se = {}, {}

    return model, {"ame": ame, "ame_se": ame_se, "n": len(pdf),
                   "testability": testability, "pdf": pdf}


# ----------------------------------------------------------------------
# Verdicts
# ----------------------------------------------------------------------

def _verdict(coef, pval, ame, expected):
    """Assign one of four verdicts. Effect size leads; significance gates."""
    if coef is None or pval is None:
        return "NOT TESTABLE", "term absent from the fitted model"

    observed = "positive" if coef > 0 else "negative"
    magnitude = abs(ame) if ame is not None else None

    if pval >= ALPHA:
        return "FAILS", f"not distinguishable from zero (p = {pval:.3g})"

    if observed != expected:
        return "REVERSES", (
            f"significant in the OPPOSITE direction to the original claim "
            f"({observed}, expected {expected})"
        )

    if magnitude is not None and magnitude < MIN_MEANINGFUL_AME:
        return "WEAKENS", (
            f"correct direction, but only {magnitude * 100:.2f} percentage "
            f"points per SD - below the {MIN_MEANINGFUL_AME * 100:.0f}pp "
            f"threshold declared in advance"
        )

    detail = f"{magnitude * 100:.2f} pp per SD" if magnitude is not None else "direction confirmed"
    return "REPLICATES", f"same direction, practically meaningful ({detail})"


def run_testability_checks(pdf, registry=REGISTRY):
    """Evaluate every declared precondition against the estimation frame.

    Run BEFORE any coefficient is inspected, so a precondition cannot be
    invoked retrospectively to explain away an inconvenient estimate.
    """
    results = {}
    for name, check in TESTABILITY_CHECKS.items():
        try:
            ok, why = check(pdf)
        except Exception as e:
            ok, why = False, f"check failed: {type(e).__name__}: {e}"
        results[name] = (ok, why)
        status = "OK" if ok else "NOT TESTABLE"
        print(f"  [{status:<12}] {name}: {why}")
    return results


def build_table(model, extras, registry=REGISTRY):
    """Assemble the replication results table."""
    if model is None:
        return pd.DataFrame()

    ame = extras.get("ame", {})
    testability = extras.get("testability", {})
    rows = []

    for f in registry:
        term = f.term
        in_model = term in model.params.index

        coef = float(model.params[term]) if in_model else None
        pval = float(model.pvalues[term]) if in_model else None
        se = float(model.bse[term]) if in_model else None
        m = float(ame[term]) if term in ame else None

        # A failed precondition overrides the estimate entirely. Reporting
        # FAILS for a finding the data cannot express would assert a negative
        # result that was never tested.
        if f.requires and f.requires in testability:
            ok, why = testability[f.requires]
            if not ok:
                rows.append({
                    "id": f.id,
                    "finding": f.short,
                    "citation": f.citation,
                    "original_scale": f.original_scale,
                    "expected": f.expected_direction,
                    "coefficient": round(coef, 5) if coef is not None else None,
                    "std_error": round(se, 5) if se is not None else None,
                    "p_value": f"{pval:.2e}" if pval is not None else None,
                    "marginal_effect_pp": round(100 * m, 3) if m is not None else None,
                    "verdict": "NOT TESTABLE",
                    "reason": why,
                })
                continue

        verdict, reason = _verdict(coef, pval, m, f.expected_direction)

        rows.append({
            "id": f.id,
            "finding": f.short,
            "citation": f.citation,
            "original_scale": f.original_scale,
            "expected": f.expected_direction,
            "coefficient": round(coef, 5) if coef is not None else None,
            "std_error": round(se, 5) if se is not None else None,
            "p_value": f"{pval:.2e}" if pval is not None else None,
            "marginal_effect_pp": round(100 * m, 3) if m is not None else None,
            "verdict": verdict,
            "reason": reason,
        })

    return pd.DataFrame(rows)


def summarise(table):
    """Print the headline counts and the interesting cases."""
    if table is None or table.empty:
        return pd.DataFrame()

    counts = table["verdict"].value_counts()
    total = len(table)

    print("\n" + "-" * 70)
    print("REPLICATION SUMMARY")
    print("-" * 70)
    for v in ["REPLICATES", "WEAKENS", "FAILS", "REVERSES", "NOT TESTABLE"]:
        n = int(counts.get(v, 0))
        if n:
            print(f"  {v:<14} {n:>2} / {total}   ({100 * n / total:.0f}%)")

    interesting = table[table["verdict"].isin(["FAILS", "REVERSES", "WEAKENS"])]
    if not interesting.empty:
        print("\nFindings that do NOT straightforwardly replicate:\n")
        for _, r in interesting.iterrows():
            print(f"  [{r['id']}] {r['finding']}")
            print(f"        {r['citation']}")
            print(f"        -> {r['verdict']}: {r['reason']}\n")

    print("These are CONCEPTUAL replications: same construct and predicted")
    print("direction, modern corpus, temporal validation. They are not")
    print("reproductions of the original specifications, and a failure here")
    print("is evidence about the 2023 marketplace, not about the original work.")

    return counts.to_frame("count").reset_index().rename(columns={"index": "verdict"})


def registry_table(registry=REGISTRY):
    """The pre-registration table: what we committed to testing, and how."""
    return pd.DataFrame([{
        "id": f.id,
        "finding": f.short,
        "citation": f.citation,
        "original_claim": f.original_claim,
        "original_scale": f.original_scale,
        "construct": f.construct,
        "model_term": f.term,
        "expected_direction": f.expected_direction,
        "moderated_by": f.moderated_by or "",
        "notes": f.notes,
    } for f in registry])


def run(train, sample_n=300_000):
    """Entry point. Returns (registry_table, results_table, summary_table)."""
    reg = registry_table()
    model, extras = estimate(train, sample_n)
    results = build_table(model, extras) if model is not None else pd.DataFrame()
    summary = summarise(results) if not results.empty else pd.DataFrame()
    return reg, results, summary
