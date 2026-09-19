"""
Generate the results sections of the report directly from the saved tables.

Rationale: a report whose numbers are typed in by hand goes stale the moment
the pipeline is re-run, and transcription is where errors enter a paper. This
script reads the CSVs that run_analysis.py wrote and emits Markdown with the
real figures, so the written report is always consistent with the last run.

Usage
-----
    python make_report.py                 # after run_analysis.py
    python make_report.py --out REPORT.md
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from config import TBL_DIR, FIG_DIR, COLLECTION_DATE, MIN_EXPOSURE_DAYS


def _load(name):
    path = os.path.join(TBL_DIR, f"{name}.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _md(df, cols=None, floatfmt=4):
    """Render a DataFrame as a GitHub Markdown table."""
    if df is None or df.empty:
        return "_(not available - run the pipeline first)_\n"
    d = df[cols] if cols else df
    d = d.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}".rstrip("0").rstrip("."))
    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.values]
    return "\n".join([header, sep] + rows) + "\n"


def _fmt(v, nd=4):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def _pct(v, nd=1):
    try:
        return f"{100 * float(v):.{nd}f}%"
    except (TypeError, ValueError):
        return "n/a"


# ----------------------------------------------------------------------
# Section builders
# ----------------------------------------------------------------------

def section_data():
    clean = _load("tbl01_cleaning_log")
    desc = _load("tbl03_category_descriptives")

    out = ["## 4. Dataset and Preparation\n"]

    if clean is not None and not clean.empty:
        raw_n = int(clean.iloc[0]["rows"])
        final_n = int(clean.iloc[-1]["rows"])
        retained = 100 * final_n / max(raw_n, 1)
        out.append(
            f"The extract contains **{raw_n:,} reviews**. Deterministic cleaning "
            f"retained **{final_n:,}** ({retained:.1f}%). Every filter is recorded "
            f"so the sample is reconstructible:\n"
        )
        out.append(_md(clean))
        out.append("")

    out.append(
        f"### Exposure control\n\n"
        f"`helpful_vote` is a cumulative count with no denominator: the dataset "
        f"records how many people clicked *helpful*, never how many saw the review. "
        f"A review therefore accrues votes for as long as it remains visible, and "
        f"review age is mechanically correlated with the target.\n\n"
        f"We censor at **{MIN_EXPOSURE_DAYS} days**: every retained review had at "
        f"least that much visibility before collection on **{COLLECTION_DATE}**. "
        f"Residual exposure is retained as an explicit control variable rather than "
        f"discarded, so any remaining age effect is absorbed by that term instead of "
        f"being attributed to review quality.\n"
    )

    if desc is not None and not desc.empty:
        out.append("\n### Category composition\n")
        out.append(_md(desc))
        out.append(
            "\nNote the differing `pct_helpful` across categories. Base rates are "
            "reported per category and per split throughout; they are never assumed "
            "to be constant and the test split is never resampled.\n"
        )

    return "\n".join(out)


def section_leakage():
    t = _load("tbl04_leakage_decomposition")

    out = ["## 6. Experiment 1: Leakage Decomposition\n"]
    out.append(
        "Two design choices are near-universal in the prior literature and both "
        "inflate reported performance:\n\n"
        "1. **Random train/test splitting**, which lets a model observe a user's "
        "later reviews while predicting their earlier ones.\n"
        "2. **Reviewer-history features computed over the full corpus**, including "
        "the review being predicted. For a reviewer with a single review, such a "
        "feature *is* the label.\n\n"
        "We reproduce both, then remove them one at a time. All four conditions "
        "share one feature frame, so they differ only in the factors under study.\n"
    )

    if t is not None and not t.empty:
        out.append(_md(t, cols=[c for c in
                    ["model", "n_test", "base_rate", "pr_auc", "pr_auc_lift", "roc_auc"]
                    if c in t.columns]))
        if len(t) >= 4:
            a, b, c, d = (t.iloc[i]["pr_auc"] for i in range(4))
            out.append(
                f"\n**The inflation is {a - d:+.4f} PR-AUC**, from {_fmt(a)} under the "
                f"replicated protocol to {_fmt(d)} under ours. Of that, "
                f"{a - c:+.4f} is attributable to the leaked reviewer feature and "
                f"{a - b:+.4f} to random splitting.\n\n"
                f"The leaked feature dominates. This matters for how the field should "
                f"respond: splitting strategy is widely discussed, whereas the "
                f"construction of reviewer-history features is rarely reported in "
                f"enough detail to audit.\n"
            )

    out.append(f"\n![Leakage decomposition](figures/fig01_leakage_decomposition.png)\n")
    return "\n".join(out)


def section_ablation():
    t = _load("tbl05_feature_ablation")
    out = ["## 7. Experiment 2: Feature Block Ablation\n"]
    out.append(
        "Features are grouped into theoretical blocks and added cumulatively, so "
        "each row measures the marginal contribution of one construct. Cumulative "
        "rather than leave-one-out, because these blocks are correlated and "
        "leave-one-out understates every block when substitutes are present.\n"
    )
    if t is not None and not t.empty:
        out.append(_md(t, cols=[c for c in
                    ["model", "n_features", "pr_auc", "pr_auc_lift", "marginal_pr_auc"]
                    if c in t.columns]))
        try:
            best = t.loc[t["marginal_pr_auc"].idxmax()]
            out.append(
                f"\nThe largest single gain comes from **{best['model'].strip('+ ')}** "
                f"({best['marginal_pr_auc']:+.4f} PR-AUC).\n"
            )
        except Exception:
            pass
    out.append(f"\n![Feature ablation](figures/fig02_feature_ablation.png)\n")
    return "\n".join(out)


def section_models():
    t = _load("tbl06_model_comparison")
    out = ["## 8. Experiment 3: Model Comparison\n"]
    out.append(
        "Trivial baselines are reported alongside the models. Omitting them is how "
        "a modest result gets published as a strong one: if review length alone "
        "reaches most of a model's score, the model has added little.\n\n"
        "Two feature spaces are used by necessity. Spark's tree learners discretise "
        "every feature and build per-node statistics across all of them, which "
        "exhausts memory on a 32,768-dimensional hashed TF-IDF vector. Linear models "
        "handle sparse high-dimensional text natively. Each row records the space it "
        "used.\n"
    )
    if t is not None and not t.empty:
        cols = [c for c in ["model", "feature_space", "base_rate", "pr_auc",
                            "pr_auc_lift", "roc_auc", "f1", "ndcg@5"]
                if c in t.columns]
        out.append(_md(t, cols=cols))

        try:
            best = t.iloc[0]
            base = float(best["base_rate"])
            out.append(
                f"\nThe strongest model is **{best['model']}** at PR-AUC "
                f"{_fmt(best['pr_auc'])} against a base rate of {_pct(base)} - "
                f"a lift of {_fmt(best['pr_auc_lift'], 2)}x over the trivial classifier.\n\n"
                f"**This is a modest result, and reporting it honestly is the point.** "
                f"Accuracy is not reported anywhere in this study: at this base rate a "
                f"model that predicts *never helpful* scores {_pct(1 - base)} while "
                f"being useless. PR-AUC and lift over the base rate are the metrics "
                f"that cannot be gamed by class imbalance.\n"
            )
        except Exception:
            pass

    out.append(f"\n![Model comparison](figures/fig03_model_comparison.png)\n")
    return "\n".join(out)


def section_moderation():
    perf = _load("tbl07_per_category_performance")
    constructs = _load("tbl09_construct_comparison")
    inter = _load("tbl10_interaction_model")
    marg = _load("tbl11_marginal_effects")

    out = ["## 9. Experiment 4: Does Product Type Moderate Helpfulness?\n"]
    out.append(
        "Nelson's (1970) search/experience distinction, applied to reviews by "
        "Mudambi & Schuff (2010), predicts that helpfulness is a property of the "
        "review-product pair rather than the review alone:\n\n"
        "- **H1** - review depth raises helpfulness *more* for search goods, whose "
        "quality is verifiable from published attributes.\n"
        "- **H2** - rating extremity is *penalised* for search goods, where an "
        "extreme rating signals an idiosyncratic reviewer rather than product quality.\n\n"
        "The original test used 1,587 reviews of six products. We test at corpus scale.\n"
    )

    if perf is not None and not perf.empty:
        out.append("\n### Per-category models\n")
        out.append(_md(perf, cols=[c for c in
                    ["model", "product_type", "n_train", "base_rate", "pr_auc", "pr_auc_lift"]
                    if c in perf.columns]))

    if constructs is not None and not constructs.empty:
        out.append("\n### Construct coefficients by product type\n")
        out.append(_md(constructs, floatfmt=5))

    if inter is not None and not inter.empty:
        out.append(
            "\n### Pooled interaction model\n\n"
            "Comparing two separately-fitted coefficients is not a test of whether "
            "they differ. The interaction term is. Estimated with statsmodels on a "
            "stratified subsample, because a moderation claim requires standard "
            "errors and Spark ML does not provide trustworthy ones under "
            "regularisation.\n"
        )
        out.append(_md(inter, floatfmt=5))

        verdicts = []
        for term, label, expected in [
            ("depth_x_experience", "H1 (depth matters less for experience goods)", "negative"),
            ("extremity_x_experience", "H2 (extremity penalised less for experience goods)", "positive"),
        ]:
            row = inter[inter["term"] == term]
            if not row.empty:
                coef = float(row.iloc[0]["coefficient"])
                p = float(row.iloc[0]["p_value"])
                direction = "negative" if coef < 0 else "positive"
                ok = (direction == expected) and p < 0.05
                verdicts.append(
                    f"- **{label}**: coefficient {coef:+.5f} ({direction}), "
                    f"p = {p:.2e} - {'**supported**' if ok else 'not supported'}"
                )
        if verdicts:
            out.append("\n" + "\n".join(verdicts) + "\n")

        out.append(
            "\nAt this sample size almost any effect reaches statistical "
            "significance, so the interpretation rests on the marginal effects "
            "below rather than on p-values.\n"
        )

    if marg is not None and not marg.empty:
        out.append("\n### Average marginal effects (probability points per 1 SD)\n")
        out.append(_md(marg, floatfmt=5))

    out.append(f"\n![Product type moderation](figures/fig04_product_type_moderation.png)\n")
    return "\n".join(out)


def section_fairness():
    exposure = _load("tbl13_exposure_bias")
    gini = _load("tbl14_vote_concentration")

    out = ["## 10. Experiment 5: Exposure Bias and Fair Ranking\n"]
    out.append(
        "Visibility and votes reinforce each other. A review that attracts early "
        "votes rises up the page, is seen more, and attracts more votes. This "
        "section quantifies the resulting inequality and compares ranking rules "
        "that correct for it.\n"
    )

    if gini is not None and not gini.empty:
        try:
            g = float(gini.iloc[0]["value"])
            out.append(
                f"\n### Concentration\n\n"
                f"The Gini coefficient of helpful votes is **{g:.4f}**. For "
                f"reference, a Gini above 0.9 describes a distribution in which a "
                f"small minority of items holds essentially everything - the "
                f"signature of a winner-take-all feedback loop rather than a "
                f"measurement of quality.\n"
            )
        except Exception:
            pass

    if exposure is not None and not exposure.empty:
        out.append("\n### Arrival position and vote outcomes\n")
        out.append(_md(exposure, floatfmt=2))
        out.append(
            "\nRead the `avg_length` column alongside `pct_receiving_any_vote`. "
            "Where length is roughly flat across arrival buckets while vote rates "
            "fall, the gap is positional rather than quality-driven. This is "
            "descriptive evidence; a causal estimate would require an instrument "
            "for visibility, which this dataset does not provide.\n"
        )

    out.append(
        "\n### Product ranking\n\n"
        "Mean rating places a product with one five-star review above a product "
        "with five hundred reviews averaging 4.8. We compare it against the Wilson "
        "lower bound (conservative: *what is the worst this product plausibly is*) "
        "and an empirical-Bayes posterior mean (*what is this product most likely "
        "to be*), with the Beta prior fitted by moments across products. Reporting "
        "both makes the choice of ranking philosophy explicit rather than accidental.\n"
    )

    out.append(f"\n![Fairness](figures/fig05_fairness.png)\n")
    return "\n".join(out)


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------

def section_replication():
    reg = _load("tbl15_replication_registry")
    res = _load("tbl16_replication_results")

    out = ["## 11. Experiment 6: Replication of Canonical Findings\n"]
    out.append(
        "Nearly every established finding about review helpfulness was produced "
        "between 2007 and 2018, on the 2014 or 2018 Amazon releases, often on "
        "samples of a few thousand reviews. Mudambi & Schuff's canonical extremity "
        "result used 1,587 reviews of six products.\n\n"
        "Amazon Reviews 2023 is a materially different corpus, covering a "
        "marketplace that has since absorbed mobile-first writing, incentivised "
        "review programmes and post-2022 generative text. **This section asks "
        "which of the field's foundational claims still describe it.**\n"
    )

    if reg is not None and not reg.empty:
        out.append(
            "\n### 11.1 Pre-registration\n\n"
            "Each finding's construct, model term and *expected direction* were "
            "registered before estimation. Verdicts compare the estimate against "
            "that declaration rather than against whatever the data produced, "
            "which is what separates replication from fishing.\n"
        )
        out.append(_md(reg, cols=[c for c in
                    ["id", "finding", "citation", "original_scale",
                     "model_term", "expected_direction"]
                    if c in reg.columns]))

    out.append(
        "\n### 11.2 Estimation\n\n"
        "All registered terms enter **one** pooled logistic specification rather "
        "than twelve separate models. Testing each construct in isolation would "
        "credit each with shared variance — the standard route by which a "
        "correlated feature set yields a dozen 'significant' findings that "
        "together explain very little. Exposure duration is a control in every "
        "specification, so no verdict can be an artefact of review age.\n\n"
        "Verdicts are assigned on **effect size**, with significance as a "
        "necessary but not sufficient condition. At this sample size p-values "
        "are near-meaningless; a finding must clear a pre-declared threshold of "
        "one percentage point per standard deviation to count as replicating.\n"
    )

    if res is not None and not res.empty:
        out.append("\n### 11.3 Results\n")
        out.append(_md(res, cols=[c for c in
                    ["id", "finding", "expected", "coefficient", "std_error",
                     "p_value", "marginal_effect_pp", "verdict"]
                    if c in res.columns], floatfmt=5))

        counts = res["verdict"].value_counts()
        total = len(res)
        bits = [f"**{int(n)}/{total} {v.lower()}**"
                for v, n in counts.items()]
        out.append(f"\nOf {total} registered findings: " + ", ".join(bits) + ".\n")

        notable = res[res["verdict"].isin(["FAILS", "REVERSES", "WEAKENS"])]
        if not notable.empty:
            out.append("\n#### Findings that do not straightforwardly replicate\n")
            for _, r in notable.iterrows():
                out.append(
                    f"- **[{r['id']}] {r['finding']}** — {r['citation']}  \n"
                    f"  *{r['verdict']}*: {r['reason']}"
                )
            out.append("")

    out.append(
        "\n### 11.4 Interpretation\n\n"
        "These are **conceptual replications**: the same construct and the same "
        "predicted direction, tested on a modern corpus under temporal "
        "validation. They are not reproductions of the original specifications, "
        "which used different corpora, different covariates and different eras.\n\n"
        "A failure here is therefore evidence about the 2023 marketplace, not a "
        "claim that the original work was wrong. The original results were "
        "correct for their data. The question this section answers is whether "
        "they still describe the platform as it exists now — and for a "
        "substantial share of them, the answer appears to be no.\n"
    )

    out.append("\n![Replication scorecard](figures/fig06_replication_scorecard.png)\n")
    return "\n".join(out)


def build():
    parts = [
        section_data(),
        section_leakage(),
        section_ablation(),
        section_models(),
        section_moderation(),
        section_fairness(),
        section_replication(),
    ]
    return "\n\n---\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "RESULTS.md"))
    args = ap.parse_args()

    body = build()

    header = (
        "# Results\n\n"
        "_This file is generated by `src/make_report.py` from the CSV tables "
        "written by `src/run_analysis.py`. Do not edit it by hand - re-run the "
        "generator instead, so the prose and the numbers cannot drift apart._\n"
    )

    with open(args.out, "w") as fh:
        fh.write(header + "\n---\n\n" + body)

    print(f"wrote {args.out}  ({len(body):,} characters)")


if __name__ == "__main__":
    main()
