"""
Main driver. Runs the full study end to end and writes every table and
figure the report needs.

Usage
-----
    cd src
    python run_analysis.py

or inside Colab / Kaggle:

    import sys; sys.path.insert(0, "src")
    from run_analysis import main
    main()
"""

import os
import sys
import time
import json
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import functions as F

from config import (
    DATA_PATH, TBL_DIR, FIG_DIR, MODEL_DIR, RANDOM_SEED,
    COLLECTION_DATE, MIN_EXPOSURE_DAYS,
)
import data_pipeline as dp
import features as feat
import models as mdl
import moderation as mod
import fairness as fair
import replication as rep
import provenance as prov
import segmentation as seg
import graph_analysis as ga
import regression as regr
import benchmark as bench


# ----------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------

RESULTS = {}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
})

TYPE_COLORS = {"search": "#2563eb", "experience": "#db2777", "unmapped": "#94a3b8"}
ACCENT = "#2563eb"


def save_table(obj, name, show=True):
    pdf = obj.toPandas() if hasattr(obj, "toPandas") else pd.DataFrame(obj)
    path = os.path.join(TBL_DIR, f"{name}.csv")
    pdf.to_csv(path, index=False)
    RESULTS[name] = pdf
    print(f"  table -> {name}.csv  ({len(pdf)} rows)")
    if show and len(pdf) <= 30:
        print(pdf.to_string(index=False))
    return pdf


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure -> {name}.png")


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def fig_leakage(table):
    """The headline figure: how much of the published result was artefact."""
    if table is None or table.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels = [m.split(".")[0] for m in table["model"]]
    colors = ["#ef4444", "#f97316", "#f97316", "#16a34a"][:len(table)]
    bars = ax.bar(labels, table["pr_auc"], color=colors)
    for bar, val, base in zip(bars, table["pr_auc"], table["base_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", fontweight="bold")
    ax.axhline(table["base_rate"].iloc[-1], ls="--", color="#64748b",
               label="base rate (trivial classifier)")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Decomposing the inflated result: each fix removes one artefact")
    ax.legend()
    save_fig(fig, "fig01_leakage_decomposition")


def fig_ablation(table):
    if table is None or table.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(range(len(table)), table["pr_auc"], marker="o", color=ACCENT, lw=2)
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels(table["model"], rotation=25, ha="right")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Cumulative contribution of each feature block")
    if "base_rate" in table.columns:
        ax.axhline(table["base_rate"].iloc[0], ls="--", color="#64748b",
                   label="base rate")
        ax.legend()
    save_fig(fig, "fig02_feature_ablation")


def fig_models(table):
    if table is None or table.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    t = table.sort_values("pr_auc")
    colors = ["#94a3b8" if "Baseline" in m or "Trivial" in m else ACCENT
              for m in t["model"]]
    ax.barh(t["model"], t["pr_auc"], color=colors)
    ax.set_xlabel("PR-AUC")
    ax.set_title("Models vs baselines (grey = trivial baseline)")
    for i, v in enumerate(t["pr_auc"]):
        ax.text(v, i, f" {v:.3f}", va="center")
    save_fig(fig, "fig03_model_comparison")


def fig_moderation(coef_df):
    if coef_df is None or coef_df.empty:
        return
    constructs = {"Depth": mod.DEPTH_FEATURES, "Extremity": mod.EXTREMITY_FEATURES}
    fig, axes = plt.subplots(1, len(constructs), figsize=(12, 4.5))
    for ax, (name, feats) in zip(np.atleast_1d(axes), constructs.items()):
        sub = coef_df[coef_df["feature"].isin(feats)]
        if sub.empty:
            continue
        g = sub.groupby(["Category", "product_type"])["coefficient"].mean().reset_index()
        colors = [TYPE_COLORS.get(t, "#94a3b8") for t in g["product_type"]]
        ax.barh(g["Category"], g["coefficient"], color=colors)
        ax.axvline(0, color="#334155", lw=1)
        ax.set_title(f"{name} effect on helpfulness")
        ax.set_xlabel("Standardised coefficient")
    fig.suptitle("Blue = search goods, pink = experience goods",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    save_fig(fig, "fig04_product_type_moderation")


def fig_replication(table):
    """The replication scorecard: effect sizes with verdict colouring.

    Drawn as marginal effects rather than logit coefficients, because
    percentage points are interpretable and log-odds are not.
    """
    if table is None or table.empty or "marginal_effect_pp" not in table.columns:
        return

    t = table.dropna(subset=["marginal_effect_pp"]).copy()
    if t.empty:
        return
    t = t.sort_values("marginal_effect_pp")

    colors = {
        "REPLICATES": "#16a34a",
        "WEAKENS": "#eab308",
        "FAILS": "#94a3b8",
        "REVERSES": "#ef4444",
        "NOT TESTABLE": "#cbd5e1",
    }

    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.55 * len(t))))
    bars = ax.barh(
        [f"[{r.id}] {r.finding}" for r in t.itertuples()],
        t["marginal_effect_pp"],
        color=[colors.get(v, "#94a3b8") for v in t["verdict"]],
    )
    ax.axvline(0, color="#334155", lw=1)

    # The pre-declared practical-significance threshold, shown explicitly so
    # readers can see it was a rule and not a post-hoc judgement.
    for sign in (1, -1):
        ax.axvline(sign * 100 * rep.MIN_MEANINGFUL_AME, ls=":",
                   color="#64748b", lw=1)

    ax.set_xlabel("Average marginal effect (percentage points per 1 SD)")
    ax.set_title("Replication scorecard: canonical findings on Amazon Reviews 2023")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for v, c in colors.items()
               if v in set(t["verdict"])]
    labels = [v for v in colors if v in set(t["verdict"])]
    ax.legend(handles, labels, fontsize=9, loc="lower right")

    plt.tight_layout()
    save_fig(fig, "fig06_replication_scorecard")


def fig_fairness(prod_pdf, exposure_pdf):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    if prod_pdf is not None and not prod_pdf.empty:
        ax[0].scatter(prod_pdf["n"], prod_pdf["avg_rating"], s=6, alpha=0.25,
                      color="#94a3b8", label="naive mean rating")
        ax[0].scatter(prod_pdf["n"], prod_pdf["bayesian"] * 4 + 1, s=6, alpha=0.25,
                      color=ACCENT, label="empirical Bayes (rescaled)")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("Reviews per product (log scale)")
        ax[0].set_ylabel("Score")
        ax[0].set_title("(a) Shrinkage pulls thin products toward the mean")
        ax[0].legend(fontsize=9)

    if exposure_pdf is not None and not exposure_pdf.empty:
        ax[1].bar(exposure_pdf["arrival_bucket"],
                  exposure_pdf["pct_receiving_any_vote"], color="#db2777")
        ax[1].set_ylabel("% receiving any helpful vote")
        ax[1].set_title("(b) Late arrivals are structurally disadvantaged")
        ax[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    save_fig(fig, "fig05_fairness")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def fig_segmentation(summary):
    """Segment sizes against their share of the helpful-vote supply.

    The gap between the two bars is the inequality: a segment whose vote
    share towers over its population share is capturing attention out of
    proportion to its numbers.
    """
    if summary is None or summary.empty:
        return
    if "pct_of_reviewers" not in summary.columns:
        return

    labels = summary.get("segment", summary["prediction"].astype(str))
    x = np.arange(len(summary))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - w / 2, summary["pct_of_reviewers"], w,
           label="% of reviewers", color="#94a3b8")
    ax.bar(x + w / 2, summary["pct_of_all_votes"], w,
           label="% of all helpful votes", color=ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Percent")
    ax.set_title("Who captures the helpful votes?")
    ax.legend()
    save_fig(fig, "fig07_reviewer_segments")


def fig_benchmark(table):
    """Scaling curves, one panel per workload, log-log."""
    if table is None or table.empty:
        return
    workloads = list(table["workload"].unique())
    fig, axes = plt.subplots(1, len(workloads), figsize=(5 * len(workloads), 4.2))
    axes = np.atleast_1d(axes)

    for ax, wl in zip(axes, workloads):
        sub = table[table["workload"] == wl].sort_values("rows")
        ax.plot(sub["rows"], sub["pandas_sec"], marker="o",
                color="#db2777", label="Pandas")
        ax.plot(sub["rows"], sub["pyspark_sec"], marker="s",
                color=ACCENT, label="PySpark")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Rows"); ax.set_ylabel("Seconds")
        ax.set_title(wl)
        ax.legend(fontsize=9)

    fig.suptitle("Where the distributed framework starts to pay off",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    save_fig(fig, "fig08_scaling_benchmark")


def main(run_text_features=True, run_interaction=True, run_benchmark=True,
         data_path=None):
    """
    Run the full study.

    data_path overrides the corpus location explicitly. This parameter exists
    because config.DATA_PATH is resolved from the environment at IMPORT time,
    and build_dataset's default argument is bound when its module is defined.
    Setting AMAZON_DATA_PATH after either has happened therefore has no effect
    - a failure mode that silently falls back to whatever path config was
    built with. Passing the path in removes the ordering dependency entirely.
    """
    t_start = time.time()

    data_path = data_path or os.environ.get("AMAZON_DATA_PATH") or DATA_PATH

    banner("AMAZON REVIEW HELPFULNESS - LEAKAGE-FREE ANALYSIS")
    print(f"Data       : {data_path}")

    if not (os.path.exists(data_path) or data_path.startswith(("hdfs:", "s3:"))):
        raise FileNotFoundError(
            f"Corpus not found at {data_path}\n"
            f"Pass the path explicitly:  main(data_path='/path/to/corpus.parquet')"
        )
    print(f"Collection : {COLLECTION_DATE}")
    print(f"Exposure   : >= {MIN_EXPOSURE_DAYS} days")
    print(f"Outputs    : {TBL_DIR}")

    spark = dp.get_spark()
    print(f"Spark      : {spark.version}")

    # ---------------- Stage 1: data ----------------
    banner("STAGE 1 - DATA PREPARATION")
    df, train, val, test, clean_log = dp.build_dataset(spark, path=data_path)

    save_table(pd.DataFrame(clean_log, columns=["step", "rows"]),
               "tbl01_cleaning_log")

    # ---------------- Stage 2: features ----------------
    banner("STAGE 2 - FEATURE ENGINEERING")
    df = feat.add_features(df).cache()
    train = feat.add_features(train).cache()
    val = feat.add_features(val).cache()
    test = feat.add_features(test).cache()

    n_total = df.count()
    n_train, n_val, n_test = train.count(), val.count(), test.count()
    print(f"Analysis sample: {n_total:,} reviews")
    print(f"  train {n_train:,} | val {n_val:,} | test {n_test:,}")

    # Provenance FIRST: what this sampling design permits, before any
    # conclusion is drawn from data that may not support it.
    prov_summary, prov_checks = prov.diagnose(df)
    save_table(prov_summary, "tbl00_provenance", show=False)
    save_table(prov_checks, "tbl00b_provenance_checks", show=False)
    save_table(prov.reviewer_frequency_table(df), "tbl00c_reviewer_frequency")

    save_table(feat.feature_summary(df), "tbl02_feature_summary", show=False)

    # Descriptives by category
    desc = (
        df.groupBy("Category", "product_type")
        .agg(
            F.count("*").alias("reviews"),
            F.round(F.avg("rating"), 3).alias("avg_rating"),
            F.round(100 * F.avg("label"), 2).alias("pct_helpful"),
            F.round(F.avg("helpful_vote"), 3).alias("avg_votes"),
            F.round(F.avg("review_length"), 0).alias("avg_length"),
            F.round(100 * F.avg("verified_int"), 1).alias("pct_verified"),
        )
        .orderBy(F.desc("reviews"))
    )
    save_table(desc, "tbl03_category_descriptives")

    # ---------------- Stage 3: leakage ----------------
    leak_table = mdl.experiment_leakage(spark, df, train, val, test)
    save_table(leak_table, "tbl04_leakage_decomposition")
    fig_leakage(leak_table)

    # ---------------- Stage 4: ablation ----------------
    abl_table = mdl.experiment_ablation(train, test, use_text=run_text_features)
    save_table(abl_table, "tbl05_feature_ablation")
    fig_ablation(abl_table)

    # ---------------- Stage 5: models ----------------
    model_table, trained, te_feat = mdl.experiment_models(
        train, test, use_text=run_text_features)
    save_table(model_table, "tbl06_model_comparison")
    fig_models(model_table)

    # Persist the best model for the demo tool
    try:
        best_name = model_table.iloc[0]["model"]
        if best_name in trained:
            model, feature_model = trained[best_name]
            model.write().overwrite().save(os.path.join(MODEL_DIR, "classifier"))
            feature_model.write().overwrite().save(os.path.join(MODEL_DIR, "features"))
            with open(os.path.join(MODEL_DIR, "meta.json"), "w") as fh:
                json.dump({"best_model": best_name,
                           "threshold": float(model_table.iloc[0].get("threshold", 0.5))},
                          fh, indent=2)
            print(f"\n  saved best model ({best_name}) -> {MODEL_DIR}")
    except Exception as e:
        print(f"  model saving skipped: {type(e).__name__}: {e}")

    # ---------------- Stage 6: moderation ----------------
    perf_table, coef_df = mod.per_category_models(train, test)
    save_table(perf_table, "tbl07_per_category_performance")
    save_table(coef_df, "tbl08_per_category_coefficients", show=False)

    construct_table = mod.compare_constructs(coef_df)
    save_table(construct_table, "tbl09_construct_comparison")
    fig_moderation(coef_df)

    if run_interaction:
        summary, sm_model = mod.interaction_model(train)
        if summary is not None:
            save_table(summary, "tbl10_interaction_model")
            me = mod.marginal_effects(sm_model)
            if me is not None:
                save_table(me, "tbl11_marginal_effects", show=False)

    # ---------------- Stage 7: fairness ----------------
    banner("EXPERIMENT 5 - EXPOSURE BIAS AND FAIR RANKING")
    prod, prior = fair.rank_products(df)
    prod = prod.cache()

    rankings = fair.ranking_comparison(prod)
    for key, tbl in rankings.items():
        save_table(tbl, f"tbl12_ranking_{key}", show=False)

    exposure_pdf = fair.exposure_bias_analysis(df)
    save_table(exposure_pdf, "tbl13_exposure_bias")

    gini = fair.gini_coefficient(df)
    save_table(pd.DataFrame([{"metric": "gini_helpful_votes", "value": gini}]),
               "tbl14_vote_concentration")

    prod_sample = prod.sample(False, min(1.0, 20000 / max(prod.count(), 1)),
                              seed=RANDOM_SEED).toPandas()
    fig_fairness(prod_sample, exposure_pdf)

    # ---------------- Stage 8: replication ----------------
    # The headline contribution: do the field's canonical findings still hold?
    reg_tbl, rep_tbl, rep_summary = rep.run(train)
    save_table(reg_tbl, "tbl15_replication_registry", show=False)
    if not rep_tbl.empty:
        save_table(rep_tbl, "tbl16_replication_results", show=False)
        print("\n" + rep_tbl[["id", "finding", "expected", "coefficient",
                              "marginal_effect_pp", "verdict"]].to_string(index=False))
    if not rep_summary.empty:
        save_table(rep_summary, "tbl17_replication_summary")
    fig_replication(rep_tbl)

    # ---------------- Stage 9: clustering (Week 3) ----------------
    sil_tbl, seg_tbl = seg.run(df)
    if not sil_tbl.empty:
        save_table(sil_tbl, "tbl18_segmentation_silhouette")
    if not seg_tbl.empty:
        save_table(seg_tbl, "tbl19_reviewer_segments", show=False)
        fig_segmentation(seg_tbl)

    # ---------------- Stage 10: graph analytics (Week 4) ----------------
    try:
        g_summary, g_top, g_comps = ga.run(df)
        if not g_summary.empty:
            save_table(g_summary, "tbl20_graph_summary")
        if not g_top.empty:
            save_table(g_top, "tbl21_top_reviewers_pagerank", show=False)
        if not g_comps.empty:
            save_table(g_comps, "tbl22_graph_components", show=False)
    except Exception as e:
        print(f"\n[graph analytics] FAILED: {type(e).__name__}: {e}")

    # ---------------- Stage 11: regression (Week 3) ----------------
    try:
        reg_tbl = regr.run(train, test, use_text=False)
        if not reg_tbl.empty:
            save_table(reg_tbl, "tbl23_regression_comparison")
    except Exception as e:
        print(f"\n[regression] FAILED: {type(e).__name__}: {e}")

    # ---------------- Stage 12: scaling benchmark ----------------
    if run_benchmark:
        try:
            bench_tbl, cross_tbl = bench.run(spark, df)
            if not bench_tbl.empty:
                save_table(bench_tbl, "tbl24_scaling_benchmark", show=False)
                fig_benchmark(bench_tbl)
            if not cross_tbl.empty:
                save_table(cross_tbl, "tbl25_benchmark_crossover")
        except Exception as e:
            print(f"\n[benchmark] FAILED: {type(e).__name__}: {e}")

    # ---------------- Done ----------------
    banner("COMPLETE")
    elapsed = time.time() - t_start
    print(f"Runtime: {elapsed / 60:.1f} minutes")
    print(f"\n{len(RESULTS)} tables in {TBL_DIR}")
    for name in sorted(RESULTS):
        print(f"  {name}")
    print(f"\nFigures in {FIG_DIR}")

    spark.stop()
    return RESULTS


if __name__ == "__main__":
    main()
