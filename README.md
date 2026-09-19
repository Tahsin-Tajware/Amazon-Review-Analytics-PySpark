# Predicting Review Helpfulness Before the First Vote

**A leakage-free study of Amazon Reviews 2023 using PySpark**

CSE 4262 — Data Analytics Lab · Gr-03 / Gr-06

---

## What this project does

**Which of the online review literature's canonical findings still hold on a 2023 corpus?**

Almost everything the field believes about what makes a review helpful was established between 2007 and 2018, on the 2014 or 2018 Amazon releases, often on a few thousand reviews. Mudambi & Schuff's canonical extremity result used 1,587 reviews of six products. Amazon Reviews 2023 covers a marketplace that has since absorbed mobile-first writing, incentivised review programmes and post-2022 generative text. **Nobody has re-tested those claims on it.**

This project registers twelve findings from the literature — each with its citation, original claim, model term and *expected direction*, fixed before estimation — and assigns each a verdict: **replicates**, **weakens**, **fails**, or **reverses**.

Getting that answer honestly requires first fixing a validity problem that affects much of the prior work. Helpful votes accumulate with exposure: a 2015 review has had eight years to collect clicks, a 2023 review has had weeks. Models given any time-correlated feature learn to detect review *age* and report strong performance without learning anything about review *quality*. A recent study on this same dataset ([arXiv:2412.02884](https://arxiv.org/abs/2412.02884)) reported 96.91% accuracy from a reviewer-history feature that, for single-review reviewers, is the target restated.

We eliminate that class of error **structurally**, measure how much of the prior performance it accounted for, and only then run the replication.

## Headline results

See **[RESULTS.md](RESULTS.md)** for all tables and figures, generated directly from the pipeline output.

| Finding | Where |
| --- | --- |
| **Which canonical findings replicate, fail, or reverse** | **RESULTS §11** |
| How much of prior performance is methodological artefact | RESULTS §6 |
| Which feature blocks actually contribute | RESULTS §7 |
| Honest model performance against trivial baselines | RESULTS §8 |
| Whether product type moderates helpfulness | RESULTS §9 |
| How unequally helpful votes are distributed | RESULTS §10 |

The full write-up, including objectives, limitations and self-evaluation, is in **[REPORT.md](REPORT.md)**.

For the Week 7 individual viva, the prepared defence and the numbers worth memorising are in **[VIVA_PREP.md](VIVA_PREP.md)**.

> `tests/synthetic_reviews.csv` is not shipped (27 MB). Regenerate it with
> `python tests/make_synthetic.py` — the seed is fixed, so the validation run is reproducible.

---

## Quick start

```bash
git clone <your-repo-url>
cd <repo>
pip install -r requirements.txt
```

### Run the study

```bash
python src/run_analysis.py       # ~tables and figures into output/
python src/make_report.py        # regenerate RESULTS.md from those tables
```

Point it at your own copy of the data with an environment variable:

```bash
AMAZON_DATA_PATH=/path/to/Amazon_Reviews.csv python src/run_analysis.py
```

### Try the demo

```bash
python review_scorer.py
```

Paste a draft review and it estimates the probability that readers will mark it helpful, then explains which characteristics drove the estimate and what would improve it.

```
$ python review_scorer.py --text "I bought this case for my iPhone 14 Pro about
  three months ago. The drop protection is genuinely good..." --rating 4 --category "Cell Phones"

  [####################....................]  51.8%

  Reasonable. Some room to improve.

   > Length      70 words is on the short side. Adding a specific use
                 case or comparison typically helps.
   + Specificity Varied vocabulary, which tends to signal a substantive review.
   > Images      No photos. Adding one is among the cheapest improvements available.
```

### In Colab or Kaggle

```python
import sys; sys.path.insert(0, "src")
from run_analysis import main
main()
```

---

## Repository structure

```
├── README.md                  ← you are here
├── REPORT.md                  ← the full written report
├── RESULTS.md                 ← generated: all results, tables, figures
├── requirements.txt
├── review_scorer.py           ← interactive demonstration tool
├── notebooks/
│   └── analysis.ipynb         ← runnable end-to-end notebook
├── src/
│   ├── config.py              ← every methodological choice, with justification
│   ├── data_pipeline.py       ← loading, cleaning, exposure control, temporal split
│   ├── features.py            ← six feature blocks, no lookahead
│   ├── evaluation.py          ← PR-AUC, lift, within-product NDCG
│   ├── models.py              ← leakage decomposition, ablation, model comparison
│   ├── moderation.py          ← product-type moderation, interaction model
│   ├── fairness.py            ← exposure bias, Wilson and empirical-Bayes ranking
│   ├── run_analysis.py        ← the driver
│   └── make_report.py         ← generates RESULTS.md from the output tables
├── tests/
│   └── make_synthetic.py      ← synthetic data with planted effects, for validation
└── output/
    ├── results/               ← CSV tables
    ├── figures/               ← PNG figures
    └── models/                ← persisted best model
```

**`src/config.py` is worth reading first.** Every parameter a reviewer might question lives there in one place with its justification, rather than scattered through the pipeline as magic numbers.

---

## The three design decisions that matter

### 1. Exposure control

`helpful_vote` is a cumulative count with no denominator — the data records how many people clicked *helpful*, never how many saw the review.

We censor reviews with under 365 days of visibility before the September 2023 collection date, so every retained review had a comparable opportunity to accumulate votes. Residual exposure is kept as an explicit control on a log scale, so remaining age effects are absorbed by that term instead of being credited to text quality.

### 2. Temporal splitting

Train on the earliest 70%, validate on the next 15%, test on the final 15%. The model is trained on the past and evaluated on the future, which is the only arrangement matching deployment.

Base rates differ across the resulting splits. That is genuine temporal drift and we report it. **The test split is never resampled** — class weights apply to training data only, because a rebalanced test set produces a PR-AUC comparable to nothing.

### 3. Reviewer history without lookahead

Prior-review features use an expanding window bounded at the row *before* the current one. A reviewer's first review gets no history, flagged explicitly rather than filled with an imputed mean.

```python
w_user = (Window.partitionBy("user_id")
          .orderBy("timestamp")
          .rowsBetween(Window.unboundedPreceding, -1))   # ← the -1 is the whole point
```

---

## Why accuracy is never reported

At the observed base rate, a model that predicts "never helpful" for every review scores around 90% accuracy while being completely useless.

Primary metric is **PR-AUC**, reported alongside **PR-AUC lift** — PR-AUC divided by the base rate — which makes "this model learned almost nothing" impossible to disguise. We also report **within-product NDCG@5**, because the platform's real task is ordering the reviews on one product page, not classifying reviews in isolation.

---

## Validation

Before running on real data, the pipeline was executed against a synthetic corpus with **deliberately planted effects of known size and direction**, to confirm the code recovers truth rather than producing plausible-looking output.

```bash
python tests/make_synthetic.py
AMAZON_DATA_PATH=tests/synthetic_reviews.csv python src/run_analysis.py
```

| Effect planted | Recovered |
| --- | --- |
| Leaky reviewer feature inflates results | ✓ inflation detected and attributed |
| Depth matters more for search goods | ✓ interaction recovered, correct sign |
| Extremity penalised for search goods | ✓ interaction recovered, correct sign |
| Exposure confound | ✓ neutralised — exposure-only baseline falls to the base rate |

**The replication registry was validated the same way**, and this is the result worth quoting:

> Run against the synthetic corpus, the registry returned `REPLICATES` for **exactly the five findings whose effects had been planted**, and `FAILS` for **exactly the seven that had not**.
>
> Zero false positives. Zero false negatives.

That is the strongest correctness evidence obtainable without an external gold standard, and it is why the verdicts can be read as measurements rather than as artefacts of model specification.

Reproducible with a fixed seed.

---

## Requirements

Python 3.9+, Java 11 or 17 (for Spark). See `requirements.txt`.

Runs on a free Colab CPU instance. Runtime scales with corpus size; the synthetic 60k-row validation run completes in about 8 minutes.

---

## Data

Amazon Reviews 2023 — McAuley Lab, UCSD
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

The full release holds 571.54M reviews across 33 categories through September 2023. This project uses a three-category extract (Cell Phones, Beauty, Video Games), chosen because those three span the search/experience product distinction the moderation analysis tests.

Publicly available for research. Reviewer identifiers are pseudonymous; no re-identification is attempted.

---

## License and citation

Course project, provided for educational use.

If the pipeline is useful to you, please cite the underlying dataset:

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An
          and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```
