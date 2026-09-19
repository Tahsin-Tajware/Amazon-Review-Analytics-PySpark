# Predicting Review Helpfulness Before the First Vote

### A leakage-free study of Amazon Reviews 2023 using PySpark

**Course:** CSE 4262 — Data Analytics Lab
**Institution:** Department of Computer Science and Engineering
**Group:** Gr-03 / Gr-06
**Submission date:** _[fill in]_

**Group members:** _[names and IDs]_

**Supervisor:** _[name]_

---

## Abstract

Online marketplaces must decide where to place a review the moment it is posted, before any reader has voted on it. This project asks whether that decision can be made from the review itself, and whether the answer depends on the kind of product being sold.

We identify a validity problem that affects much of the existing literature on this task. Helpful votes accumulate with exposure: a review posted in 2015 has had eight years to collect clicks, one posted in 2023 has had weeks. Models given any time-correlated feature learn to detect review age and report strong performance without having learned anything about review quality. A recent study on this same dataset reported 96.91% accuracy from a reviewer-history feature that, for single-review reviewers, is the prediction target restated.

We build a pipeline that eliminates this class of error structurally rather than by convention, quantify how much prior performance is attributable to it, and re-evaluate the task honestly. We then test whether the determinants of helpfulness differ between search goods and experience goods, and quantify the inequality in how helpful votes are distributed.

All numerical results referenced here are in [`RESULTS.md`](RESULTS.md), which is generated directly from the analysis output so that prose and figures cannot drift apart.

---

## 1. Introduction

Amazon hosts hundreds of millions of reviews. Nobody reads more than a handful per product, so which reviews appear first determines what most shoppers actually learn. Amazon surfaces reviews partly by how many readers marked them "helpful."

This creates a bootstrapping problem. A brand-new review has zero votes. To place it well, the platform must estimate its value from the review itself. That estimation problem is the subject of this project.

It matters to three groups. **Shoppers** get better information when useful reviews surface quickly. **Reviewers** — particularly new ones — deserve to be judged on what they wrote rather than on when they happened to arrive. **The platform** needs a ranking signal that works before social proof exists.

Data analytics is suited to the problem because the relationship between a review's text and its eventual reception is statistical rather than deterministic, and because the corpus is large enough that patterns invisible in a sample of a few thousand reviews become measurable.

---

## 2. Problem Statement

**The existing problem.** Review ranking is circular. Reviews that receive early votes are shown more prominently, which earns them more votes. A review arriving later on an established product competes for attention it will never receive, regardless of its quality. New contributors are therefore structurally disadvantaged, and the platform's own signal reinforces the disadvantage.

**Why it matters.** If helpfulness ranking is a popularity feedback loop rather than a quality measurement, then the reviews shoppers see are not the most useful ones — they are the oldest ones that got lucky early.

**What is currently missing.** Two things. First, a model that can estimate helpfulness at the moment of posting, using only what is knowable then. Second — and this is the contribution we consider most important — an honest account of how well that can actually be done. The published literature reports strong performance, but much of it is obtained under experimental protocols that let information about the future leak into training.

**How analytics supports the decision.** A calibrated pre-vote helpfulness estimate would let a platform give promising new reviews an initial placement, breaking the cold-start loop. This project establishes what such an estimate can and cannot deliver.

---

## 3. Project Objectives

1. **To construct a leakage-free analysis pipeline** for Amazon Reviews 2023, with temporal validation and exposure control enforced structurally rather than by convention.
2. **To quantify the inflation** in reported performance attributable to random splitting and to reviewer-history features computed with lookahead.
3. **To engineer and evaluate** linguistic, structural, affective, contextual and reviewer-history feature blocks, measuring the marginal contribution of each.
4. **To compare classification models** under honest evaluation, using metrics appropriate to an imbalanced target and including trivial baselines.
5. **To test whether product type moderates** the determinants of helpfulness, contrasting search goods against experience goods.
6. **To quantify exposure bias** in the distribution of helpful votes and compare fairer product ranking rules.
7. **To re-test the canonical findings of the review literature** on this corpus, with each claim pre-registered and assigned a replication verdict.
8. **To deliver a working demonstration** that scores a draft review and explains its estimate.

A self-assessment against each objective appears in Section 13.

---

## 4. Dataset and Preparation

See [`RESULTS.md` §4](RESULTS.md) for the cleaning log, retention rates and category composition produced by the actual run.

**Source.** Amazon Reviews 2023, McAuley Lab, UCSD — https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

The full release contains 571.54M reviews across 33 categories, collected through September 2023. This project uses three complete category dumps — Cell Phones and Accessories (20.8M), Video Games (4.6M) and All Beauty (701.5K) — downloaded directly from the McAuley Lab distribution. The three categories were not chosen arbitrarily: they span the search/experience distinction that Objective 5 tests.

### 4.0 Sampling design — and why it is the first thing reported

An earlier iteration of this project used a Kaggle-distributed extract of the same three categories. That extract turned out to be a **review-level random sample**, and this is not a cosmetic detail:

| | Kaggle extract | This corpus |
| --- | --- | --- |
| Reviews per reviewer | 1.029 | **1.707** |
| Single-review reviewers | 97.4% | 69.1% |
| Reviews per product | 2.56 | **5.67** |

Sampling reviews independently destroys user-level structure. If a corpus is sampled at rate *p* and a reviewer has *k* reviews, the probability they appear at least twice is 1−(1−p)^k−kp(1−p)^(k−1), which for p = 0.02 and k = 10 is about 1.6%. Near-universal singleton reviewers are therefore what that procedure *produces*, whatever the marketplace actually looks like.

Three analyses depend on reviewer history — the reviewer feature block, reviewer segmentation, and the reviewer–product graph — and none of them are testable on such a file. A null result there would say nothing about the world, only that the predictor was almost constant.

**We therefore rebuilt the corpus from complete category dumps.** Where a category exceeded the compute budget (Cell Phones, at 20.8M reviews), we subsampled **by user**: a random set of reviewers was drawn and *every* review each of them wrote was retained. This changes the size of the data without changing its structure, which review-level sampling cannot claim.

`src/provenance.py` runs this diagnostic before any modelling and prints an explicit table of which analyses the sampling design supports. `src/replication.py` enforces the same constraint at the verdict level: a finding whose data precondition fails is reported **NOT TESTABLE**, never **FAILS**.

**Format.** CSV. Publicly available for research use.

**Target variable.** `helpful_vote` — a cumulative count of readers who marked the review helpful. Binarised at ≥ 1 for classification.

### 4.1 Data attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `asin` | string | Product identifier |
| `parent_asin` | string | Product group identifier; variants of one product share it |
| `user_id` | string | Pseudonymous reviewer identifier |
| `rating` | double | Star rating, 1–5 |
| `title` | string | Review headline |
| `text` | string | Review body |
| `timestamp` | long | Posting time, milliseconds since epoch |
| `helpful_vote` | int | Cumulative helpful votes — **the target** |
| `verified_purchase` | boolean | Whether Amazon confirmed the purchase |
| `images` | string | Serialised list of attached review images |
| `Category` | string | Product category |

### 4.2 The central methodological choice

`helpful_vote` is a count with no denominator. The dataset records how many people clicked *helpful*; it does not record how many people saw the review. Exposure is unobserved, and it grows with time.

Two consequences follow. A recent review labelled "not helpful" may simply be a review nobody has seen yet — the label is censored, not negative. And any feature correlated with time becomes a proxy for exposure, letting a model score well by detecting age.

We address this in two ways:

**Censoring.** Reviews with fewer than 365 days of visibility before the September 2023 collection date are excluded, so every retained review had a comparable minimum opportunity to accumulate votes.

**Conditioning.** Residual exposure duration is retained as an explicit control feature on a log scale, so that remaining age effects are absorbed by that term rather than attributed to text quality.

### 4.3 Temporal splitting

The data is split strictly by time — earliest 70% for training, next 15% for validation, final 15% for testing. The model is trained on the past and evaluated on the future, which is the only arrangement matching deployment.

Cut points are data-driven quantiles rather than fixed dates, because review volume grows sharply over the corpus and fixed dates would leave the early split nearly empty.

**Base rates differ across the resulting splits. This is real temporal drift and is reported rather than corrected.** The test split is never resampled: class weighting is applied to training data only. A test set rebalanced toward the positive class produces a precision-recall score that cannot be compared to anything, including the deployment setting.

### 4.4 Reviewer history without lookahead

Reviewer and product track records are computed over an expanding window bounded at the row before the current one. A reviewer's first review correctly receives no history, flagged with an explicit indicator rather than an imputed mean — imputing a population mean there would smuggle corpus-level information into first-time reviewers.

This is the precise step whose omission invalidated arXiv:2412.02884. That study computed each reviewer's average helpful votes across all of their reviews, including the one being predicted.

---

## 5. Methodology

```
Problem definition
  → Dataset acquisition
  → Data cleaning (logged, reconstructible)
  → Exposure control (censoring + conditioning)
  → Target construction
  → Temporal split (train / validation / test)
  → Feature engineering (six theoretical blocks, no lookahead)
  → Experiment 1: leakage decomposition
  → Experiment 2: feature block ablation
  → Experiment 3: model comparison against baselines
  → Experiment 4: product-type moderation
  → Experiment 5: exposure bias and fair ranking
  → Model persistence and demonstration tool
  → Report generation
```

### 5.1 Feature blocks

Features are grouped so that ablations add or remove a whole theoretical construct at a time. This is what turns a results table into an argument: a claim about *depth* is a claim about a block, not about one column.

| Block | Contents | Construct |
| --- | --- | --- |
| Structural | length, word and sentence counts, title length, images | Effort and depth |
| Lexical | type-token ratio, long-word ratio, readability, punctuation | Sophistication and clarity |
| Affective | rating, extremity, sentiment polarity and ratios | Emotional stance |
| Contextual | verified purchase, deviation from product consensus | Fit to the product page |
| Reviewer | prior review count, prior average helpfulness and length | Author track record |
| Exposure | log exposure days | Visibility control |
| Text | hashed TF-IDF over title and body | Content |

Sentiment uses a compact opinion lexicon applied with native Spark functions rather than a Python UDF, which avoids serialising millions of review strings into Python workers — the usual reason PySpark text pipelines stall. A limitation of this choice is noted in Section 12.

### 5.2 Evaluation

**Accuracy is not reported anywhere in this study.** At the observed base rate, a model predicting "never helpful" achieves roughly 90% accuracy while being useless. Any paper reporting accuracy on this task is reporting the base rate.

| Metric | Why |
| --- | --- |
| **PR-AUC** | Primary. Sensitive to performance on the minority class. |
| **PR-AUC lift** | PR-AUC divided by base rate. Makes "learned almost nothing" visible. |
| ROC-AUC | Secondary. Threshold-free ranking quality. |
| F1 / precision / recall | Reported at the *swept optimal* threshold, not 0.5. |
| Lift @ 10% | If we surface the top 10% of reviews, how much better than random? |
| **NDCG@5 within product** | The deployment-realistic metric: ordering reviews on one product page. |

Within-product NDCG deserves emphasis. A platform does not classify reviews in isolation; it orders the reviews on a single product page. Reviews competing on the same page also share broadly the same visibility conditions, so their relative vote counts reflect relative quality far more cleanly than absolute counts.

---

## 6–11. Results

All experimental results, tables and figures are in **[`RESULTS.md`](RESULTS.md)**, generated directly from the pipeline output.

- §6 — Leakage decomposition
- §7 — Feature block ablation
- §8 — Model comparison
- §9 — Product-type moderation
- §10 — Exposure bias and fair ranking
- **§11 — Replication of canonical findings** ← the study's principal result

### On Experiment 6

The first five experiments establish that this corpus can be analysed without the validity problems that affect much of the prior literature. Experiment 6 uses that capability for the question that motivates the study.

Twelve findings drawn from the review literature — Mudambi & Schuff's depth and extremity effects, Korfiatis et al. on readability, Ghose & Ipeirotis on reviewer history, Sen & Lerman's negativity bias, and others — are each registered with a citation, the original claim, the scale at which it was established, the model term carrying the construct, and the **expected direction**. All of this is fixed before estimation.

Every registered term then enters a single pooled logistic specification, and each finding receives one of four verdicts: **replicates**, **weakens**, **fails**, or **reverses**.

Two design choices deserve note. First, one model rather than twelve: separate models would credit each construct with variance it shares with the others, which is the standard route by which a correlated feature set produces a dozen "significant" results that jointly explain very little. Second, verdicts turn on **effect size**, not significance. At this sample size almost any non-zero effect is significant, so a finding must clear a threshold of one percentage point per standard deviation — declared in advance — to count as replicating.

---

## 11. Tools and Technologies

| Component | Choice |
| --- | --- |
| Language | Python 3.9+ |
| Distributed processing | Apache Spark 3.5 (PySpark) |
| Machine learning | Spark MLlib |
| Statistical inference | statsmodels |
| Visualisation | Matplotlib |
| Environment | Google Colab / Kaggle |
| Version control | Git |

**On the choice of PySpark.** Spark is used for the full pipeline because the operations that dominate runtime — windowed history computation over user and product partitions, corpus-wide TF-IDF, grouped aggregation across millions of rows — are exactly what its execution model is built for. The expanding-window history features in particular are awkward and memory-hungry in Pandas and natural in Spark.

Statistical inference is the deliberate exception. Spark MLlib does not provide trustworthy standard errors under regularisation, and a moderation claim requires them, so the interaction model is estimated with statsmodels on a stratified subsample. Using the right tool for each task is a methodological choice, not an inconsistency.

---

## 12. Ethical, Privacy and Validity Considerations

**Personal data.** The dataset is public and released for research. Reviewer identifiers are pseudonymous hashes. No names, addresses or contact details are present. No attempt is made to re-identify reviewers, and no reviewer-level results are reported that could single out an individual.

**Fairness — a finding, not a disclaimer.** Section 10 shows that helpful votes are extremely concentrated and that late-arriving reviews receive systematically fewer votes at comparable length. A ranking system trained on this signal will learn to reproduce that disadvantage. We report this as a measured property of the platform's feedback mechanism rather than as boilerplate.

**Dual use.** A model that predicts which reviews will be found helpful could be used to optimise fraudulent reviews for visibility. We consider the risk modest — the model's performance is low in absolute terms, and its guidance amounts to "write in detail, be specific, attach a photo," which is also what genuine reviewers should do. We note it explicitly rather than ignoring it.

### Limitations

1. **Category-level product typing is a simplification.** Assigning an entire category to "search" or "experience" ignores within-category variation; not every phone accessory is utilitarian. A continuous proxy such as price would be more defensible and is the first extension we would make.
2. **The sentiment lexicon is compact and domain-general.** A trained sentiment model would measure the affective construct better. Results involving sentiment should be read as indicative.
3. **Exposure is controlled, not measured.** View counts are not published. Censoring and conditioning reduce the confound; they cannot eliminate it.
4. **Exposure-bias evidence is descriptive.** Establishing that arrival position *causes* lower vote rates would require an instrument for visibility, which this dataset does not provide.
5. **Syllable counting for readability is approximated** by vowel groups rather than a pronunciation dictionary.
6. **Tree models and linear models use different feature spaces** for the memory reasons given in §8, so their comparison is informative rather than strictly controlled.

---

## 13. Self-Evaluation Against Objectives

### Headline numbers

| Result | Value |
| --- | --- |
| Corpus analysed | 2,610,062 reviews (3.26M before cleaning) |
| Reviews per reviewer | 1.707 — user structure intact |
| Leakage inflation | **+0.205 PR-AUC** (0.547 → 0.342) |
| Canonical findings replicating | **5 of 12** |
| Findings that *reverse* | **3 of 12** |
| Gini of helpful votes | 0.928 — top 1% hold 49.2% |
| Best model | GBT, PR-AUC 0.409 (2.22× base rate) |
| Largest single effect | Images, **+17.97pp** |
| PySpark vs Pandas, windowed history | **175× faster** at 500K rows |

| # | Objective | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Leakage-free pipeline | **Met** | `src/data_pipeline.py`; temporal split and prior-only windows enforced in code |
| 2 | Quantify inflation | **Met** | RESULTS §6 — four-condition decomposition with attribution |
| 3 | Feature blocks and marginal contribution | **Met** | RESULTS §7 — cumulative ablation |
| 4 | Model comparison under honest evaluation | **Met** | RESULTS §8 — models plus trivial baselines, PR-AUC primary |
| 5 | Product-type moderation | **Met** | RESULTS §9 — per-category models and pooled interaction model |
| 6 | Exposure bias and fair ranking | **Met** | RESULTS §10 — Gini, arrival analysis, Wilson and empirical-Bayes rankings |
| 7 | Replication of canonical findings | **Met** | RESULTS §11 — 12 pre-registered findings with verdicts |
| 8 | Working demonstration | **Met** | `review_scorer.py` |

**Validation of the pipeline itself.** Before running on real data, the full pipeline was executed against a synthetic corpus containing deliberately planted effects of known size and direction. It recovered each one — the leakage inflation, the depth-by-product-type interaction, the extremity-by-product-type interaction, and the neutralisation of the exposure confound. The generator is in `tests/make_synthetic.py` and is reproducible with a fixed seed.

**Validation of the replication module specifically.** Run against that same synthetic corpus, the registry returned `REPLICATES` for exactly the five findings whose effects had been planted (F01 depth, F02 extremity, F03 the depth-by-type interaction, F06 verified purchase, F10 images) and `FAILS` for exactly the seven that had not been planted. **Zero false positives and zero false negatives against known ground truth.** This is the strongest correctness evidence available without an external gold standard, and it is why the verdicts in §11 can be read as measurements rather than as artefacts of specification.

**What we would do differently.** Price as a continuous moderator instead of category-level product typing; a trained sentiment model instead of a lexicon; and sentence embeddings alongside TF-IDF.

---

## 14. Expected Challenges and How They Were Addressed

| Challenge | Response |
| --- | --- |
| Severe class imbalance | PR-AUC as the primary metric; class weights on training data only; threshold swept rather than fixed at 0.5 |
| Target confounded with review age | Exposure censoring plus explicit conditioning |
| Lookahead in reviewer features | Expanding windows bounded at the previous row; explicit no-history indicators |
| Tree learners exhausting memory on sparse text | Trees restricted to the dense numeric space; text handled by linear models; documented rather than hidden |
| Spark text pipelines stalling on Python UDFs | All text features computed with native Spark expressions |
| No standard errors from Spark MLlib | Pooled interaction model estimated in statsmodels on a stratified subsample |
| Verifying correctness without ground truth | Synthetic-data validation with planted effects of known size |

---

## 15. Work Distribution

| Member | Responsibility |
| --- | --- |
| _[name]_ | Data pipeline, cleaning, exposure control, temporal splitting |
| _[name]_ | Feature engineering and ablation experiments |
| _[name]_ | Model training, evaluation metrics, leakage decomposition |
| _[name]_ | Moderation analysis, fairness section, report and demonstration |

_Adjust to reflect actual contributions._

---

## 16. References

1. Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). *Bridging Language and Items for Retrieval and Recommendation.* arXiv:2403.03952.
2. Kirimlioglu, E., Kung, H., & Orlando, D. (2024). *Were You Helpful — Predicting Helpful Votes from Amazon Reviews.* arXiv:2412.02884.
3. Mudambi, S. M., & Schuff, D. (2010). What makes a helpful online review? A study of customer reviews on Amazon.com. *MIS Quarterly*, 34(1), 185–200.
4. Nelson, P. (1970). Information and consumer behavior. *Journal of Political Economy*, 78(2), 311–329.
5. Saumya, S., Roy, P. K., & Singh, J. P. (2023). Review helpfulness prediction on e-commerce websites: A comprehensive survey. *Engineering Applications of Artificial Intelligence*, 126, 107075.
6. Krishnamoorthy, S. (2015). Linguistic features for review helpfulness prediction. *Expert Systems with Applications*, 42(7), 3751–3759.
7. Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158), 209–212.
8. Apache Software Foundation. *Spark MLlib Guide.* https://spark.apache.org/docs/latest/ml-guide.html

---

## Appendix: Reproducing this study

```bash
pip install -r requirements.txt
python src/run_analysis.py      # runs everything, writes tables and figures
python src/make_report.py       # regenerates RESULTS.md from those tables
python review_scorer.py         # interactive demonstration
```

Pipeline validation against synthetic data with known planted effects:

```bash
python tests/make_synthetic.py
AMAZON_DATA_PATH=tests/synthetic_reviews.csv python src/run_analysis.py
```
