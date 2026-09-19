## 4. Dataset and Preparation

The extract contains **3,256,371 reviews**. Deterministic cleaning retained **2,838,837** (87.2%). Every filter is recorded so the sample is reconstructible:

| step | rows |
| --- | --- |
| raw | 3256371 |
| valid_rating | 3256371 |
| valid_timestamp | 3256371 |
| deduplicated | 3218798 |
| has_text | 3215036 |
| text_at_least_20_chars | 2838837 |
| valid_target | 2838837 |


### Exposure control

`helpful_vote` is a cumulative count with no denominator: the dataset records how many people clicked *helpful*, never how many saw the review. A review therefore accrues votes for as long as it remains visible, and review age is mechanically correlated with the target.

We censor at **365 days**: every retained review had at least that much visibility before collection on **2023-09-01**. Residual exposure is retained as an explicit control variable rather than discarded, so any remaining age effect is absorbed by that term instead of being attributed to review quality.


### Category composition

| Category | product_type | reviews | avg_rating | pct_helpful | avg_votes | avg_length | pct_verified |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cell Phones | search | 1397313 | 3.947 | 16.1 | 0.612 | 196 | 94.9 |
| Video Games | experience | 1047218 | 3.975 | 28.14 | 1.463 | 351 | 84.7 |
| Beauty | experience | 165531 | 3.915 | 28.68 | 1.05 | 192 | 89.9 |


Note the differing `pct_helpful` across categories. Base rates are reported per category and per split throughout; they are never assumed to be constant and the test split is never resampled.


---

## 6. Experiment 1: Leakage Decomposition

Two design choices are near-universal in the prior literature and both inflate reported performance:

1. **Random train/test splitting**, which lets a model observe a user's later reviews while predicting their earlier ones.
2. **Reviewer-history features computed over the full corpus**, including the review being predicted. For a reviewer with a single review, such a feature *is* the label.

We reproduce both, then remove them one at a time. All four conditions share one feature frame, so they differ only in the factors under study.

| model | n_test | base_rate | pr_auc | pr_auc_lift | roc_auc |
| --- | --- | --- | --- | --- | --- |
| A. Replication (random split + leaky feature) | 522623 | 0.2171 | 0.5468 | 2.518 | 0.7845 |
| B. Temporal split, leaky feature retained | 392649 | 0.1847 | 0.4106 | 2.223 | 0.7328 |
| C. Random split, honest features | 522623 | 0.2171 | 0.4756 | 2.19 | 0.7351 |
| D. Temporal split, honest features (ours) | 392649 | 0.1847 | 0.3417 | 1.85 | 0.6795 |


**The inflation is +0.2051 PR-AUC**, from 0.5468 under the replicated protocol to 0.3417 under ours. Of that, +0.0712 is attributable to the leaked reviewer feature and +0.1362 to random splitting.

The leaked feature dominates. This matters for how the field should respond: splitting strategy is widely discussed, whereas the construction of reviewer-history features is rarely reported in enough detail to audit.


![Leakage decomposition](figures/fig01_leakage_decomposition.png)


---

## 7. Experiment 2: Feature Block Ablation

Features are grouped into theoretical blocks and added cumulatively, so each row measures the marginal contribution of one construct. Cumulative rather than leave-one-out, because these blocks are correlated and leave-one-out understates every block when substitutes are present.

| model | n_features | pr_auc | pr_auc_lift | marginal_pr_auc |
| --- | --- | --- | --- | --- |
| + exposure | 1 | 0.1965 | 1.064 | 0.1965 |
| + structural | 10 | 0.3452 | 1.869 | 0.1487 |
| + lexical | 18 | 0.3466 | 1.877 | 0.0014 |
| + affective | 25 | 0.349 | 1.89 | 0.0024 |
| + contextual | 30 | 0.3741 | 2.026 | 0.0251 |
| + reviewer | 35 | 0.3748 | 2.029 | 0.0007 |
| + text (TF-IDF) | 32803 | 0.3881 | 2.101 | 0.0133 |


The largest single gain comes from **exposure** (+0.1965 PR-AUC).


![Feature ablation](figures/fig02_feature_ablation.png)


---

## 8. Experiment 3: Model Comparison

Trivial baselines are reported alongside the models. Omitting them is how a modest result gets published as a strong one: if review length alone reaches most of a model's score, the model has added little.

Two feature spaces are used by necessity. Spark's tree learners discretise every feature and build per-node statistics across all of them, which exhausts memory on a 32,768-dimensional hashed TF-IDF vector. Linear models handle sparse high-dimensional text natively. Each row records the space it used.

| model | feature_space | base_rate | pr_auc | pr_auc_lift | roc_auc | f1 | ndcg@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gradient Boosted Trees (numeric) | numeric + category | 0.1847 | 0.4092 | 2.216 | 0.7433 | 0.4415 | 0.6444 |
| Random Forest (numeric) | numeric + category | 0.1847 | 0.3946 | 2.137 | 0.7352 | 0.4339 | 0.6396 |
| Logistic Regression (L2) + text | numeric + category + TF-IDF | 0.1847 | 0.3881 | 2.101 | 0.7307 | 0.4318 | 0.6213 |
| Logistic Regression (L1) + text | numeric + category + TF-IDF | 0.1847 | 0.3796 | 2.055 | 0.73 | 0.43 | 0.6359 |
| Baseline: review length only | single feature | 0.1847 | 0.3577 | 1.937 | 0.6866 |  |  |
| Trivial (predict majority) | single feature | 0.1847 | 0.1847 | 1 | 0.5 |  |  |
| Baseline: exposure days only | single feature | 0.1847 | 0.1713 | 0.928 | 0.4699 |  |  |
| Baseline: rating only | single feature | 0.1847 | 0.1646 | 0.891 | 0.4423 |  |  |


The strongest model is **Gradient Boosted Trees (numeric)** at PR-AUC 0.4092 against a base rate of 18.5% - a lift of 2.22x over the trivial classifier.

**This is a modest result, and reporting it honestly is the point.** Accuracy is not reported anywhere in this study: at this base rate a model that predicts *never helpful* scores 81.5% while being useless. PR-AUC and lift over the base rate are the metrics that cannot be gamed by class imbalance.


![Model comparison](figures/fig03_model_comparison.png)


---

## 9. Experiment 4: Does Product Type Moderate Helpfulness?

Nelson's (1970) search/experience distinction, applied to reviews by Mudambi & Schuff (2010), predicts that helpfulness is a property of the review-product pair rather than the review alone:

- **H1** - review depth raises helpfulness *more* for search goods, whose quality is verifiable from published attributes.
- **H2** - rating extremity is *penalised* for search goods, where an extreme rating signals an idiosyncratic reviewer rather than product quality.

The original test used 1,587 reviews of six products. We test at corpus scale.


### Per-category models

| model | product_type | n_train | base_rate | pr_auc | pr_auc_lift |
| --- | --- | --- | --- | --- | --- |
| Beauty | experience | 101799 | 0.2184 | 0.328 | 1.502 |
| Cell Phones | search | 935713 | 0.1644 | 0.3562 | 2.167 |
| Video Games | experience | 787673 | 0.2149 | 0.4386 | 2.041 |


### Construct coefficients by product type

| construct | search_goods_coef | experience_goods_coef | difference | hypothesis |
| --- | --- | --- | --- | --- |
| depth | 0.19196 | 0.19782 | -0.00587 | H1: depth effect larger for search goods |
| extremity | 0.0428 | 0.04154 | 0.00126 | H2: extremity penalised for search goods |


### Pooled interaction model

Comparing two separately-fitted coefficients is not a test of whether they differ. The interaction term is. Estimated with statsmodels on a stratified subsample, because a moderation claim requires standard errors and Spark ML does not provide trustworthy ones under regularisation.

| term | coefficient | std_error | z | p_value | odds_ratio |
| --- | --- | --- | --- | --- | --- |
| const | -1.57111 | 0.01891 | -83.062 | 0 | 0.2078 |
| log_review_length | 0.7527 | 0.01507 | 49.955 | 0 | 2.1227 |
| rating_extremity | 0.10045 | 0.00892 | 11.26 | 0 | 1.1057 |
| log_exposure_days | 0.216 | 0.00617 | 34.981 | 0 | 1.2411 |
| type_token_ratio | -0.02937 | 0.01159 | -2.534 | 0.01128 | 0.9711 |
| sentiment_polarity | -0.14683 | 0.00563 | -26.09 | 0 | 0.8634 |
| log_prior_review_count | 0.02069 | 0.00566 | 3.656 | 0.00026 | 1.0209 |
| verified_int | -0.13813 | 0.01776 | -7.776 | 0 | 0.871 |
| has_images | 1.21536 | 0.02908 | 41.788 | 0 | 3.3715 |
| is_experience | 0.55703 | 0.01254 | 44.425 | 0 | 1.7455 |
| depth_x_experience | -0.00369 | 0.01308 | -0.282 | 0.77787 | 0.9963 |
| extremity_x_experience | -0.02385 | 0.01171 | -2.037 | 0.04168 | 0.9764 |


- **H1 (depth matters less for experience goods)**: coefficient -0.00369 (negative), p = 7.78e-01 - not supported
- **H2 (extremity penalised less for experience goods)**: coefficient -0.02385 (negative), p = 4.17e-02 - not supported


At this sample size almost any effect reaches statistical significance, so the interpretation rests on the marginal effects below rather than on p-values.


### Average marginal effects (probability points per 1 SD)

| term | marginal_effect | std_error | p_value |
| --- | --- | --- | --- |
| log_review_length | 0.12038 | 0.00234 | 0 |
| rating_extremity | 0.01607 | 0.00143 | 0 |
| log_exposure_days | 0.03454 | 0.00098 | 0 |
| type_token_ratio | -0.0047 | 0.00185 | 0.01133 |
| sentiment_polarity | -0.02348 | 0.0009 | 0 |
| log_prior_review_count | 0.00331 | 0.00091 | 0.00026 |
| verified_int | -0.02209 | 0.00284 | 0 |
| has_images | 0.19437 | 0.00466 | 0 |
| is_experience | 0.08908 | 0.00198 | 0 |
| depth_x_experience | -0.00059 | 0.00209 | 0.77783 |
| extremity_x_experience | -0.00381 | 0.00187 | 0.04166 |


![Product type moderation](figures/fig04_product_type_moderation.png)


---

## 10. Experiment 5: Exposure Bias and Fair Ranking

Visibility and votes reinforce each other. A review that attracts early votes rises up the page, is seen more, and attracts more votes. This section quantifies the resulting inequality and compares ranking rules that correct for it.


### Concentration

The Gini coefficient of helpful votes is **0.9283**: the top 1% of reviews hold **49.2%** of all helpful votes, and the top 10% hold **87.7%**. For reference, a Gini above 0.9 describes a distribution in which a small minority of items holds essentially everything - the signature of a winner-take-all feedback loop rather than a measurement of quality.


### Arrival position and vote outcomes

| arrival_bucket | reviews | avg_helpful_votes | pct_receiving_any_vote | avg_length | avg_rating | avg_exposure_days |
| --- | --- | --- | --- | --- | --- | --- |
| 1-5 (earliest) | 992729 | 1.39 | 28.52 | 254 | 3.88 | 2064 |
| 6-20 | 548936 | 0.97 | 21.72 | 272 | 3.96 | 2091 |
| 21-50 | 353815 | 0.71 | 18.19 | 269 | 4 | 2043 |
| 51-200 | 434136 | 0.61 | 15.16 | 258 | 4.05 | 1995 |
| 200+ (latest) | 280446 | 0.46 | 12.34 | 230 | 4.03 | 1778 |


Read the `avg_length` column alongside `pct_receiving_any_vote`. Where length is roughly flat across arrival buckets while vote rates fall, the gap is positional rather than quality-driven. This is descriptive evidence; a causal estimate would require an instrument for visibility, which this dataset does not provide.


### Product ranking

Mean rating places a product with one five-star review above a product with five hundred reviews averaging 4.8. We compare it against the Wilson lower bound (conservative: *what is the worst this product plausibly is*) and an empirical-Bayes posterior mean (*what is this product most likely to be*), with the Beta prior fitted by moments across products. Reporting both makes the choice of ranking philosophy explicit rather than accidental.


![Fairness](figures/fig05_fairness.png)


---

## 11. Experiment 6: Replication of Canonical Findings

Nearly every established finding about review helpfulness was produced between 2007 and 2018, on the 2014 or 2018 Amazon releases, often on samples of a few thousand reviews. Mudambi & Schuff's canonical extremity result used 1,587 reviews of six products.

Amazon Reviews 2023 is a materially different corpus, covering a marketplace that has since absorbed mobile-first writing, incentivised review programmes and post-2022 generative text. **This section asks which of the field's foundational claims still describe it.**


### 11.1 Pre-registration

Each finding's construct, model term and *expected direction* were registered before estimation. Verdicts compare the estimate against that declaration rather than against whatever the data produced, which is what separates replication from fishing.

| id | finding | citation | original_scale | model_term | expected_direction |
| --- | --- | --- | --- | --- | --- |
| F01 | Review depth increases helpfulness | Mudambi & Schuff (2010), MIS Quarterly 34(1) | 1,587 reviews across 6 products | log_review_length | positive |
| F02 | Rating extremity reduces helpfulness | Mudambi & Schuff (2010), MIS Quarterly 34(1) | 1,587 reviews across 6 products | rating_extremity | negative |
| F03 | Depth matters more for search goods | Mudambi & Schuff (2010), MIS Quarterly 34(1) | 1,587 reviews across 6 products | depth_x_experience | negative |
| F04 | Readability increases helpfulness | Korfiatis, Garcia-Bariocanal & Sanchez-Alonso (2012), ECRA 11(3) | ~37,000 Amazon UK reviews | flesch_reading_ease | positive |
| F05 | Longer reviews are not necessarily better | Fink, Rosenfeld & Ravid (2018), Int. J. Information Management 39 | Amazon reviews, multiple categories | review_length_sq | negative |
| F06 | Verified purchase increases helpfulness | Common assumption; e.g. Filieri et al. (2018), J. Business Research | Varies; typically survey or small-sample studies | verified_int | positive |
| F07 | Reviewer track record predicts helpfulness | Ghose & Ipeirotis (2011), IEEE TKDE 23(10) | ~400 products, Amazon | log_prior_review_count | positive |
| F08 | Lexical diversity increases helpfulness | Krishnamoorthy (2015), Expert Systems with Applications 42(7) | ~1,500 reviews, 2 product categories | type_token_ratio | positive |
| F09 | Deviating from consensus attracts attention | Adapted from Moe & Trusov (2011), J. Marketing Research 48(3) | Bath & body retailer panel data | rating_deviation_from_product | positive |
| F10 | Images increase helpfulness | Ceylan, Diehl & Proserpio (2024), J. Marketing 88(2), and related | Experimental plus field data | has_images | positive |
| F11 | Negative sentiment increases helpfulness | Sen & Lerman (2007), J. Interactive Marketing 21(4) | Laboratory experiments | sentiment_polarity | negative |
| F12 | Early reviews accumulate disproportionate votes | Adapted from Godes & Silva (2012), Marketing Science 31(3) | Book reviews, Amazon | log_arrival_position | negative |


### 11.2 Estimation

All registered terms enter **one** pooled logistic specification rather than twelve separate models. Testing each construct in isolation would credit each with shared variance — the standard route by which a correlated feature set yields a dozen 'significant' findings that together explain very little. Exposure duration is a control in every specification, so no verdict can be an artefact of review age.

Verdicts are assigned on **effect size**, with significance as a necessary but not sufficient condition. At this sample size p-values are near-meaningless; a finding must clear a pre-declared threshold of one percentage point per standard deviation to count as replicating.


### 11.3 Results

Testability preconditions, evaluated before estimation: reviewer history OK (40.1% of reviews have prior reviewer history); product history OK (75.0% have a product consensus to deviate from). Both pass, so no finding is reported NOT TESTABLE.

| id | finding | expected | coefficient | std_error | p_value | marginal_effect_pp | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F01 | Review depth increases helpfulness | positive | 0.79396 | 0.0125 | 0 | 12.464 | REPLICATES |
| F02 | Rating extremity reduces helpfulness | negative | 0.10583 | 0.0074 | 0 | 1.661 | REVERSES |
| F03 | Depth matters more for search goods | negative | -0.0399 | 0.01087 | 0.00024 | -0.626 | WEAKENS |
| F04 | Readability increases helpfulness | positive | 0.0257 | 0.00582 | 0.00001 | 0.403 | WEAKENS |
| F05 | Longer reviews are not necessarily better | negative | 0.07719 | 0.00429 | 0 | 1.212 | REVERSES |
| F06 | Verified purchase increases helpfulness | positive | -0.05272 | 0.015 | 0.00044 | -0.828 | REVERSES |
| F07 | Reviewer track record predicts helpfulness | positive | 0.02109 | 0.00474 | 0.00001 | 0.331 | WEAKENS |
| F08 | Lexical diversity increases helpfulness | positive | 0.03265 | 0.01008 | 0.0012 | 0.513 | FAILS |
| F09 | Deviating from consensus attracts attention | positive | 0.111 | 0.00481 | 0 | 1.743 | REPLICATES |
| F10 | Images increase helpfulness | positive | 1.14448 | 0.02423 | 0 | 17.966 | REPLICATES |
| F11 | Negative sentiment increases helpfulness | negative | -0.09086 | 0.00708 | 0 | -1.426 | REPLICATES |
| F12 | Early reviews accumulate disproportionate votes | negative | -0.47634 | 0.00583 | 0 | -7.478 | REPLICATES |


Of 12 registered findings: **5/12 replicates**, **3/12 reverses**, **3/12 weakens**, **1/12 fails**.


#### Findings that do not straightforwardly replicate

- **[F02] Rating extremity reduces helpfulness** — Mudambi & Schuff (2010), MIS Quarterly 34(1)  
  *REVERSES*: significant in the OPPOSITE direction to the original claim (positive, expected negative)
- **[F03] Depth matters more for search goods** — Mudambi & Schuff (2010), MIS Quarterly 34(1)  
  *WEAKENS*: correct direction, but only 0.63 percentage points per SD - below the 1pp threshold declared in advance
- **[F04] Readability increases helpfulness** — Korfiatis, Garcia-Bariocanal & Sanchez-Alonso (2012), ECRA 11(3)  
  *WEAKENS*: correct direction, but only 0.40 percentage points per SD - below the 1pp threshold declared in advance
- **[F05] Longer reviews are not necessarily better** — Fink, Rosenfeld & Ravid (2018), Int. J. Information Management 39  
  *REVERSES*: significant in the OPPOSITE direction to the original claim (positive, expected negative)
- **[F06] Verified purchase increases helpfulness** — Common assumption; e.g. Filieri et al. (2018), J. Business Research  
  *REVERSES*: significant in the OPPOSITE direction to the original claim (negative, expected positive)
- **[F07] Reviewer track record predicts helpfulness** — Ghose & Ipeirotis (2011), IEEE TKDE 23(10)  
  *WEAKENS*: correct direction, but only 0.33 percentage points per SD - below the 1pp threshold declared in advance
- **[F08] Lexical diversity increases helpfulness** — Krishnamoorthy (2015), Expert Systems with Applications 42(7)  
  *FAILS*: not distinguishable from zero (p = 0.0012)


### 11.4 Interpretation

These are **conceptual replications**: the same construct and the same predicted direction, tested on a modern corpus under temporal validation. They are not reproductions of the original specifications, which used different corpora, different covariates and different eras.

A failure here is therefore evidence about the 2023 marketplace, not a claim that the original work was wrong. The original results were correct for their data. The question this section answers is whether they still describe the platform as it exists now — and for a substantial share of them, the answer appears to be no.


![Replication scorecard](figures/fig06_replication_scorecard.png)
