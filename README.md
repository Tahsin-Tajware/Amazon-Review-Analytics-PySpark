# Amazon Customer Review Analytics with PySpark

A distributed analytics pipeline over 521,607 real Amazon product reviews, built with PySpark,
Spark SQL and Spark's window-function API, and benchmarked against Pandas.

Built for CSE 4262 Data Analytics Lab, Ahsanullah University of Science and Technology.

---

## Overview

At half a million records a single-machine Pandas script becomes slow and memory-constrained,
which is the gap distributed engines are built to close. This project answers nine descriptive,
business-relevant questions about a real review corpus in PySpark, and closes with a timed
benchmark so the case for distributed processing rests on measurement rather than assumption.

Verified-purchase status is treated as something to measure, not a trust signal. The badge covers
93.1% of this extract, but it can still sit on a manipulated review when a seller supplies a free
unit in exchange for a positive one.

## Dataset

| Property | Value |
|---|---|
| Source | Amazon Reviews 2023, McAuley Lab, UC San Diego |
| Records | 521,607 reviews |
| Span | September 1999 to September 2023 |
| Categories | Cell Phones (79.7%), Video Games (17.6%), Beauty (2.7%) |
| Verified | 485,784 (93.1%) |
| Unique reviewers / products | 504,856 / 227,791 |

The CSV is not committed here (175 MB, over GitHub's file limit). It is published as a Kaggle
Dataset:

**[Amazon Reviews 2023 Three Category Extract](https://www.kaggle.com/datasets/tahsintajware/amazon-reviews-2023-three-category-extract)**

Original source: [McAuley Lab on Hugging Face](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).

## Repository layout

```
.
├── notebooks/                        staged pipeline, notebooks 01 to 08
├── da-lab-project-checkpoint1.ipynb  self-contained run, stages 01 to 04
├── da-lab-project-final.ipynb        self-contained run, all nine objectives
├── da_common.py                      schema, cleaning rules, plot style, savers
├── run.py                            executes the staged pipeline in order
├── SETUP_kaggle.md                   running it on Kaggle
├── requirements.txt
└── Project_Proposal.pdf
```

Two ways to run the same analysis. The `notebooks/` pipeline splits the work into eight stages
that share `da_common.py` and hand data between each other, which is how the project is organised
for development. The two root notebooks are self-contained single-file versions that need no
setup beyond the dataset, which is how it was run and submitted.

## Pipeline

```
Amazon_Reviews.csv
        |
  [01] audit and profiling
        |
  [02] preprocessing + feature engineering  -->  reviews_analytical.parquet
        |
        +--> [03] satisfaction, verified purchase
        +--> [04] helpfulness, temporal trends
        +--> [05] product, customer analytics
        +--> [06] text analytics, lexicon sentiment
        +--> [07] window-function ranking
        |
  [08] benchmark, evaluation, consolidated results
```

Notebook 02 persists the cleaned dataset to Parquet, so notebooks 03 to 08 read identical rows
without reparsing the CSV. If the Parquet is absent, each notebook rebuilds it automatically, so
any stage can be run on its own.

`da_common.py` holds the schema, cleaning rules, feature definitions, plot style and output savers,
so those are defined once and imported everywhere.

## Objectives

| # | Objective | Core Spark techniques |
|---|---|---|
| 1 | Customer satisfaction | `groupBy`, `pivot` |
| 2 | Verified-purchase analysis | Conditional aggregation on rating, helpfulness, length |
| 3 | Helpful-review analysis | Descending `orderBy`, averages by rating |
| 4 | Temporal trends | Spark SQL date functions, time-series aggregation |
| 5 | Product-level analytics | `groupBy` with a minimum-review threshold |
| 6 | Customer activity | `groupBy("user_id")`, bucketed distribution |
| 7 | Text analytics | `RegexTokenizer`, `StopWordsRemover`, `CountVectorizer`, `HashingTF` + `IDF`, VADER |
| 8 | Window-function ranking | `dense_rank`, `row_number`, `rowsBetween` |
| 9 | PySpark vs Pandas benchmark | Timed operations, cold and cached, at three data sizes |

## Key findings

- Ratings are J-shaped rather than bell-shaped: 61.3% five-star, 13.3% one-star, middle ratings
  rare. Average rating alone hides the distribution, so product ranking applies a minimum-review
  threshold.
- Non-verified reviews are roughly 3x longer than verified ones and collect around 3x the helpful
  votes, a far larger gap than the difference in star rating.
- Helpful votes reach only 17.1% of reviews, and older reviews have had longer to collect them,
  so helpfulness is a rare event confounded with review age rather than a continuous quality score.
- Five-star reviews average 164 characters against 202 to 283 for lower ratings. Satisfied
  customers confirm and leave; dissatisfied ones explain.
- The benchmark reproduces the shape the Spark literature predicts: fixed JVM and scheduling
  overhead dominates at small sizes, and Spark's relative cost falls as data grows.

## Running it

**Locally**

```bash
pip install -r requirements.txt
mkdir -p data                 # place Amazon_Reviews.csv here

python run.py --list          # show the pipeline
python run.py                 # run notebooks 01 to 08
python run.py --checkpoint1   # stages 01 to 04 only
python run.py --from 05       # resume from a given stage
```

Java 11 or newer is required. Executed copies land in `executed/`, so the originals stay clean.

**On Kaggle**

Attach the dataset linked above, then follow [SETUP_kaggle.md](SETUP_kaggle.md). The two
self-contained root notebooks need only the dataset and a Run All.

## Outputs

```
output/
├── results/      36 tables as CSV and Parquet
├── figures/      13 charts as 300 dpi PNG
└── processed/    reviews_analytical.parquet
```

`results/final_consolidated_findings.csv` carries one headline result per objective, assembled by
reading the tables the earlier notebooks wrote.

## Evaluation

The objectives are descriptive rather than predictive, so evaluation targets correctness and
reproducibility:

- Every headline aggregate is recomputed in Pandas on a sampled subset and compared with the Spark
  result.
- Row counts are logged before and after each preprocessing step.
- Benchmarks are timed cold and cached, repeated, and reported as a mean with standard deviation.
- The corpus has no sentiment label, so VADER is scored as an agreement rate against star rating
  with a row-normalised confusion matrix, not as accuracy.
- Findings are checked against the published literature on review positivity and review
  manipulation.

## Tech stack

PySpark (Spark SQL, MLlib, window functions), Pandas, NumPy, Matplotlib, PyArrow, vaderSentiment.

## References

1. Hou, Y., Li, J., He, Z., Yan, A., Chen, X., and McAuley, J. *Bridging Language and Items for Retrieval and Recommendation*. arXiv:2403.03952, 2024.
2. Chevalier, J. A., and Mayzlin, D. *The Effect of Word of Mouth on Sales: Online Book Reviews*. Journal of Marketing Research, 43(3), 2006.
3. He, S., Hollenbeck, B., and Proserpio, D. *The Market for Fake Reviews*. Marketing Science, 41(5), 2022.
4. Saumya, S., Roy, P. K., and Singh, J. P. *Review Helpfulness Prediction on E-commerce Websites: A Comprehensive Survey*. Engineering Applications of Artificial Intelligence, 126, 2023.
5. Zaharia, M. et al. *Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing*. NSDI, 2012.
