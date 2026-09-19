"""
Generate a synthetic dataset matching the Amazon Reviews 2023 extract schema.

Used to smoke-test the pipeline without the real 1M+ row file. Deliberately
builds in the structure the study is designed to detect, so that a correct
pipeline recovers it and a broken one does not:

  * helpful votes accumulate with exposure  (the confound)
  * longer reviews are more helpful         (main effect)
  * depth matters MORE for search goods     (H1)
  * extremity is penalised for search goods (H2)
  * reviewers have persistent skill         (so prior-history features work)
"""
import numpy as np, pandas as pd, random

rng = np.random.default_rng(7)
random.seed(7)

N = 60_000
CATEGORIES = {"Cell Phones": "search", "Beauty": "experience", "Video Games": "experience"}

VOCAB = ("battery life screen quality fits well arrived quickly smells nice colour "
         "payload durable plastic metal charge port cable months weeks price value "
         "recommend disappointed broke works great terrible amazing average fine "
         "comparison alternative brand size weight texture finish").split()

def make_text(n_words):
    return " ".join(random.choice(VOCAB) for _ in range(max(n_words, 4))).capitalize() + "."

start = pd.Timestamp("2014-01-01").value // 10**6
end   = pd.Timestamp("2023-08-01").value // 10**6

n_users = 6000
n_products = 3000
user_skill = rng.normal(0, 0.7, n_users)          # persistent reviewer quality
product_pop = rng.gamma(2.0, 1.0, n_products)     # product popularity

rows = []
for i in range(N):
    cat = random.choice(list(CATEGORIES))
    ptype = CATEGORIES[cat]

    uid = rng.integers(0, n_users)
    pid = rng.integers(0, n_products)

    # Recent reviews are far more common, as in the real corpus
    ts = int(start + (end - start) * (rng.beta(3.2, 1.4)))
    exposure_days = (pd.Timestamp("2023-09-01").value // 10**6 - ts) / 86_400_000

    rating = float(rng.choice([1, 2, 3, 4, 5], p=[.09, .05, .08, .20, .58]))
    extremity = abs(rating - 3.0)

    n_words = int(np.clip(rng.lognormal(3.5, 0.9), 5, 900))
    depth = np.log1p(n_words)

    # ---- latent helpfulness ----
    z = -6.2
    z += 0.34 * depth                                    # main effect
    z += 0.45 * (depth if ptype == "search" else 0.0)    # H1: depth^search
    z -= 0.30 * (extremity if ptype == "search" else 0.0)# H2: extremity penalty
    z += 0.10 * extremity if ptype == "experience" else 0
    z += 0.8 * user_skill[uid]
    z += 0.22 * np.log1p(product_pop[pid])
    verified = rng.random() < 0.82
    z += 0.25 * verified
    has_img = rng.random() < 0.06
    z += 0.45 * has_img

    p = 1 / (1 + np.exp(-z))

    # Votes accumulate with exposure: THE confound the study controls for
    exposure_factor = np.log1p(exposure_days) / np.log1p(3500)
    lam = 3.0 * p * (0.25 + 1.6 * exposure_factor)
    votes = int(rng.poisson(max(lam, 0.01)))

    rows.append({
        "asin": f"B{pid:08d}",
        "helpful_vote": votes,
        "images": "['img']" if has_img else "[]",
        "parent_asin": f"B{pid:08d}",
        "rating": rating,
        "text": make_text(n_words),
        "timestamp": ts,
        "title": make_text(rng.integers(2, 7)),
        "user_id": f"AU{uid:07d}",
        "verified_purchase": bool(verified),
        "Category": cat,
    })

df = pd.DataFrame(rows)
out = "/home/claude/project/tests/synthetic_reviews.csv"
df.to_csv(out, index=False)

print(f"wrote {out}  rows={len(df):,}")
print("pct with >=1 vote:", round(100 * (df.helpful_vote >= 1).mean(), 2))
print(df.Category.value_counts().to_string())
