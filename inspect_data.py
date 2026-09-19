import pandas as pd

DATA_PATH = "/root/.cache/kagglehub/datasets/tahsintajware/amazon-reviews-2023-three-category-extract/versions/1/Amazon_Reviews.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

print("TOTAL ROWS:", f"{len(df):,}")
print("\nCOLUMNS:", list(df.columns))

print("\nROWS PER CATEGORY:")
print(df["Category"].value_counts().to_string())

print("\nHELPFUL VOTES:")
print(f"  mean            {df['helpful_vote'].mean():.3f}")
print(f"  max             {df['helpful_vote'].max()}")
print(f"  pct with >=1    {100 * (df['helpful_vote'] >= 1).mean():.2f}%")
print(f"  pct with >=10   {100 * (df['helpful_vote'] >= 10).mean():.2f}%")

print("\nHELPFUL VOTES >=1, BY CATEGORY:")
print((100 * df.groupby("Category")["helpful_vote"].apply(lambda s: (s >= 1).mean())).round(2).to_string())

ts = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
print("\nDATE RANGE:", ts.min(), "->", ts.max())
print("\nROWS PER YEAR:")
print(ts.dt.year.value_counts().sort_index().to_string())
