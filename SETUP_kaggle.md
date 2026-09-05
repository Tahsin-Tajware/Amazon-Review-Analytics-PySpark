# Running this pipeline on Kaggle

Two things need to reach a Kaggle notebook before it can run: the dataset, and `da_common.py`.
This guide covers both, plus how to pass the cleaned dataset from notebook 02 to the ones after it.

---

## 1. Attach the dataset

Upload `Amazon_Reviews.csv` as a Kaggle Dataset once, then use **Add Data** to attach it to every
notebook in the pipeline. Nothing else is needed. `da_common.py` searches `/kaggle/input`
recursively and picks the largest matching CSV, so the exact dataset name and folder do not matter.

---

## 2. Get `da_common.py` into the notebook

Every notebook opens with a bootstrap cell that searches these locations in order:

```
/kaggle/working/repo
/kaggle/working
..
.
/kaggle/input/**/da_common.py
```

Pick whichever option below suits you.

### Option A - Upload the repository as a Kaggle Dataset (no internet needed)

1. Create a new Kaggle Dataset and upload `da_common.py` (uploading the whole repo folder is fine
   too).
2. Attach that dataset to each notebook through **Add Data**.
3. The bootstrap cell finds it under `/kaggle/input/` automatically.

This is the more reliable option, and it works with internet turned off.

### Option B - Clone from GitHub (internet must be on)

Turn on **Settings > Internet** for the notebook, then run this in a cell **above** the bootstrap
cell:

```python
!git clone -q https://github.com/<your-username>/<your-repo>.git /kaggle/working/repo
```

The bootstrap cell checks `/kaggle/working/repo` first, so nothing else changes.

---

## 3. Pass the cleaned dataset between notebooks

Notebook 02 writes `reviews_analytical.parquet` to `/kaggle/working/processed/`. Kaggle keeps
`/kaggle/working` only within a single notebook's session, so notebooks 03 to 08 need it handed
over. Two ways:

### Recommended: let each notebook rebuild it

Do nothing. `load_analytical()` looks for the Parquet, does not find it, and rebuilds it from the
raw CSV using the same cleaning code. Every notebook is self-sufficient. This costs about a minute
of extra preprocessing per notebook and is the simplest path.

### Faster: chain notebook outputs

1. Run notebook 02 and **Save Version** (Save and Run All). Its `/kaggle/working` becomes the
   notebook's output.
2. In notebooks 03 to 08, use **Add Data > Notebook Output** and attach notebook 02's output.
3. `load_analytical()` searches `/kaggle/input/**/reviews_analytical.parquet` and picks it up, so
   preprocessing is skipped entirely.

---

## 4. Suggested run order

| Session | Notebooks | Roughly |
|---|---|---|
| Checkpoint 1 | 01, 02, 03, 04 | 20 to 30 minutes total |
| Checkpoint 2 | 05, 06, 07, 08 | 25 to 40 minutes total |

Notebook 08 is the slowest because the benchmark reads the CSV repeatedly by design, and it also
loads the full file into Pandas for the head-to-head.

Notebook 08 builds its consolidated findings table by reading the result CSVs written by the
earlier notebooks. If you chained notebook outputs in step 3, attach the earlier notebooks'
outputs to notebook 08 as well so those tables are visible to it. If any table is missing, that row
is skipped rather than failing the notebook.

---

## 5. Kaggle-specific notes

- **PySpark is preinstalled.** No `pip install pyspark` is needed. The version is usually 3.5 or
  newer, and everything here works on Spark 3.4 and above.
- **Resources.** A default Kaggle CPU session gives 4 cores and about 30 GB of RAM, which is
  comfortable for 521,607 records. `spark.driver.memory` is set to 8 GB in `da_common.py`; raise it
  if you attach a larger extract.
- **Turn off the GPU.** Nothing in this pipeline uses one, and a GPU session gives fewer CPU cores.
- **vaderSentiment.** Notebook 06 tries to `pip install` it if it is missing. With internet off,
  that step fails cleanly, prints a skip message, and the rest of the notebook continues. Either
  turn internet on for that one notebook, or add vaderSentiment as a dataset.
- **Long output.** Spark prints progress bars to stderr. That is normal and can be ignored.

---

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `FileNotFoundError: da_common.py not found` | Step 2 was skipped. Attach the repo dataset or clone it. |
| `FileNotFoundError: Amazon_Reviews.csv was not found` | The dataset is not attached to this notebook. Use Add Data. |
| Notebook 08 findings table has missing rows | The earlier notebooks' result CSVs are not visible. Attach their outputs, or re-run them in the same session. |
| `OutOfMemoryError` on notebook 08 | The Pandas side of the benchmark holds the full file in memory. Restart the session so nothing else is resident, or lower `FRACTIONS` in the harness cell. |
| Java gateway errors on startup | A previous Spark session is still alive. Restart the kernel; `da_common.get_spark()` reuses any existing session, and a dead one has to be cleared first. |
