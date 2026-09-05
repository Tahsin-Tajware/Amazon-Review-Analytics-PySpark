#!/usr/bin/env python3
"""
Execute the full analytics pipeline, notebook by notebook, in order.

    python run_all.py                  run every notebook
    python run_all.py --from 03        resume from notebook 03
    python run_all.py --only 06 08     run just those notebooks
    python run_all.py --list           show the pipeline without running it

Executed copies are written to executed/ so the originals stay clean for git.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
EXECUTED = os.path.join(REPO, "executed")

PIPELINE = [
    ("01", "01_data_audit_and_profiling.ipynb",             "Data audit and profiling"),
    ("02", "02_preprocessing_and_feature_engineering.ipynb", "Preprocessing and feature engineering"),
    ("03", "03_satisfaction_and_verified_analysis.ipynb",   "Objectives 1 and 2"),
    ("04", "04_helpfulness_and_temporal_analysis.ipynb",    "Objectives 3 and 4"),
    ("05", "05_product_and_customer_analytics.ipynb",       "Objectives 5 and 6"),
    ("06", "06_text_analytics_and_sentiment.ipynb",         "Objective 7"),
    ("07", "07_window_function_ranking.ipynb",              "Objective 8"),
    ("08", "08_benchmark_and_evaluation.ipynb",             "Objective 9 and evaluation"),
]

CHECKPOINT_1 = {"01", "02", "03", "04"}


def run_notebook(filename: str, timeout: int) -> None:
    import nbformat
    from nbclient import NotebookClient

    path = os.path.join(REPO, filename)
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": REPO}})
    client.execute()
    os.makedirs(EXECUTED, exist_ok=True)
    nbformat.write(nb, os.path.join(EXECUTED, filename))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="start", metavar="NN", help="resume from this notebook number")
    p.add_argument("--only", nargs="+", metavar="NN", help="run only these notebook numbers")
    p.add_argument("--checkpoint1", action="store_true", help="run notebooks 01 to 04 only")
    p.add_argument("--list", action="store_true", help="print the pipeline and exit")
    p.add_argument("--timeout", type=int, default=3600, help="per-notebook timeout in seconds")
    args = p.parse_args()

    if args.list:
        print("Pipeline:")
        for num, fn, desc in PIPELINE:
            mark = "checkpoint 1" if num in CHECKPOINT_1 else "checkpoint 2"
            print(f"  {num}  {desc:<40} [{mark}]  {fn}")
        return 0

    selected = PIPELINE
    if args.checkpoint1:
        selected = [s for s in PIPELINE if s[0] in CHECKPOINT_1]
    elif args.only:
        wanted = set(args.only)
        selected = [s for s in PIPELINE if s[0] in wanted]
    elif args.start:
        selected = [s for s in PIPELINE if s[0] >= args.start]

    if not selected:
        print("Nothing matched the selection.")
        return 1

    os.environ.setdefault("MPLBACKEND", "Agg")
    total = time.perf_counter()
    failed = []

    for num, fn, desc in selected:
        print("\n" + "=" * 74)
        print(f"[{num}] {desc}")
        print("=" * 74)
        t0 = time.perf_counter()
        try:
            run_notebook(fn, args.timeout)
            print(f"[{num}] OK in {time.perf_counter() - t0:.1f}s")
        except Exception as e:
            failed.append((num, type(e).__name__, str(e)[:300]))
            print(f"[{num}] FAILED: {type(e).__name__}: {str(e)[:300]}")
            break

    print("\n" + "=" * 74)
    print(f"Finished in {time.perf_counter() - total:.1f}s")
    if failed:
        for num, kind, msg in failed:
            print(f"  {num} failed: {kind}: {msg}")
        return 1
    print("All selected notebooks completed. Executed copies are in executed/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
