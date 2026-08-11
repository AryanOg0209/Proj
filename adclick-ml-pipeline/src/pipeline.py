"""
End-to-end CLI orchestrator. Chains the stages so the whole pipeline can be
driven from the Linux command line with one command (mirrors the JD's
"manage project priorities, deadlines, and deliverables" via a single
reproducible entry point instead of manual notebook steps):

    python -m src.pipeline run --rows 200000 --k 4

Stages: generate synthetic data -> Spark ETL -> train clustering ->
train CTR models (decision tree / random forest + neural net).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def _step(name: str, cmd: list[str]) -> None:
    print(f"\n=== [{name}] {' '.join(cmd)} ===")
    start = time.time()
    subprocess.run(cmd, check=True)
    print(f"=== [{name}] done in {time.time() - start:.1f}s ===")


def run(rows: int, k: int, epochs: int) -> None:
    py = sys.executable
    _step("generate-data", [py, "data/generate_data.py", "--rows", str(rows)])
    _step("spark-etl", [py, "-m", "src.etl.spark_pipeline"])
    _step("train-clustering", [py, "-m", "src.models.train_clustering", "--k", str(k)])
    _step("train-ctr-model", [py, "-m", "src.models.train_ctr_model"])
    _step("train-nn-model", [py, "-m", "src.models.train_nn_model", "--epochs", str(epochs)])
    print("\nPipeline complete. Serve with: uvicorn src.serving.app:app --reload")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Run the full pipeline end to end")
    run_parser.add_argument("--rows", type=int, default=200_000)
    run_parser.add_argument("--k", type=int, default=4)
    run_parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    if args.command == "run":
        run(args.rows, args.k, args.epochs)


if __name__ == "__main__":
    main()
