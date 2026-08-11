"""
User segmentation via KMeans, run through Spark MLlib on the Spark-produced
`user_profile` parquet dataset (output of src/etl/spark_pipeline.py).

Produces `models/kmeans_segments.parquet` (user_id -> segment) and
`models/kmeans_model/` (the fitted Spark ML pipeline model) so the segment
lookup can be served without re-running Spark.

Usage:
    python -m src.models.train_clustering --features data/features/user_profile \
        --k 4 --out models
"""

from __future__ import annotations

import argparse
import os

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession

FEATURE_COLS = [
    "avg_historical_ctr",
    "observed_ctr",
    "avg_session_length",
    "impression_count",
    "distinct_categories_seen",
    "peak_hour_ratio",
]


def train(features_path: str, out_dir: str, k: int = 4, master: str = "local[*]") -> float:
    spark = SparkSession.builder.appName("adclick-clustering").master(master).getOrCreate()
    try:
        df = spark.read.parquet(features_path)

        assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="raw_features")
        scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True)
        kmeans = KMeans(featuresCol="features", predictionCol="segment", k=k, seed=42)
        pipeline = Pipeline(stages=[assembler, scaler, kmeans])

        model = pipeline.fit(df)
        predictions = model.transform(df)

        evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="segment")
        silhouette = evaluator.evaluate(predictions)

        os.makedirs(out_dir, exist_ok=True)
        predictions.select("user_id", "segment").write.mode("overwrite").parquet(
            f"{out_dir}/kmeans_segments.parquet"
        )
        model.write().overwrite().save(f"{out_dir}/kmeans_model")

        print(f"Trained KMeans(k={k}) — silhouette score: {silhouette:.4f}")
        return silhouette
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/features/user_profile")
    parser.add_argument("--out", default="models")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    train(args.features, args.out, args.k)


if __name__ == "__main__":
    main()
