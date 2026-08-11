"""
PySpark ETL job.

Reads raw impression-level events, cleans them, engineers features at scale,
computes a per-user clustering feature set (aggregations), and writes two
Parquet outputs:

  1. `features/impressions/`  -- row-level features for CTR model training
  2. `features/user_profile/` -- per-user aggregated features for clustering

Designed to run in local mode (`local[*]`) for development, but every
transformation is expressed with the DataFrame API so it scales unchanged
against a real cluster (just point `--master` at YARN/K8s and point the
input path at cloud storage).

Usage:
    python -m src.etl.spark_pipeline --input data/raw/impressions.csv \
        --output data/features
"""

from __future__ import annotations

import argparse

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def build_spark(app_name: str = "adclick-etl", master: str = "local[*]") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def load_raw(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.csv(path, header=True, inferSchema=True)


def clean(df: DataFrame) -> DataFrame:
    """Drop malformed rows and clip obviously bad values."""
    df = df.dropna(
        subset=[
            "user_id",
            "device_type",
            "hour_of_day",
            "day_of_week",
            "ad_category",
            "historical_ctr",
            "clicked",
        ]
    )
    df = df.filter((F.col("hour_of_day") >= 0) & (F.col("hour_of_day") <= 23))
    df = df.filter((F.col("historical_ctr") >= 0) & (F.col("historical_ctr") <= 1))
    df = df.withColumn(
        "session_length_sec", F.least(F.col("session_length_sec"), F.lit(5000.0))
    )
    return df


def engineer_row_features(df: DataFrame) -> DataFrame:
    """Row-level features consumed by the CTR classifier."""
    df = df.withColumn(
        "is_peak_hour",
        F.col("hour_of_day").isin(8, 9, 18, 19, 20, 21).cast("int"),
    )
    df = df.withColumn("is_weekend", F.col("day_of_week").isin(5, 6).cast("int"))
    df = df.withColumn(
        "device_tier",
        F.when(F.col("device_type") == "flagship_android", 3)
        .when(F.col("device_type") == "ios", 3)
        .when(F.col("device_type") == "mid_android", 2)
        .otherwise(1),
    )
    df = df.withColumn("session_length_log", F.log1p(F.col("session_length_sec")))
    return df


def build_user_profile(df: DataFrame) -> DataFrame:
    """Per-user aggregated behavioural profile, the feature set for clustering."""
    agg_exprs = [
        F.avg("historical_ctr").alias("avg_historical_ctr"),
        F.avg("clicked").alias("observed_ctr"),
        F.avg("session_length_sec").alias("avg_session_length"),
        F.count("*").alias("impression_count"),
        F.countDistinct("ad_category").alias("distinct_categories_seen"),
    ]
    if "is_peak_hour" in df.columns:
        agg_exprs.append(F.avg("is_peak_hour").alias("peak_hour_ratio"))

    profile = df.groupBy("user_id").agg(*agg_exprs)
    if "peak_hour_ratio" not in profile.columns:
        profile = profile.withColumn("peak_hour_ratio", F.lit(0.0))
    return profile


def rank_users_by_engagement(profile: DataFrame) -> DataFrame:
    """Adds a percentile rank column, demonstrating window-function usage."""
    window = Window.orderBy(F.col("observed_ctr").desc())
    return profile.withColumn("engagement_percentile", F.percent_rank().over(window))


def run(input_path: str, output_dir: str, master: str = "local[*]") -> None:
    spark = build_spark(master=master)
    try:
        raw = load_raw(spark, input_path)
        cleaned = clean(raw)
        row_features = engineer_row_features(cleaned)

        profile = build_user_profile(row_features)
        profile = rank_users_by_engagement(profile)

        row_features.write.mode("overwrite").parquet(f"{output_dir}/impressions")
        profile.write.mode("overwrite").parquet(f"{output_dir}/user_profile")

        print(f"row_features: {row_features.count():,} rows")
        print(f"user_profile: {profile.count():,} users")
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/impressions.csv")
    parser.add_argument("--output", default="data/features")
    parser.add_argument("--master", default="local[*]")
    args = parser.parse_args()
    run(args.input, args.output, args.master)


if __name__ == "__main__":
    main()
