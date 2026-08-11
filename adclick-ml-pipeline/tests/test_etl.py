import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")

from src.etl.spark_pipeline import (  # noqa: E402
    build_spark,
    build_user_profile,
    clean,
    engineer_row_features,
)


@pytest.fixture(scope="module")
def spark():
    s = build_spark(app_name="test", master="local[2]")
    yield s
    s.stop()


@pytest.fixture
def raw_df(spark):
    pdf = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, None],
            "device_type": ["ios", "ios", "mid_android", "mid_android", "ios"],
            "hour_of_day": [9, 25, 10, 10, 9],  # 25 is invalid -> dropped
            "day_of_week": [1, 1, 6, 6, 1],
            "ad_category": ["gaming", "gaming", "news", "news", "gaming"],
            "content_type": ["lockscreen", "lockscreen", "video", "video", "lockscreen"],
            "historical_ctr": [0.1, 0.1, 0.2, 0.2, 0.1],
            "session_length_sec": [100.0, 9000.0, 50.0, 60.0, 100.0],
            "clicked": [1, 0, 0, 1, 1],
        }
    )
    return spark.createDataFrame(pdf)


def test_clean_drops_invalid_rows(raw_df):
    cleaned = clean(raw_df)
    rows = cleaned.collect()
    # the null user_id row and the hour_of_day=25 row must both be dropped
    assert all(r["user_id"] is not None for r in rows)
    assert all(0 <= r["hour_of_day"] <= 23 for r in rows)


def test_clean_clips_session_length(raw_df):
    cleaned = clean(raw_df)
    max_session = cleaned.agg({"session_length_sec": "max"}).collect()[0][0]
    assert max_session <= 5000.0


def test_engineer_row_features_adds_columns(raw_df):
    cleaned = clean(raw_df)
    features = engineer_row_features(cleaned)
    for col in ["is_peak_hour", "is_weekend", "device_tier", "session_length_log"]:
        assert col in features.columns


def test_user_profile_aggregates_per_user(raw_df):
    cleaned = clean(raw_df)
    features = engineer_row_features(cleaned)
    profile = build_user_profile(features)
    users = {r["user_id"] for r in profile.collect()}
    assert users == {1, 2}
    row = profile.filter(profile.user_id == 1).collect()[0]
    assert row["impression_count"] == 1  # the hour=25 row for user 1 was dropped
