"""
Synthetic mobile-ad-impression data generator.

Simulates the kind of event stream a mobile ad/content platform (e.g. a lock-screen
or in-feed ad system) would log: one row per impression, with device/context
features and a binary `clicked` label. The label is generated from a nonlinear
combination of features plus noise, so downstream models actually have signal
to learn (rather than pure randomness).

Usage:
    python data/generate_data.py --rows 200000 --out data/raw/impressions.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

RNG_SEED = 42

DEVICE_TYPES = ["low_end_android", "mid_android", "flagship_android", "ios"]
OS_VERSIONS = ["v10", "v11", "v12", "v13", "v14"]
REGIONS = ["north", "south", "east", "west", "central"]
AD_CATEGORIES = ["finance", "gaming", "ecommerce", "entertainment", "news", "travel"]
CONTENT_TYPES = ["lockscreen", "in_feed", "video", "story"]


def _click_probability(df: pd.DataFrame) -> np.ndarray:
    """Hand-crafted latent function so labels correlate with features."""
    p = np.full(len(df), 0.05)

    # Peak engagement hours (commute + evening)
    p += np.where(df["hour_of_day"].isin([8, 9, 18, 19, 20, 21]), 0.06, 0.0)

    # Weekends slightly higher engagement
    p += np.where(df["day_of_week"].isin([5, 6]), 0.02, 0.0)

    # Flagship devices render richer creatives -> higher CTR
    p += np.where(df["device_type"] == "flagship_android", 0.04, 0.0)
    p += np.where(df["device_type"] == "ios", 0.05, 0.0)

    # Category affinity
    p += np.where(df["ad_category"].isin(["gaming", "entertainment"]), 0.03, 0.0)

    # Historical CTR is the strongest signal (user propensity)
    p += df["historical_ctr"] * 0.5

    # Longer sessions -> more exposed / engaged
    p += np.clip(df["session_length_sec"] / 6000.0, 0, 0.05)

    # Content type effect
    p += np.where(df["content_type"] == "lockscreen", 0.02, 0.0)

    noise = np.random.normal(0, 0.02, size=len(df))
    p = np.clip(p + noise, 0.005, 0.95)
    return p


def generate(rows: int, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    df = pd.DataFrame(
        {
            "impression_id": np.arange(1, rows + 1),
            "user_id": rng.integers(1, rows // 4 + 1, size=rows),
            "device_type": rng.choice(DEVICE_TYPES, size=rows, p=[0.30, 0.35, 0.20, 0.15]),
            "os_version": rng.choice(OS_VERSIONS, size=rows),
            "region": rng.choice(REGIONS, size=rows),
            "hour_of_day": rng.integers(0, 24, size=rows),
            "day_of_week": rng.integers(0, 7, size=rows),
            "ad_category": rng.choice(AD_CATEGORIES, size=rows),
            "content_type": rng.choice(CONTENT_TYPES, size=rows),
            "historical_ctr": np.clip(rng.normal(0.08, 0.05, size=rows), 0.0, 0.6),
            "session_length_sec": np.clip(rng.exponential(180, size=rows), 1, 5000),
        }
    )

    click_p = _click_probability(df)
    df["clicked"] = (rng.random(rows) < click_p).astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--out", type=str, default="data/raw/impressions.csv")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = generate(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out} (click rate: {df['clicked'].mean():.3%})")


if __name__ == "__main__":
    main()
