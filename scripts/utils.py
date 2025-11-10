"""
utils.py
--------
Combine human scores + automatic metrics (sentiment, latency, etc.)
into a single hybrid_score for each (query, model, prompt) row.
"""

import os
from typing import Optional

import pandas as pd

# ---------- CONFIG ----------
INPUT_PATH = "results/benchmark_with_sentiment.csv"
OUTPUT_PATH = "results/final_hybrid_scores.csv"

# Weights for the hybrid score (you can tweak these later)
WEIGHTS = {
    "human_overall_norm": 0.6,
    "sentiment_norm": 0.2,
    "latency_norm": 0.2,
}
# ----------------------------


def normalize_series(series: pd.Series, invert: bool = False) -> pd.Series:
    """
    Normalize a numeric series to [0, 1].

    If invert=True, higher original values become lower normalized values.
    (Use this for latency so that faster = closer to 1.)
    """
    s = pd.to_numeric(series, errors="coerce")
    non_null = s.dropna()

    if non_null.empty:
        return pd.Series([None] * len(series), index=series.index)

    min_v = non_null.min()
    max_v = non_null.max()

    if max_v == min_v:
        # All values equal → give everyone 0.5
        norm = pd.Series([0.5] * len(series), index=series.index)
    else:
        norm = (s - min_v) / (max_v - min_v)

    if invert:
        norm = 1 - norm

    return norm


def build_hybrid_scores(
    input_path: str = INPUT_PATH,
    output_path: str = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Read sentiment-enriched results, normalize metrics,
    and compute a final hybrid_score column.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # 1️⃣ Ensure we have a human overall score
    # Expected: a column "human_overall" on a 1–5 scale.
    # If it's missing, try to average sub-scores if they exist.
    if "human_overall" not in df.columns:
        candidate_cols = [
            c
            for c in [
                "human_accuracy",
                "human_tone",
                "human_clarity",
                "human_brevity",
                "human_relevance",
            ]
            if c in df.columns
        ]
        if candidate_cols:
            df["human_overall"] = df[candidate_cols].mean(axis=1)
            print(f"Computed human_overall from: {', '.join(candidate_cols)}")
        else:
            raise ValueError(
                "No 'human_overall' or sub-score columns found. "
                "Please add human scores before running utils.py."
            )

    # Convert 1–5 human_overall → 0–1
    df["human_overall_norm"] = (df["human_overall"] - 1.0) / 4.0

    # 2️⃣ Sentiment normalization (already in [0, 1] from evaluate.py)
    if "sentiment_norm" not in df.columns:
        raise ValueError(
            "'sentiment_norm' column not found. "
            "Run evaluate.py before utils.py."
        )

    # 3️⃣ Latency normalization: smaller latency → higher score
    if "latency_seconds" in df.columns:
        df["latency_norm"] = normalize_series(df["latency_seconds"], invert=True)
    else:
        print("[WARN] latency_seconds column not found; setting latency_norm to None.")
        df["latency_norm"] = None

    # 4️⃣ Compute the hybrid score
    # Fill NaNs with 0 for the weighted sum
    human_part = df["human_overall_norm"].fillna(0)
    sent_part = df["sentiment_norm"].fillna(0)
    lat_part = df["latency_norm"].fillna(0)

    df["hybrid_score"] = (
        WEIGHTS["human_overall_norm"] * human_part
        + WEIGHTS["sentiment_norm"] * sent_part
        + WEIGHTS["latency_norm"] * lat_part
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved final hybrid scores → {output_path}")
    return df


if __name__ == "__main__":
    build_hybrid_scores()
