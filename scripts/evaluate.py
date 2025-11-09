"""
evaluate.py
-----------
This script computes sentiment analysis for each model response from the benchmark results.
It reads the CSV with human scores (if available), adds sentiment metrics, and saves a new enriched CSV.
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# ---------- CONFIG ----------
INPUT_PATH = "results/benchmark_with_human.csv"       # file with human grades
OUTPUT_PATH = "results/benchmark_with_sentiment.csv"  # enriched file
# ----------------------------

from typing import Optional

def compute_sentiment(text: Optional[str], analyzer: SentimentIntensityAnalyzer) -> Optional[float]:
    """Return VADER compound sentiment (-1 to 1), or None if text is empty."""
    if not isinstance(text, str) or not text.strip():
        return None
    return analyzer.polarity_scores(text)["compound"]


def add_sentiment_scores(input_path: str, output_path: str) -> pd.DataFrame:
    """Compute sentiment for all model responses and save new CSV."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    analyzer = SentimentIntensityAnalyzer()

    # Compute sentiment scores
    df["sentiment_score"] = df["model_response"].apply(
        lambda txt: compute_sentiment(txt, analyzer)
    )

    # Normalize from -1..1 → 0..1 for easier weighting
    df["sentiment_norm"] = df["sentiment_score"].apply(
        lambda s: (s + 1) / 2 if pd.notna(s) else None
    )

    # Optionally add length-based metric
    df["response_length"] = df["model_response"].apply(
        lambda x: len(str(x).split()) if isinstance(x, str) else 0
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved sentiment-enriched file → {output_path}")
    return df


if __name__ == "__main__":
    add_sentiment_scores(INPUT_PATH, OUTPUT_PATH)
