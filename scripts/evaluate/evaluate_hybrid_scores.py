import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Update these paths if your repo differs
bench_path = "results/benchmark/benchmark_results.csv"
human_path = "results/human_rating/human_ratings.csv"

bench = pd.read_csv(bench_path)
human = pd.read_csv(human_path)

print("Benchmark results loaded:", bench.shape)
print("Human ratings loaded:", human.shape)


# Merge on model_response
df = bench.merge(
    human,
    on=["model_response"],  
    how="inner"
)

print("Merged dataset shape:", df.shape)

#Normalize relevant columns
scaler = MinMaxScaler()

# Human rating is 1–5
# Sentiment already has sentiment_norm in your benchmark — but we recompute for accuracy
# Latency needs normalization
# Response length needs normalization

df[["human_norm", "sentiment_norm_final", "latency_norm", "length_norm"]] = scaler.fit_transform(
    df[[
        "human_rating",
        "sentiment_score",
        "latency_seconds",
        "response_length"
    ]]
)

df["hybrid_score"] = (
    0.55 * df["human_norm"] +                # human judgment dominates
    0.20 * df["sentiment_norm_final"] +      # tone/emotional quality
    0.20 * (1 - df["latency_norm"]) +        # faster = better
    0.05 * (1 - df["length_norm"])           # shorter = better (brevity)
)

#Summary of results

summary = df.groupby(["model", "prompt_name"]).agg({
    "human_rating": "mean",
    "sentiment_score": "mean",
    "latency_seconds": "mean",
    "hybrid_score": "mean"
}).reset_index()

summary = summary.sort_values("hybrid_score", ascending=False)

print("\n===== FINAL HYBRID SCORE RANKING =====")
print(summary)

#save outputs

df.to_csv("results/evaluation/merged_full_evaluation.csv", index=False)
summary.to_csv("results/evaluation/final_prompt_ranking.csv", index=False)

print("\nSaved outputs:")
print(" - ../results/evaluation/merged_full_evaluation.csv")
print(" - ../results/evaluation/final_prompt_ranking.csv")
