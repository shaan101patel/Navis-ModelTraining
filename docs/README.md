# Navis-ModelTraining

This project focuses on optimizing large language model (LLM) responses for customer support queries through prompt engineering. It includes dataset preparation, model benchmarking across multiple OpenAI models (e.g., GPT-4o, GPT-4o-mini, GPT-3.5-turbo), automated sentiment analysis, and a hybrid evaluation framework combining human ratings with automated metrics like latency, response length, and sentiment scores. The goal is to balance response quality, speed, and cost-efficiency for business applications.

## Table of Contents
- Overview
- Features
- Installation
- Data Preparation
- Benchmarking
- Evaluation
- Results
- Usage
- Contributing
- License

## Overview
The project uses prompt engineering to fine-tune LLM outputs for customer support without full supervised fine-tuning. It processes a dataset of customer queries and expected responses, benchmarks models, and evaluates performance using a hybrid score that weighs human judgment (55%), sentiment (20%), latency (20%), and brevity (5%). This approach emphasizes scalability, cost-effectiveness, and quick iteration.

Key components:
- **Data Cleaning**: Cleans and prepares the Bitext customer support dataset.
- **Benchmarking**: Runs queries against models and records responses, latency, and token usage.
- **Sentiment Analysis**: Adds VADER-based sentiment scores to responses.
- **Hybrid Evaluation**: Merges human ratings with automated metrics for ranking.

## Features
- **Prompt Engineering**: Customizable system prompts for tone (e.g., friendly vs. professional).
- **Multi-Model Support**: Benchmarks GPT-4o, GPT-4o-mini, and GPT-3.5-turbo.
- **Automated Metrics**: Tracks latency, token counts, sentiment, and response length.
- **Human-in-the-Loop Evaluation**: Integrates manual scoring (1-5 scale) for relevance, clarity, and empathy.
- **Business Alignment**: Prioritizes cost, speed, and quality trade-offs.
- **Extensible**: Supports adding new models, prompts, or metrics.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Navis-ModelTraining
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Key libraries: `pandas`, `openai`, `vaderSentiment`, `textblob`, `scikit-learn`, `python-dotenv`.

3. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`.

4. Ensure Python 3.8+ is installed.

## Data Preparation
The dataset is sourced from Bitext's customer support data. Cleaning involves removing duplicates, normalizing text, and adding features like sentiment and response length.

- **Raw Data**: Located in raw_bitext_customer_support_data.csv.
- **Cleaning Script**: Run data_cleaning.ipynb to generate cleaned data.
  - Removes URLs, normalizes spaces, drops duplicates.
  - Adds `query_id`, `sentiment` (via TextBlob), and `response_length`.
  - Output: customer_queries_cleaned.csv.
- **Sampling**: Stratified sampling for variety, saved as sample_queries.csv.

## Benchmarking
Benchmarking sends customer queries to models using system prompts and records responses.

- **Script**: benchmark.py.
- **Key Function**: `run_benchmark` loads data, templates, and models, then calls the API.
- **Inputs**:
  - Dataset: sample_queries.csv.
  - Prompts: templates.md (e.g., `# expanded_friendly`).
- **Outputs**: benchmark_sample_queries.csv with columns like `model_response`, `latency_seconds`, `total_tokens`.
- **Configuration**: Adjust `MODELS`, `MAX_ROWS`, and `MAX_PROMPTS` in the script.

Run via:
```bash
python scripts/benchmarking/benchmark.py
```

## Evaluation
Evaluation combines human ratings with automated metrics into a hybrid score.

- **Sentiment Addition**: add_sentiment.py adds VADER sentiment scores to benchmark results.
  - Input: benchmark_sample_queries.csv.
  - Output: benchmark_with_sentiment.csv.
- **Hybrid Scoring**: evaluate_hybrid_scores.py merges with human ratings and computes scores.
  - Human Ratings: Assumed in human_ratings.csv (scale: 1-5, see results/human_rating/human_weighting_scoring.txt).
  - Formula: `hybrid_score = 0.55 * human_norm + 0.20 * sentiment_norm + 0.20 * (1 - latency_norm) + 0.05 * (1 - length_norm)`.
  - Outputs: merged_full_evaluation.csv and final_prompt_ranking.csv.

Run via:
```bash
python scripts/evaluate/evaluate_hybrid_scores.py
```

## Results
- **Benchmark Results**: Raw outputs in benchmark.
- **Evaluated Rankings**: See final_prompt_ranking.csv for model/prompt rankings by hybrid score.
- **Insights**: Faster models (e.g., GPT-4o-mini) may score higher on latency, while larger models excel in quality.

## Usage
1. Prepare data using data_cleaning.ipynb.
2. Run benchmarking: `python scripts/benchmarking/benchmark.py`.
3. Add sentiment: `python scripts/benchmarking/add_sentiment.py`.
4. Evaluate: `python scripts/evaluate/evaluate_hybrid_scores.py`.
5. Analyze results in results for business decisions (e.g., cost vs. quality trade-offs).

For custom prompts, edit templates.md.

## Contributing
- Fork the repo and submit pull requests.
- Report issues via GitHub.
- Follow the project's focus on prompt engineering and hybrid evaluation.

## License
This project is licensed under the MIT License. See LICENSE for details.
