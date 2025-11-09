import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

# ---------------- CONFIG ----------------
DATA_PATH = "data/cleaned/sample_queries.csv"
OUTPUT_PATH = "results/benchmark_sample_queries.csv"
TEMPLATE_PATH = "prompts/template.md"

# Start with cheaper models; add more once things look good
MODELS = [
    "gpt-4o-mini",
     "gpt-4o",
     "gpt-3.5-turbo",
]

# To control cost while testing
MAX_ROWS = 20      # None = use all 50
MAX_PROMPTS = None  # None = use all prompt templates
# ----------------------------------------


def get_client() -> OpenAI:
    """Create an OpenAI client using the API key from .env."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment/.env")
    return OpenAI(api_key=api_key)


def load_dataset(path: str, max_rows: int | None = None) -> pd.DataFrame:
    """Load the cleaned dataset of customer queries."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["customer_query"])
    if max_rows is not None:
        df = df.head(max_rows)
    return df


def load_prompt_templates(path: str, max_prompts: int | None = None):
    """
    Load system prompt templates from prompts/template.md.

    Expected format:

        # prompt_name
        Prompt text line 1
        Prompt text line 2
        ---
        # another_name
        Another prompt...

    Returns a list of dicts: [{"name": ..., "text": ...}, ...]
    """
    file_path = Path(path)
    if not file_path.exists():
        print(f"[WARN] Prompt template file not found at {path}. Using a default template.")
        return [
            {
                "name": "default",
                "text": (
                    "You are a helpful, professional customer service agent. "
                    "Answer clearly, politely, and concisely."
                ),
            }
        ]

    raw = file_path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in raw.split("---") if b.strip()]

    templates = []
    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue

        first = lines[0].strip()
        if first.startswith("#"):
            name = first.lstrip("#").strip()
            text = "\n".join(lines[1:]).strip()
        else:
            # no explicit name; generate one
            name = f"template_{len(templates) + 1}"
            text = "\n".join(lines).strip()

        if text:
            templates.append({"name": name, "text": text})

    if max_prompts is not None:
        templates = templates[:max_prompts]

    if not templates:
        # safety fallback
        templates = [
            {
                "name": "default",
                "text": (
                    "You are a helpful, professional customer service agent. "
                    "Answer clearly, politely, and concisely."
                ),
            }
        ]

    print(f"Loaded {len(templates)} prompt template(s) from {path}")
    return templates


def call_model(
    client: OpenAI,
    model: str,
    system_prompt: str,
    customer_query: str,
):
    """
    Call the model and return:
      response_text, latency_seconds, prompt_tokens, completion_tokens
    """
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": customer_query},
        ],
    )
    end = time.perf_counter()
    latency = end - start

    message = response.choices[0].message.content
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

    return message, latency, prompt_tokens, completion_tokens


def run_benchmark() -> None:
    client = get_client()
    df = load_dataset(DATA_PATH, MAX_ROWS)
    templates = load_prompt_templates(TEMPLATE_PATH, MAX_PROMPTS)

    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    print(f"Using models: {', '.join(MODELS)}")

    results: list[dict] = []

    for _, row in df.iterrows():
        query_id = row.get("query_id")
        customer_query = row.get("customer_query")
        expected_response = row.get("expected_response")
        category = row.get("category")

        if not isinstance(customer_query, str) or not customer_query.strip():
            continue

        for tmpl in templates:
            prompt_name = tmpl["name"]
            system_prompt = tmpl["text"]

            for model_name in MODELS:
                try:
                    (
                        response_text,
                        latency,
                        prompt_tokens,
                        completion_tokens,
                    ) = call_model(
                        client,
                        model_name,
                        system_prompt,
                        customer_query,
                    )

                    total_tokens = (
                        (prompt_tokens or 0) + (completion_tokens or 0)
                        if (prompt_tokens is not None and completion_tokens is not None)
                        else None
                    )

                    results.append(
                        {
                            "query_id": query_id,
                            "model": model_name,
                            "prompt_name": prompt_name,
                            "system_prompt": system_prompt,
                            "customer_query": customer_query,
                            "expected_response": expected_response,
                            "model_response": response_text,
                            "category": category,
                            "latency_seconds": latency,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        }
                    )

                    print(
                        f"[OK] id={query_id} prompt={prompt_name} "
                        f"model={model_name} latency={latency:.2f}s"
                    )

                except Exception as e:
                    print(
                        f"[ERROR] id={query_id} prompt={prompt_name} "
                        f"model={model_name}: {e}"
                    )
                    results.append(
                        {
                            "query_id": query_id,
                            "model": model_name,
                            "prompt_name": prompt_name,
                            "system_prompt": system_prompt,
                            "customer_query": customer_query,
                            "expected_response": expected_response,
                            "model_response": None,
                            "category": category,
                            "latency_seconds": None,
                            "prompt_tokens": None,
                            "completion_tokens": None,
                            "total_tokens": None,
                            "error": str(e),
                        }
                    )

    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(results_df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_benchmark()
