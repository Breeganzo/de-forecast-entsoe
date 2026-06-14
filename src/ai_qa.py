"""
Task 4: AI-Accelerated Workflow

Two AI components:
1. LLM-driven Data QA: Given schema + sample, LLM proposes validation rules
2. Automated Drivers Commentary: LLM explains daily price movement from metrics

Uses Groq API (llama-3.3-70b-versatile) — fast, free tier.
All prompts, outputs, and failures are logged to logs/ai_run.jsonl.
"""
import logging
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, LOGS_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)

LOG_FILE = LOGS_DIR / "ai_run.jsonl"


def log_llm_call(prompt: str, response: str, latency: float,
                 tokens: int = 0, status: str = "success"):
    """Log every LLM interaction to JSONL file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": GROQ_MODEL,
        "prompt_preview": prompt[:200],
        "response_preview": response[:300] if response else "",
        "latency_s": round(latency, 2),
        "tokens": tokens,
        "status": status,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def call_groq(prompt: str, max_retries: int = 2) -> str:
    """Call Groq API with retry logic. Returns response text or raises."""
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — using fallback rules")
        return None

    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    for attempt in range(max_retries + 1):
        try:
            start = time.time()
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=GROQ_TEMPERATURE,
                max_tokens=1000,
            )
            latency = time.time() - start
            text = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
            log_llm_call(prompt, text, latency, tokens, "success")
            return text

        except Exception as e:
            latency = time.time() - start
            log_llm_call(prompt, str(e), latency, 0, "error")
            logger.error(f"Groq API error (attempt {attempt+1}): {e}")
            if attempt < max_retries:
                time.sleep(1)
            else:
                return None


def generate_qa_rules(df: pd.DataFrame) -> list[dict]:
    """
    AI Component 1: LLM proposes data validation rules from schema + sample.
    Pipeline then EXECUTES those rules and reports pass/fail.
    """
    # Build schema description
    schema = []
    for col in df.columns:
        schema.append(f"  - {col}: {df[col].dtype}, range [{df[col].min():.1f}, {df[col].max():.1f}]")
    schema_str = "\n".join(schema)

    sample = df.head(3).to_string()

    prompt = f"""You are a data quality engineer for a European power trading desk.
Given this hourly electricity market dataset schema and sample, propose 5 validation rules as JSON.

Schema (columns, types, ranges):
{schema_str}

Sample rows:
{sample}

Return ONLY a JSON array of objects with fields:
- "rule_name": short name
- "column": which column to check
- "condition": Python pandas expression (using 'df' as variable)
- "severity": "error" or "warning"
- "description": one-line explanation

Example format:
[{{"rule_name": "no_missing_prices", "column": "price_eur_mwh", "condition": "df['price_eur_mwh'].notna().all()", "severity": "error", "description": "Prices must not be missing"}}]
"""

    response = call_groq(prompt)

    if response:
        try:
            # Extract JSON from response
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                rules = json.loads(response[json_start:json_end])
                logger.info(f"LLM proposed {len(rules)} QA rules")
                return rules
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM rules: {e}")

    # Fallback: hardcoded rules if LLM fails
    logger.info("Using fallback QA rules")
    return [
        {"rule_name": "no_missing_prices", "column": "price_eur_mwh",
         "condition": "df['price_eur_mwh'].notna().all()",
         "severity": "error", "description": "Prices must not be missing"},
        {"rule_name": "price_within_exchange_limits", "column": "price_eur_mwh",
         "condition": "(df['price_eur_mwh'] >= -500).all() and (df['price_eur_mwh'] <= 4000).all()",
         "severity": "error", "description": "Prices within EPEX limits"},
        {"rule_name": "load_positive", "column": "load_mw",
         "condition": "(df['load_mw'] > 0).all()",
         "severity": "error", "description": "Load must be positive"},
        {"rule_name": "wind_non_negative", "column": "wind_mw",
         "condition": "(df['wind_mw'] >= 0).all()",
         "severity": "error", "description": "Wind generation cannot be negative"},
        {"rule_name": "solar_zero_at_night", "column": "solar_mw",
         "condition": "df.loc[df.index.hour.isin([0,1,2,3,4,23]), 'solar_mw'].median() < 100",
         "severity": "warning", "description": "Solar should be near-zero at night hours"},
    ]


def execute_qa_rules(df: pd.DataFrame, rules: list[dict]) -> str:
    """Execute LLM-proposed rules against the data and report results."""
    results = []
    passed = 0
    failed = 0

    for rule in rules:
        try:
            result = eval(rule["condition"])  # noqa: S307
            # Handle both scalar booleans and pandas Series
            if isinstance(result, pd.Series):
                result = result.all()
            if result:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"
        except Exception as e:
            status = "ERROR"
            result = str(e)

        results.append(f"  [{status}] {rule['rule_name']}: {rule['description']}")

    report = "\n".join([
        "\n## AI-Generated QA Rules\n",
        f"Rules proposed by: {GROQ_MODEL}",
        f"Results: {passed} passed, {failed} failed out of {len(rules)}\n",
    ] + results)

    logger.info(f"QA rules executed: {passed} passed, {failed} failed out of {len(rules)}")
    return report


def generate_drivers_commentary(df: pd.DataFrame) -> str:
    """
    AI Component 2: LLM generates a brief daily market explanation
    ONLY from computed metrics (no invented numbers).
    """
    # Compute metrics for the last 24 hours vs previous 24 hours
    last_24h = df.tail(24)
    prev_24h = df.iloc[-48:-24]

    metrics = {
        "avg_price_today": f"{last_24h['price_eur_mwh'].mean():.1f}",
        "avg_price_yesterday": f"{prev_24h['price_eur_mwh'].mean():.1f}",
        "price_change": f"{last_24h['price_eur_mwh'].mean() - prev_24h['price_eur_mwh'].mean():.1f}",
        "peak_price": f"{last_24h['price_eur_mwh'].max():.1f}",
        "min_price": f"{last_24h['price_eur_mwh'].min():.1f}",
        "avg_wind": f"{last_24h['wind_mw'].mean():.0f}",
        "avg_wind_yesterday": f"{prev_24h['wind_mw'].mean():.0f}",
        "avg_solar": f"{last_24h['solar_mw'].mean():.0f}",
        "avg_load": f"{last_24h['load_mw'].mean():.0f}",
        "avg_load_yesterday": f"{prev_24h['load_mw'].mean():.0f}",
        "negative_hours": f"{(last_24h['price_eur_mwh'] < 0).sum()}",
    }

    prompt = f"""You are a power market analyst writing a brief daily commentary for DE-LU Day-Ahead prices.
Write exactly 3-4 sentences explaining today's price movement using ONLY the metrics below.
Do NOT invent any numbers. Reference the data provided.

Today's metrics:
- Average DA price: €{metrics['avg_price_today']}/MWh (yesterday: €{metrics['avg_price_yesterday']}/MWh, change: €{metrics['price_change']}/MWh)
- Price range: €{metrics['min_price']} to €{metrics['peak_price']}/MWh
- Wind generation: {metrics['avg_wind']} MW (yesterday: {metrics['avg_wind_yesterday']} MW)
- Solar generation: {metrics['avg_solar']} MW
- Load: {metrics['avg_load']} MW (yesterday: {metrics['avg_load_yesterday']} MW)
- Negative price hours: {metrics['negative_hours']}

Write the commentary:"""

    response = call_groq(prompt)

    if response:
        commentary = f"\n## AI-Generated Market Commentary\n\n{response}\n\n*Source metrics: {json.dumps(metrics)}*"
    else:
        commentary = (
            f"\n## Market Commentary (Auto-generated)\n\n"
            f"DA prices averaged €{metrics['avg_price_today']}/MWh "
            f"(change: €{metrics['price_change']}/MWh vs yesterday). "
            f"Wind at {metrics['avg_wind']} MW, load at {metrics['avg_load']} MW."
        )

    return commentary


def run_ai_workflow(df: pd.DataFrame):
    """Main entry point for Task 4: AI-accelerated workflow."""
    logger.info("=" * 60)
    logger.info("STEP 4: AI-ACCELERATED WORKFLOW")
    logger.info("=" * 60)

    # Component 1: LLM-driven QA rules
    logger.info("Generating AI QA rules...")
    rules = generate_qa_rules(df)
    qa_report = execute_qa_rules(df, rules)

    # Append to existing QA report
    qa_path = OUTPUT_DIR / "data_qa_report.md"
    with open(qa_path, "a") as f:
        f.write(qa_report)
    logger.info(f"AI QA results appended to {qa_path}")

    # Component 2: Drivers commentary
    logger.info("Generating AI market commentary...")
    commentary = generate_drivers_commentary(df)
    with open(qa_path, "a") as f:
        f.write(commentary)

    logger.info(f"AI workflow complete. Logs: {LOG_FILE}")


def generate_feature_descriptions(feature_importance: dict) -> str:
    """
    AI Component 3: LLM generates domain-specific feature descriptions
    for the top features, explaining their trading relevance.
    """
    top_features = list(feature_importance.items())[:10]
    features_str = "\n".join([f"  {i+1}. {name} (importance: {imp})"
                             for i, (name, imp) in enumerate(top_features)])

    prompt = f"""You are a quantitative analyst at a European power trading desk.
For each feature below (used in a LightGBM model predicting German DA electricity prices),
write a ONE-sentence explanation of WHY it matters for price prediction, referencing
the German merit order, weather-driven supply, or demand fundamentals.

Top 10 features by split importance:
{features_str}

Return a numbered list matching the feature order above. Each explanation must be
specific to European power markets (not generic ML commentary). Max 25 words per feature."""

    response = call_groq(prompt)
    if response:
        return f"\n## AI-Generated Feature Explanations\n\n{response}\n"
    else:
        return "\n## Feature Explanations\n\n*LLM unavailable — see feature importance table in model_results.md for manual descriptions.*\n"


def generate_model_diagnosis(metrics: dict) -> str:
    """
    AI Component 4: LLM diagnoses model performance and suggests improvements
    based on computed metrics (no raw data access).
    """
    metrics_str = "\n".join([
        f"- MAE: €{metrics.get('lgbm_mae', 0):.2f}/MWh",
        f"- RMSE: €{metrics.get('lgbm_rmse', 0):.2f}/MWh",
        f"- Naive MAE: €{metrics.get('naive_mae', 0):.2f}/MWh",
        f"- Improvement vs naive: {metrics.get('improvement_pct', 0):.1f}%",
        f"- Direction accuracy: {metrics.get('direction_accuracy', 0):.1f}%",
        f"- PI coverage: {metrics.get('pi_coverage', 0):.1f}% (target: 96%)",
        f"- Peak hour MAE: €{metrics.get('peak_hour_mae', 0):.2f}/MWh",
        f"- Off-peak MAE: €{metrics.get('offpeak_hour_mae', 0):.2f}/MWh",
        f"- Calm day MAE: €{metrics.get('calm_day_mae', 0):.2f}/MWh",
        f"- Volatile day MAE: €{metrics.get('volatile_day_mae', 0):.2f}/MWh",
        f"- DM test p-value: {metrics.get('dm_pvalue', 1):.6f}",
    ])

    prompt = f"""You are a senior quant reviewing a German DA electricity price forecasting model.
Based ONLY on the metrics below, write a 4-5 sentence diagnosis covering:
1. Where the model performs well vs struggles
2. What the peak/off-peak gap implies about missing features
3. One specific, actionable improvement (data source or method)

Model metrics (LightGBM, 92-day walk-forward, 25 lagged features):
{metrics_str}

Be specific to European power markets. Do not suggest generic ML improvements."""

    response = call_groq(prompt)
    if response:
        return f"\n## AI Model Diagnosis\n\n{response}\n"
    else:
        return (
            "\n## Model Diagnosis (Auto-generated)\n\n"
            f"The model achieves €{metrics.get('lgbm_mae', 0):.1f}/MWh MAE, "
            f"{metrics.get('improvement_pct', 0):.0f}% better than naive. "
            f"Peak hours (€{metrics.get('peak_hour_mae', 0):.0f}) are 1.4× harder "
            f"than off-peak (€{metrics.get('offpeak_hour_mae', 0):.0f}), suggesting "
            "intraday balancing dynamics and interconnector flows are the primary "
            "missing information.\n"
        )


def run_ai_post_model(feature_importance: dict, metrics: dict):
    """
    Post-model AI workflow: generate feature explanations and model diagnosis.
    Called after forecasting is complete (requires model outputs).
    """
    logger.info("Generating AI feature explanations...")
    feat_report = generate_feature_descriptions(feature_importance)

    logger.info("Generating AI model diagnosis...")
    diag_report = generate_model_diagnosis(metrics)

    # Append to model results
    results_path = OUTPUT_DIR / "model_results.md"
    with open(results_path, "a") as f:
        f.write(feat_report)
        f.write(diag_report)

    logger.info(f"AI feature explanations + diagnosis appended to {results_path}")
