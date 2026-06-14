"""
European Power Fair Value Pipeline — DE-LU Market (ENTSO-E API)
================================================================
Entry point for the full forecasting pipeline.
Data sourced directly from ENTSO-E Transparency Platform REST API.

Usage:
    python3 main.py

Tasks:
    1. Data Ingestion + Quality (ENTSO-E API → cached CSV → parquet)
    2. Forecasting + Validation (walk-forward LightGBM)
    3. Prompt Curve Translation (trading signals)
    4. AI-Accelerated Workflow (Groq LLM QA + commentary)
"""
import logging
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger("main")


def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("DE-LU DAY-AHEAD PRICE FORECASTING PIPELINE")
    logger.info("Data Source: ENTSO-E Transparency Platform API")
    logger.info("=" * 60)

    # --- Task 1: Ingestion + Quality ---
    from src.ingest import run_ingestion
    from src.quality import run_quality

    df = run_ingestion()
    qa_report = run_quality(df)

    # --- Task 4: AI Workflow (runs early to augment QA) ---
    from src.ai_qa import run_ai_workflow
    run_ai_workflow(df)

    # --- Task 2: Feature Engineering + Forecasting ---
    from src.features import build_features
    from src.forecast import run_forecast

    df_feat = build_features(df)
    results = run_forecast(df_feat)

    # --- Task 3: Prompt Curve Translation ---
    from src.curve_view import run_curve_translation
    run_curve_translation(results)

    # --- Task 4b: AI Post-Model Analysis (feature explanations + diagnosis) ---
    from src.ai_qa import run_ai_post_model
    feat_imp = dict(results["results_lagged"]["feature_importance"].head(10))
    metrics = results["results_lagged"]["metrics"]
    run_ai_post_model(feat_imp, metrics)

    # --- Generate Figures ---
    from src.figures import generate_figures
    generate_figures(df, results)

    # --- Summary ---
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Outputs: outputs/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
