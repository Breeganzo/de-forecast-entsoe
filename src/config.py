"""
Configuration for DE-LU Day-Ahead power price forecasting pipeline.
Data sourced from ENTSO-E Transparency Platform REST API.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = ROOT_DIR / "logs"

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Market Configuration ---
MARKET = "DE-LU"
TIMEZONE = "Europe/Berlin"
BIDDING_ZONE_EIC = "10Y1001A1001A82H"  # ENTSO-E EIC code for DE-LU

# --- ENTSO-E API Configuration ---
ENTSOE_BASE_URL = "https://web-api.tp.entsoe.eu/api"
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")

# --- Date Range (2 years) ---
START_DATE = "2024-06-01"
END_DATE = "2026-06-01"

# --- Raw Data Files (cached from API) ---
RAW_PRICES_FILE = RAW_DIR / "entsoe_da_prices.csv"
RAW_LOAD_FILE = RAW_DIR / "entsoe_load.csv"
RAW_WIND_FILE = RAW_DIR / "entsoe_wind.csv"
RAW_SOLAR_FILE = RAW_DIR / "entsoe_solar.csv"
RAW_GAS_FILE = RAW_DIR / "entsoe_gas.csv"
RAW_WIND_FORECAST_FILE = RAW_DIR / "entsoe_wind_forecast.csv"
RAW_SOLAR_FORECAST_FILE = RAW_DIR / "entsoe_solar_forecast.csv"
RAW_LOAD_FORECAST_FILE = RAW_DIR / "entsoe_load_forecast.csv"

# --- Model Configuration ---
TRAIN_MIN_DAYS = 365       # minimum training window
TEST_DAYS = 90             # walk-forward test window
FORECAST_HORIZON = 24      # predict next 24 hours

LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Groq LLM ---
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.2

# --- Data Quality Thresholds (DE-LU market) ---
PRICE_MIN = -500       # €/MWh (exchange floor)
PRICE_MAX = 4000       # €/MWh (exchange cap)
LOAD_MIN = 25000       # MW (DE minimum realistic load)
LOAD_MAX = 85000       # MW (DE maximum realistic load)
WIND_SOLAR_MIN = 0     # MW
WIND_SOLAR_MAX = 100000  # MW (DE installed wind+solar capacity ceiling)
