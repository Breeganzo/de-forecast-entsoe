"""
Task 2 (Part 1): Feature Engineering

Builds model-ready features from hourly price/load/wind/solar data.
All features use data from t-24h or earlier to avoid look-ahead bias.

Feature categories:
  1. Calendar (hour, dow, month, is_weekend, is_peak)
  2. Lags (price/load/wind/solar at t-24h, t-48h, t-168h)
  3. Rolling (7-day mean/std for price, wind, load)
  4. Ratios (wind penetration, renewable share, momentum)
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features from raw hourly data.
    Returns DataFrame with original columns + feature columns.
    Drops rows with insufficient history (first 168 hours = 7 days).
    """
    logger.info("=" * 60)
    logger.info("STEP 2a: FEATURE ENGINEERING")
    logger.info("=" * 60)

    feat = df.copy()

    # --- 1. Calendar Features ---
    # These are always available (no lag needed)
    feat["hour"] = feat.index.hour
    feat["dow"] = feat.index.dayofweek          # 0=Mon, 6=Sun
    feat["month"] = feat.index.month
    feat["is_weekend"] = (feat["dow"] >= 5).astype(int)
    feat["is_peak"] = ((feat["hour"] >= 8) & (feat["hour"] <= 20) &
                       (feat["dow"] < 5)).astype(int)

    # --- 2. Lag Features ---
    # All shifted by at least 24h (trader makes decision at noon D-1)
    feat["price_lag_24h"] = feat["price_eur_mwh"].shift(24)
    feat["price_lag_48h"] = feat["price_eur_mwh"].shift(48)
    feat["price_lag_168h"] = feat["price_eur_mwh"].shift(168)  # same hour last week

    feat["load_lag_24h"] = feat["load_mw"].shift(24)
    feat["load_lag_168h"] = feat["load_mw"].shift(168)

    feat["wind_lag_24h"] = feat["wind_mw"].shift(24)
    feat["wind_lag_168h"] = feat["wind_mw"].shift(168)

    feat["solar_lag_24h"] = feat["solar_mw"].shift(24)
    feat["solar_lag_168h"] = feat["solar_mw"].shift(168)

    # Gas generation lags (marginal fuel — indicates price regime)
    if "gas_mw" in feat.columns:
        feat["gas_lag_24h"] = feat["gas_mw"].shift(24)
        feat["gas_lag_168h"] = feat["gas_mw"].shift(168)
        logger.info("  Added gas generation lags (gas_lag_24h, gas_lag_168h)")

    # --- 3. Rolling Features ---
    # Rolling windows use min_periods to avoid NaN propagation
    feat["price_7d_mean"] = feat["price_eur_mwh"].shift(24).rolling(168, min_periods=48).mean()
    feat["price_7d_std"] = feat["price_eur_mwh"].shift(24).rolling(168, min_periods=48).std()
    feat["wind_7d_mean"] = feat["wind_mw"].shift(24).rolling(168, min_periods=48).mean()
    feat["load_24h_mean"] = feat["load_mw"].shift(24).rolling(24, min_periods=12).mean()

    # --- 4. Ratio Features ---
    # Wind penetration: how much of load is covered by wind (lagged)
    feat["wind_penetration_lag24"] = feat["wind_lag_24h"] / feat["load_lag_24h"]
    # Renewable share: (wind + solar) / load (lagged)
    feat["renewable_share_lag24"] = (feat["wind_lag_24h"] + feat["solar_lag_24h"]) / feat["load_lag_24h"]
    # Price momentum: recent vs last week
    feat["price_spread_24_168"] = feat["price_lag_24h"] - feat["price_lag_168h"]

    # --- 5. Forward-Looking Features (available before DA auction) ---
    # These are ENTSO-E day-ahead forecasts — what the market actually sees at decision time
    # Solar forecast: NaN at night is physically correct (solar=0), fill with 0
    # Daytime gaps: forward-fill (short gaps at dawn/dusk transitions)
    if "solar_forecast_mw" in feat.columns:
        feat["solar_forecast_mw"] = feat["solar_forecast_mw"].ffill().fillna(0)
    if "load_forecast_mw" in feat.columns:
        feat["load_forecast_mw"] = feat["load_forecast_mw"].ffill()

    if "wind_forecast_mw" in feat.columns and feat["wind_forecast_mw"].notna().sum() > 100:
        feat["wind_forecast_da"] = feat["wind_forecast_mw"]
        feat["wind_forecast_error_lag24"] = feat["wind_mw"].shift(24) - feat["wind_forecast_mw"].shift(24)
        logger.info("  Added forward-looking: wind_forecast_da, wind_forecast_error_lag24")
    if "solar_forecast_mw" in feat.columns and feat["solar_forecast_mw"].notna().sum() > 100:
        feat["solar_forecast_da"] = feat["solar_forecast_mw"]
        logger.info("  Added forward-looking: solar_forecast_da")
    if "load_forecast_mw" in feat.columns and feat["load_forecast_mw"].notna().sum() > 100:
        feat["load_forecast_da"] = feat["load_forecast_mw"]
        # Residual load forecast = load - renewables (drives thermal dispatch)
        if "wind_forecast_mw" in feat.columns:
            feat["residual_load_forecast"] = feat["load_forecast_mw"] - feat["wind_forecast_mw"] - feat["solar_forecast_mw"]
            logger.info("  Added forward-looking: load_forecast_da, residual_load_forecast")

    # --- 6. Forecast Error Features (lagged — shows systematic bias) ---
    # Yesterday's forecast error often persists (auto-correlated bias)
    if "solar_forecast_mw" in feat.columns and feat["solar_forecast_mw"].notna().sum() > 100:
        feat["solar_forecast_error_lag24"] = feat["solar_mw"].shift(24) - feat["solar_forecast_mw"].shift(24)
        logger.info("  Added: solar_forecast_error_lag24")
    if "load_forecast_mw" in feat.columns and feat["load_forecast_mw"].notna().sum() > 100:
        feat["load_forecast_error_lag24"] = feat["load_mw"].shift(24) - feat["load_forecast_mw"].shift(24)
        logger.info("  Added: load_forecast_error_lag24")

    # --- Drop rows with NaN from lagging (only base features, not forward) ---
    # Forward features (solar_forecast_da etc.) can have NaN — LightGBM handles it natively
    n_before = len(feat)
    # Forward features + gas can have NaN — LightGBM handles missing natively
    forward_and_optional = (
        "wind_forecast_da", "wind_forecast_error_lag24",
        "solar_forecast_da", "solar_forecast_error_lag24",
        "load_forecast_da", "load_forecast_error_lag24",
        "residual_load_forecast", "gas_lag_24h", "gas_lag_168h",
    )
    base_cols = [c for c in get_feature_columns(feat)
                 if c not in forward_and_optional]
    feat = feat.dropna(subset=base_cols)
    n_dropped = n_before - len(feat)

    logger.info(f"Created {len(get_feature_columns(feat))} features")
    logger.info(f"Dropped {n_dropped} rows with insufficient lag history")
    logger.info(f"Final feature matrix: {feat.shape}")

    return feat


def get_feature_columns(df=None) -> list[str]:
    """Return the list of feature column names used by the model."""
    base_features = [
        # Calendar
        "hour", "dow", "month", "is_weekend", "is_peak",
        # Lags
        "price_lag_24h", "price_lag_48h", "price_lag_168h",
        "load_lag_24h", "load_lag_168h",
        "wind_lag_24h", "wind_lag_168h",
        "solar_lag_24h", "solar_lag_168h",
        # Rolling
        "price_7d_mean", "price_7d_std", "wind_7d_mean", "load_24h_mean",
        # Ratios
        "wind_penetration_lag24", "renewable_share_lag24", "price_spread_24_168",
    ]
    # Gas generation features (added if gas data available)
    gas_features = ["gas_lag_24h", "gas_lag_168h"]
    # Forward-looking features (added dynamically based on data availability)
    forward_features = [
        "wind_forecast_da", "wind_forecast_error_lag24",
        "solar_forecast_da", "solar_forecast_error_lag24",
        "load_forecast_da", "load_forecast_error_lag24",
        "residual_load_forecast",
    ]
    if df is not None:
        available_gas = [f for f in gas_features if f in df.columns]
        available_fwd = [f for f in forward_features if f in df.columns]
        return base_features + available_gas + available_fwd
    return base_features


def get_target_column() -> str:
    """Return the target variable name."""
    return "price_eur_mwh"
