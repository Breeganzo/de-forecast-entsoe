"""
Figure generation for the submission.
Produces at least 2 required figures/tables.
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.config import FIGURES_DIR

logger = logging.getLogger(__name__)


def generate_figures(df: pd.DataFrame, results: dict):
    """Generate all submission figures."""
    logger.info("=" * 60)
    logger.info("GENERATING FIGURES")
    logger.info("=" * 60)

    plt.style.use("seaborn-v0_8-whitegrid")

    # Figure 1: DA Price Time Series with fundamentals
    fig1_path = plot_price_series(df)
    logger.info(f"Figure 1: {fig1_path}")

    # Figure 2: Forecast vs Actual (test period) with prediction intervals
    fig2_path = plot_forecast_vs_actual(results)
    logger.info(f"Figure 2: {fig2_path}")

    # Figure 3: Feature importance
    fig3_path = plot_feature_importance(results)
    logger.info(f"Figure 3: {fig3_path}")

    # Figure 4: Hourly error heatmap (hour × weekday)
    fig4_path = plot_hourly_error_heatmap(results)
    logger.info(f"Figure 4: {fig4_path}")

    # Figure 5: Error distribution
    fig5_path = plot_error_distribution(results)
    logger.info(f"Figure 5: {fig5_path}")

    # Figure 6: Temporal stability (rolling MAE over test period)
    fig6_path = plot_temporal_stability(results)
    logger.info(f"Figure 6: {fig6_path}")

    # Figure 7: SHAP summary (feature attribution — what drives each prediction)
    fig7_path = plot_shap_summary(results)
    if fig7_path:
        logger.info(f"Figure 7: {fig7_path}")


def plot_price_series(df: pd.DataFrame) -> str:
    """Plot DA prices with wind/solar overlay."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Daily averages for cleaner plot
    daily = df.resample("D").mean()

    ax1.plot(daily.index, daily["price_eur_mwh"], color="navy", linewidth=0.8)
    ax1.axhline(y=0, color="red", linestyle="--", linewidth=0.5, alpha=0.7)
    ax1.set_ylabel("Day-Ahead Price (€/MWh)")
    ax1.set_title("DE-LU Day-Ahead Prices & Renewable Generation (Jun 2024 – Jun 2026)")
    ax1.legend(["DA Price", "Zero line"])

    ax2.fill_between(daily.index, 0, daily["wind_mw"] / 1000, alpha=0.6, label="Wind (GW)")
    ax2.fill_between(daily.index, daily["wind_mw"] / 1000,
                     (daily["wind_mw"] + daily["solar_mw"]) / 1000, alpha=0.6, label="Solar (GW)")
    ax2.set_ylabel("Generation (GW)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    path = FIGURES_DIR / "da_prices_and_renewables.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_forecast_vs_actual(results: dict) -> str:
    """Plot forecast vs actual for test period with prediction intervals."""
    fig, ax = plt.subplots(figsize=(14, 5))

    timestamps = results["timestamps"]
    actuals = results["actuals"]
    preds = results["preds_lgbm"]
    q10 = results["preds_q10"]
    q90 = results["preds_q90"]

    # Plot last 2 weeks for readability
    n_show = min(336, len(actuals))  # 14 days × 24h
    ts = timestamps[-n_show:]
    ax.fill_between(ts, q10[-n_show:], q90[-n_show:],
                    alpha=0.2, color="orangered", label="90% Prediction Interval")
    ax.plot(ts, actuals[-n_show:],
            color="navy", linewidth=1, label="Actual", alpha=0.8)
    ax.plot(ts, preds[-n_show:],
            color="orangered", linewidth=1, label="LightGBM Forecast", alpha=0.8)

    metrics = results["metrics"]
    ax.set_title(f"Forecast vs Actual — Last 2 Weeks of Test Period "
                 f"(MAE: €{metrics['lgbm_mae']:.1f}/MWh, Direction: {metrics['direction_accuracy']:.0f}%)")
    ax.set_ylabel("DA Price (€/MWh)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    plt.tight_layout()
    path = FIGURES_DIR / "forecast_vs_actual.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_feature_importance(results: dict) -> str:
    """Plot top features by importance."""
    feat_imp = results["feature_importance"]
    top10 = feat_imp.head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(top10)), top10.values, color="steelblue")
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10.index)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (split count)")
    ax.set_title("Top 10 Features — LightGBM Model")

    plt.tight_layout()
    path = FIGURES_DIR / "feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_hourly_error_heatmap(results: dict) -> str:
    """Plot MAE by hour-of-day × day-of-week heatmap."""
    timestamps = results["timestamps"]
    actuals = results["actuals"]
    preds = results["preds_lgbm"]

    errors = np.abs(np.array(actuals) - np.array(preds))
    hours = np.array([t.hour for t in timestamps])
    dows = np.array([t.dayofweek for t in timestamps])

    # Build 24×7 grid
    heatmap = np.zeros((24, 7))
    for h in range(24):
        for d in range(7):
            mask = (hours == h) & (dows == d)
            if mask.sum() > 0:
                heatmap[h, d] = errors[mask].mean()

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_yticks(range(24))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(24)])
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour of Day")
    ax.set_title("Forecast Error (MAE €/MWh) by Hour × Weekday\n"
                 "Darker = higher error = harder to predict")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("MAE (€/MWh)")

    # Annotate cells
    for h in range(24):
        for d in range(7):
            val = heatmap[h, d]
            color = "white" if val > heatmap.mean() + heatmap.std() else "black"
            ax.text(d, h, f"{val:.0f}", ha="center", va="center", fontsize=7, color=color)

    plt.tight_layout()
    path = FIGURES_DIR / "hourly_error_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_error_distribution(results: dict) -> str:
    """Plot error distribution with key percentiles marked."""
    actuals = results["actuals"]
    preds = results["preds_lgbm"]
    errors = actuals - preds  # signed errors (bias analysis)
    abs_errors = np.abs(errors)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: signed error distribution (bias check)
    ax1.hist(errors, bins=80, color="steelblue", alpha=0.7, edgecolor="none")
    ax1.axvline(0, color="red", linestyle="--", linewidth=1, label="Zero (no bias)")
    ax1.axvline(np.mean(errors), color="orange", linestyle="-", linewidth=1.5,
                label=f"Mean bias: €{np.mean(errors):.1f}")
    ax1.set_xlabel("Forecast Error (€/MWh)")
    ax1.set_ylabel("Count")
    ax1.set_title("Signed Error Distribution\n(left-skew = model over-predicts)")
    ax1.legend()

    # Right: absolute error CDF with key percentiles
    sorted_err = np.sort(abs_errors)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax2.plot(sorted_err, cdf * 100, color="navy", linewidth=1.5)
    # Mark key percentiles
    for pct, color in [(50, "green"), (75, "orange"), (95, "red")]:
        val = np.percentile(abs_errors, pct)
        ax2.axhline(pct, color=color, linestyle=":", alpha=0.5)
        ax2.axvline(val, color=color, linestyle=":", alpha=0.5)
        ax2.annotate(f"P{pct}: €{val:.0f}", xy=(val, pct),
                     xytext=(val + 5, pct - 5), fontsize=9, color=color)
    ax2.set_xlabel("Absolute Error (€/MWh)")
    ax2.set_ylabel("Cumulative %")
    ax2.set_title("Error CDF — Where do 50/75/95% of errors fall?")
    ax2.set_xlim(0, np.percentile(abs_errors, 99))

    plt.tight_layout()
    path = FIGURES_DIR / "error_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_temporal_stability(results: dict) -> str:
    """Plot rolling 7-day MAE over the test period to show model stability."""
    timestamps = results["timestamps"]
    actuals = np.array(results["actuals"])
    preds = np.array(results["preds_lgbm"])

    # Build daily MAE
    ts_series = pd.Series(np.abs(actuals - preds), index=pd.DatetimeIndex(timestamps))
    daily_mae = ts_series.resample("D").mean()

    fig, ax = plt.subplots(figsize=(14, 5))

    # 7-day rolling MAE
    rolling_mae = daily_mae.rolling(7, min_periods=3).mean()
    ax.plot(daily_mae.index, daily_mae.values, alpha=0.3, color="steelblue",
            linewidth=0.8, label="Daily MAE")
    ax.plot(rolling_mae.index, rolling_mae.values, color="navy",
            linewidth=2, label="7-day Rolling MAE")

    # Overall MAE line
    overall_mae = np.mean(np.abs(actuals - preds))
    ax.axhline(overall_mae, color="red", linestyle="--", linewidth=1,
               label=f"Overall MAE: €{overall_mae:.1f}/MWh")

    # Trend line to show stability/degradation
    x_num = np.arange(len(daily_mae))
    valid = ~np.isnan(daily_mae.values)
    if valid.sum() > 10:
        slope, intercept = np.polyfit(x_num[valid], daily_mae.values[valid], 1)
        trend_label = f"Trend: {'+'if slope > 0 else ''}{slope:.2f} €/day"
        ax.plot(daily_mae.index, intercept + slope * x_num,
                color="orange", linestyle=":", linewidth=1.5, label=trend_label)

    ax.set_xlabel("Date")
    ax.set_ylabel("MAE (€/MWh)")
    ax.set_title("Temporal Stability — Does Forecast Quality Degrade Over Time?")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    plt.tight_layout()
    path = FIGURES_DIR / "temporal_stability.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_shap_summary(results: dict) -> str | None:
    """
    Figure 7: SHAP summary plot — shows how each feature pushes predictions
    up or down. A trader reads this as: "What drove my forecast today?"
    """
    # Get SHAP values from the lagged model (honest baseline)
    lagged = results.get("results_lagged", results)
    shap_values = lagged.get("shap_values")
    shap_X = lagged.get("shap_X")

    if shap_values is None or shap_X is None:
        return None

    try:
        import shap

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, shap_X,
            max_display=15,
            show=False,
            plot_size=None,
        )
        plt.title("SHAP Feature Attribution — What Drives Price Predictions?",
                  fontsize=12, pad=15)
        plt.tight_layout()
        path = FIGURES_DIR / "shap_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return str(path)
    except Exception:
        return None
