"""
Task 3: Prompt Curve Translation

Converts hourly DA price forecasts into tradable views for the desk.
Includes: delivery-period averages, confidence-weighted signals,
position sizing guidance, forward-vs-model comparison, and P&L framing.
"""
import logging
import pandas as pd
import numpy as np

from src.config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def run_curve_translation(results: dict):
    """Translate forecast into prompt curve trading signals with desk-relevant framing."""
    logger.info("=" * 60)
    logger.info("STEP 3: PROMPT CURVE TRANSLATION")
    logger.info("=" * 60)

    preds = results["preds_lgbm"]
    preds_q10 = results["preds_q10"]
    preds_q90 = results["preds_q90"]
    actuals = results["actuals"]
    timestamps = results["timestamps"]

    # Create DataFrame of predictions with intervals
    pred_df = pd.DataFrame({
        "timestamp": timestamps,
        "predicted": preds,
        "q10": preds_q10,
        "q90": preds_q90,
        "actual": actuals,
    }).set_index("timestamp")

    pred_df["interval_width"] = pred_df["q90"] - pred_df["q10"]
    pred_df["hour"] = pred_df.index.hour
    pred_df["dow"] = pred_df.index.dayofweek
    pred_df["is_peak"] = ((pred_df["hour"] >= 8) & (pred_df["hour"] <= 20) & (pred_df["dow"] < 5)).astype(int)

    # --- Delivery Period Averages ---
    pred_df["week_ahead_baseload"] = pred_df["predicted"].rolling(168, min_periods=24).mean()
    pred_df["week_q10"] = pred_df["q10"].rolling(168, min_periods=24).mean()
    pred_df["week_q90"] = pred_df["q90"].rolling(168, min_periods=24).mean()

    # --- Confidence-Weighted Signal ---
    # Signal identifies MISPRICINGS: where model fair value diverges from current market
    # BUY = model thinks current market is too cheap (fair value > actual)
    # SELL = model thinks current market is too expensive (fair value < actual)
    # Threshold: only trade when edge exceeds noise (model MAE ~€18)

    # Confidence: narrow interval = high confidence, wide = low
    median_width = pred_df["interval_width"].median()
    pred_df["confidence"] = np.clip(median_width / pred_df["interval_width"], 0.2, 1.0)

    # Rolling volatility for adaptive threshold
    rolling_std = pred_df["actual"].rolling(168, min_periods=48).std()

    # Edge: model's view of fair value vs current market price
    pred_df["edge"] = pred_df["predicted"] - pred_df["actual"]

    # Signal: trade when edge is significant (>0.5σ) AND prediction interval agrees
    # BUY: model predicts higher than actual AND even pessimistic (Q5) > actual
    # SELL: model predicts lower than actual AND even optimistic (Q95) < actual
    threshold = 0.5 * rolling_std  # adaptive threshold

    pred_df["signal"] = np.where(
        (pred_df["edge"] > threshold) & (pred_df["q10"] > pred_df["actual"]),
        "BUY",
        np.where(
            (pred_df["edge"] < -threshold) & (pred_df["q90"] < pred_df["actual"]),
            "SELL",
            "HOLD"
        )
    )
    pred_df["signal_strength"] = (np.abs(pred_df["edge"]) / rolling_std) * pred_df["confidence"]

    signal_counts = pred_df["signal"].value_counts()
    logger.info(f"Trading signals: {signal_counts.to_dict()}")

    # --- Forward Price Reference ---
    # Use exponentially-weighted 7-day actual mean as proxy for prompt-week forward price.
    # EWM with halflife=72h weights recent prices more heavily, mimicking how forwards
    # react faster to recent spot than a flat rolling window.
    # In production: use EEX EPEX prompt-week baseload settlement (daily settlement price).
    pred_df["forward_proxy"] = pred_df["actual"].ewm(halflife=72, min_periods=48).mean()

    # --- Last Week Analysis (the actual trading view) ---
    last_week = pred_df.tail(168)
    week_base = last_week["predicted"].mean()
    week_q10 = last_week["q10"].mean()
    week_q90 = last_week["q90"].mean()
    week_peak = last_week.loc[last_week["is_peak"] == 1, "predicted"].mean()
    week_offpeak = last_week.loc[last_week["is_peak"] == 0, "predicted"].mean()
    week_actual = last_week["actual"].mean()
    forward_ref = last_week["forward_proxy"].mean()
    avg_confidence = last_week["confidence"].mean()

    # Edge calculation: model fair value vs forward proxy
    edge = week_base - forward_ref
    edge_direction = "LONG" if edge > 0 else "SHORT"

    # Position sizing: scale by confidence and edge magnitude
    # Base position = 10 MW, scaled by confidence (0.2–1.0) and capped
    base_position_mw = 10
    position_size = base_position_mw * avg_confidence * min(abs(edge) / 5, 2.0)
    position_size = round(min(position_size, 20), 1)  # cap at 20 MW

    # Expected P&L = |edge| × position × hours (always positive if direction is correct)
    hours_in_week = 168
    expected_pnl = abs(edge) * position_size * hours_in_week
    # Max loss: if actual = opposite bound of interval
    max_loss_per_mwh = abs(week_q10 - forward_ref) if edge > 0 else abs(week_q90 - forward_ref)
    max_loss = max_loss_per_mwh * position_size * hours_in_week

    last_signal = pred_df["signal"].iloc[-1]

    # --- Signal P&L Backtest ---
    # Simulate: on each BUY/SELL signal, take 1 MW position for next 24h
    # Entry at current actual market price, exit at avg of next 24h actuals
    # Transaction costs: €0.05/MWh bid-ask spread per side (entry + exit = €0.10/MWh round-trip)
    SPREAD_PER_SIDE = 0.05  # €/MWh — typical EEX/EPEX screen spread for prompt products
    ROUND_TRIP_COST = 2 * SPREAD_PER_SIDE  # €0.10/MWh

    pred_df["actual_next_24h_mean"] = pred_df["actual"].rolling(24).mean().shift(-24)
    pred_df["signal_pnl"] = 0.0

    buy_mask = pred_df["signal"] == "BUY"
    sell_mask = pred_df["signal"] == "SELL"

    # BUY P&L: buy at current actual + spread, profit if next 24h avg > entry
    pred_df.loc[buy_mask, "signal_pnl"] = (
        pred_df.loc[buy_mask, "actual_next_24h_mean"] - pred_df.loc[buy_mask, "actual"]
    ) - ROUND_TRIP_COST
    # SELL P&L: sell at current actual - spread, profit if next 24h avg < entry
    pred_df.loc[sell_mask, "signal_pnl"] = (
        pred_df.loc[sell_mask, "actual"] - pred_df.loc[sell_mask, "actual_next_24h_mean"]
    ) - ROUND_TRIP_COST

    # Signal performance stats
    buy_signals = pred_df[buy_mask].dropna(subset=["signal_pnl"])
    sell_signals = pred_df[sell_mask].dropna(subset=["signal_pnl"])
    all_signal_pnl = pred_df[buy_mask | sell_mask].dropna(subset=["signal_pnl"])

    buy_hit_rate = (buy_signals["signal_pnl"] > 0).mean() * 100 if len(buy_signals) > 0 else 0
    sell_hit_rate = (sell_signals["signal_pnl"] > 0).mean() * 100 if len(sell_signals) > 0 else 0
    total_hit_rate = (all_signal_pnl["signal_pnl"] > 0).mean() * 100 if len(all_signal_pnl) > 0 else 0

    avg_win = all_signal_pnl.loc[all_signal_pnl["signal_pnl"] > 0, "signal_pnl"].mean() if (all_signal_pnl["signal_pnl"] > 0).any() else 0
    avg_loss = all_signal_pnl.loc[all_signal_pnl["signal_pnl"] <= 0, "signal_pnl"].mean() if (all_signal_pnl["signal_pnl"] <= 0).any() else 0
    total_pnl = all_signal_pnl["signal_pnl"].sum()
    sharpe = all_signal_pnl["signal_pnl"].mean() / all_signal_pnl["signal_pnl"].std() * np.sqrt(252) if all_signal_pnl["signal_pnl"].std() > 0 else 0

    logger.info(f"Signal backtest: hit rate={total_hit_rate:.1f}%, avg win=€{avg_win:.2f}, avg loss=€{avg_loss:.2f}")
    logger.info(f"Total P&L (1 MW per signal): €{total_pnl:.0f}, Sharpe: {sharpe:.2f}")

    # --- Write comprehensive trading view ---
    lines = [
        "\n\n# Prompt Curve Translation\n",
        "## Method\n",
        "1. Aggregate hourly forecasts into delivery-period averages (baseload, peak, off-peak)",
        "2. Compare model fair value to forward price reference",
        "3. Generate confidence-weighted directional signal",
        "4. Size position proportional to edge magnitude × model confidence\n",
        "## Trading View (Last Week of Test)\n",
        "### Fair Value Estimates\n",
        "| Product | Model Fair Value | 90% Interval | Forward Ref | Edge |",
        "|---------|-----------------|--------------|-------------|------|",
        f"| Week-ahead baseload | €{week_base:.2f}/MWh | [€{week_q10:.2f}, €{week_q90:.2f}] | €{forward_ref:.2f} | €{edge:+.2f} |",
        f"| Week-ahead peak | €{week_peak:.2f}/MWh | - | - | - |",
        f"| Week-ahead off-peak | €{week_offpeak:.2f}/MWh | - | - | - |",
        f"| Peak/off-peak spread | €{week_peak - week_offpeak:.2f}/MWh | - | - | - |",
        "",
        "### Position Recommendation\n",
        f"- **Direction:** {edge_direction} prompt-week baseload",
        f"- **Edge:** €{edge:+.2f}/MWh (model vs forward reference)",
        f"- **Model confidence:** {avg_confidence:.0%}",
        f"- **Recommended size:** {position_size} MW",
        f"- **Product:** EEX/EPEX prompt-week baseload future",
        f"- **Expected P&L (if correct):** €{expected_pnl:,.0f} ({position_size} MW × €{abs(edge):.2f}/MWh × {hours_in_week}h)",
        f"- **Max adverse (90% interval):** €{max_loss:,.0f}",
        f"- **Risk/reward ratio:** {abs(expected_pnl/max_loss):.2f}" if max_loss != 0 else "- **Risk/reward ratio:** N/A",
        "",
        "### Signal Performance (Backtest)\n",
        "| Signal | Count | Next-24h Directional Accuracy |",
        "|--------|-------|------------------------------|",
        f"| BUY | {len(buy_signals)} | {buy_hit_rate:.1f}% profitable |",
        f"| SELL | {len(sell_signals)} | {sell_hit_rate:.1f}% profitable |",
        f"| Combined | {len(all_signal_pnl)} | {total_hit_rate:.1f}% hit rate |",
        "",
        "### P&L Backtest (1 MW per signal, 24h holding period)\n",
        f"- **Total P&L:** €{total_pnl:,.0f}",
        f"- **Avg win:** €{avg_win:.2f}/MWh",
        f"- **Avg loss:** €{avg_loss:.2f}/MWh",
        f"- **Win/loss ratio:** {abs(avg_win/avg_loss):.2f}x" if avg_loss != 0 else "- **Win/loss ratio:** N/A",
        f"- **Annualised Sharpe:** {sharpe:.2f}",
        f"- **Signals per day:** {len(all_signal_pnl) / 92:.1f}",
        f"- **Transaction cost:** €{ROUND_TRIP_COST:.2f}/MWh round-trip (€{SPREAD_PER_SIDE:.2f}/MWh per side)",
        "",
        f"*Note: P&L includes €{ROUND_TRIP_COST:.2f}/MWh round-trip transaction cost (bid-ask spread). No slippage or position limits applied.*",
        f"*Forward reference uses exponentially-weighted 7-day mean (halflife=72h) as proxy for EEX prompt-week settlement.*",
        "",
        "### How to Express This View\n",
        f"1. **Primary:** {edge_direction} {position_size} MW prompt-week baseload at EEX",
        f"   - Entry: current forward (€{forward_ref:.0f}), Target: model fair value (€{week_base:.0f})",
        f"   - Stop-loss: exit if forward moves {'above' if edge < 0 else 'below'} €{(forward_ref + edge * 2):.0f} (2× edge against)",
        f"2. **Shape trade:** If peak/off-peak spread ({week_peak - week_offpeak:.0f} €/MWh) diverges from historical ({rolling_std.mean():.0f} €/MWh std), trade the spread",
        f"3. **Hourly blocks:** Specific hours where |deviation| > 2σ are candidates for block trades",
        "",
        "### Signal Invalidation Conditions\n",
        "The position should be CLOSED or signal IGNORED if:",
        "- Wind forecast revision > 5 GW from yesterday (supply fundamental shifted)",
        "- Unplanned power plant outage > 2 GW capacity (supply shock not in model)",
        "- Gas price (TTF front-month) moves > 10% intraday (marginal cost shifted)",
        "- Interconnector outage DE↔neighbouring zone (flow disrupted)",
        "- Model error over last 3 days exceeds 2× historical average MAE",
        f"- Prediction interval width > 2× median (€{2*median_width:.0f}/MWh = low confidence → reduce size or exit)",
        "",
        "### Data Limitations & Production Upgrades\n",
        "| This Prototype | Production Version |",
        "|---------------|-------------------|",
        "| Forward proxy (EWM 7-day, halflife=72h) | EEX prompt-week baseload settlement (daily) |",
        f"| Transaction cost: €{ROUND_TRIP_COST:.2f}/MWh flat | Venue-specific spread + market impact model |",
        "| DA forecasts from ENTSO-E A69 (batch) | Live streaming ENTSO-E A69 feed (real-time) |",
        "| No gas/carbon features | TTF front-month + EU ETS settlement (ICE/EEX) |",
        "| No cross-border flows | ENTSO-E scheduled commercial exchanges |",
        "| No outage data | ENTSO-E unavailability (A78) |",
        "| Static thresholds | Adaptive thresholds calibrated weekly |",
        "",
        f"### Signal Distribution (Full Test Period)\n",
        f"| Signal | Count | % |",
        f"|--------|-------|---|",
    ]
    for sig, count in signal_counts.items():
        pct = 100 * count / len(pred_df)
        lines.append(f"| {sig} | {count} | {pct:.1f}% |")

    # Append to model results
    results_path = OUTPUT_DIR / "model_results.md"
    with open(results_path, "a") as f:
        f.write("\n".join(lines))
    logger.info(f"Curve translation appended to {results_path}")
