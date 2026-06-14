# Model Results

## Model Comparison (Walk-Forward Validation, 90-day test period)

We run four models to isolate what drives performance:

| Model | Features | MAE (€/MWh) | RMSE | P95 | vs Naive |
|-------|----------|-------------|------|-----|----------|
| Seasonal Naive | — | 42.39 | 60.66 | 117.96 | baseline |
| Ridge Regression | 25 (lagged) | 30.15 | 42.18 | 84.10 | 28.9% |
| **LightGBM (lagged-only)** | 25 (lagged) | **28.57** | 40.91 | 79.61 | **32.6%** |
| LightGBM (+ DA forecasts) | 30 (lagged + forward) | 16.59 | 26.20 | 45.34 | 60.9% |

### What This Table Tells a Trader

1. **Ridge vs Naive (29% improvement):** Linear relationships between fundamentals and price already explain a significant chunk. The merit-order (residual load → marginal cost) is approximately linear within a regime.
2. **LightGBM vs Ridge (4pp additional):** Gradient boosting captures nonlinear interactions: e.g., wind penetration only crashes prices below zero when BOTH wind is high AND demand is low (weekend nights). Ridge cannot model this interaction.
3. **Forward-enhanced vs Lagged-only (28pp additional):** Day-ahead wind/solar/load forecasts add value because they represent the information the market actually prices at the DA auction. These are real ENTSO-E A69 forecasts (published pre-auction, ~12:00 D-1). The improvement quantifies the value of TSO forecasts for short-term price prediction.

## Deep Validation (Lagged-Only Model — Honest Metrics)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Direction Accuracy | 54.5% | Model predicts correct price direction (up/down) |
| Prediction Interval Coverage | 92.9% | % of actuals within [P2, P98] band (target: 96%) |
| Avg Interval Width | €136.9/MWh | Narrower = more confident |
| Calm Day MAE | €26.46/MWh | Performance on low-volatility days |
| Volatile Day MAE | €30.69/MWh | Performance on high-volatility days |
| Peak Hour MAE (8-20) | €33.20/MWh | Performance during trading hours |
| Off-peak MAE | €23.10/MWh | Performance overnight |
| Diebold-Mariano stat | 10.70 | Statistical test: LightGBM vs naive |
| DM p-value | 0.0000 | Highly significant (p<0.01) |

### Why the Model Struggles at Peak Hours

Peak MAE (€33) is 1.4× worse than off-peak (€23). This is expected because:
- Peak hours (8:00-20:00 weekdays) are when **ramping, storage arbitrage, and interconnector congestion** create price spikes
- These events are driven by real-time balancing — fundamentals alone cannot predict them
- In production: adding EEX intraday auction results + balancing market data would improve peak forecasting

### Prediction Interval Calibration

The quantile regression (P2/P98) is calibrated using **split-conformal prediction**: the first 30% of the test set computes conformity scores (how much actuals exceed raw quantile bounds), then the 96th percentile of those scores widens the intervals on the remaining data. This lightweight MAPIE-equivalent approach improves coverage toward the 96% target while keeping intervals informatively narrow. Empirical coverage: 93%.

## Validation Approach

- **Method:** Walk-forward (expanding window) — mimics live trading exactly
- **Training:** Min 365 days, expanding daily
- **Test period:** Last 90 days (Mar-May 2026)
- **Prediction:** Next-day hourly prices (24 hours ahead)
- **No leakage:** All lagged features use data from t-24h or earlier
- **Forward features:** Day-ahead forecasts (available pre-auction, 12:00 D-1)
- **Quantile regression:** P2/P98 trained alongside point forecast (wider for fat tails)
- **Submission uses lagged-only model** (honest, reproducible without DA forecast API)

## Top 10 Features — Lagged-Only Model (by importance)

| Rank | Feature | Importance | Why It Matters |
|------|---------|-----------|----------------|
| 1 | price_7d_std | 1189 | Weekly volatility = model scales uncertainty in volatile periods |
| 2 | wind_7d_mean | 1154 | Wind regime = sustained high wind depresses prices for days |
| 3 | price_7d_mean | 1102 | Weekly average = regime detection (high vs low price environment) |
| 4 | load_24h_mean | 896 | Load level = demand fundamental (high load → expensive thermal on margin) |
| 5 | price_lag_24h | 788 | Yesterday's price = strongest autoregressive signal (mean reversion) |
| 6 | wind_lag_168h | 681 | Last week wind = wind patterns have ~7-day cycles (weather systems) |
| 7 | gas_lag_24h | 677 | Gas gen indicates thermal dispatch depth — when gas ramps up, merit order shifts right, marginal cost rises |
| 8 | price_lag_48h | 646 | Two days ago = captures multi-day weather patterns |
| 9 | price_lag_168h | 588 | Same hour last week = weekly seasonality baseline |
| 10 | solar_forecast_error_lag24 | 515 | Persistent TSO solar over-prediction → market priced more supply than materialised → upward price correction |

## Top 5 Features — Forward-Enhanced Model

| Rank | Feature | Importance | Why It Matters |
|------|---------|-----------|----------------|
| 1 | residual_load_forecast | 1082 | Residual load = how much thermal needed = determines marginal cost |
| 2 | wind_forecast_da | 826 | DA wind forecast = what the market prices in at the auction |
| 3 | price_lag_24h | 791 | Yesterday's price = strongest autoregressive signal (mean reversion) |
| 4 | price_7d_std | 663 | Weekly volatility = model scales uncertainty in volatile periods |
| 5 | price_lag_168h | 628 | Same hour last week = weekly seasonality baseline |

## Target Choice Justification

**Option A (recommended):** Forecast next-day hourly DA prices.
- Hourly granularity captures peak/off-peak shape dynamics
- Weekly/monthly averages derived from hourly forecasts (see Task 3)
- Enables hour-by-hour mispricing detection for traders
- Prediction intervals enable position sizing (wider interval = less confidence = smaller size)

## Forward Features: Real ENTSO-E DA Forecasts

The forward-enhanced model uses **real** DA generation/load forecasts from ENTSO-E API:
- Wind forecast: A69 endpoint, psrType B19 (onshore) + B18 (offshore)
- Solar forecast: A69 endpoint, psrType B16
- Load forecast: A65 endpoint, processType A01

These forecasts are published before the DA auction (~12:00 D-1) and represent 
the fundamental information available to market participants at decision time.

**The submission.csv uses the lagged-only model** as the conservative baseline — 
it requires no forward data access and represents what any participant could reproduce.

# Prompt Curve Translation

## Method

1. Aggregate hourly forecasts into delivery-period averages (baseload, peak, off-peak)
2. Compare model fair value to forward price reference
3. Generate confidence-weighted directional signal
4. Size position proportional to edge magnitude × model confidence

## Trading View (Last Week of Test)

### Fair Value Estimates

| Product | Model Fair Value | 90% Interval | Forward Ref | Edge |
|---------|-----------------|--------------|-------------|------|
| Week-ahead baseload | €80.56/MWh | [€41.10, €118.62] | €92.80 | €-12.24 |
| Week-ahead peak | €34.57/MWh | - | - | - |
| Week-ahead off-peak | €109.58/MWh | - | - | - |
| Peak/off-peak spread | €-75.00/MWh | - | - | - |

### Position Recommendation

- **Direction:** SHORT prompt-week baseload
- **Edge:** €-12.24/MWh (model vs forward reference)
- **Model confidence:** 90%
- **Recommended size:** 18.0 MW
- **Product:** EEX/EPEX prompt-week baseload future
- **Expected P&L (if correct):** €37,008 (18.0 MW × €12.24/MWh × 168h)
- **Max adverse (90% interval):** €78,086
- **Risk/reward ratio:** 0.47

### Signal Performance (Backtest)

| Signal | Count | Next-24h Directional Accuracy |
|--------|-------|------------------------------|
| BUY | 39 | 94.9% profitable |
| SELL | 63 | 69.8% profitable |
| Combined | 102 | 79.4% hit rate |

### P&L Backtest (1 MW per signal, 24h holding period)

- **Total P&L:** €8,047
- **Avg win:** €104.64/MWh
- **Avg loss:** €-20.42/MWh
- **Win/loss ratio:** 5.12x
- **Annualised Sharpe:** 12.13
- **Signals per day:** 1.1
- **Transaction cost:** €0.10/MWh round-trip (€0.05/MWh per side)

*Note: P&L includes €0.10/MWh round-trip transaction cost (bid-ask spread). No slippage or position limits applied.*
*Forward reference uses exponentially-weighted 7-day mean (halflife=72h) as proxy for EEX prompt-week settlement.*

### How to Express This View

1. **Primary:** SHORT 18.0 MW prompt-week baseload at EEX
   - Entry: current forward (€93), Target: model fair value (€81)
   - Stop-loss: exit if forward moves above €68 (2× edge against)
2. **Shape trade:** If peak/off-peak spread (-75 €/MWh) diverges from historical (62 €/MWh std), trade the spread
3. **Hourly blocks:** Specific hours where |deviation| > 2σ are candidates for block trades

### Signal Invalidation Conditions

The position should be CLOSED or signal IGNORED if:
- Wind forecast revision > 5 GW from yesterday (supply fundamental shifted)
- Unplanned power plant outage > 2 GW capacity (supply shock not in model)
- Gas price (TTF front-month) moves > 10% intraday (marginal cost shifted)
- Interconnector outage DE↔neighbouring zone (flow disrupted)
- Model error over last 3 days exceeds 2× historical average MAE
- Prediction interval width > 2× median (€154/MWh = low confidence → reduce size or exit)

### Data Limitations & Production Upgrades

| This Prototype | Production Version |
|---------------|-------------------|
| Forward proxy (EWM 7-day, halflife=72h) | EEX prompt-week baseload settlement (daily) |
| Transaction cost: €0.10/MWh flat | Venue-specific spread + market impact model |
| DA forecasts from ENTSO-E A69 (batch) | Live streaming ENTSO-E A69 feed (real-time) |
| No gas/carbon features | TTF front-month + EU ETS settlement (ICE/EEX) |
| No cross-border flows | ENTSO-E scheduled commercial exchanges |
| No outage data | ENTSO-E unavailability (A78) |
| Static thresholds | Adaptive thresholds calibrated weekly |

### Signal Distribution (Full Test Period)

| Signal | Count | % |
|--------|-------|---|
| HOLD | 2105 | 95.4% |
| SELL | 63 | 2.9% |
| BUY | 39 | 1.8% |
## AI-Generated Feature Explanations

1. price_7d_std: Volatility affects merit order.
2. wind_7d_mean: Wind supply impacts prices.
3. price_7d_mean: Recent prices influence bidding.
4. load_24h_mean: Demand drives price formation.
5. price_lag_24h: Yesterday's price sets today's floor.
6. wind_lag_168h: Weekly wind patterns matter.
7. gas_lag_24h: Gas prices influence power bidding.
8. price_lag_48h: 2-day price momentum is key.
9. price_lag_168h: Weekly price trends persist.
10. solar_forecast_error_lag24: Solar forecast errors impact supply.

## AI Model Diagnosis

The model performs well in off-peak hours, with a lower MAE of €23.10/MWh, but struggles during peak hours, with a higher MAE of €33.20/MWh, indicating potential issues with capturing demand-driven price spikes. The significant gap between peak and off-peak MAE values suggests that the model may be missing features related to European power market fundamentals, such as German-Austrian cross-border transmission capacities or French nuclear availability, which can impact peak-hour pricing. The large difference in MAE between peak and off-peak hours implies that the model could benefit from incorporating more detailed data on European power grid operations, such as EEX Phelix or APX ENDEX spot prices. To improve the model, incorporating data from the European Network of Transmission System Operators for Electricity (ENTSO-E) on cross-border transmission capacities and grid congestion could help better capture peak-hour price dynamics. Additionally, the model's relatively low direction accuracy of 54.5% and significant DM test p-value suggest that there may be other market-specific factors at play that are not being adequately captured.
