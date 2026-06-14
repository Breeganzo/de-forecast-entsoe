# DE-LU Day-Ahead Price Forecasting & Prompt Curve Translation

**Anthony Breeganzo Thomas** | anthonybreeganzo02@gmail.com | June 2026

---

## 1. Data Ingestion & Quality

I built an hourly DE-LU dataset from the ENTSO-E Transparency Platform REST API (Base URL: `https://web-api.tp.entsoe.eu/api`, Postman documentation: https://documenter.getpostman.com/view/7009892/2s93JtP3F6). I chose DE-LU because it is Europe's most liquid power market with sufficient renewable penetration to make short-term forecasting commercially meaningful.

**Endpoints used:**

- Day-Ahead Prices (documentType A44, hourly, EIC 10Y1001A1001A82H)
- Actual Total Load (A65, processType A16, 15-min resampled to hourly)
- Wind Onshore + Offshore Generation (A75, psrType B19/B18)
- Solar Generation (A75, psrType B16)
- Gas Generation (A75, psrType B04) — marginal fuel in the merit order
- Day-Ahead Forecasts: wind, solar, load (A69, A65/A01) — published pre-auction

**Dataset:** 17,520 hourly rows (June 2024 – June 2026), 100% coverage, zero gaps, zero duplicates. All timestamps converted from UTC to Europe/Berlin with 4 DST transitions verified (2 spring-forward 23h days, 2 fall-back 25h days). Negative prices retained — they represent real wind curtailment events where generators pay to dump excess power (prices reached −€500/MWh in our dataset, consistent with EPEX Spot DE-LU historical behaviour).

**Data cleaning:**

- Solar forecast NaN at night (40% of rows) → filled with 0 (solar panels produce zero output at night — domain knowledge, not statistical imputation)
- 1 missing load forecast hour → forward-filled from previous hour
- No mean/median imputation applied anywhere — extreme values are genuine market events
- 168 rows dropped from feature engineering (7-day lag warmup, structural not missing)

**QA checks (6 automated):** Missingness per column, duplicate timestamps, physical bounds (price: −500 to 4000 €/MWh; load: 25–85 GW; renewables: 0–100 GW), DST transition presence, coverage completeness, hourly gap detection. Full report: `outputs/data_qa_report.md`.

---

## 2. Forecasting

**Target:** Next-day hourly DA prices (24h ahead, Option A).  
**Validation:** Walk-forward with expanding training window. For each of 92 test days, the model trains on all prior history (starting at 632 days, expanding to 723 days) and predicts the next 24 hours. No information leakage — all lagged features use data from t−24h or earlier.

**Features (30 total):** 25 lagged features (calendar encodings, price/load/wind/solar/gas lags at 24h and 168h, 7-day rolling statistics, wind penetration ratio, renewable share, price momentum, forecast error lags) plus 5 forward features (ENTSO-E DA forecasts for wind, solar, load, residual load — published before the DA auction and available to all market participants at decision time).

| Model | MAE (€/MWh) | vs Naive | Direction |
|-------|:-----------:|:--------:|:---------:|
| Seasonal Naive (same hour −7d) | 42.39 | baseline | — |
| Ridge Regression (α=100) | 30.15 | −29% | — |
| **LightGBM (25 lagged features)** | **28.57** | **−33%** | **54.5%** |
| LightGBM (30 features, + DA forecasts) | 16.59 | −61% | 79.1% |

Ridge captures the linear merit-order relationship (load → marginal cost); LightGBM adds 4 percentage points by learning nonlinear interactions (e.g., high wind × low demand → negative price risk; high gas dispatch × peak load → scarcity premium). Direction accuracy of 54.5% on lagged-only confirms the DA market is approximately efficient at 24h horizon — this is expected and healthy. With DA forecasts the direction accuracy jumps to 79.1% because TSO predictions resolve most next-day uncertainty about renewable output.

**submission.csv uses the lagged-only model** — no forward information leakage. The forward-enhanced model demonstrates what's achievable when TSO forecasts are incorporated.

**Prediction intervals:** P2/P98 quantile bounds calibrated via split-conformal prediction. First 30% of test = calibration set; the 96th percentile of conformity scores widens intervals for remaining predictions. Empirical coverage: 92.9%. Negative lower bounds (e.g., −€0.45) are correct — German DA prices genuinely go negative during high-wind/low-demand periods.

**Explainability:** SHAP TreeExplainer computed on 192 test samples shows exactly which features drove each prediction. Top drivers: price_7d_std (volatility regime), wind_7d_mean (renewable supply pressure), gas_lag_24h (marginal fuel indication).

---

## 3. Prompt Curve Translation & Trading

The model produces a daily fair-value curve (24 hourly price estimates). I translate these into desk-relevant trading signals a power trader can act on immediately:

**Signal logic:**

- **BUY:** Model fair value > market price AND P10 (pessimistic bound) > market — even in the worst case, the model thinks the market is cheap
- **SELL:** Model fair value < market price AND P90 (optimistic bound) < market — even in the best case, the model thinks the market is rich
- **HOLD:** Otherwise — model is not confident enough to trade (this is 95.4% of hours)

Edge threshold: signal only fires when mispricing exceeds 0.5× the rolling standard deviation of recent forecast residuals. Forward reference: EWM mean (halflife=72h) as prompt-week proxy.

| Metric | Value |
|--------|-------|
| Signals fired | 102 / 2,207 hours (4.6%) |
| Hit rate (directional) | 79.4% |
| BUY accuracy | 94.9% (39 signals) |
| SELL accuracy | 69.8% (63 signals) |
| Net P&L (1 MW per signal) | +€8,047 (after transaction costs) |
| Avg win / Avg loss | €104.64 / €20.42 = 5.1× |
| Sharpe ratio (annualised) | 12.13 |
| Transaction cost applied | €0.10/MWh round-trip |

**How to express:** EEX/EPEX prompt-week baseload futures. Size proportional to edge magnitude × prediction interval confidence, capped at 20 MW. Shape trades available when peak/off-peak spread deviates from model by > 1σ.

**Invalidation — close position if:** Wind forecast revision > 5 GW (supply fundamental shifted), unplanned outage > 2 GW (merit-order disruption), TTF gas intraday move > 10% (marginal cost shifted), or model error > 2× MAE over 3 consecutive days (model regime has broken down).

---

## 4. AI-Accelerated Workflow

Four programmatic LLM calls (Groq API, Llama 3.3 70B) integrated into the pipeline — not manual chat, fully auditable:

1. **QA rule generation** — LLM receives dataset schema + 10 sample rows → proposes domain-specific validation rules (e.g., "solar must be zero between 22:00–05:00") → code executes each rule against the full 17,520-row dataset
2. **Market commentary** — LLM generates a daily drivers summary from computed statistics only; it never receives raw data or permission to invent numbers
3. **Feature explanation** — Post-model, LLM explains the top-10 features in trading terms: merit-order position, dispatch logic, weather cycles
4. **Model diagnosis** — LLM receives MAE/RMSE/direction metrics → provides structured diagnosis with actionable improvement suggestions specific to European power markets

**Controls:** Every call logged to `logs/ai_run.jsonl` with full prompt, response, latency, and token count. Deterministic fallbacks execute if the LLM API is unavailable. Pipeline is fully reproducible without LLM access.

---

## 5. Limitations & Next Steps

- **No gas/carbon prices** — gas generation indicates marginal fuel regime, but TTF/EUA spot prices would capture cost directly
- **No interconnector flows** — misses cross-border price convergence dynamics across DE's 11 borders
- **Static daily retraining** — production system would incorporate intraday ID1 auction results for real-time adaptation
- **Single market** — framework designed to parallelise across bidding zones by abstracting EIC codes

---

## Reproducibility

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
cp .env.example .env   # add ENTSOE_API_KEY + optional GROQ_API_KEY
python3 main.py        # ~17 min (walk-forward on 92 test days)
```

**Outputs:** `outputs/submission.csv` (2,207 hourly forecasts), `outputs/figures/` (7 diagnostic plots incl. SHAP), `outputs/model_results.md`, `logs/ai_run.jsonl`
