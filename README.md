# European Power Fair Value: DE-LU Day-Ahead Forecasting (ENTSO-E API)

This is the API-first version of my European power fair-value case study. It builds a two-year DE-LU hourly dataset directly from the **ENTSO-E Transparency Platform REST API**, checks the data like a trading desk would, trains a walk-forward Day-Ahead price model, and translates the hourly forecast into prompt-curve views.

I chose DE-LU because it is one of Europe's most liquid power markets and has enough wind/solar volatility to make the forecasting problem commercially meaningful. The project is designed so a reviewer can run one command, regenerate the outputs, and inspect every assumption through logs and reports.

## Data Source

**ENTSO-E Transparency Platform REST API**
- Base URL: `https://web-api.tp.entsoe.eu/api`
- Full Postman Documentation: https://documenter.getpostman.com/view/7009892/2s93JtP3F6
- Auth: API key via `securityToken` parameter (register at https://transparency.entsoe.eu)
- Market: DE-LU bidding zone (EIC: `10Y1001A1001A82H`)
- Period: June 2024 – June 2026 (2 years, hourly)

### Endpoints Used

| # | Data | documentType | Resolution |
|---|------|-------------|------------|
| 1 | Day-Ahead Prices | `A44` | PT60M (hourly) |
| 2 | Actual Total Load | `A65` + `processType=A16` | PT15M → resampled hourly |
| 3 | Wind Onshore Generation | `A75` + `psrType=B19` | PT15M → resampled hourly |
| 4 | Wind Offshore Generation | `A75` + `psrType=B18` | PT15M → resampled hourly |
| 5 | Solar Generation | `A75` + `psrType=B16` | PT15M → resampled hourly |
| 6 | **Gas Generation** | `A75` + `psrType=B04` | PT15M → resampled hourly |

See `src/data_sources.py` for full endpoint documentation with example requests. The Postman link is documentation only; API keys are read from `.env` and should never be committed.

## How This Maps to the Case Study

| Case study requirement | Where it is handled |
|---|---|
| Public data ingestion + QA | `src/ingest.py`, `src/quality.py`, `outputs/data_qa_report.md` |
| Forecasting + validation | `src/features.py`, `src/forecast.py`, `outputs/model_results.md` |
| Prompt curve translation | `src/curve_view.py`, appended section in `outputs/model_results.md` |
| Programmatic AI workflow | `src/ai_qa.py`, `logs/ai_run.jsonl`, AI sections in QA report |

## Quick Start: macOS / Linux with Python 3.13

```bash
cd de-forecast-entsoe
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env → add your ENTSO-E API key + optional GROQ_API_KEY

# Run the full pipeline
# First run fetches from API (~20 min total), subsequent runs use cache (~17 min)
python3 main.py
```

If your Python executable is named `python3` instead of `python3.13`, use `python3 -m venv venv`.

## Quick Start: Windows PowerShell with Python 3.13

```powershell
cd de-forecast-entsoe
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env
# Edit .env in Notepad or VS Code and add ENTSOE_API_KEY plus optional GROQ_API_KEY

python main.py
```

If PowerShell blocks activation, run this once for your user account and then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Windows CMD alternative:

```cmd
venv\Scripts\activate.bat
python main.py
```

All outputs are written to `outputs/`. Raw API responses are cached to `data/raw/` — delete that folder to force a re-fetch.

## Project Structure

```
de-forecast-entsoe/
├── main.py                 # Pipeline orchestrator (single entry point)
├── REPORT.md               # 1-3 page case-study submission report
├── REPORT.pdf              # PDF export of the report
├── src/
│   ├── config.py           # Central configuration (paths, params, API keys)
│   ├── ingest.py           # Task 1: ENTSO-E API fetch → parse XML → DataFrame
│   ├── data_sources.py     # API endpoint documentation
│   ├── quality.py          # Task 1: 6 data quality checks → markdown report
│   ├── ai_qa.py            # Task 4: LLM-driven QA rules + market commentary
│   ├── features.py         # Task 2: 25 lagged + 5 forward features (incl. gas, forecast errors)
│   ├── forecast.py         # Task 2: Dual walk-forward (lagged + forward) + Ridge baseline + SHAP
│   ├── curve_view.py       # Task 3: Prompt curve translation + mispricing signals + P&L backtest
│   └── figures.py          # Generates 7 submission figures (incl. SHAP attribution)
├── data/
│   ├── raw/                # Cached API responses as CSV (auto-generated)
│   └── processed/          # Parquet cache (auto-generated, gitignored)
├── outputs/
│   ├── submission.csv      # Out-of-sample predictions (id, y_pred)
│   ├── data_qa_report.md   # Deterministic QA report + AI-generated QA rules
│   ├── model_results.md    # Model metrics + prompt curve translation
│   └── figures/            # PNG charts
├── logs/
│   ├── pipeline.log        # Full execution log
│   └── ai_run.jsonl        # LLM call audit trail
├── requirements.txt        # Pinned Python dependencies
├── .env.example            # Template for API keys
└── .gitignore              # Excludes venv, .env, processed cache
```

## Pipeline Execution Order

```
python3 main.py
  │
  ├── 1. ingest.py      → Fetch from ENTSO-E API (or load cache) → ~17,500 hourly rows
  ├── 2. quality.py     → 6 QA checks → outputs/data_qa_report.md
  ├── 3. ai_qa.py       → Groq API: QA rules + commentary → logs/ai_run.jsonl
  ├── 4. features.py    → Engineer 25 lagged + 5 forward features
  ├── 5. forecast.py    → Dual walk-forward (lagged+Ridge, forward) → outputs/submission.csv
  ├── 6. curve_view.py  → Mispricing signals + P&L backtest → model_results.md
  └── 7. figures.py     → 7 PNGs → outputs/figures/
```

## Requirements

- Python 3.13 recommended (Python 3.11+ should also work)
- ENTSO-E API key (free, register at https://transparency.entsoe.eu)
- Groq API key (free tier, optional — fallback rules execute without it)
- ~20 minutes runtime (first run fetches data; subsequent runs use cache)

## Data Handling

- **17,520 hourly rows** from June 2024 – May 2026 (0 missing in price/load/wind)
- **Solar forecast NaN at night → filled with 0** (physically correct: no solar production at night)
- **168 rows dropped** from feature engineering (7-day lag warmup — structural, not missing data)
- **LightGBM handles remaining NaN natively** (e.g., daytime solar gaps at dawn/dusk transitions)
- **Walk-forward drops 0 test days** — all 92 days have valid data

## What to Review First

For a quick evaluation, open these files in order:

1. `REPORT.md` — 1-3 page case-study narrative
2. `outputs/data_qa_report.md` — source coverage, DST, missingness, outliers, deterministic checks, AI QA rules
3. `outputs/model_results.md` — model metrics and prompt-curve translation
4. `logs/ai_run.jsonl` — LLM prompt/output audit trail
5. `outputs/figures/` — 7 charts: prices+renewables, forecast+PI, feature importance, hourly error heatmap, error distribution, temporal stability, SHAP attribution
