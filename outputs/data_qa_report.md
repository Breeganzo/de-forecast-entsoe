# Data Quality Report

**Market:** DE-LU (Germany/Luxembourg)
**Source:** ENTSO-E Transparency Platform REST API
**API Docs:** https://documenter.getpostman.com/view/7009892/2s93JtP3F6
**Generated:** 2026-06-13 17:19

## Summary Statistics

| Column | Count | Mean | Std | Min | 25% | 50% | 75% | Max |
|--------|-------|------|-----|-----|-----|-----|-----|-----|
| price_eur_mwh | 17520 | 91.9 | 58.4 | -500.0 | 66.3 | 95.3 | 120.2 | 899.9 |
| load_mw | 17520 | 53852.9 | 9316.8 | 32813.3 | 46188.2 | 53954.2 | 60402.6 | 79007.6 |
| wind_mw | 17520 | 15103.1 | 10771.0 | 47.2 | 6348.4 | 12603.6 | 21757.5 | 53413.0 |
| solar_mw | 17520 | 8334.7 | 12600.3 | 0.0 | 10.6 | 288.5 | 13397.8 | 56361.9 |
| wind_solar_mw | 17520 | 23437.8 | 14186.9 | 132.0 | 11502.7 | 21543.0 | 33929.2 | 66385.9 |
| gas_mw | 17520 | 6910.5 | 4218.2 | 1310.9 | 3461.9 | 5896.9 | 9373.3 | 20161.6 |
| wind_forecast_mw | 17520 | 14997.3 | 10796.4 | 248.2 | 6357.3 | 12355.6 | 21461.6 | 52510.4 |
| solar_forecast_mw | 10512 | 14032.0 | 13621.6 | 0.0 | 1665.1 | 9823.2 | 24058.3 | 56908.1 |
| load_forecast_mw | 17519 | 53995.6 | 8883.8 | 33769.8 | 46688.2 | 53690.9 | 61205.2 | 74586.0 |

## Coverage

- **Start:** 2024-06-01 00:00:00+02:00
- **End:** 2026-05-31 23:00:00+02:00
- **Expected hours:** 17520
- **Actual hours:** 17520
- **Coverage:** 100.0%

## Missingness

| Column | Missing | % |
|--------|---------|---|
| price_eur_mwh | 0 | 0.0% |
| load_mw | 0 | 0.0% |
| wind_mw | 0 | 0.0% |
| solar_mw | 0 | 0.0% |
| wind_solar_mw | 0 | 0.0% |
| gas_mw | 0 | 0.0% |
| wind_forecast_mw | 0 | 0.0% |
| solar_forecast_mw | 7008 | 40.0% |
| load_forecast_mw | 1 | 0.006% |

## Duplicates

Duplicate timestamps: **0**


## Outliers (Physical Bounds)

| Column | Below Min | Above Max | Bounds |
|--------|-----------|-----------|--------|
| price_eur_mwh | 0 | 0 | [-500, 4000] |
| load_mw | 0 | 0 | [25000, 85000] |
| wind_mw | 0 | 0 | [0, 100000] |
| solar_mw | 0 | 0 | [0, 100000] |
| wind_solar_mw | 0 | 0 | [0, 100000] |

## DST Transitions (Timezone: Europe/Berlin)

| Date | Hours | Type |
|------|-------|------|
| 2024-10-27 | 25 | fall-back (25h) |
| 2025-03-30 | 23 | spring-forward (23h) |
| 2025-10-26 | 25 | fall-back (25h) |
| 2026-03-29 | 23 | spring-forward (23h) |

## Gaps

No gaps detected — continuous hourly series.


## Notable Observations

- **Negative prices:** 1062 hours (6.06%)
  - Min negative price: €-499.99/MWh
  - These occur when renewables oversupply (wind+solar > demand)
## AI-Generated QA Rules

Rules proposed by: llama-3.3-70b-versatile
Results: 3 passed, 2 failed out of 5

  [PASS] price_range_check: Price must be within the range [-500.0, 899.9]
  [FAIL] load_range_check: Load must be within the range [32813.3, 79007.6]
  [PASS] no_negative_renewables: Solar and wind power must not be negative
  [FAIL] forecast_not_exceeding_actual: Load forecast must not exceed actual load by more than 10%
  [PASS] wind_solar_sum_check: Wind and solar sum must match the wind_solar_mw column
## AI-Generated Market Commentary

The DE-LU Day-Ahead price increased by €4.0/MWh to an average of €90.2/MWh, driven by a decrease in wind generation from 10919 MW to 5574 MW. The price range was quite volatile, spanning from €-7.8 to €162.2/MWh, with 4 hours of negative pricing. The load decreased to 42431 MW, while solar generation reached 12374 MW. Overall, the reduction in wind generation appears to have had a significant impact on today's price movement.

*Source metrics: {"avg_price_today": "90.2", "avg_price_yesterday": "86.2", "price_change": "4.0", "peak_price": "162.2", "min_price": "-7.8", "avg_wind": "5574", "avg_wind_yesterday": "10919", "avg_solar": "12374", "avg_load": "42431", "avg_load_yesterday": "44792", "negative_hours": "4"}*