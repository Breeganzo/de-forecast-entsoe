"""
Task 1 (continued): Data Quality checks.

Checks performed:
1. Missingness — % NaN per column, gap analysis
2. Duplicates — repeated timestamps
3. Outliers — values outside physical bounds
4. DST transitions — 23h and 25h days
5. Coverage — completeness per field and time period

Missing data strategy (if gaps were found):
- Gaps < 3 hours: forward-fill (hourly prices have strong autocorrelation)
- Gaps > 3 hours: exclude day from training (avoid teaching model interpolated patterns)
- Current dataset: 0% missingness, so no imputation applied.
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    PRICE_MIN, PRICE_MAX, LOAD_MIN, LOAD_MAX,
    WIND_SOLAR_MIN, WIND_SOLAR_MAX, OUTPUT_DIR, TIMEZONE
)

logger = logging.getLogger(__name__)


def check_missingness(df: pd.DataFrame) -> dict:
    """Check for missing values per column."""
    total = len(df)
    results = {}
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct = 100 * n_missing / total
        results[col] = {"missing": int(n_missing), "pct": round(pct, 3)}
        if n_missing > 0:
            logger.warning(f"  {col}: {n_missing} missing ({pct:.2f}%)")
        else:
            logger.info(f"  {col}: complete (0 missing)")
    return results


def check_duplicates(df: pd.DataFrame) -> int:
    """Check for duplicate timestamps in the index."""
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        logger.warning(f"  {n_dupes} duplicate timestamps found!")
    else:
        logger.info("  No duplicate timestamps")
    return int(n_dupes)


def check_outliers(df: pd.DataFrame) -> dict:
    """Check for values outside physical bounds."""
    bounds = {
        "price_eur_mwh": (PRICE_MIN, PRICE_MAX),
        "load_mw": (LOAD_MIN, LOAD_MAX),
        "wind_mw": (WIND_SOLAR_MIN, WIND_SOLAR_MAX),
        "solar_mw": (WIND_SOLAR_MIN, WIND_SOLAR_MAX),
        "wind_solar_mw": (WIND_SOLAR_MIN, WIND_SOLAR_MAX),
    }
    results = {}
    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            continue
        below = (df[col] < lo).sum()
        above = (df[col] > hi).sum()
        results[col] = {
            "below_min": int(below),
            "above_max": int(above),
            "bounds": [lo, hi]
        }
        if below + above > 0:
            logger.warning(f"  {col}: {below} below {lo}, {above} above {hi}")
        else:
            logger.info(f"  {col}: all within [{lo}, {hi}]")
    return results


def check_dst(df: pd.DataFrame) -> list[dict]:
    """Verify DST transitions exist (proves timezone handling is correct)."""
    daily_hours = df.groupby(df.index.date).size()
    transitions = []
    for date, hours in daily_hours.items():
        if hours != 24:
            t_type = "fall-back (25h)" if hours > 24 else "spring-forward (23h)"
            transitions.append({
                "date": str(date),
                "hours": int(hours),
                "type": t_type
            })
    logger.info(f"  DST transitions found: {len(transitions)}")
    for t in transitions:
        logger.info(f"    {t['date']}: {t['hours']}h ({t['type']})")
    return transitions


def check_coverage(df: pd.DataFrame) -> dict:
    """Check data coverage: start, end, total hours, expected vs actual."""
    start = df.index.min()
    end = df.index.max()
    # Expected hours = difference between start and end
    expected = int((end - start).total_seconds() / 3600) + 1
    actual = len(df)
    coverage_pct = 100 * actual / expected if expected > 0 else 0

    result = {
        "start": str(start),
        "end": str(end),
        "expected_hours": expected,
        "actual_hours": actual,
        "coverage_pct": round(coverage_pct, 2),
    }
    logger.info(f"  Coverage: {actual}/{expected} hours ({coverage_pct:.1f}%)")
    logger.info(f"  Range: {start} → {end}")
    return result


def check_gaps(df: pd.DataFrame) -> list[dict]:
    """Find gaps (consecutive missing hours) in the time series."""
    # Expected: every hour should be present
    expected_freq = pd.tseries.frequencies.to_offset("h")
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h", tz=TIMEZONE)
    missing_hours = full_idx.difference(df.index)

    gaps = []
    if len(missing_hours) > 0:
        # Group consecutive missing hours into gaps
        diffs = missing_hours.to_series().diff()
        gap_starts = missing_hours[0:1].append(missing_hours[diffs > pd.Timedelta("1h")])
        logger.warning(f"  {len(missing_hours)} missing hours in time series")
        for start in gap_starts[:10]:  # log first 10 gaps
            gaps.append({"start": str(start)})
            logger.warning(f"    Gap starting at: {start}")
    else:
        logger.info("  No gaps in hourly time series")

    return gaps


def generate_qa_report(df: pd.DataFrame) -> str:
    """Run all QA checks and generate markdown report."""
    logger.info("=" * 60)
    logger.info("DATA QUALITY ASSESSMENT")
    logger.info("=" * 60)

    # Run checks
    logger.info("\n[1] Missingness:")
    missing = check_missingness(df)

    logger.info("\n[2] Duplicates:")
    duplicates = check_duplicates(df)

    logger.info("\n[3] Outliers:")
    outliers = check_outliers(df)

    logger.info("\n[4] DST transitions:")
    dst = check_dst(df)

    logger.info("\n[5] Coverage:")
    coverage = check_coverage(df)

    logger.info("\n[6] Gaps:")
    gaps = check_gaps(df)

    # Generate markdown report
    report = []
    report.append("# Data Quality Report\n")
    report.append(f"**Market:** DE-LU (Germany/Luxembourg)")
    report.append(f"**Source:** ENTSO-E Transparency Platform REST API")
    report.append(f"**API Docs:** https://documenter.getpostman.com/view/7009892/2s93JtP3F6")
    report.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Summary stats
    report.append("## Summary Statistics\n")
    report.append("| Column | Count | Mean | Std | Min | 25% | 50% | 75% | Max |")
    report.append("|--------|-------|------|-----|-----|-----|-----|-----|-----|")
    desc = df.describe()
    for col in desc.columns:
        row = desc[col]
        report.append(
            f"| {col} | {row['count']:.0f} | {row['mean']:.1f} | {row['std']:.1f} "
            f"| {row['min']:.1f} | {row['25%']:.1f} | {row['50%']:.1f} "
            f"| {row['75%']:.1f} | {row['max']:.1f} |"
        )

    # Coverage
    report.append(f"\n## Coverage\n")
    report.append(f"- **Start:** {coverage['start']}")
    report.append(f"- **End:** {coverage['end']}")
    report.append(f"- **Expected hours:** {coverage['expected_hours']}")
    report.append(f"- **Actual hours:** {coverage['actual_hours']}")
    report.append(f"- **Coverage:** {coverage['coverage_pct']}%")

    # Missingness
    report.append(f"\n## Missingness\n")
    report.append("| Column | Missing | % |")
    report.append("|--------|---------|---|")
    for col, info in missing.items():
        report.append(f"| {col} | {info['missing']} | {info['pct']}% |")

    # Duplicates
    report.append(f"\n## Duplicates\n")
    report.append(f"Duplicate timestamps: **{duplicates}**\n")

    # Outliers
    report.append(f"\n## Outliers (Physical Bounds)\n")
    report.append("| Column | Below Min | Above Max | Bounds |")
    report.append("|--------|-----------|-----------|--------|")
    for col, info in outliers.items():
        report.append(
            f"| {col} | {info['below_min']} | {info['above_max']} "
            f"| [{info['bounds'][0]}, {info['bounds'][1]}] |"
        )

    # DST
    report.append(f"\n## DST Transitions (Timezone: Europe/Berlin)\n")
    if dst:
        report.append("| Date | Hours | Type |")
        report.append("|------|-------|------|")
        for t in dst:
            report.append(f"| {t['date']} | {t['hours']} | {t['type']} |")
    else:
        report.append("*No DST transitions detected — check timezone handling!*\n")

    # Gaps
    report.append(f"\n## Gaps\n")
    if gaps:
        report.append(f"Total missing hours: {len(gaps)}\n")
        for g in gaps[:10]:
            report.append(f"- Gap at: {g['start']}")
    else:
        report.append("No gaps detected — continuous hourly series.\n")

    # Negative prices (interesting for trading context)
    neg_prices = (df["price_eur_mwh"] < 0).sum()
    report.append(f"\n## Notable Observations\n")
    report.append(f"- **Negative prices:** {neg_prices} hours ({100*neg_prices/len(df):.2f}%)")
    if neg_prices > 0:
        report.append(f"  - Min negative price: €{df['price_eur_mwh'].min():.2f}/MWh")
        report.append(f"  - These occur when renewables oversupply (wind+solar > demand)")

    report_text = "\n".join(report)

    # Save report
    report_path = OUTPUT_DIR / "data_qa_report.md"
    report_path.write_text(report_text)
    logger.info(f"\nQA report saved to: {report_path}")

    return report_text


def run_quality(df: pd.DataFrame) -> str:
    """Main entry point for data quality assessment."""
    return generate_qa_report(df)
