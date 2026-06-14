"""
Task 1: Data Ingestion — Fetch hourly data from ENTSO-E Transparency Platform API.

Data Source: ENTSO-E Transparency Platform REST API
Documentation: https://documenter.getpostman.com/view/7009892/2s93JtP3F6
Base URL: https://web-api.tp.entsoe.eu/api
Auth: securityToken query parameter (register at https://transparency.entsoe.eu)

Endpoints used:
  1. Day-Ahead Prices (Art. 12.1.D): documentType=A44
  2. Actual Total Load (Art. 6.1.A): documentType=A65, processType=A16
  3. Actual Generation per Type (Art. 16.1.B&C): documentType=A75, processType=A16

All timestamps in API use UTC. Responses are XML (Publication_MarketDocument).
Max request span: 1 year. Pipeline splits 2-year range into yearly chunks.
"""
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests

from src.config import (
    ENTSOE_BASE_URL, ENTSOE_API_KEY, BIDDING_ZONE_EIC,
    START_DATE, END_DATE, TIMEZONE,
    RAW_PRICES_FILE, RAW_LOAD_FILE, RAW_WIND_FILE, RAW_SOLAR_FILE,
    RAW_GAS_FILE, RAW_WIND_FORECAST_FILE, RAW_SOLAR_FORECAST_FILE,
    RAW_LOAD_FORECAST_FILE, PROCESSED_DIR,
)

logger = logging.getLogger(__name__)

# XML namespace used in ENTSO-E responses
NS_MARKET = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}
NS_LOAD = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}


def _date_to_entsoe(date_str: str) -> str:
    """Convert 'YYYY-MM-DD' to ENTSO-E format 'yyyyMMddHHmm' (UTC midnight)."""
    dt = pd.Timestamp(date_str, tz=TIMEZONE).tz_convert("UTC")
    return dt.strftime("%Y%m%d%H%M")


def _split_date_range(start: str, end: str, max_days: int = 365) -> list[tuple[str, str]]:
    """Split a date range into chunks of max_days to respect API limits."""
    chunks = []
    current = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while current < end_ts:
        chunk_end = min(current + pd.Timedelta(days=max_days), end_ts)
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end
    return chunks


def _api_request(params: dict, description: str) -> str:
    """Make a single API request with retry logic. Returns XML text."""
    if not ENTSOE_API_KEY:
        raise RuntimeError("ENTSOE_API_KEY not set in .env")

    params["securityToken"] = ENTSOE_API_KEY

    for attempt in range(3):
        try:
            resp = requests.get(ENTSOE_BASE_URL, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 429:
                wait = 30 * (attempt + 1)
                logger.warning(f"Rate limited on {description}, waiting {wait}s...")
                time.sleep(wait)
            elif resp.status_code == 400:
                logger.error(f"Bad request for {description}: {resp.text[:200]}")
                return None
            else:
                logger.error(f"HTTP {resp.status_code} for {description}: {resp.text[:200]}")
                if attempt < 2:
                    time.sleep(5)
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on {description} (attempt {attempt + 1})")
            time.sleep(10)

    logger.error(f"Failed after 3 attempts: {description}")
    return None


def _parse_market_timeseries(xml_text: str) -> pd.DataFrame:
    """
    Parse ENTSO-E market document XML into DataFrame with columns [timestamp, value].
    Handles PT60M and PT15M resolutions.
    """
    root = ET.fromstring(xml_text)

    # Detect namespace from root tag
    ns_uri = root.tag.split("}")[0] + "}" if "}" in root.tag else ""
    ns = {"ns": ns_uri.strip("{}")} if ns_uri else {}
    prefix = "ns:" if ns else ""

    rows = []
    for ts in root.findall(f".//{prefix}TimeSeries", ns):
        for period in ts.findall(f"{prefix}Period", ns):
            start_el = period.find(f"{prefix}timeInterval/{prefix}start", ns)
            res_el = period.find(f"{prefix}resolution", ns)
            if start_el is None or res_el is None:
                continue

            start_utc = pd.Timestamp(start_el.text)
            resolution = res_el.text  # PT60M or PT15M

            if resolution == "PT60M":
                delta = timedelta(hours=1)
            elif resolution == "PT15M":
                delta = timedelta(minutes=15)
            elif resolution == "PT30M":
                delta = timedelta(minutes=30)
            else:
                logger.warning(f"Unknown resolution {resolution}, assuming PT60M")
                delta = timedelta(hours=1)

            for point in period.findall(f"{prefix}Point", ns):
                pos = int(point.find(f"{prefix}position", ns).text)
                # Try price.amount first (market docs), then quantity (load/gen docs)
                val_el = point.find(f"{prefix}price.amount", ns)
                if val_el is None:
                    val_el = point.find(f"{prefix}quantity", ns)
                if val_el is None:
                    continue

                timestamp = start_utc + delta * (pos - 1)
                rows.append({"timestamp": timestamp, "value": float(val_el.text)})

    if not rows:
        return pd.DataFrame(columns=["timestamp", "value"])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_day_ahead_prices() -> pd.Series:
    """
    Fetch Day-Ahead prices for DE-LU from ENTSO-E.
    Endpoint: documentType=A44 (Price Document)
    Resolution: PT60M (hourly)
    """
    logger.info("Fetching Day-Ahead prices from ENTSO-E API...")
    chunks = _split_date_range(START_DATE, END_DATE)
    all_dfs = []

    for i, (start, end) in enumerate(chunks):
        logger.info(f"  Chunk {i+1}/{len(chunks)}: {start} → {end}")
        params = {
            "documentType": "A44",
            "in_Domain": BIDDING_ZONE_EIC,
            "out_Domain": BIDDING_ZONE_EIC,
            "periodStart": _date_to_entsoe(start),
            "periodEnd": _date_to_entsoe(end),
        }
        xml = _api_request(params, f"DA prices {start}→{end}")
        if xml:
            df = _parse_market_timeseries(xml)
            all_dfs.append(df)
        time.sleep(2)  # Polite rate limiting

    if not all_dfs:
        raise RuntimeError("Failed to fetch any Day-Ahead price data")

    combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates("timestamp")
    combined = combined.set_index("timestamp").sort_index()

    # Convert UTC → local timezone
    series = combined["value"].tz_convert(TIMEZONE)
    series.name = "price_eur_mwh"

    # Resample to hourly (should already be hourly, but ensures alignment)
    series = series[~series.index.duplicated(keep="first")]

    logger.info(f"  Total DA prices: {len(series)} rows")
    return series


def fetch_actual_load() -> pd.Series:
    """
    Fetch actual total load for DE-LU from ENTSO-E.
    Endpoint: documentType=A65, processType=A16 (Realised)
    Resolution: PT15M → resampled to PT60M (hourly mean)
    """
    logger.info("Fetching actual load from ENTSO-E API...")
    chunks = _split_date_range(START_DATE, END_DATE)
    all_dfs = []

    for i, (start, end) in enumerate(chunks):
        logger.info(f"  Chunk {i+1}/{len(chunks)}: {start} → {end}")
        params = {
            "documentType": "A65",
            "processType": "A16",
            "outBiddingZone_Domain": BIDDING_ZONE_EIC,
            "periodStart": _date_to_entsoe(start),
            "periodEnd": _date_to_entsoe(end),
        }
        xml = _api_request(params, f"Load {start}→{end}")
        if xml:
            df = _parse_market_timeseries(xml)
            all_dfs.append(df)
        time.sleep(2)

    if not all_dfs:
        raise RuntimeError("Failed to fetch any load data")

    combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates("timestamp")
    combined = combined.set_index("timestamp").sort_index()

    # Convert UTC → local, resample 15-min → hourly
    series = combined["value"].tz_convert(TIMEZONE)
    series = series[~series.index.duplicated(keep="first")]
    series = series.resample("1h").mean()
    series.name = "load_mw"

    logger.info(f"  Total load: {len(series)} hourly rows")
    return series


def fetch_generation(psr_type: str, name: str) -> pd.Series:
    """
    Fetch actual generation for a specific PSR type from ENTSO-E.
    Endpoint: documentType=A75, processType=A16 (Realised)

    PSR Types:
      B16 = Solar
      B18 = Wind Offshore
      B19 = Wind Onshore

    Resolution: PT15M → resampled to PT60M (hourly mean)
    """
    logger.info(f"Fetching {name} generation (psrType={psr_type}) from ENTSO-E API...")
    chunks = _split_date_range(START_DATE, END_DATE)
    all_dfs = []

    for i, (start, end) in enumerate(chunks):
        logger.info(f"  Chunk {i+1}/{len(chunks)}: {start} → {end}")
        params = {
            "documentType": "A75",
            "processType": "A16",
            "in_Domain": BIDDING_ZONE_EIC,
            "psrType": psr_type,
            "periodStart": _date_to_entsoe(start),
            "periodEnd": _date_to_entsoe(end),
        }
        xml = _api_request(params, f"{name} gen {start}→{end}")
        if xml:
            df = _parse_market_timeseries(xml)
            all_dfs.append(df)
        time.sleep(2)

    if not all_dfs:
        logger.warning(f"No data fetched for {name} — returning empty series")
        return pd.Series(dtype=float, name=name)

    combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates("timestamp")
    combined = combined.set_index("timestamp").sort_index()

    series = combined["value"].tz_convert(TIMEZONE)
    series = series[~series.index.duplicated(keep="first")]
    series = series.resample("1h").mean()
    series.name = name

    logger.info(f"  Total {name}: {len(series)} hourly rows")
    return series


def align_and_merge(
    prices: pd.Series,
    load: pd.Series,
    wind: pd.Series,
    solar: pd.Series,
    gas: pd.Series = None,
    wind_forecast: pd.Series = None,
    solar_forecast: pd.Series = None,
    load_forecast: pd.Series = None,
) -> pd.DataFrame:
    """
    Merge all series on their datetime index.
    The final training frame keeps only complete hourly observations.
    Forecast series are forward-looking (available before DA auction).
    """
    df = pd.DataFrame({
        "price_eur_mwh": prices,
        "load_mw": load,
        "wind_mw": wind,
        "solar_mw": solar,
    })
    # Derived column: total renewable generation
    df["wind_solar_mw"] = df["wind_mw"] + df["solar_mw"]

    # Gas generation — the marginal fuel in German merit order
    if gas is not None and len(gas) > 0:
        df["gas_mw"] = gas
        logger.info(f"  Added gas_mw: {gas.notna().sum()} values")

    # Add forecast columns (forward-looking features available at decision time)
    if wind_forecast is not None and len(wind_forecast) > 0:
        df["wind_forecast_mw"] = wind_forecast
        logger.info(f"  Added wind_forecast_mw: {wind_forecast.notna().sum()} values")
    if solar_forecast is not None and len(solar_forecast) > 0:
        df["solar_forecast_mw"] = solar_forecast
        logger.info(f"  Added solar_forecast_mw: {solar_forecast.notna().sum()} values")
    if load_forecast is not None and len(load_forecast) > 0:
        df["load_forecast_mw"] = load_forecast
        logger.info(f"  Added load_forecast_mw: {load_forecast.notna().sum()} values")

    # Drop rows where CORE columns are NaN (inner-join on actuals)
    core_cols = ["price_eur_mwh", "load_mw", "wind_mw", "solar_mw"]
    n_before = len(df)
    df = df.dropna(subset=core_cols)
    n_dropped = n_before - len(df)

    logger.info(
        f"Aligned hourly dataset: {len(df)} complete rows "
        f"({n_dropped} non-overlapping source points excluded during alignment)"
    )
    logger.info(f"Date range: {df.index.min()} → {df.index.max()}")
    return df


def log_dst_transitions(df: pd.DataFrame) -> list[dict]:
    """Identify and log DST transition days (23h or 25h days)."""
    daily_counts = df.groupby(df.index.date).size()
    transitions = []

    for date, count in daily_counts.items():
        if count != 24:
            transitions.append({"date": str(date), "hours": int(count)})
            logger.info(f"  DST transition: {date} has {count} hours")

    if not transitions:
        logger.warning("  No DST transitions found — check timezone handling!")
    else:
        logger.info(f"  Found {len(transitions)} DST transitions")

    return transitions


def fetch_wind_forecast() -> pd.Series:
    """
    Fetch day-ahead wind generation forecast for DE-LU.
    Endpoint: documentType=A69 (Wind and solar generation forecasts)
    processType=A01 (Day ahead)
    psrType=B19 (Wind Onshore) + B18 (Wind Offshore) — fetched separately and summed.

    This is the FORECAST published BEFORE the DA auction — it represents
    what the market knows at decision time (forward-looking feature).
    """
    logger.info("Fetching DA wind forecast from ENTSO-E API...")
    chunks = _split_date_range(START_DATE, END_DATE)

    # Fetch onshore (B19) and offshore (B18) separately, then sum
    onshore_dfs = []
    offshore_dfs = []

    for i, (start, end) in enumerate(chunks):
        logger.info(f"  Chunk {i+1}/{len(chunks)}: {start} → {end}")
        # Wind onshore forecast
        params_on = {
            "documentType": "A69",
            "processType": "A01",
            "in_Domain": BIDDING_ZONE_EIC,
            "psrType": "B19",
            "periodStart": _date_to_entsoe(start),
            "periodEnd": _date_to_entsoe(end),
        }
        xml = _api_request(params_on, f"Wind onshore forecast {start}→{end}")
        if xml:
            df = _parse_market_timeseries(xml)
            onshore_dfs.append(df)
        time.sleep(2)

        # Wind offshore forecast
        params_off = {
            "documentType": "A69",
            "processType": "A01",
            "in_Domain": BIDDING_ZONE_EIC,
            "psrType": "B18",
            "periodStart": _date_to_entsoe(start),
            "periodEnd": _date_to_entsoe(end),
        }
        xml = _api_request(params_off, f"Wind offshore forecast {start}→{end}")
        if xml:
            df = _parse_market_timeseries(xml)
            offshore_dfs.append(df)
        time.sleep(2)

    if not onshore_dfs and not offshore_dfs:
        logger.warning("No wind forecast data fetched — feature will be empty")
        return pd.Series(dtype=float, name="wind_forecast_mw")

    # Process onshore
    onshore_series = pd.Series(dtype=float)
    if onshore_dfs:
        combined = pd.concat(onshore_dfs, ignore_index=True).drop_duplicates("timestamp")
        combined = combined.set_index("timestamp").sort_index()
        onshore_series = combined["value"].tz_convert(TIMEZONE)
        onshore_series = onshore_series[~onshore_series.index.duplicated(keep="first")]
        onshore_series = onshore_series.resample("1h").mean()

    # Process offshore
    offshore_series = pd.Series(dtype=float)
    if offshore_dfs:
        combined = pd.concat(offshore_dfs, ignore_index=True).drop_duplicates("timestamp")
        combined = combined.set_index("timestamp").sort_index()
        offshore_series = combined["value"].tz_convert(TIMEZONE)
        offshore_series = offshore_series[~offshore_series.index.duplicated(keep="first")]
        offshore_series = offshore_series.resample("1h").mean()

    # Sum onshore + offshore
    series = onshore_series.add(offshore_series, fill_value=0)
    series.name = "wind_forecast_mw"
    logger.info(f"  Total wind forecast (on+offshore): {len(series)} hourly rows")
    return series


def fetch_solar_forecast() -> pd.Series:
    """
    Fetch day-ahead solar generation forecast for DE-LU.
    Endpoint: documentType=A69, processType=A01, psrType=B16
    """
    logger.info("Fetching DA solar forecast from ENTSO-E API...")
    chunks = _split_date_range(START_DATE, END_DATE)
    all_dfs = []

    for i, (start, end) in enumerate(chunks):
        logger.info(f"  Chunk {i+1}/{len(chunks)}: {start} → {end}")
        params = {
            "documentType": "A69",
            "processType": "A01",
            "in_Domain": BIDDING_ZONE_EIC,
            "psrType": "B16",
            "periodStart": _date_to_entsoe(start),
            "periodEnd": _date_to_entsoe(end),
        }
        xml = _api_request(params, f"Solar forecast {start}→{end}")
        if xml:
            df = _parse_market_timeseries(xml)
            all_dfs.append(df)
        time.sleep(2)

    if not all_dfs:
        logger.warning("No solar forecast data fetched — feature will be empty")
        return pd.Series(dtype=float, name="solar_forecast_mw")

    combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates("timestamp")
    combined = combined.set_index("timestamp").sort_index()
    series = combined["value"].tz_convert(TIMEZONE)
    series = series[~series.index.duplicated(keep="first")]
    series = series.resample("1h").mean()
    series.name = "solar_forecast_mw"
    logger.info(f"  Total solar forecast: {len(series)} hourly rows")
    return series


def fetch_load_forecast() -> pd.Series:
    """
    Fetch day-ahead total load forecast for DE-LU.
    Endpoint: documentType=A65, processType=A01 (Day ahead forecast)
    """
    logger.info("Fetching DA load forecast from ENTSO-E API...")
    chunks = _split_date_range(START_DATE, END_DATE)
    all_dfs = []

    for i, (start, end) in enumerate(chunks):
        logger.info(f"  Chunk {i+1}/{len(chunks)}: {start} → {end}")
        params = {
            "documentType": "A65",
            "processType": "A01",
            "outBiddingZone_Domain": BIDDING_ZONE_EIC,
            "periodStart": _date_to_entsoe(start),
            "periodEnd": _date_to_entsoe(end),
        }
        xml = _api_request(params, f"Load forecast {start}→{end}")
        if xml:
            df = _parse_market_timeseries(xml)
            all_dfs.append(df)
        time.sleep(2)

    if not all_dfs:
        logger.warning("No load forecast data fetched — feature will be empty")
        return pd.Series(dtype=float, name="load_forecast_mw")

    combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates("timestamp")
    combined = combined.set_index("timestamp").sort_index()
    series = combined["value"].tz_convert(TIMEZONE)
    series = series[~series.index.duplicated(keep="first")]
    series = series.resample("1h").mean()
    series.name = "load_forecast_mw"
    logger.info(f"  Total load forecast: {len(series)} hourly rows")
    return series


def _cache_series(series: pd.Series, filepath) -> None:
    """Save a series to CSV for caching."""
    df = series.reset_index()
    df.columns = ["timestamp", series.name]
    df.to_csv(filepath, index=False)
    logger.info(f"  Cached {len(df)} rows → {filepath.name}")


def _load_cached(filepath, name: str) -> pd.Series | None:
    """Load a cached CSV if it exists and has data."""
    if filepath.exists():
        df = pd.read_csv(filepath)
        if len(df) > 100:  # sanity check — at least some data
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            series = df.set_index("timestamp")[name]
            series.index = series.index.tz_convert(TIMEZONE)
            logger.info(f"  Loaded from cache: {filepath.name} ({len(series)} rows)")
            return series
    return None


def run_ingestion() -> pd.DataFrame:
    """
    Main ingestion pipeline:
    1. Check for cached data (skip API calls if fresh cache exists)
    2. Fetch from ENTSO-E API if needed
    3. Cache raw data
    4. Align and merge into single DataFrame
    5. Save processed parquet
    """
    logger.info("=" * 60)
    logger.info("STEP 1a: DATA INGESTION (ENTSO-E API)")
    logger.info("=" * 60)
    logger.info(f"Market: {BIDDING_ZONE_EIC} (DE-LU)")
    logger.info(f"Period: {START_DATE} → {END_DATE}")
    logger.info(f"Source: ENTSO-E Transparency Platform REST API")
    logger.info(f"Docs: https://documenter.getpostman.com/view/7009892/2s93JtP3F6")

    # --- Check cache or fetch ---
    prices = _load_cached(RAW_PRICES_FILE, "price_eur_mwh")
    if prices is None:
        prices = fetch_day_ahead_prices()
        _cache_series(prices, RAW_PRICES_FILE)

    load = _load_cached(RAW_LOAD_FILE, "load_mw")
    if load is None:
        load = fetch_actual_load()
        _cache_series(load, RAW_LOAD_FILE)

    wind = _load_cached(RAW_WIND_FILE, "wind_mw")
    if wind is None:
        wind_onshore = fetch_generation("B19", "wind_onshore_mw")
        wind_offshore = fetch_generation("B18", "wind_offshore_mw")
        wind = wind_onshore.add(wind_offshore, fill_value=0)
        wind.name = "wind_mw"
        _cache_series(wind, RAW_WIND_FILE)

    solar = _load_cached(RAW_SOLAR_FILE, "solar_mw")
    if solar is None:
        solar = fetch_generation("B16", "solar_mw")
        _cache_series(solar, RAW_SOLAR_FILE)

    # --- Fossil gas generation (marginal fuel in German merit order) ---
    gas = _load_cached(RAW_GAS_FILE, "gas_mw")
    if gas is None:
        gas = fetch_generation("B04", "gas_mw")
        if len(gas) > 0:
            _cache_series(gas, RAW_GAS_FILE)

    # --- Forward-looking forecasts (available before DA auction) ---
    logger.info("\n  Fetching forward-looking forecasts (DA wind/solar/load)...")
    wind_fc = _load_cached(RAW_WIND_FORECAST_FILE, "wind_forecast_mw")
    if wind_fc is None:
        wind_fc = fetch_wind_forecast()
        if len(wind_fc) > 0:
            _cache_series(wind_fc, RAW_WIND_FORECAST_FILE)

    solar_fc = _load_cached(RAW_SOLAR_FORECAST_FILE, "solar_forecast_mw")
    if solar_fc is None:
        solar_fc = fetch_solar_forecast()
        if len(solar_fc) > 0:
            _cache_series(solar_fc, RAW_SOLAR_FORECAST_FILE)

    load_fc = _load_cached(RAW_LOAD_FORECAST_FILE, "load_forecast_mw")
    if load_fc is None:
        load_fc = fetch_load_forecast()
        if len(load_fc) > 0:
            _cache_series(load_fc, RAW_LOAD_FORECAST_FILE)

    # --- Merge ---
    df = align_and_merge(prices, load, wind, solar, gas, wind_fc, solar_fc, load_fc)

    # --- DST verification ---
    transitions = log_dst_transitions(df)
    logger.info(f"DST transitions found: {len(transitions)}")

    # --- Save processed ---
    parquet_path = PROCESSED_DIR / "de_lu_hourly.parquet"
    df.to_parquet(parquet_path)
    logger.info(f"Saved processed data → {parquet_path.name}")

    return df
