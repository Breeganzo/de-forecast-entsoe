"""
Data Source Documentation
=========================

Primary Source: ENTSO-E Transparency Platform REST API
------------------------------------------------------
Full Postman Collection: https://documenter.getpostman.com/view/7009892/2s93JtP3F6
Base URL: https://web-api.tp.entsoe.eu/api
Auth: securityToken query parameter
  - Register at https://transparency.entsoe.eu/ to receive an API key via email.

Market: DE-LU (Germany-Luxembourg) bidding zone
EIC Code: 10Y1001A1001A82H
Timezone: Europe/Berlin (CET/CEST)
Date Range: June 2024 – June 2026 (2 years)

Endpoints Used
--------------

1. Day-Ahead Prices (Article 12.1.D)
   documentType: A44 (Price Document)
   Resolution: PT60M (hourly)
   Request:
     GET https://web-api.tp.entsoe.eu/api
       ?securityToken=<API_KEY>
       &documentType=A44
       &in_Domain=10Y1001A1001A82H
       &out_Domain=10Y1001A1001A82H
       &periodStart=202406010000
       &periodEnd=202407010000
   Response: XML Publication_MarketDocument with TimeSeries > Period > Point
   Value field: price.amount (€/MWh)
   Max span: 1 year per request

2. Actual Total Load (Article 6.1.A)
   documentType: A65 (System total load)
   processType: A16 (Realised)
   Resolution: PT15M (quarter-hourly) → resampled to PT60M
   Request:
     GET https://web-api.tp.entsoe.eu/api
       ?securityToken=<API_KEY>
       &documentType=A65
       &processType=A16
       &outBiddingZone_Domain=10Y1001A1001A82H
       &periodStart=202406010000
       &periodEnd=202407010000
   Response: XML GL_MarketDocument with TimeSeries > Period > Point
   Value field: quantity (MW)
   Max span: 1 year per request

3. Actual Generation per Type (Article 16.1.B&C)
   documentType: A75 (Actual generation per type)
   processType: A16 (Realised)
   Resolution: PT15M (quarter-hourly) → resampled to PT60M
   PSR Types used:
     B04 = Fossil Gas (marginal fuel in German merit order)
     B16 = Solar (photovoltaic)
     B18 = Wind Offshore
     B19 = Wind Onshore
   Request (example for Wind Onshore):
     GET https://web-api.tp.entsoe.eu/api
       ?securityToken=<API_KEY>
       &documentType=A75
       &processType=A16
       &in_Domain=10Y1001A1001A82H
       &psrType=B19
       &periodStart=202406010000
       &periodEnd=202407010000
   Response: XML GL_MarketDocument with TimeSeries > Period > Point
   Value field: quantity (MW)
   Max span: 1 year per request

API Response Structure (XML)
----------------------------
All responses follow IEC 62325-451 schema:

  <Publication_MarketDocument>
    <TimeSeries>
      <Period>
        <timeInterval>
          <start>2024-06-01T00:00Z</start>    ← Always UTC
          <end>2024-06-02T00:00Z</end>
        </timeInterval>
        <resolution>PT60M</resolution>
        <Point>
          <position>1</position>               ← 1-indexed hour offset
          <price.amount>88.58</price.amount>   ← (or <quantity> for load/gen)
        </Point>
        ...
      </Period>
    </TimeSeries>
  </Publication_MarketDocument>

Key Notes:
  - All timestamps are UTC. For CET midnight, use HH=22 (summer) or HH=23 (winter).
  - Position is 1-indexed: position 1 = start time of the period.
  - Multiple TimeSeries may appear (e.g., one per generation unit for A75).
  - Rate limit: ~400 requests/minute. Pipeline uses 2s delay between calls.
  - Max 100 TimeSeries per response; max 1 year per request.

Data Pipeline Behaviour
-----------------------
1. On first run: fetches all data from API, caches to data/raw/*.csv
2. On subsequent runs: loads from cache (delete data/raw/ to force re-fetch)
3. Quarter-hourly data (PT15M) is resampled to hourly mean
4. Wind = Wind Onshore (B19) + Wind Offshore (B18)
5. Final merge uses inner join — only hours with ALL four fields are kept
6. DST transitions produce 23h (spring) and 25h (fall) days — verified in QA
"""
