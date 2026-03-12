"""
Air Quality Data Connector
Primary: Open-Meteo Air Quality API (CAMS satellite data, free, no key)
Fallback: Clean-air defaults (AQI=0)

Open-Meteo Air Quality provides:
  - PM2.5 and PM10 from CAMS global atmospheric model
  - European AQI and US AQI indices
  - Updated hourly, global coverage, no authentication required

AQI scale (US EPA):
  0–50   Good        51–100  Moderate     101–150  Unhealthy for Sensitive Groups
  151–200 Unhealthy  201–300 Very Unhealthy  301–500  Hazardous

CAMS (Copernicus Atmosphere Monitoring Service) reanalysis data:
  Inness, A., Ades, M., Agustí-Panareda, A., et al. (2019).
  "The CAMS reanalysis of atmospheric composition."
  Atmospheric Chemistry and Physics, 19(6), pp. 3515–3556.
  https://doi.org/10.5194/acp-19-3515-2019
"""
from typing import Dict, Optional
import datetime

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


class AirQualityConnector:
    """
    Fetches real-time PM2.5 / AQI data from Open-Meteo's Air Quality API.

    Uses CAMS (Copernicus Atmosphere Monitoring Service) global reanalysis/
    forecast data — no ground station proximity needed, full global coverage.

    Falls back to AQI=0 (clean air) on any network failure.
    """

    # Open-Meteo Air Quality endpoint — completely free, no key required
    AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

    def __init__(self, api_key: str = ""):
        # api_key kept for interface compatibility but not used
        self.api_key = api_key.strip()
        self._cache: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, lat: float, lon: float, radius_m: int = 25_000) -> Dict:
        """
        Fetch current air quality for the given lat/lon.

        Returns a dict with:
            aqi (float 0–500), pm25 (float μg/m³), pm10 (float μg/m³),
            advisory (str), station_name (str), station_distance_m (float)
        Falls back to clean-air defaults on any error.
        """
        if self._cache is not None:
            return self._cache

        if not _REQUESTS_AVAILABLE:
            print("  [AQ] 'requests' not installed — using AQI=0 (clean air).")
            return self._default()

        try:
            result = self._fetch_openmeteo(lat, lon)
            self._cache = result
            return result
        except Exception as e:
            print(f"  [AQ] WARNING: Could not fetch air quality ({e}). Using AQI=0.")
            return self._default()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_openmeteo(self, lat: float, lon: float) -> Dict:
        """
        Query Open-Meteo Air Quality API for current PM2.5, PM10, and AQI.
        Fetches 1 day of hourly data and extracts the most recent non-null value.
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "pm2_5,pm10,us_aqi",
            "timezone": "auto",
            "forecast_days": 1,
        }
        resp = requests.get(self.AQ_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        times   = hourly.get("time", [])
        pm25_series = hourly.get("pm2_5", [])
        pm10_series = hourly.get("pm10", [])
        aqi_series  = hourly.get("us_aqi", [])

        # Pick the most recent non-null hour
        pm25, pm10, aqi = 0.0, 0.0, 0.0
        for i in reversed(range(len(times))):
            v_aqi  = aqi_series[i]  if i < len(aqi_series)  else None
            v_pm25 = pm25_series[i] if i < len(pm25_series) else None
            v_pm10 = pm10_series[i] if i < len(pm10_series) else None
            if v_aqi is not None:
                aqi  = max(0.0, float(v_aqi))
                pm25 = max(0.0, float(v_pm25)) if v_pm25 is not None else 0.0
                pm10 = max(0.0, float(v_pm10)) if v_pm10 is not None else 0.0
                break

        advisory = self._aqi_to_advisory(aqi)
        tz = data.get("timezone_abbreviation", "UTC")
        print(f"  [AQ] Open-Meteo CAMS ({tz}) — "
              f"PM2.5={pm25:.1f} μg/m³, PM10={pm10:.1f} μg/m³, "
              f"US AQI={aqi:.0f} ({advisory})")

        return {
            "aqi": round(aqi, 1),
            "pm25": round(pm25, 2),
            "pm10": round(pm10, 2),
            "advisory": advisory,
            "station_name": "Open-Meteo CAMS",
            "station_distance_m": 0.0,
        }

    # ------------------------------------------------------------------
    # AQI conversion (US EPA breakpoints for PM2.5 — kept for reference)
    # ------------------------------------------------------------------

    def _pm25_to_aqi(self, pm25: float) -> float:
        """
        Convert PM2.5 concentration (μg/m³) to US AQI using EPA linear interpolation.
        (Not used when Open-Meteo provides us_aqi directly.)
        """
        breakpoints = [
            (0.0,   12.0,   0,   50),
            (12.1,  35.4,  51,  100),
            (35.5,  55.4, 101,  150),
            (55.5, 150.4, 151,  200),
            (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500),
        ]
        pm25 = max(0.0, pm25)
        for c_lo, c_hi, i_lo, i_hi in breakpoints:
            if c_lo <= pm25 <= c_hi:
                return i_lo + (pm25 - c_lo) * (i_hi - i_lo) / (c_hi - c_lo)
        return 500.0

    def _aqi_to_advisory(self, aqi: float) -> str:
        if aqi <= 50:   return "Good"
        if aqi <= 100:  return "Moderate"
        if aqi <= 150:  return "Unhealthy for Sensitive Groups"
        if aqi <= 200:  return "Unhealthy"
        if aqi <= 300:  return "Very Unhealthy"
        return "Hazardous"

    def _default(self) -> Dict:
        return {
            "aqi": 0.0, "pm25": 0.0, "pm10": 0.0,
            "advisory": "Good",
            "station_name": "N/A", "station_distance_m": 0.0,
        }
