"""
Fire Weather Index (FWI) Data Connector
Uses Open-Meteo Forecast API — free, no API key required, global coverage.

Fetches weather variables and derives the full Canadian FWI system:
  FFMC → DMC → DC → ISI → BUI → FWI

FWI Danger Scale:
  0–5   Very Low   5–10  Low   10–20  Moderate
  20–30 High       30–40 Very High   40+   Extreme

Canadian Forest Fire Weather Index system:
  Van Wagner, C.E. (1987).
  Development and Structure of the Canadian Forest Fire Weather Index System.
  Forestry Technical Report 35. Canadian Forestry Service, Ottawa.

  Van Wagner, C.E. & Pickett, T.L. (1985).
  Equations and FORTRAN Program for the Canadian Forest Fire Weather Index System.
  Forestry Technical Report 33. Canadian Forestry Service, Ottawa.
"""
import math
from typing import Dict, Optional

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


class FWIConnector:
    """
    Fetches fire weather data from Open-Meteo and computes Canadian FWI components.

    No API key required. Caches the last successful result for the session so
    that repeated calls within a simulation run do not re-hit the network.
    """

    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self):
        self._cache: Optional[Dict] = None

    def fetch(self, lat: float, lon: float) -> Dict:
        """
        Fetch FWI components for the given lat/lon.

        Returns a dict with keys:
            ffmc, dmc, dc, isi, bui, fwi,
            fire_danger_index, temperature_max, humidity_min,
            wind_max_kmh, precipitation, risk_level (str)
        Falls back to low-risk defaults on any network/parse error.
        """
        if self._cache is not None:
            return self._cache

        if not _REQUESTS_AVAILABLE:
            print("  [FWI] 'requests' not installed — using default values.")
            return self._default()

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "relative_humidity_2m_min",
                "windspeed_10m_max",
                "precipitation_sum",
                "et0_fao_evapotranspiration",
            ],
            "hourly": ["soil_moisture_0_1cm"],
            "forecast_days": 1,
            "timezone": "auto",
        }

        try:
            resp = requests.get(self.FORECAST_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            daily = data.get("daily", {})
            hourly = data.get("hourly", {})

            temp_max  = self._first(daily.get("temperature_2m_max"))  or 20.0
            temp_min  = self._first(daily.get("temperature_2m_min"))  or 14.0
            rh_min    = self._first(daily.get("relative_humidity_2m_min")) or 40.0
            wind_max  = self._first(daily.get("windspeed_10m_max"))   or 10.0  # km/h
            precip    = self._first(daily.get("precipitation_sum"))   or 0.0
            et0       = self._first(daily.get("et0_fao_evapotranspiration")) or 3.0
            soil_mois = self._first(hourly.get("soil_moisture_0_1cm"))

            result = self._compute_fwi(temp_max, rh_min, wind_max, precip, et0, soil_mois)

            print(f"  [FWI] FWI={result['fwi']:.1f} ({result['risk_level']}), "
                  f"FFMC={result['ffmc']:.1f}, Wind={wind_max:.1f} km/h, "
                  f"Temp={temp_max:.1f}°C, RH={rh_min:.0f}%, Precip={precip:.1f}mm")

            self._cache = result
            return result

        except Exception as e:
            print(f"  [FWI] WARNING: API error ({e}). Using default low-risk values.")
            return self._default()

    # ------------------------------------------------------------------
    # Canadian FWI System (simplified, Van Wagner 1987)
    # ------------------------------------------------------------------

    def _compute_fwi(self, temp: float, rh: float, wind_kmh: float,
                     precip: float, et0: float,
                     soil_moisture: Optional[float]) -> Dict:
        """
        Compute FWI system components from daily weather parameters.

        References:
        - Van Wagner, C.E. (1987). Development and structure of the Canadian
          Forest Fire Weather Index System. Forestry Technical Report 35.
        - All equations use the simplified daily startup formulas.
        """
        wind_mps = wind_kmh / 3.6  # convert to m/s for ISI

        # ---- FFMC (Fine Fuel Moisture Code) --------------------------------
        # Previous-day FFMC assumed moderate (85) for fresh simulation
        ffmc_prev = 85.0
        m0 = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)

        # Rain effect on fine fuel moisture
        if precip > 0.5:
            rf = precip - 0.5
            if m0 <= 150:
                m0 = m0 + 42.5 * rf * math.exp(-100.0 / (251.0 - m0)) * (1.0 - math.exp(-6.93 / rf))
            else:
                m0 = m0 + 42.5 * rf * math.exp(-100.0 / (251.0 - m0)) * (1.0 - math.exp(-6.93 / rf))
            m0 = min(250.0, m0)

        # Equilibrium moisture content
        Ed = 0.942 * (rh ** 0.679) + 11.0 * math.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh))
        Ew = 0.618 * (rh ** 0.753) + 10.0 * math.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh))

        if m0 > Ed:
            k_d = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * math.sqrt(wind_mps) * (1.0 - (rh / 100.0) ** 8)
            m = Ed + (m0 - Ed) * 10.0 ** (-k_d)
        elif m0 < Ew:
            k_w = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) + 0.0694 * math.sqrt(wind_mps) * (1.0 - ((100.0 - rh) / 100.0) ** 8)
            m = Ew - (Ew - m0) * 10.0 ** (-k_w)
        else:
            m = m0

        m = max(0.0, min(250.0, m))
        ffmc = 59.5 * (250.0 - m) / (147.2 + m)
        ffmc = max(0.0, min(101.0, ffmc))

        # ---- DMC (Duff Moisture Code) --------------------------------------
        dmc_prev = 15.0  # assumed moderate start
        if precip > 1.5:
            re = 0.92 * precip - 1.27
            mo = 20.0 + math.exp(5.6348 - dmc_prev / 43.43)
            b = 100.0 / (0.5 + 0.3 * dmc_prev) if dmc_prev <= 33 else (
                14.0 - 1.3 * math.log(dmc_prev) if dmc_prev <= 65 else
                6.2 * math.log(dmc_prev) - 17.2
            )
            mr = mo + 1000.0 * re / (48.77 + b * re)
            pr = 244.72 - 43.43 * math.log(mr - 20.0) if mr > 20.0 else 0.0
            dmc_prev = max(0.0, pr)

        # Day-length factor (simplified: use 9 hours = mid-latitude)
        Le = 9.0
        K = max(0.0, 1.894 * (temp + 1.1) * (100.0 - rh) * Le * 1e-6)
        dmc = dmc_prev + 100.0 * K
        if soil_moisture is not None:
            # High soil moisture reduces effective drying
            dmc *= (1.0 - min(float(soil_moisture), 0.8) * 0.5)
        dmc = max(0.0, dmc)

        # ---- DC (Drought Code) --------------------------------------------
        dc_prev = 100.0  # assumed moderate start
        if precip > 2.8:
            rd = 0.83 * precip - 1.27
            qr = 800.0 * math.exp(-dc_prev / 400.0)
            qr += 3.937 * rd
            dc_rain = 400.0 * math.log(800.0 / qr) if qr > 0 else dc_prev
            dc_prev = max(0.0, dc_rain)

        # Drying factor (simplified day length 9h)
        V = 0.36 * (temp + 2.8) + 9.0 / 2.0
        V = max(0.0, V)
        dc = dc_prev + 0.5 * V
        dc = max(0.0, dc)

        # ---- ISI (Initial Spread Index) ------------------------------------
        f_W = math.exp(0.05039 * wind_kmh)
        m_isi = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
        f_F = 91.9 * math.exp(-0.1386 * m_isi) * (1.0 + (m_isi ** 5.31) / 4.93e7)
        isi = 0.208 * f_W * f_F
        isi = max(0.0, isi)

        # ---- BUI (Buildup Index) ------------------------------------------
        if dmc == 0 and dc == 0:
            bui = 0.0
        elif dmc <= 0.4 * dc:
            bui = 0.8 * dmc * dc / (dmc + 0.4 * dc) if (dmc + 0.4 * dc) > 0 else 0.0
        else:
            bui = dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc + 1e-9)) * (0.92 + (0.0114 * dmc) ** 1.7)
        bui = max(0.0, bui)

        # ---- FWI (Fire Weather Index) ------------------------------------
        if bui > 80.0:
            f_D = 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui))
        else:
            f_D = 0.626 * (bui ** 0.809) + 2.0

        B = 0.1 * isi * f_D
        if B > 1.0:
            fwi = math.exp(2.72 * ((0.434 * math.log(B)) ** 0.647))
        else:
            fwi = B
        fwi = max(0.0, fwi)

        # ---- Risk label ---------------------------------------------------
        risk_level = (
            "Very Low" if fwi < 5 else
            "Low"      if fwi < 10 else
            "Moderate" if fwi < 20 else
            "High"     if fwi < 30 else
            "Very High" if fwi < 40 else
            "Extreme"
        )

        return {
            "ffmc":              round(ffmc, 1),
            "dmc":               round(dmc, 1),
            "dc":                round(dc, 1),
            "isi":               round(isi, 2),
            "bui":               round(bui, 1),
            "fwi":               round(fwi, 1),
            "fire_danger_index": round(min(fwi, 100.0), 1),
            "temperature_max":   round(temp, 1),
            "humidity_min":      round(rh, 1),
            "wind_max_kmh":      round(wind_kmh, 1),
            "precipitation":     round(precip, 2),
            "risk_level":        risk_level,
        }

    # ------------------------------------------------------------------
    def _first(self, lst) -> Optional[float]:
        """Return the first non-None element of a list, or None."""
        if not lst:
            return None
        for v in lst:
            if v is not None:
                return float(v)
        return None

    def _default(self) -> Dict:
        """Low-risk defaults used when the API is unavailable."""
        return {
            "ffmc": 70.0, "dmc": 10.0, "dc": 50.0,
            "isi": 2.0, "bui": 15.0, "fwi": 5.0,
            "fire_danger_index": 5.0,
            "temperature_max": 20.0, "humidity_min": 40.0,
            "wind_max_kmh": 10.0, "precipitation": 2.0,
            "risk_level": "Low",
        }
