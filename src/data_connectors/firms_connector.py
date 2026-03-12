"""
NASA FIRMS (Fire Information for Resource Management System) Connector
Free API — requires a free MAP_KEY from https://firms.modaps.eosdis.nasa.gov/api/

Provides:
  - Recent active fire hotspots (VIIRS 375m resolution, ~3h latency)
  - Historical ignition locations for building ignition density maps
  - Fire Radiative Power (FRP) as fire intensity proxy

VIIRS 375 m active fire detection algorithm:
  Schroeder, W., Oliva, P., Giglio, L. & Csiszar, I.A. (2014).
  "The New VIIRS 375 m active fire detection data product: Algorithm description
  and initial assessment."
  Remote Sensing of Environment, 143, pp. 85–96.
  https://doi.org/10.1016/j.rse.2013.12.008
"""
import io
from typing import List, Tuple, Optional

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


class FIRMSConnector:
    """
    Fetches active fire hotspots from NASA FIRMS VIIRS NRT feed.

    Usage:
        connector = FIRMSConnector(map_key="YOUR_FREE_KEY")
        hotspots = connector.fetch_hotspots(lat, lon, radius_deg=0.1, days=1)
        density  = connector.build_ignition_density(lat, lon, radius_deg, grid_shape)
    """

    BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    SOURCE   = "VIIRS_SNPP_NRT"  # 375m, ~3h latency globally

    def __init__(self, map_key: str = ""):
        self.map_key = map_key.strip()
        self._hotspot_cache: Optional[List] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_hotspots(
        self,
        lat: float,
        lon: float,
        radius_deg: float = 0.2,
        days: int = 2,
    ) -> List[Tuple[float, float, float]]:
        """
        Fetch recent fire hotspots within a bounding box.

        Args:
            lat, lon:    Centre of the area of interest
            radius_deg:  Half-width of the bounding box in degrees
            days:        Number of past days to query (1–10)

        Returns:
            List of (lat, lon, frp) tuples.
            frp = Fire Radiative Power in MW (proxy for fire intensity).
            Returns [] if no key is set or the request fails.
        """
        if self._hotspot_cache is not None:
            return self._hotspot_cache

        if not self.map_key:
            print("  [FIRMS] No MAP_KEY set — skipping hotspot fetch. "
                  "Get a free key at https://firms.modaps.eosdis.nasa.gov/api/")
            return []

        if not _REQUESTS_AVAILABLE:
            print("  [FIRMS] 'requests' not installed — skipping.")
            return []

        w = round(lon - radius_deg, 4)
        s = round(lat - radius_deg, 4)
        e = round(lon + radius_deg, 4)
        n = round(lat + radius_deg, 4)
        bbox = f"{w},{s},{e},{n}"
        days = max(1, min(days, 10))

        url = f"{self.BASE_URL}/{self.map_key}/{self.SOURCE}/{bbox}/{days}"

        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            hotspots = self._parse_csv(resp.text)
            print(f"  [FIRMS] {len(hotspots)} active hotspots found "
                  f"(last {days}d, bbox={bbox})")
            self._hotspot_cache = hotspots
            return hotspots
        except Exception as e:
            print(f"  [FIRMS] WARNING: API error ({e}). No hotspot data available.")
            return []

    def build_ignition_density(
        self,
        lat: float,
        lon: float,
        radius_deg: float,
        grid_shape: Tuple[int, int],
        days: int = 7,
    ):
        """
        Build a 2-D ignition density grid from FIRMS hotspot history.

        Each hotspot is mapped to a grid cell and contributes +1.0 to that cell.
        The result is normalised to [0, 1].

        Returns np.ndarray of shape grid_shape, dtype float32.
        If no data, returns a zero grid.
        """
        if not _NUMPY_AVAILABLE:
            return None

        import numpy as np
        density = np.zeros(grid_shape, dtype=np.float32)
        hotspots = self.fetch_hotspots(lat, lon, radius_deg, days)

        if not hotspots:
            return density

        min_lat = lat - radius_deg
        max_lat = lat + radius_deg
        min_lon = lon - radius_deg
        max_lon = lon + radius_deg
        rows, cols = grid_shape

        for h_lat, h_lon, _frp in hotspots:
            r = int((max_lat - h_lat) / (max_lat - min_lat) * rows)
            c = int((h_lon - min_lon) / (max_lon - min_lon) * cols)
            r = max(0, min(rows - 1, r))
            c = max(0, min(cols - 1, c))
            density[r, c] += 1.0

        # Normalise to [0, 1]
        max_val = density.max()
        if max_val > 0:
            density /= max_val

        return density

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_csv(self, text: str) -> List[Tuple[float, float, float]]:
        """Parse FIRMS CSV response into (lat, lon, frp) tuples."""
        hotspots = []
        lines = text.strip().splitlines()
        if len(lines) < 2:
            return hotspots

        header = [h.strip().lower() for h in lines[0].split(",")]
        try:
            lat_idx = header.index("latitude")
            lon_idx = header.index("longitude")
            frp_idx = header.index("frp") if "frp" in header else None
        except ValueError:
            return hotspots

        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) <= max(lat_idx, lon_idx):
                continue
            try:
                h_lat = float(parts[lat_idx])
                h_lon = float(parts[lon_idx])
                frp   = float(parts[frp_idx]) if frp_idx is not None and frp_idx < len(parts) else 0.0
                hotspots.append((h_lat, h_lon, frp))
            except (ValueError, IndexError):
                continue

        return hotspots
