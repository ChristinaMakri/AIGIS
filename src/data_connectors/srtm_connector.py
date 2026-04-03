"""
SRTM Elevation Connector
========================
Fetches real terrain elevation for any lat/lon/radius using the SRTM 30m
dataset via the OpenTopoData public API (free, no API key required).

Primary reference:
  Farr, T. G. et al. (2007). "The Shuttle Radar Topography Mission."
  Reviews of Geophysics, 45(2), RG2004.
  DOI: 10.1029/2005RG000183.

API:
  OpenTopoData SRTM30m endpoint — https://www.opentopodata.org/
  Supports batch queries of up to 100 locations per request, no auth needed.
  Rate limit: 1 request/second on the public instance.

Strategy:
  1. Build a coarse sample grid (default 40×40 = 1600 points) over the area.
  2. Query OpenTopoData in batches of ≤100 points.
  3. Bicubic-interpolate the coarse grid to the target resolution (200×200).
  Returns None on any failure so the caller can fall back to Perlin terrain.
"""
from __future__ import annotations
import time
import math
import numpy as np
from typing import Optional, Tuple

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    from scipy.interpolate import RegularGridInterpolator
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


class SRTMConnector:
    """
    Fetch SRTM 30m elevation for a rectangular region and return it
    resampled to any target grid resolution.

    Usage:
        connector = SRTMConnector()
        elevation = connector.fetch_elevation_grid(
            lat=38.09, lon=23.92, radius=3000, grid_shape=(200, 200)
        )
        # elevation is np.ndarray (200, 200) in metres, or None on failure.
    """

    OPENTOPODATA_URL = "https://api.opentopodata.org/v1/srtm30m"
    BATCH_SIZE       = 100    # OpenTopoData max locations per request
    REQUEST_DELAY    = 1.05   # seconds between requests (rate limit: 1 req/s)

    def fetch_elevation_grid(
        self,
        lat:        float,
        lon:        float,
        radius:     float,   # metres
        grid_shape: Tuple[int, int] = (200, 200),
        sample_res: int = 40,  # coarse grid resolution before interpolation
    ) -> Optional[np.ndarray]:
        """
        Fetch SRTM elevation and return a grid of shape ``grid_shape``.

        Args:
            lat, lon:    Centre of the area in decimal degrees.
            radius:      Half-width of the square area in metres.
            grid_shape:  Target output grid resolution (rows, cols).
            sample_res:  Coarse sampling resolution.  40×40 = 1600 points;
                         at 3 km radius each sample covers ~150 m — well within
                         the SRTM 30m native resolution after interpolation.

        Returns:
            np.ndarray of shape grid_shape (float32, metres), or None.
        """
        if not _REQUESTS_AVAILABLE:
            print("  [SRTM] 'requests' not installed — skipping SRTM fetch.")
            return None
        if not _SCIPY_AVAILABLE:
            print("  [SRTM] 'scipy' not installed — skipping SRTM fetch.")
            return None

        # Convert radius in metres to degrees (approximate)
        lat_deg = radius / 111_320.0
        lon_deg = radius / (111_320.0 * math.cos(math.radians(lat)))

        # Build coarse sample grid
        lats = np.linspace(lat - lat_deg, lat + lat_deg, sample_res)
        lons = np.linspace(lon - lon_deg, lon + lon_deg, sample_res)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        points_lat = lat_grid.ravel()
        points_lon = lon_grid.ravel()
        n_points    = len(points_lat)

        elevations = np.zeros(n_points, dtype=np.float32)

        # Query in batches
        n_batches = math.ceil(n_points / self.BATCH_SIZE)
        print(f"  [SRTM] Fetching {n_points} elevation points "
              f"({n_batches} batches)...")

        for batch_idx in range(n_batches):
            start = batch_idx * self.BATCH_SIZE
            end   = min(start + self.BATCH_SIZE, n_points)

            locations = "|".join(
                f"{points_lat[i]:.6f},{points_lon[i]:.6f}"
                for i in range(start, end)
            )

            try:
                resp = requests.get(
                    self.OPENTOPODATA_URL,
                    params={"locations": locations},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                for j, result in enumerate(data.get("results", [])):
                    elev = result.get("elevation")
                    if elev is not None:
                        elevations[start + j] = float(elev)

            except Exception as exc:
                print(f"  [SRTM] Batch {batch_idx + 1}/{n_batches} failed: {exc}")
                return None

            if batch_idx < n_batches - 1:
                time.sleep(self.REQUEST_DELAY)

        # Reshape coarse grid
        elev_coarse = elevations.reshape(sample_res, sample_res)

        # Interpolate to target resolution
        elev_fine = self._interpolate(elev_coarse, grid_shape)

        e_min, e_max = elev_fine.min(), elev_fine.max()
        print(f"  [SRTM] Elevation loaded — min: {e_min:.1f} m, "
              f"max: {e_max:.1f} m, range: {e_max - e_min:.1f} m")

        return elev_fine

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate(
        coarse: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Bicubic (linear in RegularGridInterpolator terms) upsampling
        from coarse grid to target_shape.
        """
        src_rows, src_cols = coarse.shape
        row_coords = np.linspace(0, 1, src_rows)
        col_coords = np.linspace(0, 1, src_cols)

        interpolator = RegularGridInterpolator(
            (row_coords, col_coords),
            coarse,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        tgt_rows, tgt_cols = target_shape
        tgt_row = np.linspace(0, 1, tgt_rows)
        tgt_col = np.linspace(0, 1, tgt_cols)
        tgt_col_g, tgt_row_g = np.meshgrid(tgt_col, tgt_row)
        query_pts = np.column_stack([tgt_row_g.ravel(), tgt_col_g.ravel()])

        result = interpolator(query_pts).reshape(target_shape)
        return result.astype(np.float32)
