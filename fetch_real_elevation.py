"""
Fetch real elevation data from Open-Elevation API (SRTM dataset).

SRTM (Shuttle Radar Topography Mission) provides global elevation data:
- Resolution: ~30 meters (1 arc-second)
- Coverage: 56°S to 60°N (covers most populated areas)
- Free and open source

Alternative APIs:
- Open-Elevation: https://open-elevation.com (free, SRTM data)
- OpenTopoData: https://www.opentopodata.org (free, multiple datasets)
- Google Elevation API: (requires API key, paid)
"""
import requests
import numpy as np
from pathlib import Path
import time
from typing import Tuple


def fetch_elevation_grid(latitude, longitude, radius_meters, grid_size=100):
    """
    Fetch real elevation data for a geographic area.

    Uses Open-Elevation API with SRTM dataset (Shuttle Radar Topography Mission).

    Args:
        latitude: Center latitude
        longitude: Center longitude
        radius_meters: Radius in meters
        grid_size: Output grid size (grid_size x grid_size)

    Returns:
        elevation_grid: 2D numpy array with elevation in meters
        min_elevation: Minimum elevation in the area
        max_elevation: Maximum elevation in the area
    """
    print(f"🏔️  Fetching real elevation data...")
    print(f"   Location: ({latitude:.4f}, {longitude:.4f})")
    print(f"   Radius: {radius_meters}m")
    print(f"   Grid: {grid_size}x{grid_size}")
    print(f"   Source: SRTM (Shuttle Radar Topography)")

    # Calculate bounding box
    # Approximation: 1 degree latitude ≈ 111,320 meters
    lat_offset = radius_meters / 111320.0
    lon_offset = radius_meters / (111320.0 * np.cos(np.radians(latitude)))

    min_lat = latitude - lat_offset
    max_lat = latitude + lat_offset
    min_lon = longitude - lon_offset
    max_lon = longitude + lon_offset

    print(f"\n   Bounding box:")
    print(f"     Lat: {min_lat:.4f} to {max_lat:.4f}")
    print(f"     Lon: {min_lon:.4f} to {max_lon:.4f}")

    # Generate grid of sample points
    lats = np.linspace(max_lat, min_lat, grid_size)  # Top to bottom
    lons = np.linspace(min_lon, max_lon, grid_size)  # Left to right

    elevation_grid = np.zeros((grid_size, grid_size))

    # API endpoint
    # Using Open-Elevation (free, no API key required)
    api_url = "https://api.open-elevation.com/api/v1/lookup"

    print(f"\n   Fetching elevation for {grid_size * grid_size} points...")
    print(f"   (This may take a minute...)")

    # Fetch in batches to avoid overwhelming the API
    batch_size = 100  # API limit per request
    total_points = grid_size * grid_size
    fetched = 0

    for row in range(grid_size):
        for col_start in range(0, grid_size, batch_size):
            col_end = min(col_start + batch_size, grid_size)

            # Build locations for this batch
            locations = []
            for col in range(col_start, col_end):
                locations.append({
                    "latitude": lats[row],
                    "longitude": lons[col]
                })

            # Make API request
            try:
                response = requests.post(
                    api_url,
                    json={"locations": locations},
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                # Extract elevations
                if "results" in data:
                    for i, result in enumerate(data["results"]):
                        col = col_start + i
                        elevation = result.get("elevation", 0)
                        elevation_grid[row, col] = elevation

                fetched += len(locations)
                progress = (fetched / total_points) * 100
                print(f"   Progress: {progress:.0f}% ({fetched}/{total_points} points)", end="\r")

                # Rate limiting - be nice to the free API
                time.sleep(0.1)

            except Exception as e:
                print(f"\n   ⚠️  API error (using interpolation): {e}")
                # Fill with interpolated values on error
                if col_start > 0:
                    # Interpolate from previous values
                    for col in range(col_start, col_end):
                        elevation_grid[row, col] = elevation_grid[row, col_start - 1]
                else:
                    # Use default elevation
                    for col in range(col_start, col_end):
                        elevation_grid[row, col] = 100.0

    print(f"\n   ✅ Elevation data fetched")

    # Calculate statistics
    min_elev = elevation_grid.min()
    max_elev = elevation_grid.max()
    mean_elev = elevation_grid.mean()

    print(f"\n   📊 Elevation Statistics:")
    print(f"      Min:    {min_elev:.1f} m")
    print(f"      Max:    {max_elev:.1f} m")
    print(f"      Mean:   {mean_elev:.1f} m")
    print(f"      Range:  {max_elev - min_elev:.1f} m")

    return elevation_grid, min_elev, max_elev


def main():
    """Main execution"""
    print("=" * 70)
    print("🏔️  REAL ELEVATION DATA COLLECTION")
    print("=" * 70)
    print("\nSource: SRTM (Shuttle Radar Topography Mission)")
    print("API: Open-Elevation (free, no API key required)")
    print("Resolution: ~30 meters")

    # Default location: Athens, Greece
    latitude = 38.0364
    longitude = 23.7281
    radius = 5000  # 5km radius
    grid_size = 100  # 100x100 grid

    print(f"\n📍 Target location: Athens, Greece")

    # Fetch elevation data
    elevation_grid, min_elev, max_elev = fetch_elevation_grid(
        latitude, longitude, radius, grid_size
    )

    # Save to file
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    output_file = data_dir / "elevation_data.npz"
    np.savez(
        output_file,
        elevation_grid=elevation_grid,
        min_elevation=min_elev,
        max_elevation=max_elev,
        latitude=latitude,
        longitude=longitude,
        radius=radius,
        grid_size=grid_size
    )

    print(f"\n💾 Elevation data saved to: {output_file}")

    # Visual representation of terrain
    print(f"\n🗺️  Terrain Profile:")
    # Create simple ASCII visualization
    num_rows = 10
    for i in range(num_rows):
        row_idx = int((i / num_rows) * grid_size)
        row_slice = elevation_grid[row_idx, :]
        # Normalize to 0-10 scale for visualization
        normalized = ((row_slice - min_elev) / (max_elev - min_elev + 0.001) * 10).astype(int)
        visual = "".join(["█" if v > 7 else "▓" if v > 5 else "▒" if v > 3 else "░" for v in normalized])
        print(f"      {visual[:50]}")

    print("\n" + "=" * 70)
    print("✅ Elevation data collection complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Simulation will automatically load this data")
    print("  2. Fire spread will use real terrain slopes")
    print("  3. More accurate Rothermel fire model")


if __name__ == "__main__":
    main()
