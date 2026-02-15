"""
Fetch real population data from OpenStreetMap (OSM) using building footprints.

Estimates population density by:
1. Downloading building footprints from OSM
2. Filtering for residential buildings
3. Estimating population (2.5 people per residential building average)
4. Creating a population density grid

Requires: osmnx, geopandas, scipy
"""
import osmnx as ox
import numpy as np
import geopandas as gpd
from pathlib import Path
from scipy.ndimage import gaussian_filter


def get_population_from_osm(lat, lon, radius, grid_size=100):
    """
    Fetch real population data from OpenStreetMap.

    Args:
        lat: Center latitude
        lon: Center longitude
        radius: Radius in meters
        grid_size: Output grid size (grid_size x grid_size)

    Returns:
        population_grid: 2D numpy array with population density
        total_population: Total estimated population
    """
    print(f"🌍 Fetching population data from OpenStreetMap...")
    print(f"   Location: ({lat:.4f}, {lon:.4f})")
    print(f"   Radius: {radius}m")

    # Fetch building footprints from OSM
    try:
        tags = {'building': True}
        buildings = ox.features_from_point((lat, lon), tags=tags, dist=radius)
        print(f"   ✅ Downloaded {len(buildings)} buildings")
    except Exception as e:
        print(f"   ❌ Error fetching buildings: {e}")
        return np.zeros((grid_size, grid_size)), 0

    # Filter for residential buildings
    residential_types = [
        'residential', 'house', 'apartments', 'detached',
        'semidetached_house', 'terrace', 'bungalow', 'yes',
        'apartment', 'dormitory', 'farm'
    ]

    if 'building' in buildings.columns:
        residential_buildings = buildings[
            buildings['building'].isin(residential_types) |
            buildings['building'].str.contains('house|apartment', case=False, na=False)
        ]
    else:
        # If no building type specified, assume 60% are residential
        residential_buildings = buildings.sample(frac=0.6, random_state=42)

    print(f"   🏠 Residential buildings: {len(residential_buildings)}")

    # Estimate population (2.5 people per residential building on average)
    # This is a global average - varies by region
    population_per_building = 2.5
    estimated_population = len(residential_buildings) * population_per_building

    print(f"   👥 Estimated population: {int(estimated_population)} people")

    # Create bounding box
    lat_min = lat - (radius / 111320)  # 1 degree lat ≈ 111.32 km
    lat_max = lat + (radius / 111320)
    lon_min = lon - (radius / (111320 * np.cos(np.radians(lat))))
    lon_max = lon + (radius / (111320 * np.cos(np.radians(lat))))

    # Create population grid
    population_grid = np.zeros((grid_size, grid_size))

    if len(residential_buildings) > 0:
        # Get building centroids
        centroids = residential_buildings.geometry.centroid

        for centroid in centroids:
            # Convert to grid coordinates
            if centroid is not None and not centroid.is_empty:
                lon_c = centroid.x
                lat_c = centroid.y

                # Map to grid indices
                grid_x = int((lon_c - lon_min) / (lon_max - lon_min) * (grid_size - 1))
                grid_y = int((lat_c - lat_min) / (lat_max - lat_min) * (grid_size - 1))

                # Ensure within bounds
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    population_grid[grid_y, grid_x] += population_per_building

        # Smooth the grid using Gaussian filter to spread population
        population_grid = gaussian_filter(population_grid, sigma=2)

    return population_grid, int(estimated_population)


def main():
    """Main execution"""
    print("=" * 70)
    print("👥 REAL POPULATION DATA COLLECTION")
    print("=" * 70)
    print("\nSource: OpenStreetMap (OSM)")
    print("Method: Building footprint analysis")

    # Default location: Athens, Greece (can be customized)
    latitude = 38.0364
    longitude = 23.7281
    radius = 5000  # 5km radius

    print(f"\n📍 Target location: Athens, Greece")
    print(f"   Coordinates: ({latitude}, {longitude})")
    print(f"   Radius: {radius}m")

    # Fetch population data
    population_grid, total_population = get_population_from_osm(
        latitude, longitude, radius, grid_size=100
    )

    # Save to file
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    output_file = data_dir / "population_data.npz"
    np.savez(
        output_file,
        population_grid=population_grid,
        total_population=total_population,
        latitude=latitude,
        longitude=longitude,
        radius=radius
    )

    print(f"\n💾 Population data saved to: {output_file}")
    print(f"\n📊 POPULATION SUMMARY")
    print("=" * 70)
    print(f"  Grid size: {population_grid.shape}")
    print(f"  Total population: {total_population} people")
    print(f"  Max density: {population_grid.max():.1f} people/cell")
    print(f"  Mean density: {population_grid.mean():.1f} people/cell")
    print(f"  Non-zero cells: {np.count_nonzero(population_grid)}")

    print("\n" + "=" * 70)
    print("✅ Population data collection complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Simulation will automatically load this data")
    print("  2. Population density affects casualty predictions")
    print("  3. To change location, edit latitude/longitude in this script")


if __name__ == "__main__":
    main()
