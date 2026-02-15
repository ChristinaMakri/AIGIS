"""
Fetch real-time active fire detections from NASA FIRMS (Fire Information for Resource Management System).

Data sources:
- MODIS (Moderate Resolution Imaging Spectroradiometer): 1km resolution, 4 times daily
- VIIRS (Visible Infrared Imaging Radiometer Suite): 375m resolution, twice daily

API Documentation: https://firms.modaps.eosdis.nasa.gov/api/
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time


def fetch_firms_active_fires(latitude, longitude, radius_km=50, days=1, source='VIIRS_NOAA20_NRT'):
    """
    Fetch active fire detections from NASA FIRMS within a geographic area.

    Args:
        latitude: Center latitude
        longitude: Center longitude
        radius_km: Search radius in kilometers
        days: Number of days to look back (1-10)
        source: Data source ('VIIRS_NOAA20_NRT', 'MODIS_NRT', 'VIIRS_SNPP_NRT')

    Returns:
        DataFrame with active fire detections

    API Key: Get free key from https://firms.modaps.eosdis.nasa.gov/api/area/
    Note: This uses the demo endpoint. For production, register for an API key.
    """
    print(f"🛰️  Fetching active fires from NASA FIRMS...")
    print(f"   Location: ({latitude:.4f}, {longitude:.4f})")
    print(f"   Radius: {radius_km} km")
    print(f"   Period: Last {days} day(s)")
    print(f"   Source: {source}")

    # NASA FIRMS API endpoint (using public demo)
    # For production, replace MAP_KEY with your registered API key
    base_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

    # Demo key (limited to 10 requests/minute)
    # Register for free at: https://firms.modaps.eosdis.nasa.gov/api/
    map_key = "DEMO_KEY"  # Replace with your API key for production

    # Calculate bounding box
    # Approximation: 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 111 km * cos(latitude)
    import math
    lat_offset = radius_km / 111.0
    lon_offset = radius_km / (111.0 * math.cos(math.radians(latitude)))

    min_lat = latitude - lat_offset
    max_lat = latitude + lat_offset
    min_lon = longitude - lon_offset
    max_lon = longitude + lon_offset

    # Build URL
    url = f"{base_url}/{map_key}/{source}/{min_lon},{min_lat},{max_lon},{max_lat}/{days}"

    try:
        print(f"\n   Requesting data from FIRMS API...")
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            # Parse CSV response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))

            print(f"   ✅ Retrieved {len(df)} active fire detections")

            if len(df) > 0:
                # Add distance from center
                df['distance_km'] = df.apply(
                    lambda row: haversine_distance(
                        latitude, longitude, row['latitude'], row['longitude']
                    ), axis=1
                )

                # Sort by distance
                df = df.sort_values('distance_km')

                # Print summary
                print(f"\n   📊 Fire Detection Summary:")
                print(f"      Closest fire: {df['distance_km'].min():.1f} km away")
                print(f"      Farthest fire: {df['distance_km'].max():.1f} km away")
                print(f"      Average confidence: {df['confidence'].mean():.0f}%")
                print(f"      High confidence fires: {len(df[df['confidence'] > 80])}")

            return df

        elif response.status_code == 404:
            print(f"   ℹ️  No active fires detected in this area")
            return pd.DataFrame()

        else:
            print(f"   ⚠️  API returned status {response.status_code}")
            print(f"   Message: {response.text[:200]}")
            return pd.DataFrame()

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return pd.DataFrame()


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km using Haversine formula"""
    import math

    R = 6371  # Earth radius in km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


def main():
    """Main execution"""
    print("=" * 70)
    print("🛰️  NASA FIRMS REAL-TIME FIRE DETECTION")
    print("=" * 70)
    print("\nSource: NASA Fire Information for Resource Management System")
    print("Satellites: VIIRS (375m resolution) + MODIS (1km resolution)")

    # Default location: Athens, Greece
    latitude = 38.0364
    longitude = 23.7281
    radius_km = 100  # 100km search radius

    print(f"\n📍 Target location: Athens, Greece")

    # Fetch active fires
    df = fetch_firms_active_fires(latitude, longitude, radius_km, days=1)

    if len(df) > 0:
        # Save to file
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        output_file = data_dir / "active_fires.csv"
        df.to_csv(output_file, index=False)

        print(f"\n💾 Active fire data saved to: {output_file}")

        # Display nearest fires
        print(f"\n🔥 Nearest Active Fires:")
        print("=" * 70)

        for idx, row in df.head(5).iterrows():
            print(f"\n  Fire #{idx + 1}:")
            print(f"    Location: ({row['latitude']:.4f}, {row['longitude']:.4f})")
            print(f"    Distance: {row['distance_km']:.1f} km")
            print(f"    Confidence: {row['confidence']}%")
            print(f"    Brightness: {row['bright_ti4']:.1f}K")
            print(f"    Detected: {row['acq_date']} {row['acq_time']}")

    else:
        print(f"\n✅ No active fires detected within {radius_km} km")
        print("   This is good news! The area is currently fire-free.")

    print("\n" + "=" * 70)
    print("✅ Fire detection complete!")
    print("=" * 70)
    print("\nNote: Using DEMO_KEY (limited to 10 requests/minute)")
    print("For production use, register for free API key at:")
    print("https://firms.modaps.eosdis.nasa.gov/api/")


if __name__ == "__main__":
    main()
