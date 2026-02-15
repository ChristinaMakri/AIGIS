"""
Fetch real-time weather data from Open-Meteo API (free, no API key required).

Weather parameters fetched:
- Temperature (°C)
- Wind speed and direction
- Relative humidity (%)
- Precipitation (mm)
- Cloud cover (%)
- Surface pressure (hPa)

This data is used to initialize realistic fire spread conditions.
"""
import requests
import json
from datetime import datetime
from pathlib import Path


def get_real_weather(latitude, longitude):
    """
    Fetch current weather conditions from Open-Meteo API.

    Open-Meteo is a free weather API that doesn't require authentication.
    Data sources: NOAA, DWD, Météo-France, and other national weather services.

    Args:
        latitude: Location latitude
        longitude: Location longitude

    Returns:
        Dictionary with weather parameters
    """
    print(f"🌤️  Fetching real-time weather data...")
    print(f"   Location: ({latitude:.4f}, {longitude:.4f})")
    print(f"   Source: Open-Meteo API")

    # Open-Meteo API endpoint (no API key required!)
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        'latitude': latitude,
        'longitude': longitude,
        'current_weather': 'true',
        'hourly': 'temperature_2m,relativehumidity_2m,precipitation,windspeed_10m,winddirection_10m,cloudcover,surface_pressure',
        'timezone': 'auto'
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract current weather
        current = data.get('current_weather', {})

        weather_data = {
            'temperature': current.get('temperature', 25.0),  # °C
            'wind_speed': current.get('windspeed', 5.0),  # km/h
            'wind_direction': current.get('winddirection', 90.0),  # degrees
            'time': current.get('time', datetime.now().isoformat()),
            'weathercode': current.get('weathercode', 0)
        }

        # Extract hourly data (first hour for additional parameters)
        if 'hourly' in data:
            hourly = data['hourly']
            weather_data['humidity'] = hourly.get('relativehumidity_2m', [30])[0]  # %
            weather_data['precipitation'] = hourly.get('precipitation', [0])[0]  # mm
            weather_data['cloud_cover'] = hourly.get('cloudcover', [0])[0]  # %
            weather_data['pressure'] = hourly.get('surface_pressure', [1013])[0]  # hPa

        print(f"   ✅ Weather data retrieved")
        print(f"\n   Current conditions:")
        print(f"     Temperature:  {weather_data['temperature']:.1f}°C")
        print(f"     Wind speed:   {weather_data['wind_speed']:.1f} km/h")
        print(f"     Wind dir:     {weather_data['wind_direction']:.0f}°")
        print(f"     Humidity:     {weather_data.get('humidity', 'N/A')}%")
        print(f"     Precipitation: {weather_data.get('precipitation', 0):.1f} mm")

        return weather_data

    except Exception as e:
        print(f"   ❌ Error fetching weather: {e}")
        print(f"   Using default weather parameters")

        # Return default values if API fails
        return {
            'temperature': 25.0,
            'wind_speed': 5.0,
            'wind_direction': 90.0,
            'humidity': 30.0,
            'precipitation': 0.0,
            'cloud_cover': 0,
            'pressure': 1013.0,
            'time': datetime.now().isoformat()
        }


def calculate_fire_weather_index(temperature, humidity, wind_speed, precipitation):
    """
    Calculate Fire Weather Index (FWI) - simplified version.

    FWI is used to assess fire danger. Higher values = higher fire risk.

    Factors:
    - High temperature → more fire risk
    - Low humidity → more fire risk
    - High wind → more fire risk
    - Precipitation → reduces fire risk

    Args:
        temperature: Temperature in °C
        humidity: Relative humidity in %
        wind_speed: Wind speed in km/h
        precipitation: Precipitation in mm

    Returns:
        Fire danger level: 'LOW', 'MODERATE', 'HIGH', 'EXTREME'
    """
    # Simplified FWI calculation
    fwi_score = 0

    # Temperature contribution (0-30 points)
    if temperature > 35:
        fwi_score += 30
    elif temperature > 30:
        fwi_score += 20
    elif temperature > 25:
        fwi_score += 10

    # Humidity contribution (0-30 points) - inverse relationship
    if humidity < 20:
        fwi_score += 30
    elif humidity < 30:
        fwi_score += 20
    elif humidity < 40:
        fwi_score += 10

    # Wind contribution (0-20 points)
    if wind_speed > 40:
        fwi_score += 20
    elif wind_speed > 25:
        fwi_score += 15
    elif wind_speed > 15:
        fwi_score += 10

    # Precipitation reduction (0-20 points penalty)
    if precipitation > 5:
        fwi_score -= 20
    elif precipitation > 1:
        fwi_score -= 10

    fwi_score = max(0, fwi_score)

    # Classify fire danger
    if fwi_score >= 60:
        return 'EXTREME', fwi_score
    elif fwi_score >= 40:
        return 'HIGH', fwi_score
    elif fwi_score >= 20:
        return 'MODERATE', fwi_score
    else:
        return 'LOW', fwi_score


def main():
    """Main execution"""
    print("=" * 70)
    print("🌤️  REAL-TIME WEATHER DATA FETCHING")
    print("=" * 70)
    print("\nSource: Open-Meteo API (free, no API key required)")
    print("Data: NOAA, DWD, Météo-France, and other weather services")

    # Default location: Athens, Greece
    latitude = 38.0364
    longitude = 23.7281

    print(f"\n📍 Target location: Athens, Greece")

    # Fetch weather
    weather = get_real_weather(latitude, longitude)

    # Calculate fire danger
    fire_danger, fwi_score = calculate_fire_weather_index(
        weather['temperature'],
        weather.get('humidity', 30),
        weather['wind_speed'],
        weather.get('precipitation', 0)
    )

    print(f"\n🔥 FIRE WEATHER INDEX")
    print("=" * 70)
    print(f"  FWI Score: {fwi_score}/100")
    print(f"  Fire Danger: {fire_danger}")

    # Save to file
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    output_file = data_dir / "current_weather.json"
    weather['fire_danger'] = fire_danger
    weather['fwi_score'] = fwi_score

    with open(output_file, 'w') as f:
        json.dump(weather, f, indent=2)

    print(f"\n💾 Weather data saved to: {output_file}")

    print("\n" + "=" * 70)
    print("✅ Weather data retrieval complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Simulation will use this weather data")
    print("  2. Wind speed/direction affect fire spread")
    print("  3. Temperature/humidity affect fire intensity")
    print("  4. Re-run this script to update weather conditions")


if __name__ == "__main__":
    main()
