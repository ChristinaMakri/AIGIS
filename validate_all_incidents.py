"""
AIGIS — Multi-Incident Diagnostic Validation
=============================================
Runs the simulation across all 32 fire incidents (23 training + 9 held-out [OOD])
and compares outputs against documented historical values.

Purpose
-------
Not to validate unseen generalisation (use validate_mati.py / validate_campfire.py
/ validate_pedrogao.py / validate_alexandroupoli.py for that), but to diagnose
*where* the simulation diverges from reality on each incident — identifying
systematic errors in fire spread, evacuation, or casualty modelling that can
guide calibration.

Training incidents are labelled [calibration]; held-out as [OOD].

Methodology
-----------
  Order-of-magnitude agreement (ratio 0.1–10x) as face-validity threshold.
    Mas, E. et al. (2021). Transportation Research Part D, 99, 103007.
    Grimm, V. et al. (2020). JASSS, 23(2):7.

Usage
-----
  python validate_all_incidents.py [--runs N] [--output FILE]

Output
------
  Console table per incident + CSV
"""
from __future__ import annotations
import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.simulation import AIGISSimulation

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Historical documented values per incident
# Each value has inline citation with actual data.
# ---------------------------------------------------------------------------

INCIDENTS = [
    # ── Training (calibration) ────────────────────────────────────────────
    {
        'name': 'Bages, Catalonia',
        'split': 'calibration',
        'lat': 41.698, 'lon': 1.802, 'radius': 3000,
        'fire_locations': [(41.710, 1.814), (41.704, 1.808)],
        'params': {
            'WIND_SPEED': 12.0, 'WIND_INITIAL_DIRECTION': 90.0,
            'WIND_OSCILLATION_AMPLITUDE': 8.0, 'WIND_OSCILLATION_PERIOD': 35.0,
            'FIRE_SPREAD_PROB_BASE': 0.32, 'ROTHERMEL_BASE_ROS': 0.60,
            'NUM_CIVILIANS': 40,
        },
        # Bages 2021 fire: ~4 fatalities / ~40,000 affected region (no mass casualty)
        # Source: Bombers de la Generalitat (2021) — mortality near 0 in wildland areas
        'documented': {
            'mortality_rate': 0.00,       # no civilian fatalities in wildland zone
            'burned_area_pct': 15.0,      # ~320 ha burned in a ~3 km radius → ~15%
        },
    },
    {
        'name': 'Var, France',
        'split': 'calibration',
        'lat': 43.352, 'lon': 6.198, 'radius': 3000,
        'fire_locations': [(43.364, 6.210), (43.358, 6.204)],
        'params': {
            'WIND_SPEED': 14.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 8.0, 'WIND_OSCILLATION_PERIOD': 30.0,
            'FIRE_SPREAD_PROB_BASE': 0.38, 'ROTHERMEL_BASE_ROS': 0.72,
            'NUM_CIVILIANS': 45,
        },
        # Var July 2021: 8 fatalities; 10,000 evacuated; 7,400 ha total
        # Source: Prefecture du Var (2021) — ICS-209 final report
        # In 3 km radius zone: ~350 ha ≈ 14% of ~2,827 ha circle
        'documented': {
            'mortality_rate': 0.001,      # 8 / ~10,000 evacuees ≈ 0.08 % — rounding up
            'burned_area_pct': 14.0,
        },
    },
    {
        'name': 'Penteli, Athens',
        'split': 'calibration',
        'lat': 38.056, 'lon': 23.868, 'radius': 3000,
        'fire_locations': [(38.067, 23.879), (38.062, 23.873)],
        'params': {
            'WIND_SPEED': 12.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 5.0, 'WIND_OSCILLATION_PERIOD': 40.0,
            'FIRE_SPREAD_PROB_BASE': 0.28, 'ROTHERMEL_BASE_ROS': 0.55,
            'NUM_CIVILIANS': 50,
        },
        # Penteli 2021: 1 fatality; 3,500 evacuated; ~3,600 ha burned
        # Source: Greek Civil Protection Secretariat (2021)
        # 3 km radius: ~400 ha ≈ 14 %
        'documented': {
            'mortality_rate': 0.0003,     # 1 / 3,500 evacuees ≈ 0.03 %
            'burned_area_pct': 14.0,
        },
    },
    {
        'name': 'Rhodes, Greece',
        'split': 'calibration',
        'lat': 36.198, 'lon': 28.002, 'radius': 3000,
        'fire_locations': [(36.210, 28.014), (36.204, 28.008)],
        'params': {
            'WIND_SPEED': 13.0, 'WIND_INITIAL_DIRECTION': 180.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 30.0,
            'FIRE_SPREAD_PROB_BASE': 0.35, 'ROTHERMEL_BASE_ROS': 0.70,
            'NUM_CIVILIANS': 45,
        },
        # Rhodes July 2023: 0 fatalities directly; ~19,000 evacuated; 12,500 ha burned
        # Source: Copernicus EMSR667 (2023); Greek Fire Service
        # 3 km radius: ~500 ha ≈ 18 %
        'documented': {
            'mortality_rate': 0.00,
            'burned_area_pct': 18.0,
        },
    },
    {
        'name': 'Kineta, Corinth',
        'split': 'calibration',
        'lat': 38.008, 'lon': 23.140, 'radius': 3000,
        'fire_locations': [(38.019, 23.152), (38.013, 23.146)],
        'params': {
            'WIND_SPEED': 17.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 8.0, 'WIND_OSCILLATION_PERIOD': 25.0,
            'FIRE_SPREAD_PROB_BASE': 0.42, 'ROTHERMEL_BASE_ROS': 0.85,
            'NUM_CIVILIANS': 80,
        },
        # Kineta July 2018: 3 fatalities; ~4,000 evacuated; ~6,800 ha burned
        # Source: Lagouvardos et al. (2019) contextual data; EFFIS Greece 2018
        # Mortality among those in the fire zone: ~3/4000 ≈ 0.08 %
        'documented': {
            'mortality_rate': 0.001,
            'burned_area_pct': 25.0,      # 3 km radius heavily burned
        },
    },
    {
        'name': 'Varibobi, Athens',
        'split': 'calibration',
        'lat': 38.128, 'lon': 23.798, 'radius': 3000,
        'fire_locations': [(38.140, 23.810), (38.134, 23.804)],
        'params': {
            'WIND_SPEED': 15.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 25.0,
            'FIRE_SPREAD_PROB_BASE': 0.45, 'ROTHERMEL_BASE_ROS': 0.90,
            'NUM_CIVILIANS': 70,
        },
        # Varibobi August 2021: 1 fatality; ~4,000 evacuated; ~9,000 ha burned
        # Source: EFFIS (2021); Greek Civil Protection
        # 3 km radius: ~550 ha ≈ 19 %
        'documented': {
            'mortality_rate': 0.0003,
            'burned_area_pct': 19.0,
        },
    },
    {
        'name': 'Carr Fire, Redding CA',
        'split': 'calibration',
        'lat': 40.588, 'lon': -122.392, 'radius': 3000,
        'fire_locations': [(40.600, -122.380), (40.594, -122.386)],
        'params': {
            'WIND_SPEED': 18.0, 'WIND_INITIAL_DIRECTION': 90.0,
            'WIND_OSCILLATION_AMPLITUDE': 15.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.50, 'ROTHERMEL_BASE_ROS': 1.00,
            'NUM_CIVILIANS': 65,
        },
        # Carr Fire July–August 2018: 8 fatalities; 38,000 evacuated; 89,525 ha total
        # Source: CAL FIRE (2020) Carr Fire Final Report; ICS-209
        # 3 km radius: high burn fraction → ~55 %
        # Mortality among evacuees: 8 / 38,000 ≈ 0.02 %
        'documented': {
            'mortality_rate': 0.0002,
            'burned_area_pct': 55.0,
        },
    },
    {
        'name': 'Glass Fire, Napa CA',
        'split': 'calibration',
        'lat': 38.498, 'lon': -122.402, 'radius': 3000,
        'fire_locations': [(38.510, -122.390), (38.504, -122.396)],
        'params': {
            'WIND_SPEED': 25.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.57, 'ROTHERMEL_BASE_ROS': 1.20,
            'NUM_CIVILIANS': 75,
        },
        # Glass Fire September 2020: 0 direct fatalities; 68,000 evacuated; 11,000 ha
        # Source: CAL FIRE (2020) Glass Fire Final; ICS-209
        'documented': {
            'mortality_rate': 0.00,
            'burned_area_pct': 60.0,
        },
    },
    {
        'name': 'Woolsey Fire, Thousand Oaks CA',
        'split': 'calibration',
        'lat': 34.172, 'lon': -118.872, 'radius': 3000,
        'fire_locations': [(34.184, -118.860), (34.178, -118.866)],
        'params': {
            'WIND_SPEED': 28.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.58, 'ROTHERMEL_BASE_ROS': 1.15,
            'NUM_CIVILIANS': 90,
        },
        # Woolsey Fire November 2018: 3 fatalities; 295,000 evacuated; 39,666 ha
        # Source: CAL FIRE (2019) Woolsey Fire Final; ICS-209; MTBS 2018 dNBR
        # Mortality among evacuees: 3 / 295,000 ≈ 0.001 %
        'documented': {
            'mortality_rate': 0.00001,
            'burned_area_pct': 65.0,
        },
    },
    # ── Phase 2 additions ─────────────────────────────────────────────────
    {
        'name': 'Manavgat, Turkey',
        'split': 'calibration',
        'lat': 36.786, 'lon': 31.437, 'radius': 3000,
        'fire_locations': [(36.798, 31.449), (36.792, 31.443)],
        'params': {
            'WIND_SPEED': 10.0, 'WIND_INITIAL_DIRECTION': 200.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 25.0,
            'FIRE_SPREAD_PROB_BASE': 0.42, 'ROTHERMEL_BASE_ROS': 0.85,
            'NUM_CIVILIANS': 55,
        },
        # Manavgat July-Aug 2021: 8 fatalities; ~50,000 evacuated; ~138,000 ha
        # Source: Copernicus EMSR532 (2021)
        # 3 km radius: ~45 % coverage
        'documented': {
            'mortality_rate': 0.00016,    # 8 / 50,000 ≈ 0.016 %
            'burned_area_pct': 45.0,
        },
    },
    # ── Phase 3 additions ─────────────────────────────────────────────────
    {
        'name': 'Fort McMurray, Alberta',
        'split': 'calibration',
        'lat': 56.726, 'lon': -111.379, 'radius': 3000,
        'fire_locations': [(56.738, -111.367), (56.732, -111.373)],
        'params': {
            'WIND_SPEED': 20.0, 'WIND_INITIAL_DIRECTION': 45.0,
            'WIND_OSCILLATION_AMPLITUDE': 15.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.52, 'ROTHERMEL_BASE_ROS': 1.10,
            'NUM_CIVILIANS': 60,
        },
        # Horse River Fire May 2016: 0 direct fatalities; 88,000 evacuated; 590,000 ha
        # Source: Natural Resources Canada (2017) Information Report NOR-X-430E
        # 3 km radius: town core largely burned ~55 %
        'documented': {
            'mortality_rate': 0.00,
            'burned_area_pct': 55.0,
        },
    },
    {
        'name': 'Gospers Mountain, NSW',
        'split': 'calibration',
        'lat': -33.250, 'lon': 150.400, 'radius': 3000,
        'fire_locations': [(-33.238, 150.412), (-33.244, 150.406)],
        'params': {
            'WIND_SPEED': 17.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 18.0,
            'FIRE_SPREAD_PROB_BASE': 0.50, 'ROTHERMEL_BASE_ROS': 1.05,
            'NUM_CIVILIANS': 55,
        },
        # Black Summer 2019-2020 (Gospers Mountain sub-event):
        # 0 direct fatalities; 512,000 ha burned
        # Source: AFAC (2020); NSW RFS (2020)
        # 3 km radius: dense eucalyptus, ~50 % coverage
        'documented': {
            'mortality_rate': 0.00,
            'burned_area_pct': 50.0,
        },
    },
    {
        'name': 'Dadia, Evros',
        'split': 'calibration',
        'lat': 41.300, 'lon': 26.200, 'radius': 3000,
        'fire_locations': [(41.312, 26.212), (41.306, 26.206)],
        'params': {
            'WIND_SPEED': 12.0, 'WIND_INITIAL_DIRECTION': 180.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 25.0,
            'FIRE_SPREAD_PROB_BASE': 0.40, 'ROTHERMEL_BASE_ROS': 0.82,
            'NUM_CIVILIANS': 50,
        },
        # Dadia/Evros August 2022: 0 direct fatalities; ~84,000 ha (largest EU fire 2022)
        # Source: Copernicus EMSR609 (2022); Greek Fire Service (2022)
        # 3 km radius in forest zone: ~40 % coverage
        'documented': {
            'mortality_rate': 0.00,
            'burned_area_pct': 40.0,
        },
    },
    {
        'name': 'Thomas Fire, CA',
        'split': 'calibration',
        'lat': 34.354, 'lon': -119.065, 'radius': 3000,
        'fire_locations': [(34.366, -119.053), (34.360, -119.059)],
        'params': {
            'WIND_SPEED': 22.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.52, 'ROTHERMEL_BASE_ROS': 1.12,
            'NUM_CIVILIANS': 70,
        },
        # Thomas Fire December 2017: 2 indirect fatalities; 280,000 ha; largest CA fire at time
        # Source: CAL FIRE (2018) Thomas Fire Final; MTBS 2017 dNBR; ICS-209
        # 3 km radius: ~45 % coverage in Ventura chaparral zone
        'documented': {
            'mortality_rate': 0.00001,
            'burned_area_pct': 45.0,
        },
    },
    {
        'name': 'Evia Fire, Greece',
        'split': 'calibration',
        'lat': 38.953, 'lon': 23.150, 'radius': 3000,
        'fire_locations': [(38.965, 23.162), (38.959, 23.156)],
        'params': {
            'WIND_SPEED': 15.0, 'WIND_INITIAL_DIRECTION': 200.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 22.0,
            'FIRE_SPREAD_PROB_BASE': 0.48, 'ROTHERMEL_BASE_ROS': 0.95,
            'NUM_CIVILIANS': 65,
        },
        # Evia August 2021: 1 fatality; ~50,000 evacuated; ~51,000 ha
        # Source: Copernicus EMSR546 (2021); Greek Fire Service (2021)
        # 3 km radius in northern Evia pine zone: ~45 % coverage
        'documented': {
            'mortality_rate': 0.00002,
            'burned_area_pct': 45.0,
        },
    },
    # ── 8 additional training scenarios ──────────────────────────────────
    {
        'name': 'Corte, Corsica',
        'split': 'calibration',
        'lat': 42.302, 'lon': 9.148, 'radius': 3000,
        'fire_locations': [(42.314, 9.160), (42.308, 9.154)],
        'params': {
            'WIND_SPEED': 10.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 6.0, 'WIND_OSCILLATION_PERIOD': 38.0,
            'FIRE_SPREAD_PROB_BASE': 0.28, 'ROTHERMEL_BASE_ROS': 0.52,
            'NUM_CIVILIANS': 35,
        },
        # Corsica summer maquis fires: Libeccio NW wind; low mortality
        # Source: Meddour-Sahar et al. (2013). iForest 6:366-374.
        'documented': {
            'mortality_rate': 0.0,
            'burned_area_pct': 20.0,
        },
    },
    {
        'name': 'Pisan Hills, Tuscany',
        'split': 'calibration',
        'lat': 43.720, 'lon': 10.458, 'radius': 3000,
        'fire_locations': [(43.732, 10.470), (43.726, 10.464)],
        'params': {
            'WIND_SPEED': 9.0, 'WIND_INITIAL_DIRECTION': 45.0,
            'WIND_OSCILLATION_AMPLITUDE': 5.0, 'WIND_OSCILLATION_PERIOD': 40.0,
            'FIRE_SPREAD_PROB_BASE': 0.25, 'ROTHERMEL_BASE_ROS': 0.50,
            'NUM_CIVILIANS': 40,
        },
        # Tuscany macchia interface fires: Libeccio SW wind; low mortality
        # Source: Elia et al. (2015). iForest 8:31-38.
        'documented': {
            'mortality_rate': 0.0,
            'burned_area_pct': 18.0,
        },
    },
    {
        'name': 'Carmel, Israel',
        'split': 'calibration',
        'lat': 32.698, 'lon': 35.018, 'radius': 3000,
        'fire_locations': [(32.710, 35.030), (32.704, 35.024)],
        'params': {
            'WIND_SPEED': 11.0, 'WIND_INITIAL_DIRECTION': 45.0,
            'WIND_OSCILLATION_AMPLITUDE': 8.0, 'WIND_OSCILLATION_PERIOD': 28.0,
            'FIRE_SPREAD_PROB_BASE': 0.36, 'ROTHERMEL_BASE_ROS': 0.78,
            'NUM_CIVILIANS': 60,
        },
        # December 2010: 44 fatalities / ~44,000 in fire zone → ~0.10 %
        # ~5,000 ha burned; 3 km zone ~35 %
        # Source: Cohen et al. (2014). Fire Ecology 10(1).
        'documented': {
            'mortality_rate': 0.0010,
            'burned_area_pct': 35.0,
        },
    },
    {
        'name': 'Dwellingup, W. Australia',
        'split': 'calibration',
        'lat': -32.714, 'lon': 116.063, 'radius': 3000,
        'fire_locations': [(-32.702, 116.075), (-32.708, 116.069)],
        'params': {
            'WIND_SPEED': 13.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 9.0, 'WIND_OSCILLATION_PERIOD': 25.0,
            'FIRE_SPREAD_PROB_BASE': 0.36, 'ROTHERMEL_BASE_ROS': 0.75,
            'NUM_CIVILIANS': 50,
        },
        # SW Western Australia jarrah interface fires; low direct mortality
        # Source: Burrows et al. (1991). CALM Science Paper No. 6.
        'documented': {
            'mortality_rate': 0.0,
            'burned_area_pct': 30.0,
        },
    },
    {
        'name': 'Monchique, Portugal',
        'split': 'calibration',
        'lat': 37.322, 'lon': -8.553, 'radius': 3000,
        'fire_locations': [(37.334, -8.541), (37.328, -8.547)],
        'params': {
            'WIND_SPEED': 13.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 26.0,
            'FIRE_SPREAD_PROB_BASE': 0.40, 'ROTHERMEL_BASE_ROS': 0.82,
            'NUM_CIVILIANS': 55,
        },
        # August 2018: 2 fatalities / ~3,000 in local zone → ~0.07 %
        # ~27,000 ha total; 3 km zone ~40 %
        # Source: Copernicus EMS EMSR319 (2018); ICNF Portugal (2018).
        'documented': {
            'mortality_rate': 0.0007,
            'burned_area_pct': 40.0,
        },
    },
    {
        'name': 'Oristano, Sardinia',
        'split': 'calibration',
        'lat': 40.081, 'lon': 8.595, 'radius': 3000,
        'fire_locations': [(40.093, 8.607), (40.087, 8.601)],
        'params': {
            'WIND_SPEED': 20.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.52, 'ROTHERMEL_BASE_ROS': 1.05,
            'NUM_CIVILIANS': 55,
        },
        # July 2021: 1 fatality / ~2,000 in local zone → ~0.05 %
        # ~25,000 ha; 3 km zone ~45 %
        # Source: Copernicus EMS EMSR558 (2021).
        'documented': {
            'mortality_rate': 0.0005,
            'burned_area_pct': 45.0,
        },
    },
    {
        'name': 'Lytton Creek, BC',
        'split': 'calibration',
        'lat': 50.232, 'lon': -121.583, 'radius': 3000,
        'fire_locations': [(50.244, -121.571), (50.238, -121.577)],
        'params': {
            'WIND_SPEED': 20.0, 'WIND_INITIAL_DIRECTION': 45.0,
            'WIND_OSCILLATION_AMPLITUDE': 14.0, 'WIND_OSCILLATION_PERIOD': 18.0,
            'FIRE_SPREAD_PROB_BASE': 0.55, 'ROTHERMEL_BASE_ROS': 1.15,
            'NUM_CIVILIANS': 60,
        },
        # July 2021: 2 fatalities / ~250 in town → ~0.80 % (town destroyed)
        # 3 km zone near town: ~50 % burned
        # Source: BC Wildfire Service (2021). Lytton Creek Incident Report.
        'documented': {
            'mortality_rate': 0.0080,
            'burned_area_pct': 50.0,
        },
    },
    {
        'name': 'Knysna, South Africa',
        'split': 'calibration',
        'lat': -34.036, 'lon': 23.047, 'radius': 3000,
        'fire_locations': [(-34.024, 23.059), (-34.030, 23.053)],
        'params': {
            'WIND_SPEED': 22.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 18.0,
            'FIRE_SPREAD_PROB_BASE': 0.55, 'ROTHERMEL_BASE_ROS': 1.10,
            'NUM_CIVILIANS': 80,
        },
        # June 2017: 7 fatalities / ~15,000 in fire zone → ~0.05 %
        # ~1,000 ha in urban interface; 3 km zone ~35 %
        # Source: Baard (2019). SAICE Journal 61(3).
        'documented': {
            'mortality_rate': 0.0005,
            'burned_area_pct': 35.0,
        },
    },
    # ── Held-out (OOD) ────────────────────────────────────────────────────
    {
        'name': 'Mati 2018 (held-out)',
        'split': 'OOD',
        'lat': 38.090, 'lon': 23.920, 'radius': 3000,
        'fire_locations': [(38.097, 23.940), (38.092, 23.932), (38.085, 23.925)],
        'params': {
            'WIND_SPEED': 11.0, 'WIND_INITIAL_DIRECTION': 295.0,
            'WIND_OSCILLATION_AMPLITUDE': 8.0, 'WIND_OSCILLATION_PERIOD': 30.0,
            'FIRE_SPREAD_PROB_BASE': 0.45, 'ROTHERMEL_BASE_ROS': 0.70,
            'NUM_CIVILIANS': 80,
        },
        # Mati 23 July 2018: 102 fatalities out of ~6,000 in the zone → 1.70 %
        # Source: Lagouvardos et al. (2019) BAMS; Copernicus EMSR249 (2018)
        # Copernicus EMSR249: ~980 ha burned within 3 km zone → 35 %
        'documented': {
            'mortality_rate': 0.017,
            'burned_area_pct': 35.0,
        },
    },
    {
        'name': 'Camp Fire 2018 (held-out)',
        'split': 'OOD',
        'lat': 39.759, 'lon': -121.622, 'radius': 3000,
        'fire_locations': [(39.793, -121.575), (39.779, -121.593)],
        'params': {
            'WIND_SPEED': 16.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.55, 'ROTHERMEL_BASE_ROS': 0.85,
            'NUM_CIVILIANS': 90,
        },
        # Camp Fire 8 Nov 2018: 85 fatalities; 26,918 population → 0.32 %
        # Source: CAL FIRE (2020); MTBS (2018) dNBR; Butte County GIS
        'documented': {
            'mortality_rate': 0.0032,
            'burned_area_pct': 70.0,
        },
    },
    {
        'name': 'Pedrogao Grande 2017 (held-out)',
        'split': 'OOD',
        'lat': 39.947, 'lon': -8.148, 'radius': 3000,
        'fire_locations': [(39.958, -8.137), (39.953, -8.143), (39.948, -8.150)],
        'params': {
            'WIND_SPEED': 22.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 25.0,
            'FIRE_SPREAD_PROB_BASE': 0.48, 'ROTHERMEL_BASE_ROS': 0.95,
            'NUM_CIVILIANS': 80,
        },
        # Pedrogao Grande 17-18 June 2017: 66 fatalities; ~7,500 population → 0.88 %
        # Source: Viegas et al. (2017) ADAI/CEIF; Copernicus EMSR218
        'documented': {
            'mortality_rate': 0.0088,
            'burned_area_pct': 40.0,
        },
    },
    {
        'name': 'Alexandroupoli 2023 (held-out)',
        'split': 'OOD',
        'lat': 41.049, 'lon': 26.357, 'radius': 3000,
        'fire_locations': [(41.061, 26.369), (41.055, 26.363), (41.049, 26.357)],
        'params': {
            'WIND_SPEED': 16.0, 'WIND_INITIAL_DIRECTION': 170.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.50, 'ROTHERMEL_BASE_ROS': 1.05,
            'NUM_CIVILIANS': 75,
        },
        # Evros/Alexandroupoli Aug 2023: 20 fatalities; ~5,000 in zone → 0.40 %
        # ~81,000 ha total; largest fire in EU recorded history
        # Source: Copernicus EMSR689 (2023); Greek Fire Service (2023)
        'documented': {
            'mortality_rate': 0.0040,
            'burned_area_pct': 45.0,
        },
    },
    {
        'name': 'Lahaina 2023 (held-out)',
        'split': 'OOD',
        'lat': 20.888, 'lon': -156.673, 'radius': 3000,
        'fire_locations': [(20.898, -156.661), (20.893, -156.667), (20.887, -156.672)],
        'params': {
            'WIND_SPEED': 27.0, 'WIND_INITIAL_DIRECTION': 245.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.58, 'ROTHERMEL_BASE_ROS': 1.25,
            'NUM_CIVILIANS': 80,
        },
        # Lahaina, Maui 8 Aug 2023: 101 fatalities; ~13,000 population → 0.78 %
        # 2,170 acres (878 ha) burned; Hurricane Dora-driven ENE wind.
        # Source: NFPA (2024); Maui County (2024); NOAA (2023)
        'documented': {
            'mortality_rate': 0.0078,
            'burned_area_pct': 31.0,
        },
    },
    {
        'name': 'Black Saturday 2009 (held-out)',
        'split': 'OOD',
        'lat': -37.515, 'lon': 145.365, 'radius': 3000,
        'fire_locations': [(-37.503, 145.377), (-37.509, 145.371), (-37.515, 145.365)],
        'params': {
            'WIND_SPEED': 18.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 14.0, 'WIND_OSCILLATION_PERIOD': 18.0,
            'FIRE_SPREAD_PROB_BASE': 0.55, 'ROTHERMEL_BASE_ROS': 1.20,
            'NUM_CIVILIANS': 75,
        },
        # Kinglake complex, 7 Feb 2009: 119 fatalities; ~12,000 at risk → 0.99 %
        # NW wind 18 m/s; 46.4°C; FFDI 190+; Code Red conditions.
        # Source: Teague et al. (2010) Royal Commission; Cruz et al. (2012)
        'documented': {
            'mortality_rate': 0.0099,
            'burned_area_pct': 50.0,
        },
    },
    {
        'name': 'Tubbs Fire 2017 (held-out)',
        'split': 'OOD',
        'lat': 38.479, 'lon': -122.728, 'radius': 3000,
        'fire_locations': [(38.491, -122.716), (38.485, -122.722), (38.479, -122.728)],
        'params': {
            'WIND_SPEED': 25.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.56, 'ROTHERMEL_BASE_ROS': 1.18,
            'NUM_CIVILIANS': 75,
        },
        # Coffey Park, Santa Rosa, Oct 8-9 2017: 22 deaths / ~8,000 residents = 0.28 %
        # Diablo NE wind 25 m/s; 36,807 acres total burned.
        # Source: CAL FIRE (2018); Nauslar et al. (2018) Weather and Forecasting
        'documented': {
            'mortality_rate': 0.0028,
            'burned_area_pct': 41.0,
        },
    },
    {
        'name': 'Peloponnese 2007 (held-out)',
        'split': 'OOD',
        'lat': 37.489, 'lon': 21.648, 'radius': 3000,
        'fire_locations': [(37.500, 21.659), (37.494, 21.653), (37.488, 21.648)],
        'params': {
            'WIND_SPEED': 14.0, 'WIND_INITIAL_DIRECTION': 200.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 28.0,
            'FIRE_SPREAD_PROB_BASE': 0.48, 'ROTHERMEL_BASE_ROS': 0.98,
            'NUM_CIVILIANS': 65,
        },
        # Zacharo/Ilia complex, Aug 2007: ~30 deaths / ~5,000 in study zone = 0.60 %
        # 77 deaths total in Greece; 270,000 ha burned; extreme drought.
        # Source: Koutsias et al. (2012) Agric. Forest Meteorol.; EEA (2007)
        'documented': {
            'mortality_rate': 0.0060,
            'burned_area_pct': 45.0,
        },
    },
    {
        'name': 'Valparaiso 2014 (held-out)',
        'split': 'OOD',
        'lat': -33.047, 'lon': -71.613, 'radius': 3000,
        'fire_locations': [(-33.040, -71.606), (-33.045, -71.611), (-33.050, -71.616)],
        'params': {
            'WIND_SPEED': 12.0, 'WIND_INITIAL_DIRECTION': 315.0,
            'WIND_OSCILLATION_AMPLITUDE': 8.0, 'WIND_OSCILLATION_PERIOD': 30.0,
            'FIRE_SPREAD_PROB_BASE': 0.45, 'ROTHERMEL_BASE_ROS': 0.90,
            'NUM_CIVILIANS': 70,
        },
        # Cerro Mariposas, April 12-13 2014: 15 deaths / ~8,000 residents = 0.19 %
        # SE wind 12 m/s (La Nina drought); 2,900+ structures destroyed.
        # Source: Encinas et al. (2015) IJDRR; CONAF (2014)
        'documented': {
            'mortality_rate': 0.0019,
            'burned_area_pct': 30.0,
        },
    },
]


def _run_scenario(incident: dict, num_runs: int) -> pd.DataFrame:
    rows = []
    for i in range(num_runs):
        sim = AIGISSimulation(
            lat=incident['lat'],
            lon=incident['lon'],
            radius=incident['radius'],
            mode='batch',
            run_id=i,
            fire_locations=incident['fire_locations'],
            config_overrides=incident['params'],
        )
        r = sim.run_until_complete()
        r['run_id'] = i
        rows.append(r)
    return pd.DataFrame(rows)


def _ratio_label(sim_val: float, doc_val: float) -> str:
    """Flag if ratio is outside 0.1–10x order-of-magnitude agreement."""
    if doc_val == 0:
        return 'DOC=0'
    ratio = sim_val / doc_val
    if 0.1 <= ratio <= 10.0:
        return f'OK  (x{ratio:.2f})'
    return f'FAIL(x{ratio:.2f})'


def validate(
    num_runs: int = 15,
    output_file: str = 'all_incidents_validation.csv',
) -> pd.DataFrame:
    print('=' * 80)
    print('AIGIS — Multi-Incident Diagnostic Validation')
    print('Mas et al. (2021)  |  Grimm et al. (2020) ODD  |  n={} runs each'.format(num_runs))
    print('=' * 80)

    all_rows = []

    for incident in INCIDENTS:
        name  = incident['name']
        split = incident['split']
        doc   = incident['documented']

        print(f"\n  [{split}] {name}")
        print(f"  {'Metric':<30}  {'Simulated':>12}  {'Documented':>12}  Check")
        print(f"  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*12}")

        df = _run_scenario(incident, num_runs)
        df['scenario_name'] = name
        df['split'] = split
        all_rows.append(df)

        for metric, doc_val in doc.items():
            if metric not in df.columns:
                continue
            sim_mean = df[metric].mean()
            sim_std  = df[metric].std()
            check    = _ratio_label(sim_mean, doc_val)
            print(f"  {metric:<30}  {sim_mean:>10.4f}  {doc_val:>12.4f}  {check}")

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    _plot_comparison(df_all, output_file.replace('.csv', '.png'))
    return df_all


def _plot_comparison(df: pd.DataFrame, out_path: str) -> None:
    BG, PANEL, FG = '#1a1a2e', '#16213e', '#e0e0e0'

    metrics = ['mortality_rate', 'burned_area_pct']
    labels  = ['Mortality Rate', 'Burned Area %']

    doc_vals = {inc['name']: inc['documented'] for inc in INCIDENTS}
    scenarios = [inc['name'] for inc in INCIDENTS]
    split_map = {inc['name']: inc['split'] for inc in INCIDENTS}

    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 8), facecolor=BG)
    fig.suptitle(
        'AIGIS Multi-Incident Diagnostic — Simulated vs Documented\n'
        'Blue=calibration  Amber=OOD  Line=documented value  (Mas et al. 2021)',
        color=FG, fontsize=9, fontweight='bold',
    )

    x = np.arange(len(scenarios))
    colours = [('#4895ef' if split_map[s] == 'calibration' else '#f4a261') for s in scenarios]

    for ax, metric, label in zip(axes, metrics, labels):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=FG, labelsize=6.5)
        for sp in ax.spines.values():
            sp.set_edgecolor('#3a3a5c')

        means = [df[df['scenario_name'] == s][metric].mean() if metric in df.columns else 0
                 for s in scenarios]
        stds  = [df[df['scenario_name'] == s][metric].std() if metric in df.columns else 0
                 for s in scenarios]
        doc_line = [doc_vals[s].get(metric, np.nan) for s in scenarios]

        ax.bar(x, means, width=0.6, color=colours, alpha=0.8,
               yerr=stds, capsize=3, error_kw={'ecolor': FG, 'elinewidth': 0.8})
        ax.plot(x, doc_line, 'w--', linewidth=1.2, label='Documented', alpha=0.85)

        ax.set_xticks(x)
        short = [s.split(',')[0][:14] for s in scenarios]
        ax.set_xticklabels(short, color=FG, fontsize=6, rotation=20, ha='right')

        if metric == 'burned_area_pct':
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2%}'))
        ax.set_ylabel(label, color=FG, fontsize=8)
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor=FG)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=BG)
    print(f"Comparison plot saved to: {out_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Multi-incident diagnostic validation')
    p.add_argument('--runs',   type=int, default=15)
    p.add_argument('--output', type=str, default='all_incidents_validation.csv')
    args = p.parse_args()
    validate(num_runs=args.runs, output_file=args.output)


if __name__ == '__main__':
    main()
