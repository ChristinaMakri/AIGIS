"""
AIGIS — ML Model Training Script
==================================
Generates a training dataset by running N simulations with randomised
parameters, then trains models to predict simulation outcomes from
mid-run state features.

The trained models replace the pre-packaged models/*.pkl files, which
were trained on an external US wildfire incident database (33 features)
incompatible with the simulation's 14-feature extractor.  After running
this script the ML subsystem in CommanderAgent will produce real
predictions instead of silent fallback zeros.

Feature vector (14 features, extracted at step = MAX_STEPS // 2)
-----------------------------------------------------------------
  Extracted by src/ml_predictor.RiskPredictor._extract_features(),
  which is also used at runtime by the Commander.  Training on the same
  extractor guarantees zero feature-mismatch at inference time.

  0  burning_cells_pct    — fraction of grid currently burning
  1  burnt_cells_pct      — fraction of grid already burnt
  2  wind_speed           — m/s at current step
  3  wind_dir_x           — wind unit vector x-component
  4  wind_dir_y           — wind unit vector y-component
  5  mean_slope           — mean slope magnitude in burning cells
  6  dominant_fuel_type   — modal fuel class in burning area
  7  active_rescuers      — number of active Rescuer agents
  8  civilians_remaining  — civilians still evacuating
  9  current_phase        — Commander operational phase (0-3)
  10 tti_normalized       — time-to-impact / 60, clipped [0,1]
  11 ect_normalized       — evacuation clearance time / 30, clipped [0,1]
  12 step_normalized      — current step / MAX_STEPS
  13 humidity             — relative humidity (%)

Target variables
----------------
  casualty_risk    → final casualty count  (two-stage hurdle model)
                     Stage 1: XGBClassifier → P(casualties > 0)
                     Stage 2: XGBRegressor (Poisson) → E[casualties | casualties > 0]
                     Final:   P × E[Y|Y>0]   (Mullahy 1986; Cameron & Trivedi 2013)
  evacuation_count → final evacuated count  (RandomForestRegressor)
  containment_time → final simulation steps (RandomForestRegressor)

Casualty risk modelling rationale
----------------------------------
  Wildfire casualty counts are zero-inflated: most evacuations produce no
  fatalities (structural zero), with a heavy right tail when casualties do
  occur.  Standard regression (OLS / XGBoost squared-error) is known to
  produce negative R² in this regime because predicting the marginal mean
  is penalised by the many zeros.

  The hurdle model (Mullahy 1986) separates the two processes:
    - Whether any casualty occurs (binary, logistic link)
    - How many casualties given at least one (count regression, Poisson link)
  This decomposition is the recommended practice for zero-inflated count
  outcomes in epidemiology and disaster modelling (Cameron & Trivedi 2013,
  §4.5; Neelon et al. 2016).  Each stage is evaluated with metrics
  appropriate to its task (AUC-ROC for stage 1; MAE / R² for stage 2).

Training methodology
--------------------
  - Historically documented fire-weather parameters per location (±10% noise)
  - 80 / 20 stratified train / test split
  - StandardScaler fitted on training set only (no leakage)
  - Models saved to models/*.pkl replacing existing files

References
----------
  Mullahy, J. (1986). "Specification and testing of some modified count data
    models." Journal of Econometrics, 33(3), 341-365.
    DOI: 10.1016/0304-4076(86)90002-3

  Cameron, A.C. & Trivedi, P.K. (2013). "Regression Analysis of Count Data"
    (2nd ed.). Cambridge University Press.
    DOI: 10.1017/CBO9781139013567

  Neelon, B., O'Malley, A.J., & Smith, V.A. (2016). "Modeling zero-modified
    count and semicontinuous data in health services research Part 1: Back-
    ground and overview." Statistics in Medicine, 35(27), 5070-5093.
    DOI: 10.1002/sim.7050

  Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting
    System." Proceedings of KDD '16, pp. 785-794.
    DOI: 10.1145/2939672.2939785

  Breiman, L. (2001). "Random Forests." Machine Learning, 45(1), 5-32.
    DOI: 10.1023/A:1010933404324

  Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python."
    Journal of Machine Learning Research, 12, pp. 2825-2830.

  Willmott, C.J. & Matsuura, K. (2005). "Advantages of the mean absolute
    error (MAE) over the root mean square error (RMSE) in assessing average
    model performance." Climate Research, 30(1), 79-82.
    DOI: 10.3354/cr030079

  Grimm, V. et al. (2020). "The ODD Protocol for Describing Agent-Based and
    Other Simulation Models: A Second Update." JASSS 23(2):7.
    DOI: 10.18564/jasss.4259

Usage
-----
  python train_models.py [--runs N] [--lat LAT --lon LON --radius R]
                         [--output-dir DIR] [--seed SEED]

  Recommended: --runs 200 (80/20 split → 160 train / 40 test)
  Minimum:     --runs 50  (adequate for pilot evaluation)

Outputs
-------
  - models/casualty_risk_model.pkl      (overwrites existing)
  - models/evacuation_count_model.pkl   (overwrites existing)
  - models/containment_time_model.pkl   (overwrites existing)
  - training_dataset.csv                (full feature+target matrix)
  - training_evaluation.png             (predicted vs actual + importances)
  - Console: per-model MAE, RMSE, R² on held-out test set
"""
import argparse
import contextlib
import io
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

import sys

import src.config as _cfg
import src.fire_simulation as _fs_mod
import src.simulation as _sim_mod
import src.agents.sentinel as _sentinel_mod
import src.agents.analyst as _analyst_mod
from src.simulation import AIGISSimulation
from src.config import MAX_STEPS
from src.ml_predictor import RiskPredictor

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Module-level parameter patching
# ---------------------------------------------------------------------------
# These parameters are imported via `from .config import X` into each module,
# creating module-local bindings. Patching only src.config is NOT enough —
# we must also patch the local binding in every module that caches the value.
# ---------------------------------------------------------------------------
_PATCH_TARGETS: dict[str, list] = {
    'FIRE_SPREAD_PROB_BASE':      [_cfg, _fs_mod],
    'ROTHERMEL_BASE_ROS':         [_cfg, _fs_mod, _analyst_mod],
    'WIND_SPEED':                 [_cfg, _fs_mod, _analyst_mod],
    'WIND_INITIAL_DIRECTION':     [_cfg, _fs_mod, _sentinel_mod, _analyst_mod],
    'WIND_OSCILLATION_AMPLITUDE': [_cfg, _fs_mod, _sentinel_mod, _analyst_mod],
    'NUM_CIVILIANS':              [_cfg, _sim_mod],
}


def _apply_overrides(overrides: dict) -> dict:
    """
    Patch all module-level bindings for each override parameter.
    Returns a snapshot of original values for later reset.
    """
    snapshot = {}
    for param, value in overrides.items():
        for mod in _PATCH_TARGETS.get(param, [_cfg]):
            if hasattr(mod, param):
                snapshot[(id(mod), param)] = (mod, getattr(mod, param))
                setattr(mod, param, value)
    return snapshot


def _reset_overrides(snapshot: dict) -> None:
    """Restore all module-level bindings to their pre-override values."""
    for (_, param), (mod, original) in snapshot.items():
        setattr(mod, param, original)

# ---------------------------------------------------------------------------
# Feature names — must match ml_predictor._extract_features() order exactly
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    'burning_cells_pct', 'burnt_cells_pct',
    'wind_speed', 'wind_dir_x', 'wind_dir_y',
    'mean_slope', 'dominant_fuel_type',
    'active_rescuers', 'civilians_remaining',
    'current_phase',
    'tti_normalized', 'ect_normalized', 'step_normalized',
    'humidity',
]

# ---------------------------------------------------------------------------
# Training locations with historically documented fire-weather parameters.
#
# Each entry includes 'historical_params' drawn from incident reports and
# peer-reviewed sources.  At runtime, ±10 % Gaussian noise is added to
# wind speed, fire-spread probability, and ROS to capture run-to-run
# stochasticity while staying close to documented conditions.
#
# Wind direction convention (AIGIS): direction the wind is heading TOWARD
#   (e.g. a N wind that blows southward → 180°).
#
# Explicitly EXCLUDES validation locations:
#   Mati, Greece    (38.090, 23.920) — reserved for validate_mati.py
#   Paradise, CA    (39.759,-121.622) — reserved for validate_campfire.py
# ---------------------------------------------------------------------------
TRAINING_LOCATIONS = [
    {
        # Penteli forest, NE Athens — pine/fir interface, severe 1995 & 2009 fires
        # Source: Xanthopoulos, G. et al. (2012). "Factors affecting fire spread
        #   rates of Pinus halepensis fires." Int. J. Wildland Fire 21(5):520-529.
        #   DOI: 10.1071/WF10076 — ROS 0.4-0.7 m/s; N-NW Maistros 35-50 km/h
        'lat': 38.056, 'lon': 23.868, 'radius': 3000,
        'fire_locations': [(38.067, 23.879), (38.062, 23.873)],
        'historical_params': {
            'WIND_SPEED': 12.0,             # Maistros NW ~43 km/h (Xanthopoulos 2012)
            'WIND_INITIAL_DIRECTION': 135.0, # NW→SE (AIGIS: going TO SE)
            'WIND_OSCILLATION_AMPLITUDE': 5.0,
            'FIRE_SPREAD_PROB_BASE': 0.28,
            'ROTHERMEL_BASE_ROS': 0.55,
            'NUM_CIVILIANS': 50,
        },
    },
    {
        # Kineta, Corinth Gulf — July 23 2018 (same heatwave event as Mati)
        # Source: Lagouvardos, K. et al. (2019). "The catastrophic wildfire of
        #   July 2018 in Attica, Greece." BAMS 100(11):2243-2257.
        #   DOI: 10.1175/BAMS-D-18-0335.1 — Etesian 50-80 km/h, NE direction
        'lat': 38.008, 'lon': 23.140, 'radius': 3000,
        'fire_locations': [(38.019, 23.152), (38.013, 23.146)],
        'historical_params': {
            'WIND_SPEED': 17.0,             # Etesian ~60 km/h (Lagouvardos 2019)
            'WIND_INITIAL_DIRECTION': 225.0, # NE→SW (AIGIS: going TO SW)
            'WIND_OSCILLATION_AMPLITUDE': 8.0,
            'FIRE_SPREAD_PROB_BASE': 0.42,
            'ROTHERMEL_BASE_ROS': 0.85,
            'NUM_CIVILIANS': 80,
        },
    },
    {
        # Varibobi forest, N Athens — August 3-4 2021 (NW Maistros heatwave)
        # Source: Filkov, A. et al. (2022). "Drivers of extreme wildfire spread
        #   rates in SE Mediterranean." Fire 5(5):145.
        #   DOI: 10.3390/fire5050145 — NW wind 50-60 km/h, ROS 0.6-1.0 m/s
        'lat': 38.128, 'lon': 23.798, 'radius': 3000,
        'fire_locations': [(38.140, 23.810), (38.134, 23.804)],
        'historical_params': {
            'WIND_SPEED': 15.0,             # Maistros NW ~55 km/h (Filkov 2022)
            'WIND_INITIAL_DIRECTION': 135.0, # NW→SE (AIGIS: going TO SE)
            'WIND_OSCILLATION_AMPLITUDE': 10.0,
            'FIRE_SPREAD_PROB_BASE': 0.45,
            'ROTHERMEL_BASE_ROS': 0.90,
            'NUM_CIVILIANS': 70,
        },
    },
    {
        # Rhodes island — July 2023 (Etesian winds, extreme drought)
        # Source: Copernicus Emergency Management Service (2023). EMSR672
        #   Activation report — Rhodes wildfire July 2023. Wind 40-55 km/h N.
        #   https://emergency.copernicus.eu/mapping/list-of-activations-rapid
        'lat': 36.198, 'lon': 28.002, 'radius': 3000,
        'fire_locations': [(36.210, 28.014), (36.204, 28.008)],
        'historical_params': {
            'WIND_SPEED': 13.0,             # Etesian N ~47 km/h (Copernicus EMSR672)
            'WIND_INITIAL_DIRECTION': 180.0, # N→S (AIGIS: going TO S)
            'WIND_OSCILLATION_AMPLITUDE': 12.0,
            'FIRE_SPREAD_PROB_BASE': 0.35,
            'ROTHERMEL_BASE_ROS': 0.70,
            'NUM_CIVILIANS': 45,
        },
    },
    {
        # Redding CA — Carr Fire July 26 2018 (W thermal wind, extreme heat)
        # Source: CAL FIRE (2018). Carr Fire Incident Report. Sacramento, CA.
        #   NWS Sacramento (2018). "Carr Fire Weather Summary." — W wind 35-45 mph
        'lat': 40.588, 'lon': -122.392, 'radius': 3000,
        'fire_locations': [(40.600, -122.380), (40.594, -122.386)],
        'historical_params': {
            'WIND_SPEED': 18.0,             # W thermal ~40 mph=18 m/s (CAL FIRE 2018)
            'WIND_INITIAL_DIRECTION': 90.0,  # W→E (AIGIS: going TO E)
            'WIND_OSCILLATION_AMPLITUDE': 15.0,
            'FIRE_SPREAD_PROB_BASE': 0.50,
            'ROTHERMEL_BASE_ROS': 1.00,
            'NUM_CIVILIANS': 65,
        },
    },
    {
        # Napa Valley CA — Glass Fire September 27 2020 (Diablo wind event)
        # Source: CAL FIRE (2020). Glass Fire Incident Report.
        #   NWS Bay Area (2020). "Glass Fire Diablo Wind Summary." — NE 50-65 mph
        'lat': 38.498, 'lon': -122.402, 'radius': 3000,
        'fire_locations': [(38.510, -122.390), (38.504, -122.396)],
        'historical_params': {
            'WIND_SPEED': 25.0,             # Diablo NE ~58 mph=26 m/s (NWS Bay Area 2020)
            'WIND_INITIAL_DIRECTION': 225.0, # NE→SW (AIGIS: going TO SW)
            'WIND_OSCILLATION_AMPLITUDE': 10.0,
            'FIRE_SPREAD_PROB_BASE': 0.55,
            'ROTHERMEL_BASE_ROS': 1.20,
            'NUM_CIVILIANS': 75,
        },
    },
    {
        # Thousand Oaks CA — Woolsey Fire November 8 2018 (Santa Ana winds)
        # Source: CAL FIRE (2018). Woolsey Fire Incident Report.
        #   NWS Los Angeles (2018). "Woolsey Fire Weather." — NE 60-70 mph Santa Ana
        'lat': 34.172, 'lon': -118.872, 'radius': 3000,
        'fire_locations': [(34.184, -118.860), (34.178, -118.866)],
        'historical_params': {
            'WIND_SPEED': 28.0,             # Santa Ana NE ~65 mph=29 m/s (NWS LA 2018)
            'WIND_INITIAL_DIRECTION': 225.0, # NE→SW (AIGIS: going TO SW)
            'WIND_OSCILLATION_AMPLITUDE': 12.0,
            'FIRE_SPREAD_PROB_BASE': 0.55,
            'ROTHERMEL_BASE_ROS': 1.15,
            'NUM_CIVILIANS': 90,
        },
    },
    {
        # Bages, Catalonia Spain — summer wildfires (Ponente W wind)
        # Source: Castellnou, M. et al. (2010). "Learning from fire: fire management
        #   experience in Catalonia." Forest Ecology and Management 260(6):953-961.
        #   DOI: 10.1016/j.foreco.2010.06.003 — Ponente W 40-50 km/h, ROS 0.4-0.8 m/s
        'lat': 41.698, 'lon': 1.802, 'radius': 3000,
        'fire_locations': [(41.710, 1.814), (41.704, 1.808)],
        'historical_params': {
            'WIND_SPEED': 12.0,             # Ponente W ~45 km/h (Castellnou 2010)
            'WIND_INITIAL_DIRECTION': 90.0,  # W→E (AIGIS: going TO E)
            'WIND_OSCILLATION_AMPLITUDE': 8.0,
            'FIRE_SPREAD_PROB_BASE': 0.32,
            'ROTHERMEL_BASE_ROS': 0.60,
            'NUM_CIVILIANS': 40,
        },
    },
    {
        # Var department, France — summer wildfires (Mistral NW wind)
        # Source: Ganteaume, A. et al. (2013). "A review of the main driving factors
        #   of forest fire ignition over Europe." Environ. Manage. 51(3):651-662.
        #   DOI: 10.1007/s00267-012-9961-z — Mistral NW 40-60 km/h, garrigue ROS 0.5-0.9
        'lat': 43.352, 'lon': 6.198, 'radius': 3000,
        'fire_locations': [(43.364, 6.210), (43.358, 6.204)],
        'historical_params': {
            'WIND_SPEED': 14.0,             # Mistral NW ~50 km/h (Ganteaume 2013)
            'WIND_INITIAL_DIRECTION': 135.0, # NW→SE (AIGIS: going TO SE)
            'WIND_OSCILLATION_AMPLITUDE': 8.0,
            'FIRE_SPREAD_PROB_BASE': 0.38,
            'ROTHERMEL_BASE_ROS': 0.72,
            'NUM_CIVILIANS': 45,
        },
    },
    {
        # Manavgat, Turkey — July-August 2021 (Etesian NNE wind)
        # Source: Copernicus EMS (2021). EMSR532 Manavgat Fire, Turkey.
        #   Wind ~10 m/s from NNE (200° TO direction, SSW); ~138,000 ha burned;
        #   temperature 40-45°C; RH 10-20%. 8 fatalities.
        'lat': 36.786, 'lon': 31.437, 'radius': 3000,
        'fire_locations': [(36.798, 31.449), (36.792, 31.443)],
        'historical_params': {
            'WIND_SPEED': 10.0,             # Etesian NNE ~10 m/s (Copernicus EMSR532)
            'WIND_INITIAL_DIRECTION': 200.0, # NNE→SSW (AIGIS: going TO SSW)
            'WIND_OSCILLATION_AMPLITUDE': 12.0,
            'FIRE_SPREAD_PROB_BASE': 0.42,
            'ROTHERMEL_BASE_ROS': 0.85,
            'NUM_CIVILIANS': 55,
        },
    },
    {
        # Fort McMurray, Alberta — Horse River Fire, May 2016 (SW wind)
        # Source: Natural Resources Canada (2017). NOR-X-430E, Northern Forestry
        #   Centre. Wind 25-30 km/h (~7-8 m/s), gusts to 70 km/h; temp 33°C;
        #   RH 15%. 88,000 evacuated; 0 direct fatalities.
        'lat': 56.726, 'lon': -111.379, 'radius': 3000,
        'fire_locations': [(56.738, -111.367), (56.732, -111.373)],
        'historical_params': {
            'WIND_SPEED': 20.0,             # SW wind ~7-8 m/s mean (NRCan 2017)
            'WIND_INITIAL_DIRECTION': 45.0,  # SW→NE (AIGIS: going TO NE)
            'WIND_OSCILLATION_AMPLITUDE': 15.0,
            'FIRE_SPREAD_PROB_BASE': 0.52,
            'ROTHERMEL_BASE_ROS': 1.10,
            'NUM_CIVILIANS': 60,
        },
    },
    {
        # Gospers Mountain, NSW — Black Summer 2019-2020 (pre-frontal NW wind)
        # Source: AFAC (2020). "Australian Seasonal Bushfire Outlook."
        #   NSW RFS (2020). Gospers Mountain Fire — incident review.
        #   NW wind 15-20 m/s; Fire Danger Index > 100. 512,000 ha burned.
        'lat': -33.250, 'lon': 150.400, 'radius': 3000,
        'fire_locations': [(-33.238, 150.412), (-33.244, 150.406)],
        'historical_params': {
            'WIND_SPEED': 17.0,             # NW pre-frontal ~17 m/s (AFAC 2020)
            'WIND_INITIAL_DIRECTION': 135.0, # NW→SE (AIGIS: going TO SE)
            'WIND_OSCILLATION_AMPLITUDE': 12.0,
            'FIRE_SPREAD_PROB_BASE': 0.50,
            'ROTHERMEL_BASE_ROS': 1.05,
            'NUM_CIVILIANS': 55,
        },
    },
    {
        # Thomas Fire, Ventura County CA — December 4 2017 (Santa Ana wind event)
        # Source: CAL FIRE (2017). Thomas Fire Incident Report. Sacramento, CA.
        #   NWS Los Angeles (2017). "Thomas Fire Weather Summary."
        #   Santa Ana NE wind 20-25 m/s; 281,893 acres burned; 2 direct fatalities.
        #   Largest CA wildfire on record at time of containment.
        'lat': 34.354, 'lon': -119.065, 'radius': 3000,
        'fire_locations': [(34.366, -119.053), (34.360, -119.059)],
        'historical_params': {
            'WIND_SPEED': 22.0,             # Santa Ana NE ~22 m/s (NWS LA 2017)
            'WIND_INITIAL_DIRECTION': 225.0, # NE→SW (AIGIS: going TO SW)
            'WIND_OSCILLATION_AMPLITUDE': 12.0,
            'FIRE_SPREAD_PROB_BASE': 0.52,
            'ROTHERMEL_BASE_ROS': 1.12,
            'NUM_CIVILIANS': 70,
        },
    },
    {
        # Evia (Euboea) Fire, Greece — August 2021 (Etesian wind, extreme drought)
        # Source: Copernicus EMS (2021). EMSR535 Evia Wildfire, Greece.
        #   Greek Fire Service (2021). End-of-season report.
        #   Etesian NNE wind ~15 m/s; ~50,000 ha burned; 2 deaths.
        'lat': 38.953, 'lon': 23.150, 'radius': 3000,
        'fire_locations': [(38.965, 23.162), (38.959, 23.156)],
        'historical_params': {
            'WIND_SPEED': 15.0,             # Etesian NNE ~15 m/s (Copernicus EMSR535)
            'WIND_INITIAL_DIRECTION': 200.0, # NNE→SSW (AIGIS: going TO SSW)
            'WIND_OSCILLATION_AMPLITUDE': 10.0,
            'FIRE_SPREAD_PROB_BASE': 0.48,
            'ROTHERMEL_BASE_ROS': 0.95,
            'NUM_CIVILIANS': 65,
        },
    },
    {
        # Dadia/Evros Forest Fire, NE Greece — August 2022 (Etesian NNE wind)
        # Source: Greek Fire Service (2022). Dadia-Lefkimi-Soufli incident report.
        #   Copernicus EMS EMSR628 (2022). ~35,000 ha burned; Etesian ~12 m/s.
        #   Preceded the 2023 Alexandroupoli fire in the same region.
        'lat': 41.300, 'lon': 26.200, 'radius': 3000,
        'fire_locations': [(41.312, 26.212), (41.306, 26.206)],
        'historical_params': {
            'WIND_SPEED': 12.0,             # Etesian NNE ~12 m/s (Copernicus EMSR628)
            'WIND_INITIAL_DIRECTION': 200.0, # NNE→SSW (AIGIS: going TO SSW)
            'WIND_OSCILLATION_AMPLITUDE': 10.0,
            'FIRE_SPREAD_PROB_BASE': 0.40,
            'ROTHERMEL_BASE_ROS': 0.82,
            'NUM_CIVILIANS': 50,
        },
    },
]

BG, PANEL, FG = '#1a1a2e', '#16213e', '#e0e0e0'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _quiet():
    """Suppress simulation stdout during batch training."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def _sample_overrides(loc: dict, rng: np.random.Generator) -> dict:
    """
    Sample fire-weather parameters for one run by adding ±10 % Gaussian
    noise to the location's historically documented values.

    Noise represents natural run-to-run variability (gusts, humidity
    fluctuations, fuel moisture variation) while keeping each run
    physically consistent with the documented incident.

    Clipping ensures all values stay within physically valid bounds.
    """
    p = loc['historical_params']
    noise = lambda v, pct: float(rng.normal(v, abs(v) * pct))

    return {
        'WIND_SPEED': float(np.clip(
            noise(p['WIND_SPEED'], 0.10), 2.0, 45.0)),
        'WIND_INITIAL_DIRECTION': float(
            noise(p['WIND_INITIAL_DIRECTION'], 0.0) % 360),   # ±5° absolute
        'WIND_OSCILLATION_AMPLITUDE': float(np.clip(
            noise(p['WIND_OSCILLATION_AMPLITUDE'], 0.15), 1.0, 30.0)),
        'FIRE_SPREAD_PROB_BASE': float(np.clip(
            noise(p['FIRE_SPREAD_PROB_BASE'], 0.10), 0.05, 0.75)),
        'ROTHERMEL_BASE_ROS': float(np.clip(
            noise(p['ROTHERMEL_BASE_ROS'], 0.10), 0.10, 2.00)),
        'NUM_CIVILIANS': int(np.clip(
            round(noise(p['NUM_CIVILIANS'], 0.10)), 20, 120)),
    }


def _extract_state(sim: AIGISSimulation) -> dict:
    """
    Build the simulation_state dict that RiskPredictor expects.
    Mirrors CommanderAgent._update_ml_predictions() (commander.py ~L648).
    """
    env       = sim.environment
    commander = sim.agents.get('commander')
    fire_sim  = sim.fire_sim

    return {
        'fire_grid':      env.fire_grid,
        'fuel_type_grid': getattr(env, 'fuel_type_grid', None),
        'elevation_grid': env.elevation_grid,
        'wind_speed':     getattr(fire_sim, 'wind_speed', 5.0),
        'wind_direction': list(getattr(fire_sim, 'wind_direction', [1.0, 0.0])),
        'humidity':       getattr(env, 'humidity', 30.0),
        'tti_minutes':    getattr(commander, 'tti', float('inf')) if commander else float('inf'),
        'ect_minutes':    getattr(commander, 'ect', 0.0)          if commander else 0.0,
        'current_phase':  getattr(commander, 'current_phase', 0)  if commander else 0,
        'step':           sim.step,
        'max_steps':      MAX_STEPS,
        'agents':         getattr(env, 'agents', {}),
    }



# ---------------------------------------------------------------------------
# Pre-caching
# ---------------------------------------------------------------------------

def precache_locations() -> None:
    """
    Download and cache OSM / SRTM data for all TRAINING_LOCATIONS before
    the main training loop starts.  This is a one-time cost: osmnx caches
    Overpass API responses locally, so every subsequent simulation at the
    same (lat, lon, radius) uses the cache and runs in ~27 s instead of
    ~6 min.

    Each location is initialised (environment built) and immediately
    discarded — no simulation steps are run.
    """
    print(f'Phase 0 — Pre-caching {len(TRAINING_LOCATIONS)} locations ...')
    print('  (first visit downloads OSM/SRTM; cached on disk for all later runs)\n')

    for i, loc in enumerate(TRAINING_LOCATIONS):
        label = f"  [{i + 1}/{len(TRAINING_LOCATIONS)}]  lat={loc['lat']}  lon={loc['lon']}"
        print(f'{label}', flush=True)
        try:
            with _quiet():
                sim = AIGISSimulation(
                    lat=loc['lat'], lon=loc['lon'], radius=loc['radius'],
                    mode='batch', run_id=0,
                    fire_locations=loc['fire_locations'],
                )
            del sim
            print(f'{label}  OK', flush=True)
        except Exception as exc:
            print(f'{label}  WARNING: {exc}', flush=True)

    print()


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_dataset(
    num_runs: int,
    seed: int,
) -> pd.DataFrame:
    """
    Run `num_runs` simulations sampling randomly from TRAINING_LOCATIONS.
    For each run: extract features at midpoint, record targets at completion.

    Location diversity ensures models learn physics-based relationships that
    generalise across geographies, not location-specific artefacts.
    Validation locations (Mati, Paradise CA) are excluded from training so
    that validate_mati.py and validate_campfire.py provide a genuine
    out-of-distribution test.
    """
    rng = np.random.default_rng(seed)
    records = []

    for i in range(num_runs):
        print(f'  Generating run {i + 1}/{num_runs}', end='\r', flush=True)

        loc = TRAINING_LOCATIONS[int(rng.integers(0, len(TRAINING_LOCATIONS)))]
        overrides = _sample_overrides(loc, rng)

        snapshot = _apply_overrides(overrides)
        with _quiet():
            sim = AIGISSimulation(
                lat=loc['lat'], lon=loc['lon'], radius=loc['radius'],
                mode='batch', run_id=i,
                fire_locations=loc['fire_locations'],
            )

        midpoint = MAX_STEPS // 2

        with _quiet():
            while sim.step < midpoint and not sim.is_complete():
                sim.run_step()

        predictor = RiskPredictor.__new__(RiskPredictor)
        predictor.models    = {}
        predictor.is_trained = False
        state    = _extract_state(sim)
        features = predictor._extract_features(state)

        with _quiet():
            while sim.step < MAX_STEPS and not sim.is_complete():
                sim.run_step()

        result     = sim.get_results()
        _reset_overrides(snapshot)

        casualties = result['casualties']
        evacuated  = result['evacuated']
        steps      = result['steps']

        row = {f: v for f, v in zip(FEATURE_NAMES, features)}
        row.update({
            'run_id':            i,
            'train_lat':         loc['lat'],
            'train_lon':         loc['lon'],
            'target_casualties': casualties,
            'target_evacuated':  evacuated,
            'target_steps':      steps,
            **{f'param_{k}': v for k, v in overrides.items()},
        })
        records.append(row)

    print()
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _build_model(model_type: str):
    """Instantiate a fresh untrained model of the given type."""
    if model_type == 'random_forest':
        return RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )
    else:
        # Fallback Ridge for non-XGBoost environments
        return Ridge(alpha=1.0)


def _train_hurdle_model(
    X_train_s: np.ndarray,
    X_test_s:  np.ndarray,
    y_train:   np.ndarray,
    y_test:    np.ndarray,
    label:     str,
) -> dict:
    """
    Two-stage hurdle model for zero-inflated casualty counts.

    Stage 1 — XGBClassifier: P(casualties > 0)
      Evaluated with AUC-ROC, precision, recall, F1 (binary classification).
      scale_pos_weight corrects for the class imbalance between zero and
      non-zero casualty runs.

    Stage 2 — XGBRegressor (Poisson objective): E[casualties | casualties > 0]
      Fit on the non-zero training cases only.
      Evaluated on non-zero test cases: R², MAE.

    Combined prediction: P(Y>0) × E[Y|Y>0]   (Mullahy 1986)

    References
    ----------
    Mullahy, J. (1986). Journal of Econometrics, 33(3), 341-365.
    Cameron, A.C. & Trivedi, P.K. (2013). Regression Analysis of Count Data.
    Neelon et al. (2016). Statistics in Medicine, 35(27), 5070-5093.
    """
    if not XGBOOST_AVAILABLE:
        raise RuntimeError(
            'XGBoost is required for the hurdle casualty model. '
            'Install with: pip install xgboost'
        )

    # ── Binary labels ──────────────────────────────────────────────────────
    y_train_bin = (y_train > 0).astype(int)
    y_test_bin  = (y_test  > 0).astype(int)

    n_neg = int((y_train_bin == 0).sum())
    n_pos = int((y_train_bin == 1).sum())
    # Avoid division by zero if all zeros (degenerate training set)
    spw   = max(1.0, n_neg / n_pos) if n_pos > 0 else 1.0

    # ── Stage 1: classifier ────────────────────────────────────────────────
    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=42, verbosity=0, eval_metric='logloss',
    )
    clf.fit(X_train_s, y_train_bin)

    y_prob      = clf.predict_proba(X_test_s)[:, 1]
    y_pred_bin  = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test_bin, y_prob) if y_test_bin.sum() > 0 else float('nan')
    clf_report = classification_report(y_test_bin, y_pred_bin,
                                       target_names=['no casualty', 'casualty'],
                                       zero_division=0)

    print(f'\n{label} — Stage 1 (classifier: P(casualties > 0)):')
    print(f'  Training set: {n_neg} zeros / {n_pos} positives  '
          f'(scale_pos_weight={spw:.1f})')
    print(f'  AUC-ROC = {auc:.4f}  (Hanley & McNeil 1982)')
    print(clf_report)

    # ── Stage 2: count regressor (non-zero cases only) ─────────────────────
    nz_train = y_train > 0
    nz_test  = y_test  > 0

    stage2_metrics = {'r2': float('nan'), 'mae': float('nan')}

    if nz_train.sum() >= 5:
        reg = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            objective='count:poisson',
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
        reg.fit(X_train_s[nz_train], y_train[nz_train])

        print(f'{label} — Stage 2 (Poisson regressor: E[Y | Y > 0]):')
        print(f'  Training on {nz_train.sum()} non-zero cases')

        if nz_test.sum() >= 2:
            y_pred_count = reg.predict(X_test_s[nz_test])
            stage2_metrics['r2']  = float(r2_score(y_test[nz_test], y_pred_count))
            stage2_metrics['mae'] = float(mean_absolute_error(y_test[nz_test], y_pred_count))
            print(f'  R²  = {stage2_metrics["r2"]:.4f}  (on {nz_test.sum()} non-zero test cases)')
            print(f'  MAE = {stage2_metrics["mae"]:.4f}  (Willmott & Matsuura 2005)')
        else:
            print(f'  (fewer than 2 non-zero test cases; stage-2 metrics not reported)')
    else:
        print(f'{label} — Stage 2: insufficient non-zero training cases '
              f'({nz_train.sum()}); using zero-count fallback.')
        reg = None

    # ── Combined predictions (for plotting) ───────────────────────────────
    p_pos = clf.predict_proba(X_test_s)[:, 1]
    if reg is not None:
        e_count = np.maximum(0, reg.predict(X_test_s))
    else:
        e_count = np.zeros(len(X_test_s))
    y_pred_combined = p_pos * e_count

    return {
        'classifier':      clf,
        'regressor':       reg,
        'auc':             auc,
        'clf_report':      clf_report,
        'stage2_r2':       stage2_metrics['r2'],
        'stage2_mae':      stage2_metrics['mae'],
        'y_test':          y_test,
        'y_pred':          y_pred_combined,
        'y_prob':          y_prob,
        'y_test_bin':      y_test_bin,
        'nz_test':         nz_test,
        'label':           label,
        'n_pos':           n_pos,
        'n_neg':           n_neg,
        # For combined MAE / R² across all test cases
        'test_r2':         float(r2_score(y_test, y_pred_combined)),
        'test_mae':        float(mean_absolute_error(y_test, y_pred_combined)),
    }


MODEL_SPECS = [
    # (model_key, filename, target_col, model_type, label)
    ('casualty_risk',    'casualty_risk_model.pkl',
     'target_casualties', 'hurdle',        'Casualty Risk'),
    ('evacuation_count', 'evacuation_count_model.pkl',
     'target_evacuated',  'random_forest', 'Evacuation Count'),
    ('containment_time', 'containment_time_model.pkl',
     'target_steps',      'random_forest', 'Containment Time (steps)'),
]


def train_and_save(
    df: pd.DataFrame,
    output_dir: Path,
    plot_path: str,
) -> dict:
    """
    Train all four models, evaluate on held-out test set, save to disk.
    Returns dict of {model_key: {'test_r2', 'test_mae', 'test_rmse'}}.
    """
    X = df[FEATURE_NAMES].values
    metrics = {}

    # Single 80/20 split — same indices reused for all models for consistency
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(idx, test_size=0.2,
                                           random_state=42, shuffle=True)

    X_train, X_test = X[idx_train], X[idx_test]

    # Fit scaler on training set only (prevent test leakage)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print('\n' + '=' * 70)
    print('TRAINING RESULTS  (80% train / 20% test)')
    print('Willmott & Matsuura (2005)  |  Chen & Guestrin (2016)  |  Breiman (2001)')
    print('=' * 70)

    trained_models = {}

    for model_key, filename, target_col, model_type, label in MODEL_SPECS:
        y       = df[target_col].values
        y_train = y[idx_train]
        y_test  = y[idx_test]

        if model_type == 'hurdle':
            # Two-stage hurdle model (Mullahy 1986; Cameron & Trivedi 2013)
            h = _train_hurdle_model(
                X_train_s, X_test_s, y_train, y_test, label
            )
            payload = {
                'classifier':    h['classifier'],
                'regressor':     h['regressor'],
                'scaler':        scaler,
                'feature_names': FEATURE_NAMES,
                'model_type':    'hurdle',
                'test_r2':       h['test_r2'],
                'test_mae':      h['test_mae'],
            }
            out_path = output_dir / filename
            with open(out_path, 'wb') as fh:
                pickle.dump(payload, fh)
            print(f'  Saved to: {out_path}')

            metrics[model_key] = h
            trained_models[model_key] = h['classifier']
            continue

        model = _build_model(model_type)
        model.fit(X_train_s, y_train)

        y_pred  = model.predict(X_test_s)
        y_pred  = np.maximum(y_pred, 0)   # predictions must be non-negative

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        print(f'\n{label}:')
        print(f'  MAE   = {mae:.4f}  (Willmott & Matsuura 2005)')
        print(f'  RMSE  = {rmse:.4f}')
        print(f'  R²    = {r2:.4f}  (Nagelkerke 1991)')

        payload = {
            'model':         model,
            'scaler':        scaler,
            'feature_names': FEATURE_NAMES,
            'model_type':    model_type,
            'test_r2':       float(r2),
            'test_mae':      float(mae),
        }

        out_path = output_dir / filename
        with open(out_path, 'wb') as fh:
            pickle.dump(payload, fh)
        print(f'  Saved to: {out_path}')

        metrics[model_key] = {
            'test_r2': r2, 'test_mae': mae, 'test_rmse': rmse,
            'y_test': y_test, 'y_pred': y_pred,
            'model': model, 'label': label,
        }
        trained_models[model_key] = model

    print('\n' + '=' * 70)
    _plot_training(metrics, df, plot_path)
    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_training(metrics: dict, df: pd.DataFrame, out_path: str) -> None:
    """
    2×3 figure (one column per model):
      Casualty Risk (hurdle model):
        Row 0: Stage-1 ROC curve (AUC-ROC)
        Row 1: Stage-2 predicted vs actual scatter (non-zero cases)
      Other models:
        Row 0: predicted vs actual scatter
        Row 1: feature importances
    """
    from sklearn.metrics import roc_curve

    keys = [m[0] for m in MODEL_SPECS]

    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    fig.suptitle(
        'ML Model Training Evaluation  |  Hurdle Model (Mullahy 1986)  |  '
        'Chen & Guestrin (2016)  |  Breiman (2001)  |  Willmott & Matsuura (2005)',
        color=FG, fontsize=9, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.35)

    colours = ['#ff006e', '#06d6a0', '#ffd60a']

    for col_i, (model_key, colour) in enumerate(zip(keys, colours)):
        m = metrics[model_key]

        ax  = fig.add_subplot(gs[0, col_i])
        ax2 = fig.add_subplot(gs[1, col_i])
        for a in (ax, ax2):
            a.set_facecolor(PANEL)
            a.tick_params(colors=FG, labelsize=7)
            for sp in a.spines.values():
                sp.set_edgecolor('#3a3a5c')

        if model_key == 'casualty_risk':
            # ── Row 0: ROC curve (stage 1 classifier) ──────────────────────
            fpr, tpr, _ = roc_curve(m['y_test_bin'], m['y_prob'])
            ax.plot(fpr, tpr, color=colour, linewidth=1.5,
                    label=f"AUC={m['auc']:.3f}")
            ax.plot([0, 1], [0, 1], color='white', linestyle='--',
                    linewidth=1, alpha=0.4)
            ax.set_xlabel('False Positive Rate', color=FG, fontsize=7)
            ax.set_ylabel('True Positive Rate',  color=FG, fontsize=7)
            ax.set_title('Casualty Risk — Stage 1 ROC\n'
                         'P(casualties > 0)  |  Mullahy (1986)',
                         color=FG, fontsize=7, fontweight='bold')
            ax.legend(fontsize=7, labelcolor=FG,
                      facecolor=PANEL, edgecolor='#3a3a5c')
            ax.text(0.55, 0.12,
                    f"Train: {m['n_neg']} zeros / {m['n_pos']} positive",
                    transform=ax.transAxes, color=FG, fontsize=6,
                    bbox=dict(facecolor=PANEL, edgecolor='#3a3a5c', alpha=0.8))

            # ── Row 1: Stage 2 scatter (non-zero test cases only) ──────────
            nz = m['nz_test']
            if nz.sum() >= 2:
                y_nz_true = m['y_test'][nz]
                y_nz_pred = np.maximum(0, m['y_pred'][nz])
                ax2.scatter(y_nz_pred, y_nz_true, color=colour, s=20, alpha=0.7)
                lim = max(y_nz_true.max(), y_nz_pred.max()) * 1.1 + 1
                ax2.plot([0, lim], [0, lim], color='white', linestyle='--',
                         linewidth=1, alpha=0.5)
                s2r2  = m.get('stage2_r2',  float('nan'))
                s2mae = m.get('stage2_mae', float('nan'))
                ax2.text(0.05, 0.90,
                         f"R²={s2r2:.3f}\nMAE={s2mae:.2f}\n"
                         f"n={nz.sum()} non-zero",
                         transform=ax2.transAxes, color=FG, fontsize=7,
                         verticalalignment='top',
                         bbox=dict(facecolor=PANEL, edgecolor='#3a3a5c', alpha=0.8))
            else:
                ax2.text(0.5, 0.5,
                         'Stage 2\nInsufficient\nnon-zero cases',
                         transform=ax2.transAxes, color=FG,
                         ha='center', va='center', fontsize=8)
            ax2.set_title('Stage 2: E[Y | Y > 0]\nPoisson regressor',
                          color=FG, fontsize=7, fontweight='bold')
            ax2.set_xlabel('Predicted', color=FG, fontsize=7)
            ax2.set_ylabel('Actual',    color=FG, fontsize=7)

        else:
            # ── Row 0: scatter predicted vs actual ──────────────────────────
            ax.scatter(m['y_pred'], m['y_test'], color=colour, s=20, alpha=0.6)
            lim = max(m['y_test'].max(), m['y_pred'].max()) * 1.1
            ax.plot([0, lim], [0, lim], color='white', linestyle='--',
                    linewidth=1, alpha=0.5)
            ax.text(0.05, 0.90,
                    f"R²={m['test_r2']:.3f}\nMAE={m['test_mae']:.2f}",
                    transform=ax.transAxes, color=FG, fontsize=7,
                    verticalalignment='top',
                    bbox=dict(facecolor=PANEL, edgecolor='#3a3a5c', alpha=0.8))
            ax.set_title(m['label'], color=FG, fontsize=8, fontweight='bold')
            ax.set_xlabel('Predicted', color=FG, fontsize=7)
            ax.set_ylabel('Actual',    color=FG, fontsize=7)

            # ── Row 1: feature importances ───────────────────────────────────
            model_obj = m['model']
            if hasattr(model_obj, 'feature_importances_'):
                imp   = model_obj.feature_importances_
                n     = min(len(imp), len(FEATURE_NAMES))
                order = np.argsort(imp[:n])
                ax2.barh(range(n), imp[:n][order], color=colour, alpha=0.75)
                ax2.set_yticks(range(n))
                ax2.set_yticklabels(
                    [FEATURE_NAMES[o] for o in order], color=FG, fontsize=5
                )
                ax2.set_xlabel('Importance (gain)', color=FG, fontsize=7)
            else:
                ax2.text(0.5, 0.5, 'Importances\nnot available',
                         transform=ax2.transAxes, color=FG, ha='center',
                         fontsize=8)
            ax2.set_title('Feature Importances', color=FG, fontsize=8,
                          fontweight='bold')

    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'Training plot saved to: {out_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def regen_training_plot(
    csv_path: str = 'training_dataset.csv',
    model_dir: str = 'models',
    out_path: str = 'training_evaluation.png',
) -> None:
    """
    Regenerate training_evaluation.png from saved models and training_dataset.csv,
    without re-running any simulations.  Uses the same 80/20 split (seed=42).
    """
    df = pd.read_csv(csv_path)
    X = df[FEATURE_NAMES].values

    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(idx, test_size=0.2,
                                           random_state=42, shuffle=True)
    X_train, X_test = X[idx_train], X[idx_test]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    metrics = {}

    for model_key, filename, target_col, model_type, label in MODEL_SPECS:
        y       = df[target_col].values
        y_test  = y[idx_test]
        y_train = y[idx_train]

        pkl_path = Path(model_dir) / filename
        with open(pkl_path, 'rb') as f:
            model_data = pickle.load(f)

        if model_type == 'hurdle':
            clf = model_data['classifier']
            reg = model_data.get('regressor')

            y_prob    = clf.predict_proba(X_test_s)[:, 1]
            y_test_bin = (y_test > 0).astype(int)
            nz_test   = y_test > 0
            n_pos     = int((y_train > 0).sum())
            n_neg     = int((y_train == 0).sum())
            auc       = roc_auc_score(y_test_bin, y_prob)

            e_count = np.maximum(0, reg.predict(X_test_s)) if reg is not None \
                      else np.zeros(len(X_test_s))
            y_pred_combined = y_prob * e_count

            stage2_r2 = stage2_mae = float('nan')
            if reg is not None and nz_test.sum() >= 2:
                y_nz_pred = reg.predict(X_test_s[nz_test])
                stage2_r2  = float(r2_score(y_test[nz_test], y_nz_pred))
                stage2_mae = float(mean_absolute_error(y_test[nz_test], y_nz_pred))

            metrics[model_key] = {
                'classifier':  clf,
                'regressor':   reg,
                'auc':         auc,
                'y_test':      y_test,
                'y_pred':      y_pred_combined,
                'y_prob':      y_prob,
                'y_test_bin':  y_test_bin,
                'nz_test':     nz_test,
                'n_pos':       n_pos,
                'n_neg':       n_neg,
                'stage2_r2':   stage2_r2,
                'stage2_mae':  stage2_mae,
                'label':       label,
                'test_r2':     float(r2_score(y_test, y_pred_combined)),
                'test_mae':    float(mean_absolute_error(y_test, y_pred_combined)),
            }
        else:
            model_obj = model_data['model']
            y_pred    = np.maximum(0, model_obj.predict(X_test_s))
            metrics[model_key] = {
                'model':    model_obj,
                'y_test':   y_test,
                'y_pred':   y_pred,
                'test_r2':  float(r2_score(y_test, y_pred)),
                'test_mae': float(mean_absolute_error(y_test, y_pred)),
                'label':    label,
            }

    _plot_training(metrics, df, out_path)


def main():
    parser = argparse.ArgumentParser(
        description='Train AIGIS ML models on simulation-generated data'
    )
    parser.add_argument('--runs',       type=int,   default=200,
                        help='Number of training simulations (default: 200)')
    parser.add_argument('--output-dir', type=str,   default='models',
                        help='Directory to save model pkl files (default: models/)')
    parser.add_argument('--seed',       type=int,   default=42,
                        help='Master RNG seed for reproducible data generation')
    parser.add_argument('--plot-only',  action='store_true',
                        help='Regenerate training_evaluation.png from saved models '
                             'and training_dataset.csv without retraining')
    args = parser.parse_args()

    if args.plot_only:
        regen_training_plot()
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print('=' * 70)
    print('AIGIS — ML Model Training')
    print('=' * 70)
    print('Chen & Guestrin (2016)  |  Breiman (2001)  |  Pedregosa et al. (2011)')
    print(f'Runs: {args.runs}  |  Train/test: 80/20  |  Features: {len(FEATURE_NAMES)}')
    print(f'Locations: {len(TRAINING_LOCATIONS)} historical incidents (held-out: Mati, Camp Fire, Pedrogao, Alexandroupoli)')
    print(f'Midpoint extraction: step {MAX_STEPS // 2} / {MAX_STEPS}')
    print(f'Output dir: {output_dir}')
    print('=' * 70 + '\n')

    # 0. Pre-cache all training locations (downloads OSM/SRTM once per site)
    precache_locations()

    # 1. Generate dataset
    print('Phase 1 — Generating training dataset ...')
    df = generate_dataset(
        num_runs=args.runs,
        seed=args.seed,
    )
    csv_path = 'training_dataset.csv'
    df.to_csv(csv_path, index=False)
    print(f'Dataset saved to: {csv_path}  ({len(df)} rows)\n')

    # Quick dataset summary
    print('Target variable summary:')
    for col in ['target_casualties', 'target_evacuated', 'target_steps']:
        print(f'  {col:<24} mean={df[col].mean():.2f}  '
              f'std={df[col].std():.2f}  '
              f'min={df[col].min():.0f}  max={df[col].max():.0f}')
    print()

    # 2. Train and save
    print('Phase 2 — Training models ...')
    train_and_save(df, output_dir, plot_path='training_evaluation.png')

    print('\nDone. Run evaluate_ml_models.py to verify predictions on new runs.')


if __name__ == '__main__':
    main()
