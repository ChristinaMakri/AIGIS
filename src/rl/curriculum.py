"""
Training curriculum across 23 historical fire scenarios.
=========================================================
Implements difficulty-ordered curriculum learning:
  Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009).
  "Curriculum Learning." ICML-09, pp. 41–48.
  [Start with easy scenarios; add harder ones as training progresses.]

Scenarios are ordered by fire intensity (ROTHERMEL_BASE_ROS × WIND_SPEED):
  Phase 1 (easy,   5 scenarios): Bages, Var, Penteli, Corsica, Tuscany
  Phase 2 (medium, 8 scenarios): Manavgat, Rhodes, Kineta, Varibobi, Dadia,
                                 Carmel, Dwellingup, Monchique
  Phase 3 (hard,  10 scenarios): Fort McMurray, Gospers Mtn, Carr, Glass,
                                 Woolsey, Thomas, Evia, Oristano,
                                 Lytton, Knysna

Held out (never used in training — reserved for validation only):
  Mati 2018, Camp Fire 2018, Pedrogao Grande 2017, Alexandroupoli 2023,
  Lahaina 2023, Black Saturday 2009, Tubbs 2017, Peloponnese 2007,
  Valparaiso 2014
"""
from __future__ import annotations
from typing import Optional
import numpy as np

# ---------------------------------------------------------------------------
# Scenario definitions (mirrors TRAINING_LOCATIONS in train_models.py)
# ---------------------------------------------------------------------------

SCENARIOS = [
    # ── Phase 1: Easy ────────────────────────────────────────────────────
    {
        'name': 'Bages, Catalonia',
        'phase': 1,
        'lat': 41.698, 'lon': 1.802, 'radius': 3000,
        'fire_locations': [(41.710, 1.814), (41.704, 1.808)],
        'params': {
            'WIND_SPEED': 12.0, 'WIND_INITIAL_DIRECTION': 90.0,
            'WIND_OSCILLATION_AMPLITUDE': 8.0,  'WIND_OSCILLATION_PERIOD': 35.0,
            'FIRE_SPREAD_PROB_BASE': 0.32, 'ROTHERMEL_BASE_ROS': 0.60,
            'NUM_CIVILIANS': 40,
        },
    },
    {
        'name': 'Var, France',
        'phase': 1,
        'lat': 43.352, 'lon': 6.198, 'radius': 3000,
        'fire_locations': [(43.364, 6.210), (43.358, 6.204)],
        'params': {
            'WIND_SPEED': 14.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 8.0,  'WIND_OSCILLATION_PERIOD': 30.0,
            'FIRE_SPREAD_PROB_BASE': 0.38, 'ROTHERMEL_BASE_ROS': 0.72,
            'NUM_CIVILIANS': 45,
        },
    },
    {
        'name': 'Penteli, Athens',
        'phase': 1,
        'lat': 38.056, 'lon': 23.868, 'radius': 3000,
        'fire_locations': [(38.067, 23.879), (38.062, 23.873)],
        'params': {
            'WIND_SPEED': 12.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 5.0,  'WIND_OSCILLATION_PERIOD': 40.0,
            'FIRE_SPREAD_PROB_BASE': 0.28, 'ROTHERMEL_BASE_ROS': 0.55,
            'NUM_CIVILIANS': 50,
        },
    },
    {
        'name': 'Corte, Corsica',
        'phase': 1,
        'lat': 42.302, 'lon': 9.148, 'radius': 3000,
        'fire_locations': [(42.314, 9.160), (42.308, 9.154)],
        # Libeccio NW wind ~36 km/h; maquis shrubland.
        # Ref: Meddour-Sahar et al. (2013). iForest 6:366-374.
        'params': {
            'WIND_SPEED': 10.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 6.0,  'WIND_OSCILLATION_PERIOD': 38.0,
            'FIRE_SPREAD_PROB_BASE': 0.28, 'ROTHERMEL_BASE_ROS': 0.52,
            'NUM_CIVILIANS': 35,
        },
    },
    {
        'name': 'Pisan Hills, Tuscany',
        'phase': 1,
        'lat': 43.720, 'lon': 10.458, 'radius': 3000,
        'fire_locations': [(43.732, 10.470), (43.726, 10.464)],
        # Libeccio SW wind ~32 km/h; Mediterranean mixed macchia.
        # Ref: Elia et al. (2015). iForest 8:31-38.
        'params': {
            'WIND_SPEED': 9.0,  'WIND_INITIAL_DIRECTION': 45.0,
            'WIND_OSCILLATION_AMPLITUDE': 5.0,  'WIND_OSCILLATION_PERIOD': 40.0,
            'FIRE_SPREAD_PROB_BASE': 0.25, 'ROTHERMEL_BASE_ROS': 0.50,
            'NUM_CIVILIANS': 40,
        },
    },
    # ── Phase 2: Medium ──────────────────────────────────────────────────
    {
        'name': 'Manavgat, Turkey',
        'phase': 2,
        'lat': 36.786, 'lon': 31.437, 'radius': 3000,
        'fire_locations': [(36.798, 31.449), (36.792, 31.443)],
        # Copernicus EMSR532 (2021): Etesian NNE wind from 20°→ TO 200° (SSW).
        # Wind speed: ~10 m/s; temperature: 40–45 °C; RH: 10–20 %.
        # Reference: Copernicus EMS (2021). EMSR532 Manavgat Fire, Turkey.
        #   https://emergency.copernicus.eu/mapping/list-of-activations-rapid
        # ~138,000 ha burned July–August 2021; 8 fatalities.
        'params': {
            'WIND_SPEED': 10.0, 'WIND_INITIAL_DIRECTION': 200.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 25.0,
            'FIRE_SPREAD_PROB_BASE': 0.42, 'ROTHERMEL_BASE_ROS': 0.85,
            'NUM_CIVILIANS': 55,
        },
    },
    {
        'name': 'Rhodes, Greece',
        'phase': 2,
        'lat': 36.198, 'lon': 28.002, 'radius': 3000,
        'fire_locations': [(36.210, 28.014), (36.204, 28.008)],
        'params': {
            'WIND_SPEED': 13.0, 'WIND_INITIAL_DIRECTION': 180.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 30.0,
            'FIRE_SPREAD_PROB_BASE': 0.35, 'ROTHERMEL_BASE_ROS': 0.70,
            'NUM_CIVILIANS': 45,
        },
    },
    {
        'name': 'Kineta, Corinth',
        'phase': 2,
        'lat': 38.008, 'lon': 23.140, 'radius': 3000,
        'fire_locations': [(38.019, 23.152), (38.013, 23.146)],
        'params': {
            'WIND_SPEED': 17.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 8.0,  'WIND_OSCILLATION_PERIOD': 25.0,
            'FIRE_SPREAD_PROB_BASE': 0.42, 'ROTHERMEL_BASE_ROS': 0.85,
            'NUM_CIVILIANS': 80,
        },
    },
    {
        'name': 'Varibobi, Athens',
        'phase': 2,
        'lat': 38.128, 'lon': 23.798, 'radius': 3000,
        'fire_locations': [(38.140, 23.810), (38.134, 23.804)],
        'params': {
            'WIND_SPEED': 15.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 25.0,
            'FIRE_SPREAD_PROB_BASE': 0.45, 'ROTHERMEL_BASE_ROS': 0.90,
            'NUM_CIVILIANS': 70,
        },
    },
    {
        'name': 'Dadia, Evros',
        'phase': 2,
        'lat': 41.300, 'lon': 26.200, 'radius': 3000,
        'fire_locations': [(41.312, 26.212), (41.306, 26.206)],
        # Dadia-Lefkimi-Soufli fire, August 2022: Etesian NNE wind ~12 m/s.
        # ~35,000 ha burned. Reference: Copernicus EMS EMSR628 (2022).
        'params': {
            'WIND_SPEED': 12.0, 'WIND_INITIAL_DIRECTION': 200.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 28.0,
            'FIRE_SPREAD_PROB_BASE': 0.40, 'ROTHERMEL_BASE_ROS': 0.82,
            'NUM_CIVILIANS': 50,
        },
    },
    {
        'name': 'Carmel, Israel',
        'phase': 2,
        'lat': 32.698, 'lon': 35.018, 'radius': 3000,
        'fire_locations': [(32.710, 35.030), (32.704, 35.024)],
        # December 2010: SW wind ~11 m/s; 44 fatalities; ~5,000 ha.
        # Ref: Cohen et al. (2014). Fire Ecology 10(1).
        'params': {
            'WIND_SPEED': 11.0, 'WIND_INITIAL_DIRECTION': 45.0,
            'WIND_OSCILLATION_AMPLITUDE': 8.0,  'WIND_OSCILLATION_PERIOD': 28.0,
            'FIRE_SPREAD_PROB_BASE': 0.36, 'ROTHERMEL_BASE_ROS': 0.78,
            'NUM_CIVILIANS': 60,
        },
    },
    {
        'name': 'Dwellingup, W. Australia',
        'phase': 2,
        'lat': -32.714, 'lon': 116.063, 'radius': 3000,
        'fire_locations': [(-32.702, 116.075), (-32.708, 116.069)],
        # Fremantle Doctor sea breeze inversion; NE→SW ~13 m/s; jarrah forest.
        # Ref: Burrows et al. (1991). CALM Science Paper No. 6.
        'params': {
            'WIND_SPEED': 13.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 9.0,  'WIND_OSCILLATION_PERIOD': 25.0,
            'FIRE_SPREAD_PROB_BASE': 0.36, 'ROTHERMEL_BASE_ROS': 0.75,
            'NUM_CIVILIANS': 50,
        },
    },
    {
        'name': 'Monchique, Portugal',
        'phase': 2,
        'lat': 37.322, 'lon': -8.553, 'radius': 3000,
        'fire_locations': [(37.334, -8.541), (37.328, -8.547)],
        # August 2018: NE wind ~13 m/s; ~27,000 ha burned; 2 fatalities.
        # (Distinct from held-out Pedrogao 2017.)
        # Ref: Copernicus EMS EMSR319 (2018); ICNF Portugal (2018).
        'params': {
            'WIND_SPEED': 13.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 26.0,
            'FIRE_SPREAD_PROB_BASE': 0.40, 'ROTHERMEL_BASE_ROS': 0.82,
            'NUM_CIVILIANS': 55,
        },
    },
    # ── Phase 3: Hard ────────────────────────────────────────────────────
    {
        'name': 'Fort McMurray, Alberta',
        'phase': 3,
        'lat': 56.726, 'lon': -111.379, 'radius': 3000,
        'fire_locations': [(56.738, -111.367), (56.732, -111.373)],
        # Horse River Fire, May 2016: SW wind pushing fire NE.
        # SW wind = FROM 225° → TO 45° (NE). AIGIS TO convention.
        # Documented: wind 25–30 km/h (~7–8 m/s), gusts to 70 km/h;
        # temperature 33 °C; RH 15 %.  0 direct fatalities; 88,000 evacuated.
        # Reference: Natural Resources Canada (2017). "The Fort McMurray
        #   Horse River Wildfire: Alberta Fire Review."  Information Report
        #   NOR-X-430E, Northern Forestry Centre, Edmonton, AB.
        'params': {
            'WIND_SPEED': 20.0, 'WIND_INITIAL_DIRECTION': 45.0,
            'WIND_OSCILLATION_AMPLITUDE': 15.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.52, 'ROTHERMEL_BASE_ROS': 1.10,
            'NUM_CIVILIANS': 60,
        },
    },
    {
        'name': 'Gospers Mountain, NSW',
        'phase': 3,
        'lat': -33.250, 'lon': 150.400, 'radius': 3000,
        'fire_locations': [(-33.238, 150.412), (-33.244, 150.406)],
        # Black Summer 2019–2020: pre-frontal NW wind driving fire SE.
        # NW wind = FROM 315° → TO 135° (SE). AIGIS TO convention.
        # Documented: NW wind 15–20 m/s; Fire Danger Index > 100 (AFAC 2020).
        # 512,000 ha burned (Gospers Mountain sub-event); 0 direct fatalities.
        # Reference: AFAC (2020). "Australian Seasonal Bushfire Outlook."
        #   Australasian Fire and Emergency Service Authorities Council.
        #   See also: NSW RFS (2020). Gospers Mountain Fire — incident review.
        'params': {
            'WIND_SPEED': 17.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 18.0,
            'FIRE_SPREAD_PROB_BASE': 0.50, 'ROTHERMEL_BASE_ROS': 1.05,
            'NUM_CIVILIANS': 55,
        },
    },
    {
        'name': 'Carr Fire, Redding CA',
        'phase': 3,
        'lat': 40.588, 'lon': -122.392, 'radius': 3000,
        'fire_locations': [(40.600, -122.380), (40.594, -122.386)],
        'params': {
            'WIND_SPEED': 18.0, 'WIND_INITIAL_DIRECTION': 90.0,
            'WIND_OSCILLATION_AMPLITUDE': 15.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.50, 'ROTHERMEL_BASE_ROS': 1.00,
            'NUM_CIVILIANS': 65,
        },
    },
    {
        'name': 'Glass Fire, Napa CA',
        'phase': 3,
        'lat': 38.498, 'lon': -122.402, 'radius': 3000,
        'fire_locations': [(38.510, -122.390), (38.504, -122.396)],
        'params': {
            'WIND_SPEED': 25.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.57, 'ROTHERMEL_BASE_ROS': 1.20,
            'NUM_CIVILIANS': 75,
        },
    },
    {
        'name': 'Woolsey Fire, Thousand Oaks CA',
        'phase': 3,
        'lat': 34.172, 'lon': -118.872, 'radius': 3000,
        'fire_locations': [(34.184, -118.860), (34.178, -118.866)],
        'params': {
            'WIND_SPEED': 28.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.58, 'ROTHERMEL_BASE_ROS': 1.15,
            'NUM_CIVILIANS': 90,
        },
    },
    {
        'name': 'Thomas Fire, Ventura CA',
        'phase': 3,
        'lat': 34.354, 'lon': -119.065, 'radius': 3000,
        'fire_locations': [(34.366, -119.053), (34.360, -119.059)],
        # December 4 2017: Santa Ana NE wind 20-25 m/s; 281,893 acres burned.
        # Reference: CAL FIRE (2017). Thomas Fire Incident Report.
        'params': {
            'WIND_SPEED': 22.0, 'WIND_INITIAL_DIRECTION': 225.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.52, 'ROTHERMEL_BASE_ROS': 1.12,
            'NUM_CIVILIANS': 70,
        },
    },
    {
        'name': 'Evia Fire, Greece',
        'phase': 3,
        'lat': 38.953, 'lon': 23.150, 'radius': 3000,
        'fire_locations': [(38.965, 23.162), (38.959, 23.156)],
        # August 2021: Etesian NNE wind ~15 m/s; ~50,000 ha; 2 deaths.
        # Reference: Copernicus EMS EMSR535 (2021); Greek Fire Service (2021).
        'params': {
            'WIND_SPEED': 15.0, 'WIND_INITIAL_DIRECTION': 200.0,
            'WIND_OSCILLATION_AMPLITUDE': 10.0, 'WIND_OSCILLATION_PERIOD': 22.0,
            'FIRE_SPREAD_PROB_BASE': 0.48, 'ROTHERMEL_BASE_ROS': 0.95,
            'NUM_CIVILIANS': 65,
        },
    },
    {
        'name': 'Oristano, Sardinia',
        'phase': 3,
        'lat': 40.081, 'lon': 8.595, 'radius': 3000,
        'fire_locations': [(40.093, 8.607), (40.087, 8.601)],
        # July 2021: Maestrale NW wind ~20 m/s; ~25,000 ha; 1 fatality; RH < 15%.
        # Ref: Copernicus EMS EMSR558 (2021). Oristano-Cuglieri fire, Sardinia.
        'params': {
            'WIND_SPEED': 20.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 20.0,
            'FIRE_SPREAD_PROB_BASE': 0.52, 'ROTHERMEL_BASE_ROS': 1.05,
            'NUM_CIVILIANS': 55,
        },
    },
    {
        'name': 'Lytton Creek, BC',
        'phase': 3,
        'lat': 50.232, 'lon': -121.583, 'radius': 3000,
        'fire_locations': [(50.244, -121.571), (50.238, -121.577)],
        # July 2021: post-heat-dome (49.6°C record); SW wind ~20 m/s; RH < 10%;
        # ~83,000 ha burned; 2 fatalities (town of Lytton destroyed).
        # Ref: BC Wildfire Service (2021). Lytton Creek Wildfire Incident Report.
        'params': {
            'WIND_SPEED': 20.0, 'WIND_INITIAL_DIRECTION': 45.0,
            'WIND_OSCILLATION_AMPLITUDE': 14.0, 'WIND_OSCILLATION_PERIOD': 18.0,
            'FIRE_SPREAD_PROB_BASE': 0.55, 'ROTHERMEL_BASE_ROS': 1.15,
            'NUM_CIVILIANS': 60,
        },
    },
    {
        'name': 'Knysna, South Africa',
        'phase': 3,
        'lat': -34.036, 'lon': 23.047, 'radius': 3000,
        'fire_locations': [(-34.024, 23.059), (-34.030, 23.053)],
        # June 2017: NW Berg wind ~22 m/s; 7 fatalities; ~1,000 homes destroyed.
        # Ref: Baard (2019). SAICE Journal 61(3); Forsyth et al. (2010) SAEON.
        'params': {
            'WIND_SPEED': 22.0, 'WIND_INITIAL_DIRECTION': 135.0,
            'WIND_OSCILLATION_AMPLITUDE': 12.0, 'WIND_OSCILLATION_PERIOD': 18.0,
            'FIRE_SPREAD_PROB_BASE': 0.55, 'ROTHERMEL_BASE_ROS': 1.10,
            'NUM_CIVILIANS': 80,
        },
    },
]


class ScenarioCurriculum:
    """
    Manages scenario sampling according to curriculum phase.

    Phase 1 (episodes 0   → phase1_end):   easy only
    Phase 2 (episodes ph1 → phase2_end):   easy + medium
    Phase 3 (episodes ph2 → end):          all 23 scenarios

    Bengio et al. (2009) — start simple, add complexity gradually.
    """

    def __init__(
        self,
        phase1_end: int = 2000,
        phase2_end: int = 6000,
        rng_seed: Optional[int] = None,
    ):
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end
        self.rng = np.random.default_rng(rng_seed)
        self._episode = 0

    @property
    def current_curriculum_phase(self) -> int:
        if self._episode < self.phase1_end:
            return 1
        if self._episode < self.phase2_end:
            return 2
        return 3

    def sample(self) -> dict:
        """Return a scenario dict appropriate for the current episode count."""
        max_phase = self.current_curriculum_phase
        pool = [s for s in SCENARIOS if s['phase'] <= max_phase]
        idx  = int(self.rng.integers(0, len(pool)))
        self._episode += 1
        return pool[idx]

    def advance(self, n: int = 1) -> None:
        self._episode += n
