"""
Observation builders — one function per RL agent role.
=======================================================
Each function extracts a normalized flat vector from the live simulation
state.  All values are in [0, 1] or [-1, 1] so the neural network receives
well-conditioned inputs (LeCun et al. 1998, "Efficient BackProp").

Observation dimensions (must match PPOAgent.OBS_DIMS):
  Firefighter : 24
  Rescuer     : 22
  Commander   : 26   (expanded from 20; +6 inter-agent dims, Yu et al. 2022)
"""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # avoid circular imports


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wind_vec(environment) -> tuple[float, float]:
    """Return (wind_dx, wind_dy) from the live fire simulation, or (0,0)."""
    fs = getattr(environment, 'fire_simulation', None)
    if fs is None:
        return 0.0, 0.0
    wv = getattr(fs, 'wind_direction', None)
    if wv is None or len(wv) < 2:
        return 0.0, 0.0
    return float(wv[0]), float(wv[1])


def _wind_speed(environment) -> float:
    fs = getattr(environment, 'fire_simulation', None)
    if fs is None:
        return 0.0
    return float(getattr(fs, 'wind_speed', 0.0)) / 30.0  # normalise by 30 m/s


def _nearest_fire(grid_pos, fire_grid):
    """
    Return (delta_row/200, delta_col/200, dist/200) to nearest burning cell.
    Returns (0,0,1) when no fire is active.
    """
    burning = np.argwhere(fire_grid == 1)
    if len(burning) == 0:
        return 0.0, 0.0, 1.0
    r, c = grid_pos
    diffs = burning - np.array([r, c])
    dists = np.linalg.norm(diffs, axis=1)
    idx   = np.argmin(dists)
    dr, dc = diffs[idx]
    d      = dists[idx]
    return float(dr / 200), float(dc / 200), float(d / 200)


def _local_fire_stats(grid_pos, fire_grid, radius: int = 5) -> tuple[float, float, float, float]:
    """
    Within a (2r+1)×(2r+1) local window: (mean_state, max_state, burning_frac, fuel_frac).
    State values: 0=clear, 1=burning, 2=burnt, 3=fuel (some grids use 3 for unburned fuel).
    """
    r, c = grid_pos
    h, w = fire_grid.shape
    r0, r1 = max(0, r - radius), min(h, r + radius + 1)
    c0, c1 = max(0, c - radius), min(w, c + radius + 1)
    patch = fire_grid[r0:r1, c0:c1].astype(float)
    n = patch.size
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(patch.mean() / 3),
        float(patch.max() / 3),
        float((patch == 1).sum() / n),
        float((patch == 3).sum() / n),
    )


def _grid_frac(fire_grid) -> tuple[float, float]:
    """(burning_pct, burnt_pct) of whole grid."""
    n = fire_grid.size
    return (
        float((fire_grid == 1).sum() / n),
        float((fire_grid == 2).sum() / n),
    )


# ---------------------------------------------------------------------------
# Per-role observation builders
# ---------------------------------------------------------------------------

def build_firefighter_obs(agent, environment, sim_step: int, max_steps: int) -> np.ndarray:
    """
    Build 24-dim observation for a FirefighterAgent.

    Index  Feature
    -----  -------
    0–3    Local fire patch stats (mean, max, burning_frac, fuel_frac) — 5-cell radius
    4–5    Own grid position (row/200, col/200)
    6–8    Nearest fire: delta_row/200, delta_col/200, dist/200
    9–10   Wind direction (dx, dy)
    11     Wind speed / 30
    12     Mean elevation slope in local patch / 45 deg (proxy from elev grid)
    13     Dominant fuel type / 10 (modal value of fuel_type_grid)
    14     Water level: current_water / water_capacity
    15     is_refilling (binary)
    16–17  Grid-wide burning_pct, burnt_pct
    18     Smoke at own cell (smoke_grid / max_smoke)
    19     Step normalised: step / max_steps
    20–23  Nearest other firefighter: delta_row/200, delta_col/200 + 2 padding zeros
    """
    obs = np.zeros(24, dtype=np.float32)
    fg  = environment.fire_grid
    gp  = agent.grid_position if agent.grid_position else (100, 100)

    # 0–3 local patch
    obs[0], obs[1], obs[2], obs[3] = _local_fire_stats(gp, fg, radius=5)
    # 4–5 own position
    obs[4] = gp[0] / 200.0
    obs[5] = gp[1] / 200.0
    # 6–8 nearest fire
    obs[6], obs[7], obs[8] = _nearest_fire(gp, fg)
    # 9–11 wind
    obs[9], obs[10] = _wind_vec(environment)
    obs[11] = _wind_speed(environment)
    # 12 mean slope proxy (gradient magnitude of elevation around agent)
    eg = getattr(environment, 'elevation_grid', None)
    if eg is not None:
        r0 = max(0, gp[0] - 3); r1 = min(eg.shape[0], gp[0] + 4)
        c0 = max(0, gp[1] - 3); c1 = min(eg.shape[1], gp[1] + 4)
        patch = eg[r0:r1, c0:c1]
        grad = np.sqrt(np.gradient(patch.astype(float))[0] ** 2 +
                       np.gradient(patch.astype(float))[1] ** 2)
        obs[12] = float(np.clip(grad.mean() / 45.0, 0, 1))
    # 13 fuel type
    ftg = getattr(environment, 'fuel_type_grid', None)
    if ftg is not None:
        r, c = gp
        obs[13] = float(ftg[r, c]) / 10.0
    # 14–15 resources
    obs[14] = agent.current_water / agent.water_capacity
    obs[15] = float(agent.is_refilling)
    # 16–17 grid fractions
    obs[16], obs[17] = _grid_frac(fg)
    # 18 smoke
    sg = getattr(environment, 'smoke_grid', None)
    if sg is not None:
        max_smoke = float(sg.max()) if sg.max() > 0 else 1.0
        obs[18] = float(sg[gp[0], gp[1]]) / max_smoke
    # 19 time
    obs[19] = float(sim_step) / float(max(max_steps, 1))
    # 20–23 nearest other firefighter (from environment agent list)
    ff_agents = getattr(environment, '_rl_firefighter_agents', [])
    others = [a for a in ff_agents if a is not agent and a.grid_position]
    if others:
        dists = [np.linalg.norm(np.array(a.grid_position) - np.array(gp)) for a in others]
        nearest = others[int(np.argmin(dists))]
        dr = (nearest.grid_position[0] - gp[0]) / 200.0
        dc = (nearest.grid_position[1] - gp[1]) / 200.0
        obs[20], obs[21] = float(dr), float(dc)

    return obs


def build_rescuer_obs(agent, environment, sim_step: int, max_steps: int,
                      civilians: list, commander_phase: int) -> np.ndarray:
    """
    Build 22-dim observation for a RescuerAgent.

    Index  Feature
    -----  -------
    0–1    Own position (row/200, col/200)
    2–5    Nearest high-panic civilian: delta_r/200, delta_c/200, panic/100, dist/200
    6      Active civilians ratio / NUM_CIVILIANS
    7–8    Nearest fire direction (delta_r/200, delta_c/200)
    9      Nearest fire distance / 200
    10–11  Wind direction (dx, dy)
    12     Wind speed / 30
    13     Fuel level / 100
    14     Mission status encoded (0=IDLE, 0.5=MOVING, 1=ARRIVED)
    15     Max temperature along current path (if available) / 100
    16     Evacuated ratio (safe / total_civilians)
    17     Burning pct
    18     Mean smoke / max_smoke
    19     Own cell temperature / 100
    20     Step normalised
    21     Commander phase / 3
    """
    obs = np.zeros(22, dtype=np.float32)
    fg  = environment.fire_grid
    gp  = agent.grid_position if agent.grid_position else (100, 100)

    # 0–1 position
    obs[0] = gp[0] / 200.0
    obs[1] = gp[1] / 200.0

    # 2–5 nearest high-panic civilian
    active_civs = [c for c in civilians if getattr(c, 'status', '') not in ('evacuated', 'casualty')]
    if active_civs:
        # Highest panic first
        active_civs.sort(key=lambda c: getattr(c, 'panic_level', 0), reverse=True)
        target = active_civs[0]
        tgp = getattr(target, 'grid_position', None) or (100, 100)
        dr  = (tgp[0] - gp[0]) / 200.0
        dc  = (tgp[1] - gp[1]) / 200.0
        d   = np.sqrt(dr**2 + dc**2)
        obs[2] = float(np.clip(dr,  -1, 1))
        obs[3] = float(np.clip(dc,  -1, 1))
        obs[4] = float(getattr(target, 'panic_level', 0)) / 100.0
        obs[5] = float(np.clip(d, 0, 1))

    # 6 active civilian ratio
    total = max(len(civilians), 1)
    obs[6] = len(active_civs) / total

    # 7–9 nearest fire
    obs[7], obs[8], obs[9] = _nearest_fire(gp, fg)

    # 10–12 wind
    obs[10], obs[11] = _wind_vec(environment)
    obs[12] = _wind_speed(environment)

    # 13 fuel
    obs[13] = float(getattr(agent, 'fuel', 100)) / 100.0

    # 14 mission status
    status_map = {'IDLE': 0.0, 'MOVING': 0.5, 'ARRIVED': 1.0, 'ABORTED': 0.0}
    obs[14] = status_map.get(getattr(agent, 'mission_status', 'IDLE'), 0.0)

    # 15 path temperature risk
    tg = getattr(environment, 'temperature_grid', None)
    if tg is not None and tg.max() > 0:
        path = getattr(agent, 'current_path', [])
        if path:
            path_temps = []
            for node in path[:10]:  # sample first 10 nodes
                nd = environment.graph.nodes.get(node, {})
                if 'y' in nd and 'x' in nd:
                    r, c = environment.latlon_to_grid(nd['y'], nd['x'])
                    path_temps.append(float(tg[r, c]))
            if path_temps:
                obs[15] = min(float(max(path_temps)) / 100.0, 1.0)

    # 16 evacuation ratio
    safe = sum(1 for c in civilians if getattr(c, 'status', '') == 'evacuated')
    obs[16] = safe / total

    # 17 burning pct
    obs[17], _ = _grid_frac(fg)

    # 18 mean smoke
    sg = getattr(environment, 'smoke_grid', None)
    if sg is not None and sg.max() > 0:
        obs[18] = float(sg.mean()) / float(sg.max())

    # 19 own temperature
    if tg is not None:
        obs[19] = float(np.clip(tg[gp[0], gp[1]] / 100.0, 0, 1))

    # 20 time
    obs[20] = float(sim_step) / float(max(max_steps, 1))

    # 21 phase
    obs[21] = float(commander_phase) / 3.0

    return obs


def build_commander_obs(agent, environment, sim_step: int, max_steps: int,
                        civilians: list) -> np.ndarray:
    """
    Build 26-dim observation for CommanderAgent.

    Dims 0-19 are the original 20-dim commander observation.
    Dims 20-25 add inter-agent state sharing recommended by:
      Yu, C. et al. (2022). "The Surprising Effectiveness of PPO in Cooperative
      Multi-Agent Games." NeurIPS 2022.  arXiv:2103.01955.
      "Sharing sub-agent state between the central critic and commander
      significantly improves coordination in CTDE architectures."

    Index  Feature
    -----  -------
    0      burning_cells_pct
    1      burnt_cells_pct
    2      wind_speed / 30
    3      wind_dir_x
    4      wind_dir_y
    5      mean elevation slope (proxy)
    6      dominant_fuel_type / 10
    7      active_rescuers / NUM_RESCUERS (from config)
    8      civilians_remaining / total
    9      current_phase / 3
    10     tti_normalised (clip to [0,1])
    11     ect_normalised (clip to [0,1])
    12     step_normalised
    13     humidity / 100 (from fire_sim if available)
    --- original coordination features ---
    14     evacuation_rate (evacuated / total)
    15     casualty_rate
    16     rescuers_idle / NUM_RESCUERS
    17     firefighters_idle / NUM_FIREFIGHTERS
    18     rescuer_refusal_rate (recent refusals / max 10)
    19     fwi_score / 100
    --- added inter-agent state sharing (Yu et al. 2022) ---
    20     mean firefighter water level (mean current_water / water_capacity)
    21     min  firefighter water level (min  current_water / water_capacity)
    22     fraction of rescuers in MOVING status (/ NUM_RESCUERS)
    23     nearest firefighter delta_row / 200 (from commander position)
    24     nearest firefighter delta_col / 200
    25     mean civilian panic level / 1.0
    """
    obs = np.zeros(26, dtype=np.float32)
    fg  = environment.fire_grid
    n   = fg.size

    burning_pct, burnt_pct = _grid_frac(fg)
    obs[0] = burning_pct
    obs[1] = burnt_pct
    obs[2] = _wind_speed(environment)
    obs[3], obs[4] = _wind_vec(environment)

    # 5 mean slope
    eg = getattr(environment, 'elevation_grid', None)
    if eg is not None:
        gy, gx = np.gradient(eg.astype(float))
        obs[5] = float(np.clip(np.sqrt(gx**2 + gy**2).mean() / 45.0, 0, 1))

    # 6 dominant fuel type
    ftg = getattr(environment, 'fuel_type_grid', None)
    if ftg is not None:
        vals, counts = np.unique(ftg, return_counts=True)
        obs[6] = float(vals[np.argmax(counts)]) / 10.0

    # 7 active rescuers ratio
    from ..config import NUM_RESCUERS, NUM_FIREFIGHTERS
    rescuer_agents = getattr(environment, '_rl_rescuer_agents', [])
    if not rescuer_agents:
        rescuer_agents = getattr(environment, '_all_rescuers', [])
    active_rsc = sum(1 for r in rescuer_agents if getattr(r, 'mission_status', 'IDLE') != 'IDLE')
    obs[7] = active_rsc / max(NUM_RESCUERS, 1)

    # 8 civilians remaining
    total_civ = max(len(civilians), 1)
    active_civ = sum(1 for c in civilians if getattr(c, 'status', '') not in ('evacuated', 'casualty'))
    obs[8] = active_civ / total_civ

    # 9 phase
    obs[9] = float(getattr(agent, 'current_phase', 0)) / 3.0

    # 10–11 TTI / ECT normalised
    tti = float(getattr(agent, 'tti', max_steps))
    ect = float(getattr(agent, 'ect', 0))
    obs[10] = float(np.clip(tti / max_steps, 0, 1))
    obs[11] = float(np.clip(ect / max_steps, 0, 1))

    # 12 time
    obs[12] = float(sim_step) / float(max(max_steps, 1))

    # 13 humidity
    fs = getattr(environment, 'fire_simulation', None)
    humidity = float(getattr(fs, 'humidity', 50)) if fs else 50.0
    obs[13] = humidity / 100.0

    # 14 evacuation rate
    evacuated = sum(1 for c in civilians if getattr(c, 'status', '') == 'evacuated')
    obs[14] = evacuated / total_civ

    # 15 casualty rate
    casualties = sum(1 for c in civilians if getattr(c, 'status', '') == 'casualty')
    obs[15] = casualties / total_civ

    # 16 rescuers idle ratio
    idle_rsc = sum(1 for r in rescuer_agents if getattr(r, 'mission_status', 'IDLE') == 'IDLE')
    obs[16] = idle_rsc / max(NUM_RESCUERS, 1)

    # 17 firefighters idle ratio
    ff_agents = getattr(environment, '_rl_firefighter_agents', [])
    if not ff_agents:
        ff_agents = getattr(environment, '_all_firefighters', [])
    idle_ff = sum(1 for f in ff_agents if getattr(f, 'mission_status', 'IDLE') == 'IDLE')
    obs[17] = idle_ff / max(NUM_FIREFIGHTERS, 1)

    # 18 rescuer refusal rate (track in environment)
    refusals = float(getattr(environment, '_rl_recent_refusals', 0))
    obs[18] = min(refusals / 10.0, 1.0)

    # 19 FWI
    obs[19] = float(np.clip(getattr(agent, 'fwi_score', 0) / 100.0, 0, 1))

    # ---- dims 20-25: inter-agent state sharing (Yu et al. 2022) -----------
    # 20-21 firefighter water levels
    ff_agents = getattr(environment, '_rl_firefighter_agents', [])
    if not ff_agents:
        ff_agents = getattr(environment, '_all_firefighters', [])
    if ff_agents:
        water_fracs = [
            float(getattr(f, 'current_water', 0)) / max(float(getattr(f, 'water_capacity', 1)), 1.0)
            for f in ff_agents
        ]
        obs[20] = float(np.mean(water_fracs))
        obs[21] = float(np.min(water_fracs))

    # 22 fraction of rescuers MOVING
    rescuer_agents = getattr(environment, '_rl_rescuer_agents', [])
    if not rescuer_agents:
        rescuer_agents = getattr(environment, '_all_rescuers', [])
    from ..config import NUM_RESCUERS
    moving_rsc = sum(1 for r in rescuer_agents if getattr(r, 'mission_status', 'IDLE') == 'MOVING')
    obs[22] = moving_rsc / max(NUM_RESCUERS, 1)

    # 23-24 nearest firefighter relative position (for spatial coordination)
    cmd_gp = getattr(agent, 'grid_position', None) or (100, 100)
    if ff_agents:
        dists = [
            np.linalg.norm(np.array(getattr(f, 'grid_position', (100, 100))) - np.array(cmd_gp))
            for f in ff_agents if getattr(f, 'grid_position', None)
        ]
        if dists:
            nearest_ff = ff_agents[int(np.argmin(dists))]
            nfp = getattr(nearest_ff, 'grid_position', (100, 100))
            obs[23] = float((nfp[0] - cmd_gp[0]) / 200.0)
            obs[24] = float((nfp[1] - cmd_gp[1]) / 200.0)

    # 25 mean civilian panic level
    if civilians:
        panics = [float(getattr(c, 'panic_level', 0)) for c in civilians
                  if getattr(c, 'status', '') not in ('evacuated', 'casualty')]
        obs[25] = float(np.mean(panics)) if panics else 0.0

    return obs


def build_global_state(obs_ff: np.ndarray, obs_rsc: np.ndarray,
                       obs_cmd: np.ndarray) -> np.ndarray:
    """
    Concatenate all three agent observations into a global state vector
    for the centralized critic (Lowe et al. 2017).
    Shape: (24 + 22 + 26,) = (72,)
    Commander obs expanded from 20 → 26 (inter-agent coordination dims,
    Yu et al. 2022).
    """
    return np.concatenate([obs_ff, obs_rsc, obs_cmd], axis=0).astype(np.float32)
