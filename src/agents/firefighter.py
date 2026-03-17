"""
Firefighter Agent - Utility-Based Architecture
Actively suppresses fires through water drops and creating fire lines

Architecture: Utility-Based Agent with Resource Management
- Assesses fire suppression opportunities
- Evaluates utility of different suppression strategies
- Manages water/retardant resources
- Coordinates with other firefighters
"""
import numpy as np
from typing import Tuple, List, Optional, Dict
from .base_agent import Agent
from ..message import Message
from .. import config as _cfg_module

# RL policy (lazy-loaded on first use)
_rl_policy = None

def _get_rl_policy():
    global _rl_policy
    if _rl_policy is None:
        import os, torch
        from ..rl.ppo import PPOAgent
        agent = PPOAgent('firefighter', global_state_dim=_cfg_module.RL_GLOBAL_STATE_DIM)
        path = os.path.join(_cfg_module.RL_POLICY_DIR, 'firefighter.pt')
        if os.path.exists(path):
            agent.load(path)
        _rl_policy = agent
    return _rl_policy


class FirefighterAgent(Agent):
    """
    ═══════════════════════════════════════════════════════════════════════
    AGENT:        Firefighter
    ARCHITECTURE: BDI — Utility-Based Intention Selection
                  CNP Contractor role (Smith 1980)
    ───────────────────────────────────────────────────────────────────────
    BELIEFS
      • target_fire         (row,col) of assigned or self-found burning cell
      • suppression_strategy  chosen action: water_drop | fire_line |
                              backburn | patrol | refill | return_to_base
      • current_water       remaining water (gallons; 0–5000)
      • is_refilling        currently at base refilling
      • current_mission     active CNP mission {mission_id, target_grid}
      • mission_status      IDLE | ASSIGNED | SUPPRESSING

    DESIRES
      • Suppress active fires to reduce spread and TTI
      • Manage water resources efficiently
      • Accept only missions that can be completed (resource check)

    INTENTIONS  (utility-based intention selection — decide())
      Utility = w_threat × Threat + w_efficiency × Efficiency
                + w_coordination × Coordination
      Best strategy selected each step; then executed in act().

      water_drop:    extinguish target cell (probabilistic 80% success)
      fire_line:     remove fuel ahead of spread direction
      backburn:      controlled pre-burn to create barrier
      patrol:        monitor when no target available
      refill:        recharge at base (10 steps)

    COMMUNICATION
      SENDS
        → PROPOSE   commander  {cost,eta,path_risk,target,target_grid}
              CNP bid in response to FIRE_SUPPRESSION_CFP
        → REFUSE    commander  {reason}
              reject CFP (refilling / no water)
        → CONFIRM   commander  {mission_id, status:'COMPLETED'}
              suppression mission completed
        → INFORM    analyst    {type:'SUPPRESSION_UPDATE', row, col}
              notify Analyst that cell (row,col) was extinguished so it
              drops that fire_report and recomputes TTI correctly
      RECEIVES
        ← CFP             commander  {type:'FIRE_SUPPRESSION_CFP',
                                       target_location, target_grid,
                                       mission_id, priority}
        ← ACCEPT_PROPOSAL commander  {mission_id, target_grid}
        ← REJECT_PROPOSAL commander  {mission_id}

    BIBLIOGRAPHY
      [1] Rothermel, R.C. (1972). A Mathematical Model for Predicting Fire
          Spread in Wildland Fuels. USDA Forest Service Research Paper INT-115.
          Fire spread direction informs fire_line placement strategy.
      [2] Anderson, H.E. (1982). Aids to Determining Fuel Models.
          USDA Forest Service GTR INT-122.
          Fuel type affects water-drop and fire-line effectiveness.
      [3] Smith, R.G. (1980). "The Contract Net Protocol."
          IEEE Trans. Computers, C-29(12), pp. 1104–1113.
          CFP → PROPOSE → ACCEPT/REJECT → CONFIRM cycle.
      [4] Rao, A.S. & Georgeff, M.P. (1995). "BDI agents: From theory to
          practice." ICMAS-95, pp. 312–319. AAAI Press.
          Utility function embedded in BDI intention-selection step.
    ═══════════════════════════════════════════════════════════════════════
    """

    def __init__(self, agent_id: str, position: Tuple[float, float]):
        """
        Initialize Firefighter agent.

        Args:
            agent_id: Unique identifier
            position: (latitude, longitude) position
        """
        super().__init__(agent_id, position)

        # === RESOURCE STATE ===
        self.water_capacity = 5000  # gallons
        self.current_water = 5000  # Start full
        self.water_per_drop = 500  # gallons per drop
        self.refill_time = 10  # steps to refill
        self.is_refilling = False
        self.refill_counter = 0

        # === SUPPRESSION EFFECTIVENESS ===
        self.drop_success_rate = 0.80  # 80% chance to extinguish
        self.fire_line_width = 2  # cells wide
        self.backburn_effectiveness = 0.90

        # === UTILITY WEIGHTS ===
        self.w_threat = 0.5  # Weight for fire threat level
        self.w_efficiency = 0.3  # Weight for water efficiency
        self.w_coordination = 0.2  # Weight for team coordination

        # === MISSION STATE ===
        self.target_fire = None  # (row, col) of target fire cell
        self.suppression_strategy = None  # 'water', 'fire_line', 'backburn'

        # === CNP MISSION TRACKING ===
        self.current_mission: Optional[dict] = None
        self.mission_status: str = "IDLE"

        # === RL INTEGRATION ===
        # Observation vector set by the MARL training loop each step.
        # None → BDI fallback is used.
        self._rl_obs: Optional[np.ndarray] = None

    def perceive(self, environment) -> None:
        """
        Perceive fire state and coordinate with other firefighters.

        Perception includes:
        - Fire locations and intensity
        - Wind direction (affects suppression strategy)
        - Other firefighter locations (avoid overlap)
        - Population at risk
        """
        # Scan for active fires
        burning_cells = np.argwhere(environment.fire_grid == 1)

        if len(burning_cells) > 0:
            claimed = getattr(environment, 'claimed_fire_cells', set())

            # Prefer unclaimed cells; fall back to any cell if all are claimed
            available = [tuple(c) for c in burning_cells if tuple(c) not in claimed]
            candidates = available if available else [tuple(c) for c in burning_cells]

            distances = [
                np.linalg.norm(np.array(self.grid_position) - np.array(cell))
                for cell in candidates
            ]
            closest_idx = np.argmin(distances)
            self.target_fire = candidates[closest_idx]

            # Claim this cell so the next firefighter skips it
            claimed.add(self.target_fire)
            environment.claimed_fire_cells = claimed

        # Check messages for coordination (Contract Net Protocol)
        for message in self.messages_inbox:
            if message.performative == "CFP":
                # Commander is requesting a firefighting mission.
                # Only respond to FIRE_SUPPRESSION_CFP; ignore other CFP types.
                if message.content.get('type') != 'FIRE_SUPPRESSION_CFP':
                    continue
                if not self.is_refilling and self.current_water >= self.water_per_drop:
                    target_grid = message.content.get('target_grid')
                    # Compute distance-based cost: closer fires are cheaper to suppress
                    if self.grid_position and target_grid:
                        dist = np.linalg.norm(
                            np.array(self.grid_position) - np.array(target_grid)
                        )
                    else:
                        dist = 0.0
                    cost = dist + (self.water_capacity - self.current_water) * 0.01
                    propose = Message(
                        sender=self.agent_id,
                        receiver=message.sender,
                        performative="PROPOSE",
                        content={
                            'cost': cost,
                            'eta': max(1, int(dist)),
                            'path_risk': 0.0,
                            'target': message.content.get('target_location'),
                            'target_grid': target_grid,
                        },
                        conversation_id=message.conversation_id
                    )
                    self.send_message(propose)
                else:
                    reason = 'refilling' if self.is_refilling else 'no_water'
                    refuse = Message(
                        sender=self.agent_id,
                        receiver=message.sender,
                        performative="REFUSE",
                        content={'reason': reason},
                        conversation_id=message.conversation_id
                    )
                    self.send_message(refuse)

            elif message.performative == "ACCEPT_PROPOSAL":
                # Commander accepted our bid — store mission and override target
                self.current_mission = {
                    'mission_id': message.content.get('mission_id'),
                    'target_grid': message.content.get('target_grid'),
                    'commander': message.sender,
                }
                target_grid = message.content.get('target_grid')
                if target_grid:
                    self.target_fire = tuple(target_grid)
                self.mission_status = "ASSIGNED"

            elif message.performative == "REJECT_PROPOSAL":
                # Commander chose another unit — stay idle
                if self.mission_status == "ASSIGNED":
                    self.mission_status = "IDLE"
                    self.current_mission = None

    # RL action index → suppression strategy string
    _RL_ACTION_MAP = {
        0: 'water_drop',
        1: 'fire_line',
        2: 'backburn',
        3: 'patrol',
        4: 'return_to_base',
    }

    def _get_bdi_valid_actions(self) -> list:
        """
        Return indices of actions that are safe under BDI safety rules.
        Masks out physically impossible or BDI-unsafe actions before PPO argmax.

        Sardina, S. & Thangarajah, J. (2011). "On the deployment of BDI agents
        in the presence of learning algorithms." Proc. 22nd IJCAI, pp. 1810-1815.
        Action masking enforces hard BDI constraints on RL-selected actions;
        the RL policy optimises over the feasible sub-space only.

        Actions: 0=water_drop, 1=fire_line, 2=backburn, 3=patrol, 4=return_to_base
        """
        all_actions = list(range(5))
        invalid: set = set()

        # No water remaining: cannot water_drop (0) or backburn (2)
        if self.current_water < self.water_per_drop:
            invalid.update([0, 2])

        # No fire target identified: cannot attack (0=water_drop, 1=fire_line, 2=backburn)
        if self.target_fire is None:
            invalid.update([0, 1, 2])

        # Currently refilling at base: can only wait to finish (only return_to_base = 4)
        if self.is_refilling:
            invalid.update([0, 1, 2, 3])

        valid = [a for a in all_actions if a not in invalid]
        return valid if valid else all_actions   # never fully block

    def decide(self) -> None:
        """
        Choose suppression strategy.
        Uses trained PPO policy (Schulman et al. 2017) with BDI action masking
        (Sardina & Thangarajah 2011) when obs is available; falls back to BDI
        utility function (Rao & Georgeff 1995) pre-training.
        """
        if self._rl_obs is not None:
            valid   = self._get_bdi_valid_actions()
            action  = _get_rl_policy().best_action_masked(self._rl_obs, valid)
            self.suppression_strategy = self._RL_ACTION_MAP.get(action, 'patrol')
            return

        # ── BDI fallback (pre-training only) ──────────────────────────
        if self.is_refilling:
            self.suppression_strategy = 'refill'
            return

        if self.current_water < self.water_per_drop:
            self.suppression_strategy = 'return_to_base'
            return

        if self.target_fire is None:
            self.suppression_strategy = 'patrol'
            return

        utilities = {
            'water_drop': self._calculate_water_drop_utility(),
            'fire_line': self._calculate_fire_line_utility(),
            'backburn': self._calculate_backburn_utility()
        }
        self.suppression_strategy = max(utilities, key=utilities.get)

    def _calculate_water_drop_utility(self) -> float:
        """Calculate utility of water drop action"""
        if self.target_fire is None or self.current_water < self.water_per_drop:
            return -float('inf')

        # Threat: Distance to fire
        distance = np.linalg.norm(np.array(self.grid_position) - np.array(self.target_fire))
        threat_score = 100 / (distance + 1)

        # Efficiency: Water remaining / drops available
        efficiency_score = (self.current_water / self.water_per_drop) * 10

        # Coordination: Check if other firefighters targeting same fire
        coordination_score = 50  # Simplified: always moderate

        utility = (self.w_threat * threat_score +
                  self.w_efficiency * efficiency_score +
                  self.w_coordination * coordination_score)

        return utility

    def _calculate_fire_line_utility(self) -> float:
        """Calculate utility of creating fire line"""
        if self.target_fire is None:
            return -float('inf')

        # Fire line is useful for predictable spread
        threat_score = 60  # Medium threat response
        efficiency_score = 70  # Doesn't use water
        coordination_score = 80  # Good for team coordination

        utility = (self.w_threat * threat_score +
                  self.w_efficiency * efficiency_score +
                  self.w_coordination * coordination_score)

        return utility

    def _calculate_backburn_utility(self) -> float:
        """Calculate utility of controlled backburn"""
        # Backburn is risky but effective when time permits
        threat_score = 40  # Lower immediate threat
        efficiency_score = 90  # Very efficient
        coordination_score = 60  # Requires careful coordination

        utility = (self.w_threat * threat_score +
                  self.w_efficiency * efficiency_score +
                  self.w_coordination * coordination_score)

        return utility

    def act(self, environment) -> None:
        """
        Execute suppression action based on chosen strategy.
        """
        # Handle refilling
        if self.is_refilling:
            self.refill_counter += 1
            if self.refill_counter >= self.refill_time:
                self.current_water = self.water_capacity
                self.is_refilling = False
                self.refill_counter = 0
                print(f"  💧 {self.agent_id}: Water refilled")
            return

        # Execute strategy
        if self.suppression_strategy == 'water_drop':
            self._execute_water_drop(environment)
        elif self.suppression_strategy == 'fire_line':
            self._execute_fire_line(environment)
        elif self.suppression_strategy == 'backburn':
            self._execute_backburn(environment)
        elif self.suppression_strategy == 'return_to_base':
            self.is_refilling = True
            print(f"  🚁 {self.agent_id}: Returning to base for refill")
        elif self.suppression_strategy == 'patrol':
            pass  # Monitoring mode

    def _execute_water_drop(self, environment) -> None:
        """Drop water on target fire"""
        if self.target_fire is None or self.current_water < self.water_per_drop:
            return

        row, col = self.target_fire

        # Check if fire still burning
        if environment.fire_grid[row, col] == 1:
            # Use water
            self.current_water -= self.water_per_drop

            # Attempt to extinguish (probabilistic)
            if np.random.random() < self.drop_success_rate:
                # Successfully extinguished
                environment.fire_grid[row, col] = 2  # Burnt out
                environment.temperature_grid[row, col] = 50.0  # Cooling
                print(f"  💦 {self.agent_id}: Extinguished fire at ({row}, {col}) "
                      f"[Water: {self.current_water}/{self.water_capacity}]")

                # ── SUPPRESSION_UPDATE → Analyst ─────────────────────────────
                # Notify Analyst so it removes this cell from active fire_reports
                # and recomputes TTI without counting an already-dead fire.
                suppression_msg = Message(
                    sender=self.agent_id,
                    receiver="analyst",
                    performative="INFORM",
                    content={
                        'type': 'SUPPRESSION_UPDATE',
                        'row': int(row),
                        'col': int(col),
                        'timestamp': environment.step_count,
                    }
                )
                self.send_message(suppression_msg)

                # ── CONFIRM → Commander (CNP mission complete) ────────────────
                if self.current_mission:
                    confirm_msg = Message(
                        sender=self.agent_id,
                        receiver=self.current_mission['commander'],
                        performative="CONFIRM",
                        content={
                            'mission_id': self.current_mission['mission_id'],
                            'status': 'COMPLETED',
                        }
                    )
                    self.send_message(confirm_msg)
                    self.current_mission = None
                    self.mission_status = "IDLE"

            else:
                # Reduced intensity but not extinguished
                print(f"  💦 {self.agent_id}: Reduced fire intensity at ({row}, {col})")

            # Clear target
            self.target_fire = None

    def _execute_fire_line(self, environment) -> None:
        """
        Create fire line by removing fuel perpendicular to the wind direction.

        Rothermel (1972) shows fire spreads fastest along the wind vector.
        An effective fire line must be placed perpendicular to that vector,
        ahead of the fire front (1-3 cells downwind of the target cell), so
        the advancing front runs into a fuel gap it cannot cross.

        Reference:
          Rothermel, R.C. (1972). A Mathematical Model for Predicting Fire
          Spread in Wildland Fuels. USDA Forest Service Research Paper INT-115.
          Wind-aligned spread direction informs optimal fire-line orientation.
        """
        if self.target_fire is None:
            return

        fire_row, fire_col = self.target_fire

        # Retrieve wind direction from the fire simulation (unit vector [dx, dy])
        wind_vec = None
        fire_sim = getattr(environment, 'fire_simulation', None)
        if fire_sim is not None:
            wind_vec = getattr(fire_sim, 'wind_direction', None)

        if wind_vec is not None and np.linalg.norm(wind_vec) > 1e-6:
            # Perpendicular to wind: rotate 90 degrees
            # wind = [dx, dy] → perp = [-dy, dx]
            wind_dx, wind_dy = wind_vec[0], wind_vec[1]
            perp_dr = int(round(-wind_dx))  # row offset perpendicular to wind
            perp_dc = int(round(wind_dy))   # col offset perpendicular to wind

            # Place line 2 cells downwind of the fire (ahead of spread direction)
            anchor_row = int(np.clip(fire_row + int(round(wind_dy * 2)),
                                     0, environment.grid_shape[0] - 1))
            anchor_col = int(np.clip(fire_col + int(round(wind_dx * 2)),
                                     0, environment.grid_shape[1] - 1))

            for offset in range(-self.fire_line_width, self.fire_line_width + 1):
                nr = int(np.clip(anchor_row + perp_dr * offset,
                                 0, environment.grid_shape[0] - 1))
                nc = int(np.clip(anchor_col + perp_dc * offset,
                                 0, environment.grid_shape[1] - 1))
                if environment.fire_grid[nr, nc] == 3:  # Fuel
                    environment.fire_grid[nr, nc] = 0   # Remove fuel
                    print(f"  [{self.agent_id}]: fire line (wind-perp) at ({nr}, {nc})")
        else:
            # Fallback: axis-aligned line when wind data unavailable
            for offset in range(-self.fire_line_width, self.fire_line_width + 1):
                new_row = int(np.clip(fire_row + offset,
                                      0, environment.grid_shape[0] - 1))
                if environment.fire_grid[new_row, fire_col] == 3:
                    environment.fire_grid[new_row, fire_col] = 0
                    print(f"  [{self.agent_id}]: fire line at ({new_row}, {fire_col})")

        self.target_fire = None

    def _execute_backburn(self, environment) -> None:
        """Controlled burn to remove fuel in fire's path"""
        # Simplified implementation
        # In reality, this is a complex and risky operation
        if self.target_fire is None:
            return

        fire_row, fire_col = self.target_fire

        # Burn fuel cells ahead of main fire
        # This removes fuel and creates a barrier
        for dr in [-3, -2, -1]:  # Burn cells ahead
            new_row = fire_row + dr
            if 0 <= new_row < environment.grid_shape[0]:
                if environment.fire_grid[new_row, fire_col] == 3:
                    # Controlled burn
                    environment.fire_grid[new_row, fire_col] = 2  # Burnt
                    print(f"  🔥 {self.agent_id}: Backburn at ({new_row}, {fire_col})")

        self.target_fire = None
