"""
AmbulanceAgent — Goal-Based Architecture
Medical extraction: navigates to injured civilians, then routes to nearest hospital.

Mirrors the Rescuer pattern but with a two-leg mission:
  Leg 1: current position → injured civilian / danger zone
  Leg 2: pickup location  → nearest hospital node

Participates in Contract Net Protocol (CNP) as a bidder:
  Commander issues AMBULANCE_CFP → Ambulance bids → Commander awards → Ambulance executes

Contract Net Protocol (CNP):
  Smith, R.G. (1980).
  "The Contract Net Protocol: High-level communication and control in a
  distributed problem solver."
  IEEE Transactions on Computers, C-29(12), pp. 1104–1113.
  https://doi.org/10.1109/TC.1980.1675516
"""
import numpy as np
import networkx as nx
from typing import Tuple, Optional, List, Dict
from .base_agent import Agent
from ..message import Message
from ..config import AMBULANCE_MAX_SPEED, AMBULANCE_RISK_THRESHOLD, RESCUER_PATH_RECALC_INTERVAL


class AmbulanceAgent(Agent):
    """
    ═══════════════════════════════════════════════════════════════════════
    AGENT:        Ambulance
    ARCHITECTURE: BDI — Two-Phase Goal Stack
                  CNP Contractor role (Smith 1980)
    ───────────────────────────────────────────────────────────────────────
    BELIEFS
      • current_mission     active mission data
      • mission_status      IDLE | TO_SCENE | TO_HOSPITAL | RETURNING
      • current_node        current road-network node
      • scene_node          target injury-scene node
      • hospital_node       destination hospital node
      • current_path        planned A* route for current leg
      • safety_threshold    maximum tolerable path risk (normalised 0–1)
      • _environment_ref    environment stored in perceive() for decide()

    DESIRES
      • Extract injured civilians from danger zones
      • Transport casualties to hospital
      • Refuse missions through active fire (preserve unit)

    INTENTIONS  (explicit two-goal stack — Rao & Georgeff 1995)
      Leg 1 (TO_SCENE):    navigate to injury scene via A* shortest path
      Leg 2 (TO_HOSPITAL): navigate from scene to nearest hospital
      Bid:  Cost = path_length + risk × 50

    COMMUNICATION
      SENDS
        → PROPOSE   commander  {cost,eta,scene_node,hospital_node}
              CNP bid in response to AMBULANCE_CFP
        → REFUSE    commander  {reason}
              reject CFP (busy / no path / dangerous)
        → CONFIRM   commander  {mission_id,type:'AMBULANCE_COMPLETE'}
              mission completed successfully
      RECEIVES
        ← CFP             commander  {type:'AMBULANCE_CFP',
                                       scene_node,hospital_node}
        ← ACCEPT_PROPOSAL commander  {mission_id,scene_node,hospital_node}
        ← REJECT_PROPOSAL commander  {}
        ← INFORM          civilian   {type:'INJURY_REPORT',
                                       agent_id, node, lat, lon}
              direct dispatch: skip CFP and self-assign to injured civilian

    BIBLIOGRAPHY
      [1] Smith, R.G. (1980). "The Contract Net Protocol: High-level
          communication and control in a distributed problem solver."
          IEEE Trans. Computers, C-29(12), pp. 1104–1113.
          DOI: 10.1109/TC.1980.1675516
      [2] Rao, A.S. & Georgeff, M.P. (1995). "BDI agents: From theory to
          practice." ICMAS-95, pp. 312–319. AAAI Press.
          Two-phase goal stack is the BDI goal-adoption mechanism.
      [3] Inness, A. et al. (2019). "The CAMS reanalysis of atmospheric
          composition." Atmos. Chem. Phys., 19(6), pp. 3515–3556.
          DOI: 10.5194/acp-19-3515-2019
          Grounds INJURY_REPORT dispatch: ambulances respond to civilian
          smoke-inhalation casualties detected via the smoke_grid.
    ═══════════════════════════════════════════════════════════════════════
    """

    def __init__(self, agent_id: str, position: Tuple[float, float]):
        super().__init__(agent_id, position)

        # ===== MOVEMENT =====
        self.max_speed = AMBULANCE_MAX_SPEED
        self.current_node: Optional[int] = None
        self.current_path: List[int] = []
        self.mission_status = "IDLE"  # IDLE | TO_SCENE | TO_HOSPITAL | RETURNING

        # ===== MISSION STATE =====
        self.current_mission: Optional[Dict] = None
        self.scene_node: Optional[int] = None    # Where the casualty is
        self.hospital_node: Optional[int] = None  # Destination hospital

        # ===== SAFETY =====
        self.safety_threshold = AMBULANCE_RISK_THRESHOLD

        # ===== PERFORMANCE =====
        self.path_recalc_interval = RESCUER_PATH_RECALC_INTERVAL
        self.steps_since_recalc = 0
        self.recalc_offset = np.random.randint(0, self.path_recalc_interval)

        # Environment reference stored by perceive() for use in decide()/_recalculate_path()
        self._environment_ref = None

    # ------------------------------------------------------------------
    # Perceive → Decide → Act
    # ------------------------------------------------------------------

    def perceive(self, environment) -> None:
        """Process CNP messages from Commander."""
        self._environment_ref = environment  # Store for use in decide()/_recalculate_path()
        for message in self.messages_inbox:

            # CFP: Commander asks for bids on a medical mission
            if message.performative == "CFP" and message.content.get('type') == 'AMBULANCE_CFP':
                self._handle_cfp(message, environment)

            # ACCEPT: Commander accepted our bid
            elif message.performative == "ACCEPT_PROPOSAL":
                mission_data = message.content
                self.current_mission = mission_data
                self.scene_node = mission_data.get('scene_node')
                self.hospital_node = mission_data.get('hospital_node')
                self.mission_status = "TO_SCENE"
                self.current_path = []  # Force path recalculation

            # REJECT: Commander chose another unit
            elif message.performative == "REJECT_PROPOSAL":
                pass  # Stay IDLE, wait for next CFP

            # INJURY_REPORT: Civilian incapacitated by smoke — self-dispatch
            # Direct dispatch without CFP: civilian casualty needs immediate response.
            # Grounded in Inness et al. (2019): smoke inhalation is the primary
            # non-fire fatality driver in wildland-urban interface fires.
            elif (message.performative == "INFORM"
                  and message.content.get('type') == 'INJURY_REPORT'
                  and self.mission_status == "IDLE"):
                scene_node = message.content.get('node')
                if scene_node is None:
                    # Derive nearest node from lat/lon
                    lat = message.content.get('lat')
                    lon = message.content.get('lon')
                    if lat is not None and lon is not None:
                        try:
                            scene_node = environment.get_nearest_node(lat, lon)
                        except Exception:
                            scene_node = None
                if scene_node is not None:
                    # Pick nearest hospital, or fall back to any safe node
                    hospital_node = None
                    if environment.hospital_nodes:
                        hospital_node = environment.hospital_nodes[0]
                    elif environment.safe_nodes:
                        hospital_node = next(iter(environment.safe_nodes))
                    self.current_mission = {
                        'mission_id': f"injury_{message.content.get('agent_id')}",
                        'scene_node': scene_node,
                        'hospital_node': hospital_node,
                    }
                    self.scene_node = scene_node
                    self.hospital_node = hospital_node
                    self.mission_status = "TO_SCENE"
                    self.current_path = []
                    print(f"  [Ambulance {self.agent_id}]: responding to "
                          f"injured civilian {message.content.get('agent_id')}")

    def decide(self) -> None:
        """Goal stack: decide next movement based on current mission leg."""
        if self.mission_status == "IDLE":
            return

        # Path recalculation throttle
        self.steps_since_recalc += 1
        recalc_due = (self.steps_since_recalc >= self.path_recalc_interval + self.recalc_offset)

        if not self.current_path or recalc_due:
            self._recalculate_path()
            self.steps_since_recalc = 0

    def act(self, environment) -> None:
        """Execute movement along current path."""
        # Initialise node from position
        if self.current_node is None and environment.graph.number_of_nodes() > 0:
            self.current_node = environment.get_nearest_node(*self.position)

        if self.mission_status == "IDLE":
            return

        if not self.current_path:
            self._advance_mission_leg(environment)
            return

        # Move along path at max_speed steps
        steps = max(1, int(self.max_speed))
        for _ in range(steps):
            if not self.current_path:
                break
            next_node = self.current_path[0]

            # Safety check: abort if path goes through active fire
            if self._node_in_fire(next_node, environment):
                self._abort_mission()
                return

            self.current_node = next_node
            self.current_path.pop(0)

            # Update lat/lon from node data
            node_data = environment.graph.nodes.get(self.current_node, {})
            if 'y' in node_data and 'x' in node_data:
                self.position = (node_data['y'], node_data['x'])
                self.grid_position = environment.latlon_to_grid(*self.position)

        # Check if we reached the current leg's target
        if not self.current_path:
            self._advance_mission_leg(environment)

    # ------------------------------------------------------------------
    # CFP handling (Contract Net Protocol)
    # ------------------------------------------------------------------

    def _handle_cfp(self, message: Message, environment) -> None:
        """Evaluate a Call For Proposal and respond with a bid or refusal."""
        if self.mission_status != "IDLE":
            # Already on a mission — refuse
            self._send_refuse(message, "already_on_mission")
            return

        if self.current_node is None:
            self.current_node = environment.get_nearest_node(*self.position)

        scene_node    = message.content.get('scene_node')
        hospital_node = message.content.get('hospital_node')

        if scene_node is None:
            self._send_refuse(message, "no_scene_node")
            return

        # Compute path to scene
        try:
            path_to_scene = environment.get_shortest_path(
                self.current_node, scene_node
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self._send_refuse(message, "no_path_to_scene")
            return

        # Risk check: reject if path goes through active fire
        path_risk = self._assess_path_risk(path_to_scene, environment)
        if path_risk > self.safety_threshold:
            self._send_refuse(message, "path_too_dangerous")
            return

        # Bid = path length + risk penalty
        path_len = len(path_to_scene)
        bid_cost = path_len + path_risk * 50.0
        eta = path_len / max(self.max_speed, 1.0)

        proposal = Message(
            sender=self.agent_id,
            receiver=message.sender,
            performative="PROPOSE",
            content={
                'cost':          bid_cost,
                'eta':           eta,
                'scene_node':    scene_node,
                'hospital_node': hospital_node,
            },
            conversation_id=message.conversation_id,
        )
        self.send_message(proposal)

    def _send_refuse(self, cfp: Message, reason: str) -> None:
        refuse = Message(
            sender=self.agent_id,
            receiver=cfp.sender,
            performative="REFUSE",
            content={'reason': reason},
            conversation_id=cfp.conversation_id,
        )
        self.send_message(refuse)

    # ------------------------------------------------------------------
    # Path management
    # ------------------------------------------------------------------

    def _recalculate_path(self) -> None:
        """Recompute A* path for the current mission leg."""
        if self._environment_ref is None or self.current_node is None:
            return

        env = self._environment_ref
        target = self.scene_node if self.mission_status == "TO_SCENE" else self.hospital_node

        if target is None:
            return

        try:
            self.current_path = env.get_shortest_path(
                self.current_node, target
            )
            # Remove current node (already there)
            if self.current_path and self.current_path[0] == self.current_node:
                self.current_path.pop(0)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.current_path = []

    def _advance_mission_leg(self, environment) -> None:
        """Called when the current path is exhausted — advance to the next leg."""
        self._environment_ref = environment

        if self.mission_status == "TO_SCENE":
            # Arrived at scene — now route to hospital
            if self.hospital_node is not None:
                self.mission_status = "TO_HOSPITAL"
                self.current_path = []
                self._recalculate_path()
            else:
                # No hospital — return to base
                self._complete_mission(environment)

        elif self.mission_status == "TO_HOSPITAL":
            # Arrived at hospital — mission complete
            self._complete_mission(environment)

    def _complete_mission(self, environment) -> None:
        """Mark mission as complete and notify Commander."""
        mission_id = self.current_mission.get('mission_id') if self.current_mission else None
        if mission_id:
            confirm = Message(
                sender=self.agent_id,
                receiver="commander",
                performative="CONFIRM",
                content={'mission_id': mission_id, 'type': 'AMBULANCE_COMPLETE'},
            )
            self.send_message(confirm)

        self.mission_status = "IDLE"
        self.current_mission = None
        self.scene_node = None
        self.hospital_node = None
        self.current_path = []

    def _abort_mission(self) -> None:
        """Abort current mission (path blocked by fire)."""
        self.mission_status = "IDLE"
        self.current_mission = None
        self.current_path = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assess_path_risk(self, path: List[int], environment) -> float:
        """Return the maximum normalised fire temperature along the path."""
        if not hasattr(environment, 'temperature_grid'):
            return 0.0
        max_temp = 0.0
        for node in path:
            node_data = environment.graph.nodes.get(node, {})
            if 'y' in node_data and 'x' in node_data:
                r, c = environment.latlon_to_grid(node_data['y'], node_data['x'])
                t = float(environment.temperature_grid[r, c])
                if t > max_temp:
                    max_temp = t
        # Normalise: FIRE_TEMP_BURNING is the max (100°C default)
        from ..config import FIRE_TEMP_BURNING
        return max_temp / max(float(FIRE_TEMP_BURNING), 1.0)

    def _node_in_fire(self, node: int, environment) -> bool:
        """Return True if the node is inside an actively burning cell."""
        node_data = environment.graph.nodes.get(node, {})
        if 'y' not in node_data:
            return False
        r, c = environment.latlon_to_grid(node_data['y'], node_data['x'])
        temp = float(environment.temperature_grid[r, c])
        from ..config import FIRE_TEMP_BURNING
        return temp > self.safety_threshold * float(FIRE_TEMP_BURNING)
