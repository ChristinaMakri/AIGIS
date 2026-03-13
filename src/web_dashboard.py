"""
AIGIS Web Dashboard — real-time browser-based simulation monitor.

Architecture
------------
  Simulation thread  →  _state_queue  →  Flask SSE endpoint  →  Browser

The WebDashboard class is the same update(sim) / finalize(sim) interface as
the matplotlib Dashboard, so main.py can swap in either with one flag.

The Flask dev server runs on a background daemon thread.  The simulation
loop runs on the main thread and pushes a compact JSON snapshot every
DASHBOARD_UPDATE_INTERVAL steps via a thread-safe queue.

Server-Sent Events (SSE) are used for the push channel — no WebSocket
library dependency required.  The client reconnects automatically if the
server restarts.

Usage
-----
    dash = WebDashboard(port=5000)
    dash.start_server()            # opens http://localhost:5000 in browser
    while not sim.is_complete():
        sim.run_step()
        if sim.step % DASHBOARD_UPDATE_INTERVAL == 0:
            dash.update(sim)
    dash.finalize(sim)
    dash.stop()
"""
import json
import queue
import threading
import time
import webbrowser
from typing import TYPE_CHECKING, Optional

import numpy as np
from flask import Flask, Response, render_template, stream_with_context

from .config import (
    DASHBOARD_UPDATE_INTERVAL,
    CIVILIAN_PANIC_RATIONAL,
    CIVILIAN_PANIC_CONFUSED,
)

if TYPE_CHECKING:
    from .simulation import AIGISSimulation

_PHASE_NAMES = {0: "Monitor", 1: "Pre-Alert", 2: "Evacuate", 3: "Shelter-in-Place"}
_PHASE_COLS  = {0: "#3a86ff", 1: "#ffbe0b", 2: "#ff006e", 3: "#8338ec"}

# Maximum simultaneous SSE subscribers (typically 1 browser tab)
_MAX_SUBSCRIBERS = 8
# Grid downsampling factor: send every Nth row/col to keep payload small
_GRID_DOWNSAMPLE = 4


def _downsample(grid: np.ndarray, factor: int) -> list:
    """Return a 2-D list downsampled by factor in both dimensions."""
    s = grid[::factor, ::factor]
    return s.tolist()


def _snapshot(sim: "AIGISSimulation") -> dict:
    """
    Build a compact JSON-serialisable snapshot from the current simulation state.
    Called on the simulation thread; must be fast.
    """
    env = sim.environment
    step = sim.step

    # ── Fire grid (downsampled) ─────────────────────────────────────────────
    fire_grid = _downsample(env.fire_grid.astype(np.int8), _GRID_DOWNSAMPLE)

    # ── Smoke grid (normalised, downsampled) ────────────────────────────────
    smoke_max = float(env.smoke_grid.max()) if env.smoke_grid is not None else 0.0
    smoke_norm = (env.smoke_grid / smoke_max) if smoke_max > 0 else env.smoke_grid
    smoke_grid = _downsample(
        (smoke_norm * 100).astype(np.int8),
        _GRID_DOWNSAMPLE,
    ) if env.smoke_grid is not None else []

    # ── Civilians ───────────────────────────────────────────────────────────
    civ_data = []
    for c in sim.agents.get("civilians", []):
        if c.grid_position:
            r, col = c.grid_position
            civ_data.append({
                "r": r, "c": col,
                "panic": round(float(c.panic_level), 3),
                "active": c.is_active,
                "evacuated": getattr(c, "is_evacuated", False),
                "injured": getattr(c, "is_injured", False),
            })

    # ── Firefighters ────────────────────────────────────────────────────────
    ff_data = []
    for ff in sim.agents.get("firefighters", []):
        if ff.grid_position:
            r, col = ff.grid_position
            water_pct = round(
                float(getattr(ff, "water_level", 0))
                / max(float(getattr(ff, "water_capacity", 1)), 1e-9)
                * 100,
                1,
            )
            ff_data.append({"r": r, "c": col, "water_pct": water_pct})

    # ── Rescuers ────────────────────────────────────────────────────────────
    rs_data = []
    for rs in sim.agents.get("rescuers", []):
        if rs.grid_position:
            r, col = rs.grid_position
            rs_data.append({"r": r, "c": col})

    # ── History arrays (step, evac, cas, fire_active, fire_burnt, aqi) ──────
    history = getattr(sim, "_dashboard_history", {
        "steps": [], "evac": [], "cas": [], "fire_active": [],
        "fire_burnt": [], "aqi": [], "smoke_injured": [],
    })

    # ── Commander phase ─────────────────────────────────────────────────────
    commander = sim.agents.get("commander")
    phase = int(commander.current_phase) if commander else 0

    # ── Firefighter CNP refusals ─────────────────────────────────────────────
    cnp_refusals = int(getattr(sim, "_cnp_refusals", 0))

    # ── Panic distribution ───────────────────────────────────────────────────
    all_panics = [
        float(c.panic_level)
        for c in sim.agents.get("civilians", [])
        if c.is_active
    ]

    # ── Grid dimensions ─────────────────────────────────────────────────────
    gh, gw = env.grid_shape

    fire_stats = sim.fire_sim.get_fire_statistics()

    return {
        "step": step,
        "grid_rows": gh,
        "grid_cols": gw,
        "downsample": _GRID_DOWNSAMPLE,
        "fire_grid": fire_grid,
        "smoke_grid": smoke_grid,
        "civilians": civ_data,
        "firefighters": ff_data,
        "rescuers": rs_data,
        "history": history,
        "phase": phase,
        "phase_name": _PHASE_NAMES.get(phase, "Unknown"),
        "phase_color": _PHASE_COLS.get(phase, "#ffffff"),
        "cnp_refusals": cnp_refusals,
        "panics": all_panics,
        "aqi": float(getattr(env, "current_aqi", 0.0)),
        "fire_active": int(fire_stats.get("burning_cells", 0)),
        "fire_burnt": int(fire_stats.get("burnt_cells", 0)),
        "evacuated": int(sim.count_evacuated()),
        "casualties": int(sum(
            1 for c in sim.agents.get("civilians", [])
            if not c.is_active and not getattr(c, "is_evacuated", False)
        )),
        "total_civilians": len(sim.agents.get("civilians", [])),
        "complete": False,
    }


class WebDashboard:
    """
    Real-time web dashboard for AIGIS.

    Starts a Flask dev server on a background thread.  The simulation loop
    calls update(sim) each interval; the SSE endpoint fans updates out to
    all connected browser tabs.

    Replay: all snapshots are stored so a browser that connects after the
    simulation ends (or mid-run) sees the full history replayed at speed,
    then receives the final "complete" frame.
    """

    def __init__(self, port: int = 5000, auto_open: bool = True):
        self._port = port
        self._auto_open = auto_open
        self._queues: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._server_thread: Optional[threading.Thread] = None
        self._started = False
        # All serialised SSE messages in order — replayed to late-joiners
        self._history: list[str] = []

        # Build Flask app pointing at our templates folder
        import os
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        self._app = Flask(__name__, template_folder=template_dir)
        self._app.logger.disabled = True
        import logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

        self._register_routes()

    # ------------------------------------------------------------------
    # Public API (matches matplotlib Dashboard interface)
    # ------------------------------------------------------------------

    def start_server(self) -> None:
        """Start the Flask server in a background daemon thread."""
        if self._started:
            return
        self._started = True

        def _run():
            self._app.run(
                host="0.0.0.0",
                port=self._port,
                threaded=True,
                use_reloader=False,
                debug=False,
            )

        self._server_thread = threading.Thread(target=_run, daemon=True)
        self._server_thread.start()
        time.sleep(0.8)  # allow Flask to bind

        url = f"http://localhost:{self._port}"
        print(f"  Web dashboard: {url}")
        if self._auto_open:
            try:
                webbrowser.open(url)
            except Exception:
                pass

    def update(self, sim: "AIGISSimulation") -> None:
        """Push a snapshot to all connected SSE subscribers."""
        # Build or extend the per-simulation history
        if not hasattr(sim, "_dashboard_history"):
            sim._dashboard_history = {
                "steps": [], "evac": [], "cas": [],
                "fire_active": [], "fire_burnt": [],
                "aqi": [], "smoke_injured": [],
            }

        fire_stats = sim.fire_sim.get_fire_statistics()
        evac = sim.count_evacuated()
        cas = sum(
            1 for c in sim.agents.get("civilians", [])
            if not c.is_active and not getattr(c, "is_evacuated", False)
        )
        smoke_inj = sum(
            1 for c in sim.agents.get("civilians", [])
            if getattr(c, "is_injured", False)
        )

        h = sim._dashboard_history
        h["steps"].append(sim.step)
        h["evac"].append(evac)
        h["cas"].append(cas)
        h["fire_active"].append(int(fire_stats.get("burning_cells", 0)))
        h["fire_burnt"].append(int(fire_stats.get("burnt_cells", 0)))
        h["aqi"].append(float(getattr(sim.environment, "current_aqi", 0.0)))
        h["smoke_injured"].append(smoke_inj)

        snap = _snapshot(sim)
        self._broadcast(snap)

    def finalize(self, sim: "AIGISSimulation") -> None:
        """Send final snapshot with complete=True."""
        self.update(sim)
        if hasattr(sim, "_dashboard_history"):
            snap = _snapshot(sim)
            snap["complete"] = True
            self._broadcast(snap)

    def stop(self) -> None:
        """Broadcast a stop signal. Server daemon thread dies with process."""
        self._broadcast({"complete": True, "step": -1})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _broadcast(self, payload: dict) -> None:
        data = json.dumps(payload, default=float)
        msg = f"data: {data}\n\n"
        with self._lock:
            self._history.append(msg)
            dead = []
            for q in self._queues:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._queues.remove(q)

    def _register_routes(self) -> None:
        app = self._app
        dashboard = self  # capture for closures

        @app.route("/")
        def index():
            return render_template("index.html")

        @app.route("/stream")
        def stream():
            q: queue.Queue = queue.Queue(maxsize=256)

            with dashboard._lock:
                # Snapshot history at connection time for replay
                replay = list(dashboard._history)
                already_complete = any(
                    '"complete": true' in m or '"complete":true' in m
                    for m in replay
                )
                if not already_complete:
                    if len(dashboard._queues) >= _MAX_SUBSCRIBERS:
                        dashboard._queues.pop(0)
                    dashboard._queues.append(q)

            def generate():
                yield "data: {\"heartbeat\": true}\n\n"
                # Replay all stored frames at ~30 ms intervals so the browser
                # sees the full simulation history even if it connects late.
                for msg in replay:
                    yield msg
                    time.sleep(0.03)
                # If simulation was already done, we're finished
                if already_complete:
                    return
                # Otherwise stream live updates
                try:
                    while True:
                        try:
                            msg = q.get(timeout=20)
                            yield msg
                            if '"complete": true' in msg or '"complete":true' in msg:
                                break
                        except queue.Empty:
                            yield ": keep-alive\n\n"
                finally:
                    with dashboard._lock:
                        if q in dashboard._queues:
                            dashboard._queues.remove(q)

            return Response(
                stream_with_context(generate()),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
