"""
AIGIS — MARL Training Script
=============================
Trains three Independent PPO agents (Firefighter, Rescuer, Commander)
with a centralized critic across 9 historical fire scenarios via curriculum.

Algorithm: Independent PPO + shared centralized critic (CTDE).
  Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms."
    arXiv:1707.06347.
  Lowe, R. et al. (2017). "Multi-Agent Actor-Critic for Mixed Cooperative-
    Competitive Environments." NeurIPS. arXiv:1706.02275.
  Tan, M. (1993). "Multi-Agent RL: Independent vs. Cooperative Agents."
    ICML-93, pp. 330–337.

Curriculum: difficulty-ordered scenario sampling.
  Bengio, Y. et al. (2009). "Curriculum Learning." ICML-09, pp. 41–48.

Evaluation: held-out Mati 2018 + Camp Fire 2018 (never seen during training).
  Grimm, V. et al. (2020). ODD Protocol. JASSS 23(2):7.

Usage
-----
  python train_marl.py [--episodes N] [--save-every N] [--device cpu|cuda]

Outputs
-------
  models/rl/firefighter.pt
  models/rl/rescuer.pt
  models/rl/commander.pt
  marl_training_log.csv
  marl_training_curves.png
"""
from __future__ import annotations
import argparse
import contextlib
import io
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

import src.config as _cfg
import src.fire_simulation as _fs_mod
import src.simulation as _sim_mod
import src.agents.sentinel as _sentinel_mod
import src.agents.analyst as _analyst_mod

from src.simulation import AIGISSimulation
from src.rl.ppo import PPOAgent
from src.rl.curriculum import ScenarioCurriculum
from src.rl.observations import (
    build_firefighter_obs, build_rescuer_obs, build_commander_obs, build_global_state
)
from src.rl.rewards import (
    step_reward_firefighter, step_reward_rescuer, step_reward_commander,
    terminal_reward_firefighter, terminal_reward_rescuer, terminal_reward_commander,
)

BG, PANEL, FG = '#1a1a2e', '#16213e', '#e0e0e0'

# ---------------------------------------------------------------------------
# Parameter patching (same mechanism as train_models.py)
# ---------------------------------------------------------------------------

_PATCH_TARGETS: dict = {
    'FIRE_SPREAD_PROB_BASE':      [_cfg, _fs_mod],
    'ROTHERMEL_BASE_ROS':         [_cfg, _fs_mod, _analyst_mod],
    'WIND_SPEED':                 [_cfg, _fs_mod, _analyst_mod],
    'WIND_INITIAL_DIRECTION':     [_cfg, _fs_mod, _sentinel_mod, _analyst_mod],
    'WIND_OSCILLATION_AMPLITUDE': [_cfg, _fs_mod, _sentinel_mod, _analyst_mod],
    'WIND_OSCILLATION_PERIOD':    [_cfg, _fs_mod],
    'NUM_CIVILIANS':              [_cfg, _sim_mod],
}


def _apply_overrides(params: dict) -> dict:
    snapshot = {}
    for param, value in params.items():
        for mod in _PATCH_TARGETS.get(param, [_cfg]):
            if hasattr(mod, param):
                snapshot[(id(mod), param)] = (mod, getattr(mod, param))
                setattr(mod, param, value)
    return snapshot


def _reset_overrides(snapshot: dict) -> None:
    for (_, param), (mod, original) in snapshot.items():
        setattr(mod, param, original)


@contextlib.contextmanager
def _quiet():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    scenario: dict,
    agents: dict,            # {'ff': PPOAgent, 'rsc': PPOAgent, 'cmd': PPOAgent}
    max_steps: int,
    training: bool = True,
    run_id: int = 0,
) -> dict:
    """
    Run one episode under the given scenario with the three RL agents.
    Collects transitions and returns episode statistics.

    Returns dict with keys: mortality_rate, evacuation_success_rate,
    burned_area_pct, total_reward_ff, total_reward_rsc, total_reward_cmd.
    """
    snapshot = _apply_overrides(scenario['params'])

    with _quiet():
        sim = AIGISSimulation(
            lat=scenario['lat'],
            lon=scenario['lon'],
            radius=scenario['radius'],
            mode='batch',
            run_id=run_id,
            fire_locations=scenario['fire_locations'],
        )

    env = sim.environment
    all_agents = sim.agents

    # Expose agent lists on the environment for observation builders
    env._rl_firefighter_agents = all_agents.get('firefighters', [])
    env._rl_rescuer_agents      = all_agents.get('rescuers', [])
    env._rl_recent_refusals     = 0

    ff_agent  = env._rl_firefighter_agents[0] if env._rl_firefighter_agents else None
    rsc_agent = env._rl_rescuer_agents[0]      if env._rl_rescuer_agents      else None
    cmd_agent = all_agents.get('commander')

    # Overwrite MAX_STEPS temporarily
    orig_max = sim.max_steps if hasattr(sim, 'max_steps') else _cfg.MAX_STEPS
    sim.max_steps = max_steps

    prev_burning      = int(np.sum(env.fire_grid == 1))
    prev_casualties   = 0
    prev_evacuated    = 0
    total_r = {'ff': 0.0, 'rsc': 0.0, 'cmd': 0.0}

    for step in range(max_steps):
        civilians = list(all_agents.get('civilians', []))

        # ── Build observations ──────────────────────────────────────────
        obs_ff  = build_firefighter_obs(ff_agent,  env, step, max_steps) \
                  if ff_agent  else np.zeros(PPOAgent.OBS_DIMS['firefighter'],  dtype=np.float32)
        obs_rsc = build_rescuer_obs(
                      rsc_agent, env, step, max_steps,
                      civilians, getattr(cmd_agent, 'current_phase', 0)
                  ) if rsc_agent else np.zeros(PPOAgent.OBS_DIMS['rescuer'], dtype=np.float32)
        obs_cmd = build_commander_obs(cmd_agent, env, step, max_steps, civilians) \
                  if cmd_agent else np.zeros(PPOAgent.OBS_DIMS['commander'], dtype=np.float32)

        global_obs = build_global_state(obs_ff, obs_rsc, obs_cmd)

        # ── Sample actions ──────────────────────────────────────────────
        if training:
            action_ff,  lp_ff  = agents['ff'].act(obs_ff)
            action_rsc, lp_rsc = agents['rsc'].act(obs_rsc)
            action_cmd, lp_cmd = agents['cmd'].act(obs_cmd)
            val_ff  = agents['ff'].value(global_obs)
            val_rsc = agents['rsc'].value(global_obs)
            val_cmd = agents['cmd'].value(global_obs)
        else:
            action_ff  = agents['ff'].best_action(obs_ff)
            action_rsc = agents['rsc'].best_action(obs_rsc)
            action_cmd = agents['cmd'].best_action(obs_cmd)

        # ── Inject RL decisions into agent objects ──────────────────────
        if ff_agent:
            ff_agent._rl_obs = obs_ff
        if rsc_agent:
            rsc_agent._rl_obs  = obs_rsc
            rsc_agent._rl_action = action_rsc
        if cmd_agent:
            cmd_agent._rl_obs = obs_cmd

        # ── Step simulation ─────────────────────────────────────────────
        done = bool(sim.run_step())

        # ── Compute step rewards ────────────────────────────────────────
        cur_burning   = int(np.sum(env.fire_grid == 1))
        cur_casualties  = sum(1 for c in civilians if getattr(c, 'status', '') == 'casualty')
        cur_evacuated   = sum(1 for c in civilians if getattr(c, 'status', '') == 'evacuated')
        cells_ext      = max(prev_burning - cur_burning, 0)
        delta_burning  = cur_burning - prev_burning
        new_casualties = max(cur_casualties - prev_casualties, 0)
        new_evacuated  = max(cur_evacuated  - prev_evacuated,  0)

        r_ff  = step_reward_firefighter(
            cells_extinguished=cells_ext,
            water_used=ff_agent.water_capacity - ff_agent.current_water if ff_agent else 0,
            water_capacity=ff_agent.water_capacity if ff_agent else 5000,
            delta_burning=delta_burning,
            fire_line_cells_created=getattr(ff_agent, 'fire_lines_this_step', 0),
            is_refilling=ff_agent.is_refilling if ff_agent else False,
        )
        r_rsc = step_reward_rescuer(
            civilians_rescued_this_step=new_evacuated,
            delta_distance_to_target=0.0,
            casualty_this_step=new_casualties,
            path_risk=0.0,
            is_waiting=(getattr(rsc_agent, '_rl_action', -1) == 3),
            active_civilians=len([c for c in civilians
                                  if getattr(c, 'status', '') not in ('evacuated','casualty')]),
        )
        r_cmd = step_reward_commander(
            civilians_evacuated_this_step=new_evacuated,
            casualties_this_step=new_casualties,
            tti=getattr(cmd_agent, 'tti', float('inf')),
            ect=getattr(cmd_agent, 'ect', 0.0),
            phase=getattr(cmd_agent, 'current_phase', 0),
            action=action_cmd,
            rescuers_idle=sum(1 for r in env._rl_rescuer_agents
                              if getattr(r, 'mission_status', 'IDLE') == 'IDLE'),
            total_rescuers=max(len(env._rl_rescuer_agents), 1),
            cfp_issued=getattr(cmd_agent, 'cfp_issued_this_step', False),
        )

        total_r['ff']  += r_ff
        total_r['rsc'] += r_rsc
        total_r['cmd'] += r_cmd

        if training:
            agents['ff'].buffer.add(
                obs_ff, global_obs, action_ff, lp_ff, r_ff, done, val_ff
            )
            agents['rsc'].buffer.add(
                obs_rsc, global_obs, action_rsc, lp_rsc, r_rsc, done, val_rsc
            )
            agents['cmd'].buffer.add(
                obs_cmd, global_obs, action_cmd, lp_cmd, r_cmd, done, val_cmd
            )

        prev_burning    = cur_burning
        prev_casualties = cur_casualties
        prev_evacuated  = cur_evacuated

        if done:
            break

    # ── Terminal rewards ────────────────────────────────────────────────
    result = sim.get_results()
    burned_pct = result['burned_area_pct']
    mort       = result['mortality_rate']
    evac       = result['evacuation_success_rate']

    t_ff  = terminal_reward_firefighter(burned_pct)
    t_rsc = terminal_reward_rescuer(mort, evac)
    t_cmd = terminal_reward_commander(mort, evac, burned_pct)

    if training and agents['ff'].buffer.rewards:
        agents['ff'].buffer.rewards[-1]  += t_ff
        agents['rsc'].buffer.rewards[-1] += t_rsc
        agents['cmd'].buffer.rewards[-1] += t_cmd

    total_r['ff']  += t_ff
    total_r['rsc'] += t_rsc
    total_r['cmd'] += t_cmd

    _reset_overrides(snapshot)
    sim.max_steps = orig_max

    return {
        'mortality_rate':          mort,
        'evacuation_success_rate': evac,
        'burned_area_pct':         burned_pct,
        'reward_ff':               total_r['ff'],
        'reward_rsc':              total_r['rsc'],
        'reward_cmd':              total_r['cmd'],
        'scenario':                scenario['name'],
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    total_episodes:   int = 10_000,
    save_every:       int = 500,
    log_every:        int = 50,
    device:           str = 'cpu',
    output_dir:       str = 'models/rl',
    phase1_end:       int = 2_000,
    phase2_end:       int = 6_000,
    training_steps:   int = 200,
    start_episode:    int = 0,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 70)
    print('AIGIS — MARL Training  (Independent PPO + CTDE Critic)')
    print('=' * 70)
    print('Schulman et al. (2017) PPO  |  Lowe et al. (2017) CTDE')
    print('Bengio et al. (2009) Curriculum  |  12 training scenarios (3 phases)')
    print(f'Total episodes: {total_episodes}  |  Steps/episode: {training_steps}')
    print(f'Device: {device}  |  Output: {output_dir}')
    if start_episode > 0:
        print(f'Resuming from episode {start_episode}')
    print('=' * 70 + '\n')

    global_dim = _cfg.RL_GLOBAL_STATE_DIM  # 72  (FF:24 + RSC:22 + CMD:26)

    agents = {
        'ff':  PPOAgent('firefighter', global_dim, device=device),
        'rsc': PPOAgent('rescuer',     global_dim, device=device),
        'cmd': PPOAgent('commander',   global_dim, device=device),
    }

    if start_episode > 0:
        for role, short in [('firefighter', 'ff'), ('rescuer', 'rsc'), ('commander', 'cmd')]:
            path = os.path.join(output_dir, f'{short}.pt')
            if not os.path.exists(path):
                path = os.path.join(output_dir, f'{role}.pt')
            agents[short].load(path)
            print(f'  Loaded checkpoint: {path}')

    curriculum = ScenarioCurriculum(
        phase1_end=phase1_end,
        phase2_end=phase2_end,
        rng_seed=42,
    )
    if start_episode > 0:
        curriculum.advance(start_episode)
        print(f'  Curriculum fast-forwarded to episode {start_episode} (phase {curriculum.current_curriculum_phase})\n')

    log_rows = []
    rolling_window = 100
    rewards_ff, rewards_rsc, rewards_cmd = [], [], []
    mort_hist, evac_hist = [], []

    for ep in range(start_episode + 1, total_episodes + 1):
        scenario = curriculum.sample()

        stats = run_episode(
            scenario=scenario,
            agents=agents,
            max_steps=training_steps,
            training=True,
            run_id=ep,
        )

        # PPO update after each episode
        losses = {
            'ff':  agents['ff'].update(),
            'rsc': agents['rsc'].update(),
            'cmd': agents['cmd'].update(),
        }

        rewards_ff.append(stats['reward_ff'])
        rewards_rsc.append(stats['reward_rsc'])
        rewards_cmd.append(stats['reward_cmd'])
        mort_hist.append(stats['mortality_rate'])
        evac_hist.append(stats['evacuation_success_rate'])

        log_rows.append({
            'episode':          ep,
            'scenario':         stats['scenario'],
            'curriculum':       curriculum.current_curriculum_phase,
            'mortality':        stats['mortality_rate'],
            'evacuation':       stats['evacuation_success_rate'],
            'burned_pct':       stats['burned_area_pct'],
            'reward_ff':        stats['reward_ff'],
            'reward_rsc':       stats['reward_rsc'],
            'reward_cmd':       stats['reward_cmd'],
            # Actor losses — all three roles (Yu et al. 2022 MAPPO logging standard)
            'loss_actor_ff':    losses['ff'].get('actor_loss',  0),
            'loss_actor_rsc':   losses['rsc'].get('actor_loss', 0),
            'loss_actor_cmd':   losses['cmd'].get('actor_loss', 0),
            # Critic losses — convergence diagnostic (Yu et al. 2022)
            'loss_critic_ff':   losses['ff'].get('critic_loss',  0),
            'loss_critic_rsc':  losses['rsc'].get('critic_loss', 0),
            'loss_critic_cmd':  losses['cmd'].get('critic_loss', 0),
            # Policy entropy — exploration/exploitation diagnostic (Yu et al. 2022;
            # MARL Diagnostics paper, arXiv:2312.08468)
            'entropy_ff':       losses['ff'].get('entropy',  0),
            'entropy_rsc':      losses['rsc'].get('entropy', 0),
            'entropy_cmd':      losses['cmd'].get('entropy', 0),
        })

        if ep % log_every == 0:
            n = min(rolling_window, len(mort_hist))
            print(
                f"  Ep {ep:5d}/{total_episodes}  "
                f"curriculum=P{curriculum.current_curriculum_phase}  "
                f"mort={np.mean(mort_hist[-n:]):.3%}  "
                f"evac={np.mean(evac_hist[-n:]):.3%}  "
                f"R_ff={np.mean(rewards_ff[-n:]):.1f}  "
                f"R_rsc={np.mean(rewards_rsc[-n:]):.1f}  "
                f"R_cmd={np.mean(rewards_cmd[-n:]):.1f}"
            )

        if ep % save_every == 0:
            for role, agent in agents.items():
                agent.save(os.path.join(output_dir, f'{role}.pt'))
            print(f"  [Saved checkpoints at episode {ep}]")

    # Final save
    for role, agent in agents.items():
        path = os.path.join(output_dir, f'{_role_name(role)}.pt')
        agent.save(path)
        print(f"  Policy saved: {path}")

    # Save log
    df_log = pd.DataFrame(log_rows)
    log_path = 'marl_training_log.csv'
    df_log.to_csv(log_path, index=False)
    print(f"\nTraining log saved to: {log_path}")

    _plot_training_curves(df_log, 'marl_training_curves.png')


def _role_name(short: str) -> str:
    return {'ff': 'firefighter', 'rsc': 'rescuer', 'cmd': 'commander'}[short]


def _plot_training_curves(df: pd.DataFrame, out_path: str) -> None:
    """
    8-panel training diagnostic figure (4 rows x 2 cols):
      Row 1: Mortality rate | Evacuation success rate
      Row 2: Firefighter reward | Rescuer reward
      Row 3: Commander reward | Policy entropy per role (FF / Rescuer / Cmd)
      Row 4: Critic loss per role (FF / Rescuer / Cmd) | (blank / reserved)

    Per-role entropy and critic loss are shown with separate lines per agent
    rather than a cross-agent mean.  This allows detection of asymmetric
    learning dynamics caused by the heterogeneous per-step reward scales
    (FF +1/cell vs Rescuer +5/rescue): if the Rescuer's entropy collapses
    significantly earlier than the Firefighter's, the 5x reward asymmetry
    is causing faster policy convergence and should be investigated.

    Ref: Yu et al. (2022). MAPPO. NeurIPS. arXiv:2103.01955.
         Kuba et al. (2023). HAPPO/HARL. JMLR 25(1).
    Curriculum phase boundaries marked as vertical lines (Bengio et al. 2009).
    """
    fig, axes = plt.subplots(4, 2, figsize=(12, 16), facecolor=BG)
    fig.suptitle(
        'AIGIS MARL Training  |  Independent PPO (Schulman et al. 2017)\n'
        'Curriculum: Bengio et al. (2009)  |  15 training scenarios  |  '
        'Phases: P1 easy → P2 medium → P3 hard',
        color=FG, fontsize=9, fontweight='bold',
    )

    def smooth(x, w=100):
        if len(x) < w:
            return x
        return pd.Series(x).rolling(w, min_periods=1).mean().values

    # Ensure per-role columns exist
    for col in ['entropy_ff', 'entropy_rsc', 'entropy_cmd',
                'loss_critic_ff', 'loss_critic_rsc', 'loss_critic_cmd']:
        if col not in df.columns:
            df[col] = 0.0

    # Role colour palette (consistent across entropy and critic loss panels)
    ROLE_COLOURS = {
        'ff':  '#ffd60a',   # yellow  — Firefighter
        'rsc': '#fb5607',   # orange  — Rescuer
        'cmd': '#8338ec',   # purple  — Commander
    }

    phase_eps = [800, 2400]   # curriculum phase boundaries (Bengio et al. 2009)

    def _style_ax(ax):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor('#3a3a5c')
        for ph_ep in phase_eps:
            ax.axvline(ph_ep, color='white', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.set_xlabel('Episode', color=FG, fontsize=8)

    # --- single-series panels (rows 1-3 left) ---
    single_plots = [
        (axes[0, 0], 'mortality',  'Mortality Rate',          '#ff006e'),
        (axes[0, 1], 'evacuation', 'Evacuation Success Rate', '#06d6a0'),
        (axes[1, 0], 'reward_ff',  'Firefighter Reward',      ROLE_COLOURS['ff']),
        (axes[1, 1], 'reward_rsc', 'Rescuer Reward',          ROLE_COLOURS['rsc']),
        (axes[2, 0], 'reward_cmd', 'Commander Reward',        ROLE_COLOURS['cmd']),
    ]

    for ax, col, label, colour in single_plots:
        _style_ax(ax)
        if col not in df.columns:
            ax.set_visible(False)
            continue
        vals = df[col].values
        ax.plot(df['episode'], vals, color=colour, alpha=0.25, linewidth=0.5)
        ax.plot(df['episode'], smooth(vals), color=colour, linewidth=1.5,
                label='100-ep rolling mean')
        ax.set_ylabel(label, color=FG, fontsize=8)
        ax.legend(fontsize=6, facecolor=PANEL, labelcolor=FG)

    # --- per-role entropy panel (row 3 right) ---
    ax_ent = axes[2, 1]
    _style_ax(ax_ent)
    ax_ent.set_ylabel('Policy Entropy (per role)', color=FG, fontsize=8)
    for role, col in [('Firefighter', 'entropy_ff'),
                      ('Rescuer',     'entropy_rsc'),
                      ('Commander',   'entropy_cmd')]:
        key = col.split('_')[1]
        colour = ROLE_COLOURS[key]
        vals = df[col].values
        ax_ent.plot(df['episode'], vals, color=colour, alpha=0.20, linewidth=0.5)
        ax_ent.plot(df['episode'], smooth(vals), color=colour, linewidth=1.5,
                    label=role)
    ax_ent.legend(fontsize=6, facecolor=PANEL, labelcolor=FG)

    # --- per-role critic loss panel (row 4 left) ---
    ax_cl = axes[3, 0]
    _style_ax(ax_cl)
    ax_cl.set_ylabel('Critic Loss (per role)', color=FG, fontsize=8)
    for role, col in [('Firefighter', 'loss_critic_ff'),
                      ('Rescuer',     'loss_critic_rsc'),
                      ('Commander',   'loss_critic_cmd')]:
        key = col.split('_')[2]
        colour = ROLE_COLOURS[key]
        vals = df[col].values
        ax_cl.plot(df['episode'], vals, color=colour, alpha=0.20, linewidth=0.5)
        ax_cl.plot(df['episode'], smooth(vals), color=colour, linewidth=1.5,
                   label=role)
    ax_cl.legend(fontsize=6, facecolor=PANEL, labelcolor=FG)

    # Hide unused bottom-right panel
    axes[3, 1].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=BG)
    print(f"Training curves saved to: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Train AIGIS MARL agents (Independent PPO)')
    p.add_argument('--episodes',    type=int, default=10_000)
    p.add_argument('--save-every',  type=int, default=500)
    p.add_argument('--device',      type=str, default='cpu',
                   help='cpu or cuda')
    p.add_argument('--steps',       type=int, default=200,
                   help='Max simulation steps per training episode')
    p.add_argument('--output',      type=str, default='models/rl')
    p.add_argument('--phase1-end',     type=int, default=2_000)
    p.add_argument('--phase2-end',     type=int, default=6_000)
    p.add_argument('--start-episode',  type=int, default=0,
                   help='Resume from this episode (loads checkpoints from --output dir)')
    args = p.parse_args()

    train(
        total_episodes=args.episodes,
        save_every=args.save_every,
        device=args.device,
        training_steps=args.steps,
        output_dir=args.output,
        phase1_end=args.phase1_end,
        phase2_end=args.phase2_end,
        start_episode=args.start_episode,
    )


if __name__ == '__main__':
    main()
