"""
AIGIS — Multi-Agent Reinforcement Learning (MARL) Package
==========================================================
Independent PPO with centralized critic (CTDE paradigm).

Algorithm references:
  Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms."
    arXiv:1707.06347.
  Tan, M. (1993). "Multi-Agent Reinforcement Learning: Independent vs.
    Cooperative Agents." ICML-93, pp. 330–337.
  Lowe, R. et al. (2017). "Multi-Agent Actor-Critic for Mixed Cooperative-
    Competitive Environments." NeurIPS. arXiv:1706.02275.  (CTDE shared critic)
  Schulman, J. et al. (2016). "High-Dimensional Continuous Control Using
    Generalized Advantage Estimation." ICLR. arXiv:1506.02438.  (GAE)
  Bengio, Y. et al. (2009). "Curriculum Learning." ICML-09.  (training curriculum)
  de Witt, C.S. et al. (2020). "Is Independent Learning All You Need in the
    StarCraft Multi-Agent Challenge?" arXiv:2011.09533.  (IPPO empirical support)
"""
from .ppo import PPOAgent
from .observations import build_firefighter_obs, build_rescuer_obs, build_commander_obs
from .rewards import (
    step_reward_firefighter, step_reward_rescuer, step_reward_commander,
    terminal_reward_firefighter, terminal_reward_rescuer, terminal_reward_commander,
)

__all__ = [
    'PPOAgent',
    'build_firefighter_obs', 'build_rescuer_obs', 'build_commander_obs',
    'step_reward_firefighter', 'step_reward_rescuer', 'step_reward_commander',
    'terminal_reward_firefighter', 'terminal_reward_rescuer', 'terminal_reward_commander',
]
