"""
Proximal Policy Optimization (PPO) — per-agent implementation
==============================================================
Custom PPO with centralized critic for CTDE MARL.

Primary reference:
  Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
  "Proximal Policy Optimization Algorithms."
  arXiv:1707.06347.

GAE (advantage estimation):
  Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016).
  "High-Dimensional Continuous Control Using Generalized Advantage Estimation."
  ICLR. arXiv:1506.02438.

Centralized critic (CTDE):
  Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017).
  "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments."
  NeurIPS. arXiv:1706.02275.
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Optional

# ---------------------------------------------------------------------------
# Network definitions
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """
    Policy network: obs → action probabilities.
    Two hidden layers (64 units each, ReLU activation).
    Small size is deliberate: fast CPU inference, low overfitting risk.
    """
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),      nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


class Critic(nn.Module):
    """
    Value network: global_state → V(s).
    During CTDE training the critic receives the concatenated observations
    of all three RL agents (Firefighter + Rescuer + Commander) as the
    global state — following Lowe et al. (2017).
    At test time the actor runs with local obs only; the critic is not used.
    """
    def __init__(self, global_state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),               nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores one episode of transitions for a single agent."""

    def __init__(self):
        self.obs:         List[np.ndarray] = []
        self.global_obs:  List[np.ndarray] = []
        self.actions:     List[int]        = []
        self.log_probs:   List[float]      = []
        self.rewards:     List[float]      = []
        self.dones:       List[bool]       = []
        self.values:      List[float]      = []

    def clear(self):
        self.__init__()

    def add(self, obs, global_obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.global_obs.append(global_obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generalized Advantage Estimation (GAE).
        Schulman et al. (2016) arXiv:1506.02438.
        δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        A_t = Σ_{l≥0} (γλ)^l δ_{t+l}
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)

        gae = 0.0
        for t in reversed(range(n)):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages[t] = gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return returns, advantages


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    Independent PPO agent (actor + shared-global critic).

    One instance per agent role (Firefighter, Rescuer, Commander).
    The critic uses global state (concatenated observations of all three roles)
    during training — CTDE paradigm (Lowe et al. 2017).

    Hyperparameters follow Schulman et al. (2017) recommended defaults.
    """

    # Observation dimensions per role.
    # Commander expanded from 20 → 26 to include sub-agent state sharing
    # (firefighter water levels, rescuer mission fractions, neighbour positions,
    # mean civilian panic) — Yu et al. (2022) MAPPO implementation guidelines:
    #   Yu, C. et al. (2022). "The Surprising Effectiveness of PPO in Cooperative
    #   Multi-Agent Games." NeurIPS 2022. arXiv:2103.01955.
    OBS_DIMS = {
        'firefighter': 24,
        'rescuer':      22,
        'commander':    26,   # was 20; +6 inter-agent coordination dims (Yu et al. 2022)
    }
    # Action space sizes per role
    ACTION_DIMS = {
        'firefighter': 5,   # water_drop / fire_line / backburn / patrol / return_to_base
        'rescuer':      4,   # move_highest_panic / move_nearest / move_safe_zone / wait
        'commander':    6,   # maintain / advance / hold_prealert / force_evacuate / shelter / reassure
    }

    def __init__(
        self,
        role: str,
        global_state_dim: int,
        lr: float          = 3e-4,
        gamma: float       = 0.99,
        gae_lambda: float  = 0.95,
        clip_eps: float    = 0.2,
        n_epochs: int      = 4,
        mini_batch: int    = 64,
        entropy_coef: float = 0.01,
        device: str        = 'cpu',
    ):
        assert role in self.OBS_DIMS, f"Unknown role: {role}"
        self.role            = role
        self.obs_dim         = self.OBS_DIMS[role]
        self.n_actions       = self.ACTION_DIMS[role]
        self.global_state_dim = global_state_dim
        self.gamma           = gamma
        self.gae_lambda      = gae_lambda
        self.clip_eps        = clip_eps
        self.n_epochs        = n_epochs
        self.mini_batch      = mini_batch
        self.entropy_coef    = entropy_coef
        self.device          = torch.device(device)

        self.actor  = Actor(self.obs_dim, self.n_actions).to(self.device)
        self.critic = Critic(global_state_dim).to(self.device)

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> tuple[int, float]:
        """
        Sample action from policy.
        Returns (action_index, log_probability).
        Used both during training rollouts and at test time.
        """
        assert obs.shape == (self.obs_dim,), (
            f"[PPO/{self.role}] act(): expected obs shape ({self.obs_dim},), "
            f"got {obs.shape}"
        )
        x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        probs = self.actor(x).squeeze(0)
        dist = Categorical(probs)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item())

    @torch.no_grad()
    def value(self, global_obs: np.ndarray) -> float:
        """Estimate V(s) from global state."""
        assert global_obs.shape == (self.global_state_dim,), (
            f"[PPO/{self.role}] value(): expected global_obs shape ({self.global_state_dim},), "
            f"got {global_obs.shape}"
        )
        x = torch.FloatTensor(global_obs).unsqueeze(0).to(self.device)
        return float(self.critic(x).item())

    @torch.no_grad()
    def best_action(self, obs: np.ndarray) -> int:
        """Greedy action (argmax) for evaluation / deployment."""
        x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        probs = self.actor(x).squeeze(0)
        return int(torch.argmax(probs).item())

    @torch.no_grad()
    def best_action_masked(self, obs: np.ndarray, valid_actions: list) -> int:
        """
        Greedy action with BDI safety masking: invalid actions set to -inf
        before argmax so PPO cannot select BDI-unsafe actions.

        Sardina, S. & Thangarajah, J. (2011). "On the deployment of BDI agents
        in the presence of learning algorithms." Proc. 22nd IJCAI, pp. 1810-1815.
        Action masking is the standard mechanism for enforcing hard constraints
        (safety rules) on RL-selected actions in hybrid BDI+RL architectures.
        """
        x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        logits = self.actor.net(x).squeeze(0)           # raw logits (pre-softmax)
        mask   = torch.full_like(logits, float('-inf'))
        for a in valid_actions:
            mask[a] = logits[a]
        return int(torch.argmax(mask).item())

    # ------------------------------------------------------------------
    # Training update
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        """
        PPO update over one episode's rollout.
        Clips policy ratio to [1-ε, 1+ε] (Schulman et al. 2017, eq. 7).
        Returns dict of loss statistics for logging.
        """
        buf = self.buffer
        if len(buf.rewards) == 0:
            return {}

        last_val = self.value(buf.global_obs[-1]) if not buf.dones[-1] else 0.0
        returns, advantages = buf.compute_returns_and_advantages(last_val, self.gamma, self.gae_lambda)

        # Normalize advantages (improves training stability)
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t       = torch.FloatTensor(np.array(buf.obs)).to(self.device)
        global_t    = torch.FloatTensor(np.array(buf.global_obs)).to(self.device)
        actions_t   = torch.LongTensor(buf.actions).to(self.device)
        old_lp_t    = torch.FloatTensor(buf.log_probs).to(self.device)
        returns_t   = torch.FloatTensor(returns).to(self.device)
        adv_t       = torch.FloatTensor(advantages).to(self.device)

        n = len(buf.rewards)
        total_actor_loss = total_critic_loss = total_entropy = 0.0
        update_count = 0

        for _ in range(self.n_epochs):
            # Mini-batch shuffle
            indices = np.random.permutation(n)
            for start in range(0, n, self.mini_batch):
                idx = indices[start:start + self.mini_batch]
                if len(idx) < 2:
                    continue

                obs_b    = obs_t[idx]
                glob_b   = global_t[idx]
                acts_b   = actions_t[idx]
                old_lp_b = old_lp_t[idx]
                ret_b    = returns_t[idx]
                adv_b    = adv_t[idx]

                # Actor loss (clipped surrogate objective)
                probs    = self.actor(obs_b)
                dist     = Categorical(probs)
                new_lp   = dist.log_prob(acts_b)
                ratio    = torch.exp(new_lp - old_lp_b)
                surr1    = ratio * adv_b
                surr2    = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy    = dist.entropy().mean()

                self.actor_opt.zero_grad()
                (actor_loss - self.entropy_coef * entropy).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()

                # Critic loss (MSE)
                values      = self.critic(glob_b).squeeze(-1)
                critic_loss = nn.functional.mse_loss(values, ret_b)

                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_opt.step()

                total_actor_loss  += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy     += entropy.item()
                update_count      += 1

        self.buffer.clear()

        if update_count == 0:
            return {}
        return {
            'actor_loss':  total_actor_loss  / update_count,
            'critic_loss': total_critic_loss / update_count,
            'entropy':     total_entropy     / update_count,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor':      self.actor.state_dict(),
            'critic':     self.critic.state_dict(),
            'actor_opt':  self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'role':       self.role,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        if 'actor_opt' in ckpt:
            self.actor_opt.load_state_dict(ckpt['actor_opt'])
        if 'critic_opt' in ckpt:
            self.critic_opt.load_state_dict(ckpt['critic_opt'])
        self.actor.train()
        self.critic.train()
