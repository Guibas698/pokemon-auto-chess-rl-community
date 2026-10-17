"""
PyTorch Neural Network Architectures & Maskable PPO Trainer for Pokemon Auto Chess.

Implements:
1. `RewardConditionedMaskedActorCritic`: PyTorch `nn.Module` with FiLM (Feature-wise
   Linear Modulation) conditioned on the agent's reward weight vector $\mathbf{w}^{(k)}$,
   invalid action masking (`logits.masked_fill(~mask, -1e8)`), and twin value heads
   (persona shaped critic + oracle placement critic).
2. `MaskablePPOTrainer`: Clipped-surrogate PPO with GAE($\gamma=0.99, \lambda=0.95$) and
   masked categorical distributions (Huang & Ontañón, 2020).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .gym_env import (
    ACT_BUY_0,
    ACT_BUY_5,
    ACT_DEPLOY_0,
    ACT_DEPLOY_5,
    ACT_END_TURN,
    ACT_LEVEL_UP,
    ACT_LOCK_SHOP,
    ACT_REROLL,
    ACT_SELL_0,
    ACT_SELL_5,
    NUM_ACTIONS,
    OBS_DIM,
)


class RewardConditionedMaskedActorCritic(nn.Module):
    """
    PyTorch Actor-Critic network with:
    - State trunk encoding economy, shop slots, and active 31-synergy vector (`obs[:, :73]`)
    - FiLM Conditioning Encoder mapping the reward vector $\mathbf{w}^{(k)}$ (`obs[:, 73:112]`)
      to $(\gamma, \beta)$ affine modulations
    - Action-masked Policy Head (`Discrete(22)`) with structured inductive bias
    - Twin Critic Heads: `v_shaped` (for PPO GAE updates) and `v_oracle` (true placement value)
    """

    def __init__(self, obs_dim: int = OBS_DIM, num_actions: int = NUM_ACTIONS, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = 73
        self.cond_dim = obs_dim - self.state_dim  # 39

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # FiLM generator from reward persona conditioning vector
        self.film_gen = nn.Sequential(
            nn.Linear(self.cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

        self.actor_head = nn.Linear(hidden_dim, num_actions)
        self.critic_shaped = nn.Linear(hidden_dim, 1)
        self.critic_oracle = nn.Linear(hidden_dim, 1)

    def compute_Domain_prior_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Computes a differentiable domain prior over the 22 macro actions based on the
        observation features and the agent's reward conditioning vector $\mathbf{w}^{(k)}$.
        The neural network learns residual refinements on top of this structured prior.
        """
        B = obs.shape[0]
        prior = torch.zeros((B, NUM_ACTIONS), device=obs.device, dtype=obs.dtype)

        stage = obs[:, 0] * 40.0
        hp = obs[:, 1] * 100.0
        gold = obs[:, 2] * 50.0
        level = obs[:, 4] * 9.0
        free_slots = obs[:, 9] * 4.0
        rel_power = obs[:, 11] * 2.0

        # Reward weights from conditioning slice
        w_interest = obs[:, 74]
        w_win_streak = obs[:, 75]
        w_lose_streak = obs[:, 76]
        w_reroll_evo = obs[:, 77]
        w_synergy = obs[:, 78]

        # Shop slots 0..5 priors
        for i in range(6):
            base = 18 + i * 4
            copies_owned = obs[:, base + 2] * 2.0
            syn_fit = obs[:, base + 3] * 3.0
            slot_score = (
                1.4 * copies_owned * (0.6 + w_reroll_evo)
                + 1.1 * syn_fit * (0.5 + w_synergy)
                + 0.8 * (free_slots > 0).float()
            )
            # Lose-streak economy holds back unnecessary purchases early unless it's a pair/synergy core
            hold_penalty = (w_lose_streak > 0.3).float() * (stage <= 11).float() * (copies_owned < 0.9).float() * 1.2
            prior[:, ACT_BUY_0 + i] = slot_score - hold_penalty

        # Reroll (action 6): favored by Hyper-Roll (high w_reroll_evo at low level) or when gold > 50
        hyper_roll_drive = w_reroll_evo * (level <= 5.5).float() * (gold >= 12.0).float() * 1.5
        excess_gold_roll = (gold >= 52.0).float() * 1.3
        econ_save_penalty = w_interest * (gold < 50.0).float() * (hp > 35.0).float() * 1.4
        desperation_roll = (hp <= 30.0).float() * (gold >= 4.0).float() * 1.6
        prior[:, ACT_REROLL] = hyper_roll_drive + excess_gold_roll + desperation_roll - econ_save_penalty

        # Level Up (action 7): favored by Tempo/Win-Streak or when gold > 50 or free board slots needed
        tempo_level_drive = w_win_streak * (gold >= 14.0).float() * (rel_power < 1.15).float() * 1.6
        econ_level_drive = (gold >= 54.0).float() * 1.8
        hyper_roll_level_ban = (w_reroll_evo > 0.6).float() * (level <= 5.0).float() * (stage <= 14.0).float() * 1.8
        prior[:, ACT_LEVEL_UP] = tempo_level_drive + econ_level_drive - hyper_roll_level_ban

        # Lock shop (action 8): mildly discouraged unless an expensive pair is in shop with 0 gold
        prior[:, ACT_LOCK_SHOP] = -1.2

        # Deploy bench (9..14): strongly favored when free_board_slots > 0
        for i in range(6):
            prior[:, ACT_DEPLOY_0 + i] = 2.0 * (free_slots > 0).float() - 0.3

        # Sell bench (15..20): favored when close to an interest breakpoint (e.g. gold % 10 >= 8)
        mod10 = torch.remainder(gold, 10.0)
        near_breakpoint = ((mod10 >= 8.0) & (gold < 50.0)).float()
        for i in range(6):
            prior[:, ACT_SELL_0 + i] = 0.9 * near_breakpoint * w_interest - 0.8

        # End turn (action 21): baseline anchor, favored when preserving interest breakpoints
        prior[:, ACT_END_TURN] = 0.4 + 0.9 * w_interest * (gold <= 50.0).float() * (free_slots == 0).float()
        return prior

    def forward_trunk(self, obs: torch.Tensor) -> torch.Tensor:
        state_x = obs[:, : self.state_dim]
        cond_x = obs[:, self.state_dim :]
        h = self.state_encoder(state_x)
        film = self.film_gen(cond_x)
        gamma, beta = torch.chunk(film, 2, dim=-1)
        h_mod = h * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * beta
        h_out = self.norm_out(h_mod + self.res_block(h_mod))
        return h_out

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes masked policy distribution, samples (or evaluates) `action`, and returns:
        `(action, log_prob, entropy, value_shaped, value_oracle)`
        """
        h = self.forward_trunk(obs)
        raw_logits = self.actor_head(h) + self.compute_Domain_prior_logits(obs)

        # Invalid action masking (Huang & Ontañón, 2020)
        masked_logits = raw_logits.masked_fill(~action_mask, -1e8)
        dist = Categorical(logits=masked_logits)

        if action is None:
            if deterministic:
                action = torch.argmax(masked_logits, dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        v_shaped = self.critic_shaped(h).squeeze(-1)
        v_oracle = self.critic_oracle(h).squeeze(-1)
        return action, log_prob, entropy, v_shaped, v_oracle


@dataclass
class RolloutStep:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    log_prob: float
    reward: float
    done: bool
    value: float


class MaskablePPOTrainer:
    """
    Maskable Proximal Policy Optimization (PPO) trainer for a single agent or shared
    population model in the Pokemon Auto Chess Community League.
    """

    def __init__(
        self,
        model: RewardConditionedMaskedActorCritic,
        lr: float = 1.5e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.015,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def update(self, buffer: List[RolloutStep], ppo_epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        if not buffer:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        n = len(buffer)
        obs_arr = np.array([s.obs for s in buffer], dtype=np.float32)
        mask_arr = np.array([s.mask for s in buffer], dtype=bool)
        act_arr = np.array([s.action for s in buffer], dtype=np.int64)
        old_logp_arr = np.array([s.log_prob for s in buffer], dtype=np.float32)
        rew_arr = np.array([s.reward for s in buffer], dtype=np.float32)
        done_arr = np.array([s.done for s in buffer], dtype=np.float32)
        val_arr = np.array([s.value for s in buffer], dtype=np.float32)

        # Compute GAE(gamma, lambda)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_non_terminal = 1.0 - done_arr[t]
            next_val = val_arr[t + 1] if t + 1 < n else 0.0
            delta = rew_arr[t] + self.gamma * next_val * next_non_terminal - val_arr[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + val_arr

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        t_obs = torch.from_numpy(obs_arr)
        t_mask = torch.from_numpy(mask_arr)
        t_act = torch.from_numpy(act_arr)
        t_old_logp = torch.from_numpy(old_logp_arr)
        t_adv = torch.from_numpy(advantages)
        t_ret = torch.from_numpy(returns)

        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_ent = 0.0
        num_updates = 0

        self.model.train()
        for _ in range(ppo_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                _, new_logp, entropy, new_val, _ = self.model.get_action_and_value(
                    t_obs[idx],
                    t_mask[idx],
                    action=t_act[idx],
                )
                logratio = new_logp - t_old_logp[idx]
                ratio = logratio.exp()

                mb_adv = t_adv[idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * F.mse_loss(new_val, t_ret[idx])
                ent_loss = entropy.mean()

                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += float(pg_loss.item())
                total_v_loss += float(v_loss.item())
                total_ent += float(ent_loss.item())
                num_updates += 1

        denom = max(1, num_updates)
        return {
            "policy_loss": total_pg_loss / denom,
            "value_loss": total_v_loss / denom,
            "entropy": total_ent / denom,
        }
