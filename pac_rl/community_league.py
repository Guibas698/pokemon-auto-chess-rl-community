"""
Community of RL Models (Multi-Reward League + PBT + MAP-Elites Quality-Diversity)
for Pokemon Auto Chess.

Implements:
1. `CommunityAgent`: Pairs a PyTorch `RewardConditionedMaskedActorCritic` + `MaskablePPOTrainer`
   with a specialized `RewardWeightConfig` persona $\mathbf{w}^{(k)}$.
2. `MAPElitesArchive`: Quality-Diversity grid indexed by `(Primary Synergy, Economic Archetype)`
   storing the best-discovered policy and board composition per niche based on True Placement.
3. `CommunityLeagueTrainer`: Runs multi-generation Maskable PPO training + Population-Based
   Training (PBT) reward/weight evolution + 8-player head-to-head evaluation tournaments.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from .combat_surrogate import FastCombatSimulator, compute_active_synergies
from .data_loader import PACDataRepository
from .gym_env import PokemonAutoChessGymEnv, RewardWeightConfig
from .models import MaskablePPOTrainer, RewardConditionedMaskedActorCritic, RolloutStep


@dataclass
class AgentMetrics:
    generation: int
    agent_name: str
    archetype: str
    target_synergy: str
    avg_placement: float
    top4_rate: float
    win_rate: float
    avg_stage_reached: float
    avg_2star_evos: float
    avg_3star_evos: float
    avg_shaped_return: float
    dominant_synergies: Dict[str, int] = field(default_factory=dict)


@dataclass
class EliteCell:
    synergy_niche: str
    econ_archetype: str
    agent_name: str
    avg_placement: float
    top4_rate: float
    win_rate: float
    sample_board_pokemon: List[str]
    reward_config: RewardWeightConfig


class MAPElitesArchive:
    """
    Quality-Diversity archive over `(Synergy Niche, Economic Archetype)` cells.
    Ensures the community preserves the best specialist for every type and economic style
    rather than collapsing into a single global meta composition.
    """

    def __init__(self):
        self.grid: Dict[Tuple[str, str], EliteCell] = {}

    def try_insert(
        self,
        synergy_niche: str,
        econ_archetype: str,
        agent_name: str,
        avg_placement: float,
        top4_rate: float,
        win_rate: float,
        sample_board_pokemon: List[str],
        reward_config: RewardWeightConfig,
    ) -> bool:
        key = (synergy_niche, econ_archetype)
        existing = self.grid.get(key)
        # Lower placement is better (1.0 is best, 8.0 is worst)
        if existing is None or avg_placement < existing.avg_placement:
            self.grid[key] = EliteCell(
                synergy_niche=synergy_niche,
                econ_archetype=econ_archetype,
                agent_name=agent_name,
                avg_placement=avg_placement,
                top4_rate=top4_rate,
                win_rate=win_rate,
                sample_board_pokemon=list(sample_board_pokemon),
                reward_config=copy.deepcopy(reward_config),
            )
            return True
        return False

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for (syn, arch), cell in sorted(self.grid.items(), key=lambda kv: kv[1].avg_placement):
            rows.append({
                "Synergy_Niche": syn,
                "Econ_Archetype": arch,
                "Best_Agent": cell.agent_name,
                "Avg_Placement": round(cell.avg_placement, 2),
                "Top4_Rate_%": round(cell.top4_rate * 100.0, 1),
                "Win_Rate_%": round(cell.win_rate * 100.0, 1),
                "Signature_Board": ", ".join(cell.sample_board_pokemon[:6]),
            })
        return pd.DataFrame(rows)


class CommunityAgent:
    """Single RL agent member of the Pokemon Auto Chess Community League."""

    def __init__(self, config: RewardWeightConfig, hidden_dim: int = 128, lr: float = 1.8e-3, seed: int = 42):
        torch.manual_seed(seed)
        self.config = config
        self.model = RewardConditionedMaskedActorCritic(hidden_dim=hidden_dim)
        self.trainer = MaskablePPOTrainer(self.model, lr=lr)
        self.history: List[AgentMetrics] = []
        self.last_board_names: List[str] = []


def build_default_community(seed: int = 42) -> List[CommunityAgent]:
    """
    Creates the 8 diverse RL personas spanning Economic Archetypes, Win/Lose Streak
    specialists, and Type/Synergy Specialists.
    """
    configs = [
        RewardWeightConfig(
            name="Econ_50g_Fast8_Compounder",
            archetype="STANDARD_ECON_50G",
            target_synergy=None,
            w_placement=1.20,
            w_interest=0.90,
            w_win_streak=0.25,
            w_lose_streak=0.0,
            w_reroll_evo=0.25,
            w_synergy_focus=0.45,
            w_hp_preservation=0.35,
            w_potential_shaping=0.25,
        ),
        RewardWeightConfig(
            name="Tempo_WinStreak_Snowballer",
            archetype="TEMPO_WIN_STREAK",
            target_synergy=None,
            w_placement=1.10,
            w_interest=0.20,
            w_win_streak=0.95,
            w_lose_streak=0.0,
            w_reroll_evo=0.40,
            w_synergy_focus=0.50,
            w_hp_preservation=0.65,
            w_potential_shaping=0.25,
        ),
        RewardWeightConfig(
            name="HyperRoll_3Star_Collector",
            archetype="HYPER_ROLL",
            target_synergy="NORMAL",
            w_placement=1.00,
            w_interest=0.05,
            w_win_streak=0.35,
            w_lose_streak=0.0,
            w_reroll_evo=0.95,
            w_synergy_focus=0.55,
            w_hp_preservation=0.30,
            w_potential_shaping=0.20,
        ),
        RewardWeightConfig(
            name="LoseStreak_Cashout_Pivot",
            archetype="LOSE_STREAK_CASHOUT",
            target_synergy="ARTIFICIAL",
            w_placement=1.10,
            w_interest=0.75,
            w_win_streak=0.20,
            w_lose_streak=0.85,
            w_reroll_evo=0.45,
            w_synergy_focus=0.45,
            w_hp_preservation=0.10,
            w_potential_shaping=0.20,
        ),
        RewardWeightConfig(
            name="Fire_Blaze_Aggro_Specialist",
            archetype="TEMPO_WIN_STREAK",
            target_synergy="FIRE",
            w_placement=1.10,
            w_interest=0.30,
            w_win_streak=0.75,
            w_lose_streak=0.0,
            w_reroll_evo=0.45,
            w_synergy_focus=0.90,
            w_hp_preservation=0.45,
            w_potential_shaping=0.25,
        ),
        RewardWeightConfig(
            name="Water_Mana_Control_Specialist",
            archetype="STANDARD_ECON_50G",
            target_synergy="WATER",
            w_placement=1.15,
            w_interest=0.70,
            w_win_streak=0.35,
            w_lose_streak=0.0,
            w_reroll_evo=0.40,
            w_synergy_focus=0.90,
            w_hp_preservation=0.40,
            w_potential_shaping=0.25,
        ),
        RewardWeightConfig(
            name="Fighting_Steel_Bastion_Specialist",
            archetype="STANDARD_ECON_50G",
            target_synergy="FIGHTING",
            w_placement=1.15,
            w_interest=0.55,
            w_win_streak=0.55,
            w_lose_streak=0.0,
            w_reroll_evo=0.50,
            w_synergy_focus=0.88,
            w_hp_preservation=0.60,
            w_potential_shaping=0.25,
        ),
        RewardWeightConfig(
            name="Dark_Assassin_Crit_Specialist",
            archetype="HYPER_ROLL",
            target_synergy="DARK",
            w_placement=1.05,
            w_interest=0.25,
            w_win_streak=0.45,
            w_lose_streak=0.0,
            w_reroll_evo=0.80,
            w_synergy_focus=0.92,
            w_hp_preservation=0.30,
            w_potential_shaping=0.20,
        ),
    ]
    return [CommunityAgent(cfg, seed=seed + i * 13) for i, cfg in enumerate(configs)]


class CommunityLeagueTrainer:
    """
    Coordinates multi-generation Maskable PPO training, MAP-Elites archiving, and
    Population-Based Training (PBT) across the community of RL agents.
    """

    def __init__(
        self,
        repo: PACDataRepository,
        sim: FastCombatSimulator,
        agents: Optional[List[CommunityAgent]] = None,
        seed: int = 42,
    ):
        self.repo = repo
        self.sim = sim
        self.agents = agents or build_default_community(seed=seed)
        self.env = PokemonAutoChessGymEnv(repo=repo, sim=sim, seed=seed)
        self.archive = MAPElitesArchive()
        self.rng = np.random.default_rng(seed)

    def train_and_evaluate_generation(
        self,
        generation: int,
        train_episodes_per_agent: int = 8,
        eval_episodes_per_agent: int = 8,
    ) -> List[AgentMetrics]:
        gen_metrics: List[AgentMetrics] = []

        for ag_idx, agent in enumerate(self.agents):
            self.env.set_reward_config(agent.config)
            rollout_buffer: List[RolloutStep] = []

            # 1. Collect on-policy rollouts with Maskable PPO
            agent.model.eval()
            for ep in range(train_episodes_per_agent):
                ep_seed = int(self.rng.integers(1, 1_000_000))
                obs, _ = self.env.reset(seed=ep_seed)
                done = False
                while not done:
                    mask = self.env.action_masks()
                    with torch.no_grad():
                        t_obs = torch.from_numpy(obs).unsqueeze(0)
                        t_mask = torch.from_numpy(mask).unsqueeze(0)
                        act_t, logp_t, _, val_t, _ = agent.model.get_action_and_value(
                            t_obs, t_mask, deterministic=False
                        )
                        action = int(act_t.item())
                        log_prob = float(logp_t.item())
                        value = float(val_t.item())

                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    rollout_buffer.append(
                        RolloutStep(
                            obs=obs,
                            mask=mask,
                            action=action,
                            log_prob=log_prob,
                            reward=reward,
                            done=done,
                            value=value,
                        )
                    )
                    obs = next_obs

            # 2. PPO policy & value network update
            agent.trainer.update(rollout_buffer, ppo_epochs=4, batch_size=64)

            # 3. Evaluate agent in 8-player lobbies on True Placement (1..8)
            placements: List[int] = []
            stages: List[int] = []
            evos_2: List[int] = []
            evos_3: List[int] = []
            shaped_returns: List[float] = []
            syn_counter: Dict[str, int] = {}

            agent.model.eval()
            for ep in range(eval_episodes_per_agent):
                ep_seed = 50000 + generation * 1000 + ag_idx * 50 + ep
                obs, _ = self.env.reset(seed=ep_seed)
                done = False
                ep_ret = 0.0
                info: Dict[str, Any] = {}
                while not done:
                    mask = self.env.action_masks()
                    with torch.no_grad():
                        t_obs = torch.from_numpy(obs).unsqueeze(0)
                        t_mask = torch.from_numpy(mask).unsqueeze(0)
                        act_t, _, _, _, _ = agent.model.get_action_and_value(
                            t_obs, t_mask, deterministic=True
                        )
                        action = int(act_t.item())
                    obs, r, terminated, truncated, info = self.env.step(action)
                    ep_ret += r
                    done = terminated or truncated

                placements.append(int(info.get("placement", 5)))
                stages.append(int(info.get("final_stage", self.env.stage_level)))
                evos_2.append(int(info.get("evolutions_2star", 0)))
                evos_3.append(int(info.get("evolutions_3star", 0)))
                shaped_returns.append(ep_ret)
                agent.last_board_names = [f"{u.name}({u.stars}*)" for u in self.env.board]

                active_syns = info.get("active_synergies", {})
                for s_name, s_tier in active_syns.items():
                    if s_tier > 0:
                        syn_counter[s_name] = syn_counter.get(s_name, 0) + 1

            avg_place = float(np.mean(placements))
            top4 = float(np.mean([1.0 if p <= 4 else 0.0 for p in placements]))
            win_r = float(np.mean([1.0 if p == 1 else 0.0 for p in placements]))

            dominant_syn = (
                agent.config.target_synergy
                if agent.config.target_synergy
                else (max(syn_counter.items(), key=lambda kv: kv[1])[0] if syn_counter else "FLEX_ALL")
            )

            metrics = AgentMetrics(
                generation=generation,
                agent_name=agent.config.name,
                archetype=agent.config.archetype,
                target_synergy=dominant_syn,
                avg_placement=avg_place,
                top4_rate=top4,
                win_rate=win_r,
                avg_stage_reached=float(np.mean(stages)),
                avg_2star_evos=float(np.mean(evos_2)),
                avg_3star_evos=float(np.mean(evos_3)),
                avg_shaped_return=float(np.mean(shaped_returns)),
                dominant_synergies=syn_counter,
            )
            agent.history.append(metrics)
            gen_metrics.append(metrics)

            # Update MAP-Elites Archive
            self.archive.try_insert(
                synergy_niche=dominant_syn,
                econ_archetype=agent.config.archetype,
                agent_name=agent.config.name,
                avg_placement=avg_place,
                top4_rate=top4,
                win_rate=win_r,
                sample_board_pokemon=agent.last_board_names,
                reward_config=agent.config,
            )

        # 4. Population-Based Training (PBT) soft knowledge sharing + reward weight adaptation
        self._pbt_step(gen_metrics)
        return gen_metrics

    def _pbt_step(self, gen_metrics: List[AgentMetrics]) -> None:
        """
        Population-Based Training (Jaderberg et al., 2017):
        - Ranks agents by True Placement (`avg_placement`).
        - The bottom agent softly blends 25% of the shared state encoder trunk weights from
          the top-performing elite (preserving its own specialized persona head & target synergy)
          and adapts its `w_hp_preservation` / `w_interest` balance if it died too early.
        """
        ranked = sorted(range(len(self.agents)), key=lambda i: gen_metrics[i].avg_placement)
        best_idx = ranked[0]
        worst_idx = ranked[-1]

        best_agent = self.agents[best_idx]
        worst_agent = self.agents[worst_idx]

        with torch.no_grad():
            for p_w, p_b in zip(
                worst_agent.model.state_encoder.parameters(),
                best_agent.model.state_encoder.parameters(),
            ):
                p_w.data.mul_(0.80).add_(p_b.data, alpha=0.20)

        # Adaptive reward weight tuning based on survival diagnostics
        worst_m = gen_metrics[worst_idx]
        if worst_m.avg_stage_reached < 22.0:
            # Agent is bleeding out too early -> boost HP preservation and board tempo
            worst_agent.config.w_hp_preservation = min(0.85, worst_agent.config.w_hp_preservation + 0.10)
            worst_agent.config.w_win_streak = min(0.85, worst_agent.config.w_win_streak + 0.08)
        else:
            # Agent survives early game but gets outscaled late -> boost interest & synergy cap
            worst_agent.config.w_interest = min(0.95, worst_agent.config.w_interest + 0.08)
            worst_agent.config.w_synergy_focus = min(0.95, worst_agent.config.w_synergy_focus + 0.08)

    def run_direct_community_lobby_tournament(self, num_lobbies: int = 25) -> pd.DataFrame:
        """
        Pits all 8 trained RL Community Agents directly against each other in the SAME
        8-player lobby across `num_lobbies` full games to measure the head-to-head
        meta equilibrium!
        """
        n_agents = len(self.agents)
        placements_by_agent: Dict[str, List[int]] = {ag.config.name: [] for ag in self.agents}
        wins_by_agent: Dict[str, int] = {ag.config.name: 0 for ag in self.agents}
        top4_by_agent: Dict[str, int] = {ag.config.name: 0 for ag in self.agents}
        avg_gold_by_agent: Dict[str, List[float]] = {ag.config.name: [] for ag in self.agents}
        avg_level_by_agent: Dict[str, List[float]] = {ag.config.name: [] for ag in self.agents}
        best_dps_by_agent: Dict[str, Tuple[str, float]] = {ag.config.name: ("NONE", 0.0) for ag in self.agents}
        best_tank_by_agent: Dict[str, Tuple[str, float]] = {ag.config.name: ("NONE", 0.0) for ag in self.agents}

        # Create one dedicated env instance per agent to step their shop decisions in parallel
        envs = [
            PokemonAutoChessGymEnv(repo=self.repo, sim=self.sim, reward_config=ag.config, seed=1000 + i)
            for i, ag in enumerate(self.agents)
        ]

        for lobby_idx in range(num_lobbies):
            base_seed = 90000 + lobby_idx * 31
            obs_list = []
            for i, env in enumerate(envs):
                obs, _ = env.reset(seed=base_seed + i)
                obs_list.append(obs)

            alive = [True] * n_agents
            eliminated_stage = [40] * n_agents

            for stage in range(1, 36):
                if sum(alive) <= 1:
                    break

                # Each alive agent executes its shop phase using its trained PyTorch policy
                gold_snapshot = []
                for i, ag in enumerate(self.agents):
                    if not alive[i]:
                        continue
                    env = envs[i]
                    env.stage_level = stage
                    env.actions_this_stage = 0
                    env.locked_this_turn = False
                    env._handle_special_stage_propositions()
                    if not env.shop_locked:
                        env._roll_shop()
                    else:
                        env.shop_locked = False

                    for _ in range(env.max_actions_per_stage):
                        obs = env._get_obs()
                        mask = env.action_masks()
                        with torch.no_grad():
                            t_obs = torch.from_numpy(obs).unsqueeze(0)
                            t_mask = torch.from_numpy(mask).unsqueeze(0)
                            act_t, _, _, _, _ = ag.model.get_action_and_value(
                                t_obs, t_mask, deterministic=True
                            )
                            action = int(act_t.item())
                        if action == 21:  # ACT_END_TURN
                            break
                        # Execute only the shop sub-action without triggering internal bot combat
                        env.step(action)
                        if env.actions_this_stage >= env.max_actions_per_stage:
                            break
                    env._auto_optimize_roster()
                    gold_snapshot.append((i, env.gold))

                for idx_i, g_val in gold_snapshot:
                    avg_gold_by_agent[self.agents[idx_i].config.name].append(float(g_val))

                # Pair alive RL agents head-to-head for combat!
                alive_indices = [i for i in range(n_agents) if alive[i]]
                self.rng.shuffle(alive_indices)
                for pair_k in range(0, len(alive_indices) - 1, 2):
                    i1 = alive_indices[pair_k]
                    i2 = alive_indices[pair_k + 1]
                    e1, e2 = envs[i1], envs[i2]
                    out = self.sim.simulate(e1.board, e2.board, stage_level=stage, rng=self.rng)
                    n1 = self.agents[i1].config.name
                    n2 = self.agents[i2].config.name
                    for ulog in out.unit_logs_a:
                        label = f"{ulog.name}({ulog.stars}*)"
                        if ulog.damage_dealt > best_dps_by_agent[n1][1]:
                            best_dps_by_agent[n1] = (label, round(ulog.damage_dealt, 1))
                        if ulog.damage_tanked > best_tank_by_agent[n1][1]:
                            best_tank_by_agent[n1] = (label, round(ulog.damage_tanked, 1))
                    for ulog in out.unit_logs_b:
                        label = f"{ulog.name}({ulog.stars}*)"
                        if ulog.damage_dealt > best_dps_by_agent[n2][1]:
                            best_dps_by_agent[n2] = (label, round(ulog.damage_dealt, 1))
                        if ulog.damage_tanked > best_tank_by_agent[n2][1]:
                            best_tank_by_agent[n2] = (label, round(ulog.damage_tanked, 1))

                    if out.winner >= 0:
                        e1.gold += 1
                        e1.win_streak += 1
                        e1.lose_streak = 0
                        e2.hp = max(0, e2.hp - out.damage_to_loser)
                        e2.lose_streak += 1
                        e2.win_streak = 0
                        if e2.hp == 0:
                            alive[i2] = False
                            eliminated_stage[i2] = stage
                    else:
                        e2.gold += 1
                        e2.win_streak += 1
                        e2.lose_streak = 0
                        e1.hp = max(0, e1.hp - out.damage_to_loser)
                        e1.lose_streak += 1
                        e1.win_streak = 0
                        if e1.hp == 0:
                            alive[i1] = False
                            eliminated_stage[i1] = stage

                # End-of-stage income & +2 XP for all alive agents
                for i in alive_indices:
                    if not alive[i]:
                        continue
                    e = envs[i]
                    interest = min(5, e.gold // 10)
                    streak = min(5, max(e.win_streak, e.lose_streak))
                    e.gold += 5 + interest + streak
                    if e.level < 9:
                        e.xp += 2
                        from .data_loader import EXP_TABLE
                        while e.level < 9 and e.xp >= EXP_TABLE[e.level]:
                            e.xp -= EXP_TABLE[e.level]
                            e.level += 1

            # Rank all 8 agents in this lobby by (alive, eliminated_stage, remaining HP)
            order = sorted(
                range(n_agents),
                key=lambda i: (1 if alive[i] else 0, envs[i].hp, eliminated_stage[i]),
                reverse=True,
            )
            for rank_zero, ag_i in enumerate(order):
                place = rank_zero + 1
                name = self.agents[ag_i].config.name
                placements_by_agent[name].append(place)
                avg_level_by_agent[name].append(float(envs[ag_i].level))
                if place == 1:
                    wins_by_agent[name] += 1
                if place <= 4:
                    top4_by_agent[name] += 1

        rows = []
        for ag in self.agents:
            name = ag.config.name
            places = placements_by_agent[name]
            dps_name, dps_val = best_dps_by_agent[name]
            tank_name, tank_val = best_tank_by_agent[name]
            rows.append({
                "Agent_Name": name,
                "Archetype": ag.config.archetype,
                "Target_Synergy": ag.config.target_synergy or "FLEX",
                "Avg_Placement": round(float(np.mean(places)), 2),
                "Top4_Rate_%": round(100.0 * top4_by_agent[name] / max(1, num_lobbies), 1),
                "Win_Rate_%": round(100.0 * wins_by_agent[name] / max(1, num_lobbies), 1),
                "Avg_Bank_Gold": round(float(np.mean(avg_gold_by_agent[name])), 1),
                "Avg_Final_Level": round(float(np.mean(avg_level_by_agent[name])), 2),
                "Record_DPS_Unit": f"{dps_name} ({dps_val:.0f} dmg)",
                "Record_Tank_Unit": f"{tank_name} ({tank_val:.0f} tank)",
            })
        df = pd.DataFrame(rows).sort_values("Avg_Placement", ascending=True).reset_index(drop=True)
        return df


def analyze_botv2_meta(repo: PACDataRepository, sim: FastCombatSimulator, num_duels: int = 600) -> pd.DataFrame:
    """
    Runs a statistical meta-analysis over the 726 human-authored community bots in
    `upstream/db-commands/botv2.json` at late-game Stage 25 to discover which synergies
    have the highest empirical win-rate and ELO representation in PAC.
    """
    rng = np.random.default_rng(42)
    bots = repo.bots
    if not bots:
        return pd.DataFrame()

    synergy_appearances: Dict[str, int] = {}
    synergy_wins: Dict[str, int] = {}
    synergy_elos: Dict[str, List[float]] = {}

    # 1. Catalog all 726 bots at Stage 25
    bot_boards = []
    for b in bots:
        entries = repo.get_bot_stage_board(b, 25)
        from .combat_surrogate import BoardUnit, TacticalPositioner
        units = []
        for name, x, y, n_items in entries[:9]:
            p = repo.get_pokemon(name)
            if p is not None:
                units.append(BoardUnit.from_pokemon_stats(p, x=min(5, max(0, x)), y=min(3, max(0, y - 1)), num_items=n_items))
        if units:
            units = TacticalPositioner.optimize_positions(units)
            _, tiers = compute_active_synergies(units)
            bot_boards.append((b, units, tiers))
            for syn, t_lvl in tiers.items():
                if t_lvl >= 2:
                    synergy_elos.setdefault(syn, []).append(b.elo)

    # 2. Simulate `num_duels` late-game duels between random bot pairs
    for _ in range(num_duels):
        i1, i2 = rng.choice(len(bot_boards), size=2, replace=False)
        b1, u1, t1 = bot_boards[int(i1)]
        b2, u2, t2 = bot_boards[int(i2)]
        out = sim.simulate(u1, u2, stage_level=25, rng=rng)
        for syn, lvl in t1.items():
            if lvl >= 2:
                synergy_appearances[syn] = synergy_appearances.get(syn, 0) + 1
                if out.winner >= 0:
                    synergy_wins[syn] = synergy_wins.get(syn, 0) + 1
        for syn, lvl in t2.items():
            if lvl >= 2:
                synergy_appearances[syn] = synergy_appearances.get(syn, 0) + 1
                if out.winner < 0:
                    synergy_wins[syn] = synergy_wins.get(syn, 0) + 1

    rows = []
    for syn, apps in synergy_appearances.items():
        if apps < 12:
            continue
        wins = synergy_wins.get(syn, 0)
        elos = synergy_elos.get(syn, [1100.0])
        rows.append({
            "Synergy": syn,
            "Sample_Matches": apps,
            "Combat_Win_Rate_%": round(100.0 * wins / apps, 1),
            "Avg_Bot_ELO": round(float(np.mean(elos)), 0),
            "Num_Bots_In_DB": len(elos),
        })

    df = pd.DataFrame(rows).sort_values("Combat_Win_Rate_%", ascending=False).reset_index(drop=True)
    return df


def analyze_composition_damage_and_tank_records(
    repo: PACDataRepository,
    sim: FastCombatSimulator,
    num_duels: int = 1200,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simulates `num_duels` full late-game battles across the 726 compositions in `botv2.json`
    + synthetic synergy-specialized boards to record:
    1. `comp_records_df`: For every one of the 31 PAC Synergies/Compositions:
       - Recordista de Dano (Best Carry): Pokemon, Peak Damage in 1 Fight, Avg Damage/Fight, Second Best Carry
       - Recordista de Tank (Best Frontline): Pokemon, Peak Damage Tanked in 1 Fight, Avg Tanked/Fight, Second Best Tank
    2. `top_dps_global_df`: Top 20 Highest-Damage Pokemon overall (Peak & Average DPS)
    3. `top_tank_global_df`: Top 20 Highest-Tank Pokemon overall (Peak & Average Damage Absorbed/Mitigated)
    """
    from .combat_surrogate import BoardUnit, TacticalPositioner
    from .data_loader import ALL_SYNERGIES

    rng = np.random.default_rng(777)
    bots = repo.bots
    bot_boards = []
    for b in bots:
        entries = repo.get_bot_stage_board(b, 25)
        units = []
        for name, x, y, n_items in entries[:9]:
            p = repo.get_pokemon(name)
            if p is not None:
                units.append(BoardUnit.from_pokemon_stats(p, x=min(5, max(0, x)), y=min(3, max(0, y - 1)), num_items=n_items))
        if units:
            units = TacticalPositioner.optimize_positions(units)
            _, tiers = compute_active_synergies(units)
            bot_boards.append((units, tiers))

    # Per-synergy per-unit stats: syn -> unit_label -> {"dmg": [..], "tank": [..], "spells": [..], "kills": [..], "rarity": str}
    syn_unit_stats: Dict[str, Dict[str, Dict[str, Any]]] = {s: {} for s in ALL_SYNERGIES}
    global_unit_stats: Dict[str, Dict[str, Any]] = {}

    def record_logs(logs, active_tiers):
        # Active synergies on this board (tier >= 1)
        board_syns = [s for s, t in active_tiers.items() if t >= 1]
        for ulog in logs:
            label = f"{ulog.name}({ulog.stars}*)"
            g_entry = global_unit_stats.setdefault(
                label,
                {"name": ulog.name, "stars": ulog.stars, "rarity": ulog.rarity, "types": "/".join(ulog.types), "dmg": [], "tank": [], "spells": [], "kills": []},
            )
            g_entry["dmg"].append(ulog.damage_dealt)
            g_entry["tank"].append(ulog.damage_tanked)
            g_entry["spells"].append(ulog.spells_cast)
            g_entry["kills"].append(ulog.kills)

            # Attribute this unit's performance to each synergy it belongs to that is active on the board
            for t in ulog.types:
                if t in syn_unit_stats and (t in board_syns or len(board_syns) == 0):
                    s_entry = syn_unit_stats[t].setdefault(
                        label,
                        {"name": ulog.name, "stars": ulog.stars, "rarity": ulog.rarity, "dmg": [], "tank": [], "spells": [], "kills": []},
                    )
                    s_entry["dmg"].append(ulog.damage_dealt)
                    s_entry["tank"].append(ulog.damage_tanked)
                    s_entry["spells"].append(ulog.spells_cast)
                    s_entry["kills"].append(ulog.kills)

    for _ in range(num_duels):
        i1, i2 = rng.choice(len(bot_boards), size=2, replace=False)
        u1, t1 = bot_boards[int(i1)]
        u2, t2 = bot_boards[int(i2)]
        out = sim.simulate(u1, u2, stage_level=25, rng=rng)
        record_logs(out.unit_logs_a, t1)
        record_logs(out.unit_logs_b, t2)

    comp_rows = []
    for syn in ALL_SYNERGIES:
        u_map = syn_unit_stats.get(syn, {})
        if not u_map:
            continue
        # Filter units with at least 3 combat appearances for reliable averages, fallback to all
        qualified = {k: v for k, v in u_map.items() if len(v["dmg"]) >= 3}
        if not qualified:
            qualified = u_map

        # Sort by Peak Damage Record (and Avg Damage tie-breaker)
        by_dps = sorted(
            qualified.items(),
            key=lambda kv: (0.55 * float(np.max(kv[1]["dmg"])) + 0.45 * float(np.mean(kv[1]["dmg"]))),
            reverse=True,
        )
        # Sort by Peak Tank Record (and Avg Tanked tie-breaker)
        by_tank = sorted(
            qualified.items(),
            key=lambda kv: (0.55 * float(np.max(kv[1]["tank"])) + 0.45 * float(np.mean(kv[1]["tank"]))),
            reverse=True,
        )

        best_dps_k, best_dps_v = by_dps[0]
        sec_dps_k, sec_dps_v = by_dps[1] if len(by_dps) > 1 else (best_dps_k, best_dps_v)

        best_tank_k, best_tank_v = by_tank[0]
        sec_tank_k, sec_tank_v = by_tank[1] if len(by_tank) > 1 else (best_tank_k, best_tank_v)

        comp_rows.append({
            "Synergy": syn,
            "Best_Carry_DPS": best_dps_k,
            "Max_Dmg_Record": round(float(np.max(best_dps_v["dmg"])), 1),
            "Avg_Dmg_Carry": round(float(np.mean(best_dps_v["dmg"])), 1),
            "Secondary_Carry": f"{sec_dps_k} (max {np.max(sec_dps_v['dmg']):.0f})",
            "Best_Frontline_Tank": best_tank_k,
            "Max_Tank_Record": round(float(np.max(best_tank_v["tank"])), 1),
            "Avg_Tank_Absorbed": round(float(np.mean(best_tank_v["tank"])), 1),
            "Secondary_Tank": f"{sec_tank_k} (max {np.max(sec_tank_v['tank']):.0f})",
        })

    comp_records_df = pd.DataFrame(comp_rows).sort_values("Max_Dmg_Record", ascending=False).reset_index(drop=True)

    global_rows = []
    for label, st in global_unit_stats.items():
        if len(st["dmg"]) < 3:
            continue
        global_rows.append({
            "Pokemon": label,
            "Rarity": st["rarity"],
            "Types": st["types"],
            "Matches": len(st["dmg"]),
            "Max_Damage_Record": round(float(np.max(st["dmg"])), 1),
            "Avg_Damage": round(float(np.mean(st["dmg"])), 1),
            "Max_Tank_Record": round(float(np.max(st["tank"])), 1),
            "Avg_Tanked": round(float(np.mean(st["tank"])), 1),
            "Avg_Spells_Cast": round(float(np.mean(st["spells"])), 2),
            "Avg_Kills": round(float(np.mean(st["kills"])), 2),
        })

    global_df = pd.DataFrame(global_rows)
    top_dps_global_df = global_df.sort_values("Max_Damage_Record", ascending=False).head(20).reset_index(drop=True)
    top_tank_global_df = global_df.sort_values("Max_Tank_Record", ascending=False).head(20).reset_index(drop=True)

    return comp_records_df, top_dps_global_df, top_tank_global_df

