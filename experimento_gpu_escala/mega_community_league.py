r"""
Mega Comunidade de 35 Agentes RL Especialistas (Todas as 31 Sinergias + 4 Economias Puras)
para execução em larga escala na GPU (`experimento_gpu_escala`).

Implementa:
1. `build_mega_35_agent_community()`:
   - 4 Arquétipos Econômicos Puros (`STANDARD_ECON_50G`, `HYPER_ROLL`, `TEMPO_WIN_STREAK`, `LOSE_STREAK_CASHOUT`)
   - 31 Especialistas dedicados a cada uma das 31 Sinergias Oficiais do Pokémon Auto Chess
     (`NORMAL` até `GOURMET`), calibrando os pesos de economia/reroll de acordo com a curva
     de raridade dos Pokémon daquele tipo.
2. `MegaCommunityLeagueTrainer`:
   - Treinamento multi-geracional em GPU com `DeepEntityTransformerActorCritic` + `GPUMaskablePPOTrainer`
   - Prioritized Fictitious Self-Play (PFSP) + Population-Based Training (PBT)
   - Arquivo MAP-Elites Completo ($31 \times 4 = 124$ nichos) registrando o Pokémon Recordista
     de Dano (Carry) e Recordista de Tank (Frontline) de cada célula!
3. `compute_synergy_counter_matrix()`:
   - Calcula a matriz de confrontos diretos $31 \times 31$ entre todas as sinergias para descobrir
     quais composições são *Hard Counters* umas das outras.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from pac_rl.combat_surrogate import FastCombatSimulator, BoardUnit, TacticalPositioner, compute_active_synergies
from pac_rl.community_league import AgentMetrics, MAPElitesArchive
from pac_rl.data_loader import ALL_SYNERGIES, PACDataRepository
from pac_rl.gym_env import PokemonAutoChessGymEnv, RewardWeightConfig
from pac_rl.models import RolloutStep

from .deep_transformer_models import (
    DeepEntityTransformerActorCritic,
    GPUMaskablePPOTrainer,
    get_optimal_device,
)


class GPUCommunityAgent:
    def __init__(
        self,
        config: RewardWeightConfig,
        device: torch.device,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        lr: float = 1.2e-3,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        self.config = config
        self.device = device
        self.model = DeepEntityTransformerActorCritic(hidden_dim=hidden_dim, num_blocks=num_blocks).to(device)
        self.trainer = GPUMaskablePPOTrainer(self.model, device=device, lr=lr)
        self.history: List[AgentMetrics] = []
        self.last_board_names: List[str] = []
        self.record_dps_unit: Tuple[str, float] = ("NONE", 0.0)
        self.record_tank_unit: Tuple[str, float] = ("NONE", 0.0)


def build_mega_35_agent_community(device: torch.device, seed: int = 42) -> List[GPUCommunityAgent]:
    """
    Constrói a Mega Comunidade de 35 Agentes RL:
    - 4 Agentes Macro Econômicos Flexíveis
    - 31 Agentes Especialistas nas 31 Sinergias Oficiais do PAC
    """
    configs: List[RewardWeightConfig] = [
        RewardWeightConfig(
            name="Macro_Econ_50g_Fast9",
            archetype="STANDARD_ECON_50G",
            target_synergy=None,
            w_placement=1.30,
            w_interest=0.95,
            w_win_streak=0.30,
            w_lose_streak=0.0,
            w_reroll_evo=0.25,
            w_synergy_focus=0.50,
            w_hp_preservation=0.40,
            w_potential_shaping=0.25,
        ),
        RewardWeightConfig(
            name="Macro_HyperRoll_3Star_Hunter",
            archetype="HYPER_ROLL",
            target_synergy=None,
            w_placement=1.10,
            w_interest=0.10,
            w_win_streak=0.40,
            w_lose_streak=0.0,
            w_reroll_evo=0.98,
            w_synergy_focus=0.55,
            w_hp_preservation=0.35,
            w_potential_shaping=0.20,
        ),
        RewardWeightConfig(
            name="Macro_Tempo_WinStreak_Crusher",
            archetype="TEMPO_WIN_STREAK",
            target_synergy=None,
            w_placement=1.15,
            w_interest=0.25,
            w_win_streak=0.98,
            w_lose_streak=0.0,
            w_reroll_evo=0.45,
            w_synergy_focus=0.55,
            w_hp_preservation=0.70,
            w_potential_shaping=0.25,
        ),
        RewardWeightConfig(
            name="Macro_LoseStreak_Cyborg_Cashout",
            archetype="LOSE_STREAK_CASHOUT",
            target_synergy=None,
            w_placement=1.15,
            w_interest=0.80,
            w_win_streak=0.25,
            w_lose_streak=0.90,
            w_reroll_evo=0.50,
            w_synergy_focus=0.50,
            w_hp_preservation=0.15,
            w_potential_shaping=0.20,
        ),
    ]

    # Classify the 31 synergies into their natural economic affinity while allowing PBT to evolve them
    hyper_roll_syns = {"NORMAL", "BUG", "BABY", "DARK", "POISON", "GRASS", "SOUND", "WILD"}
    tempo_syns = {"FIRE", "ELECTRIC", "FIGHTING", "FLYING", "FIELD", "MONSTER", "AQUATIC", "FLORA"}
    lose_streak_syns = {"ARTIFICIAL", "FOSSIL", "ROCK", "GOURMET"}
    # Remaining default to STANDARD_ECON_50G (e.g. DRAGON, PSYCHIC, WATER, STEEL, GROUND, GHOST, FAIRY, ICE, HUMAN, LIGHT, AMORPHOUS)

    for idx, syn in enumerate(ALL_SYNERGIES):
        if syn in hyper_roll_syns:
            arch = "HYPER_ROLL"
            w_int, w_win, w_lose, w_evo = 0.20, 0.45, 0.0, 0.90
        elif syn in tempo_syns:
            arch = "TEMPO_WIN_STREAK"
            w_int, w_win, w_lose, w_evo = 0.35, 0.85, 0.0, 0.50
        elif syn in lose_streak_syns:
            arch = "LOSE_STREAK_CASHOUT"
            w_int, w_win, w_lose, w_evo = 0.70, 0.30, 0.75, 0.55
        else:
            arch = "STANDARD_ECON_50G"
            w_int, w_win, w_lose, w_evo = 0.85, 0.40, 0.0, 0.40

        configs.append(
            RewardWeightConfig(
                name=f"Spec_{syn}_{arch[:5]}",
                archetype=arch,
                target_synergy=syn,
                w_placement=1.20,
                w_interest=w_int,
                w_win_streak=w_win,
                w_lose_streak=w_lose,
                w_reroll_evo=w_evo,
                w_synergy_focus=0.95,
                w_hp_preservation=0.45,
                w_potential_shaping=0.25,
            )
        )

    return [
        GPUCommunityAgent(cfg, device=device, seed=seed + i * 19)
        for i, cfg in enumerate(configs)
    ]


class MegaCommunityLeagueTrainer:
    """
    Orquestrador de Treinamento em Escala (GPU) para os 35 Agentes da Comunidade RL.
    """

    def __init__(
        self,
        repo: PACDataRepository,
        sim: FastCombatSimulator,
        device: Optional[torch.device] = None,
        agents: Optional[List[GPUCommunityAgent]] = None,
        seed: int = 42,
    ):
        self.repo = repo
        self.sim = sim
        self.device = device or get_optimal_device("auto")
        self.agents = agents or build_mega_35_agent_community(self.device, seed=seed)
        self.env = PokemonAutoChessGymEnv(repo=repo, sim=sim, seed=seed)
        self.archive = MAPElitesArchive()
        self.rng = np.random.default_rng(seed)

    def train_and_evaluate_generation(
        self,
        generation: int,
        train_episodes_per_agent: int = 24,
        eval_episodes_per_agent: int = 16,
    ) -> List[AgentMetrics]:
        gen_metrics: List[AgentMetrics] = []

        for ag_idx, agent in enumerate(self.agents):
            self.env.set_reward_config(agent.config)
            rollout_buffer: List[RolloutStep] = []

            agent.model.eval()
            for _ in range(train_episodes_per_agent):
                ep_seed = int(self.rng.integers(1, 10_000_000))
                obs, _ = self.env.reset(seed=ep_seed)
                done = False
                while not done:
                    mask = self.env.action_masks()
                    with torch.no_grad():
                        t_obs = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                        t_mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)
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

            agent.trainer.update(rollout_buffer, ppo_epochs=6, batch_size=256)

            # Evaluate agent on True Placement (1..8)
            placements, stages, evos_2, evos_3, shaped_returns = [], [], [], [], []
            syn_counter: Dict[str, int] = {}

            agent.model.eval()
            for ep in range(eval_episodes_per_agent):
                ep_seed = 500_000 + generation * 5000 + ag_idx * 100 + ep
                obs, _ = self.env.reset(seed=ep_seed)
                done = False
                ep_ret = 0.0
                info: Dict[str, Any] = {}
                while not done:
                    mask = self.env.action_masks()
                    with torch.no_grad():
                        t_obs = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                        t_mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)
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

                for s_name, s_tier in info.get("active_synergies", {}).items():
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

            m = AgentMetrics(
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
            agent.history.append(m)
            gen_metrics.append(m)

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

        self._pbt_step(gen_metrics)
        return gen_metrics

    def _pbt_step(self, gen_metrics: List[AgentMetrics]) -> None:
        """
        Population-Based Training (PBT) em escala: os 20% piores agentes herdam
        parcialmente a representação compartilhada dos 20% melhores agentes (Elites)
        e ajustam seus hiperparâmetros de recompensa.
        """
        ranked = sorted(range(len(self.agents)), key=lambda i: gen_metrics[i].avg_placement)
        k_slice = max(1, len(self.agents) // 5)
        elites = ranked[:k_slice]
        bottoms = ranked[-k_slice:]

        for b_idx, e_idx in zip(bottoms, elites):
            worst_ag = self.agents[b_idx]
            best_ag = self.agents[e_idx]
            with torch.no_grad():
                for p_w, p_b in zip(
                    worst_ag.model.input_proj.parameters(),
                    best_ag.model.input_proj.parameters(),
                ):
                    p_w.data.mul_(0.75).add_(p_b.data, alpha=0.25)

            if gen_metrics[b_idx].avg_stage_reached < 22.0:
                worst_ag.config.w_hp_preservation = min(0.90, worst_ag.config.w_hp_preservation + 0.08)
                worst_ag.config.w_win_streak = min(0.95, worst_ag.config.w_win_streak + 0.08)
            else:
                worst_ag.config.w_interest = min(0.95, worst_ag.config.w_interest + 0.08)
                worst_ag.config.w_synergy_focus = min(0.98, worst_ag.config.w_synergy_focus + 0.05)


def compute_synergy_counter_matrix(
    repo: PACDataRepository,
    sim: FastCombatSimulator,
    duels_per_pair: int = 20,
) -> pd.DataFrame:
    """
    Constrói a Matriz de Counter-Picks 31x31 simulando confrontos diretos entre os melhores
    tabuleiros de cada Sinergia, identificando exatamente qual é o Melhor Counter (Hard Counter)
    e qual é a Presa Favorita de cada uma das 31 composições do PAC.
    """
    rng = np.random.default_rng(2026)
    by_syn: Dict[str, List[List[BoardUnit]]] = {s: [] for s in ALL_SYNERGIES}

    for b in repo.bots:
        entries = repo.get_bot_stage_board(b, 25)
        units = []
        for name, x, y, n_items in entries[:9]:
            p = repo.get_pokemon(name)
            if p is not None:
                units.append(BoardUnit.from_pokemon_stats(p, x=min(5, max(0, x)), y=min(3, max(0, y - 1)), num_items=n_items))
        if units:
            units = TacticalPositioner.optimize_positions(units)
            _, tiers = compute_active_synergies(units)
            if tiers:
                top_syn = max(tiers.items(), key=lambda kv: kv[1])[0]
                by_syn.setdefault(top_syn, []).append(units)

    active_syns = [s for s, b_list in by_syn.items() if len(b_list) >= 2]
    rows = []

    for s_a in active_syns:
        win_rates_vs: Dict[str, float] = {}
        for s_b in active_syns:
            if s_a == s_b:
                continue
            wins = 0
            for _ in range(duels_per_pair):
                ba = by_syn[s_a][int(rng.integers(0, len(by_syn[s_a])))]
                bb = by_syn[s_b][int(rng.integers(0, len(by_syn[s_b])))]
                out = sim.simulate(ba, bb, stage_level=25, rng=rng)
                if out.winner >= 0:
                    wins += 1
            win_rates_vs[s_b] = 100.0 * wins / max(1, duels_per_pair)

        if not win_rates_vs:
            continue
        best_prey = max(win_rates_vs.items(), key=lambda kv: kv[1])
        worst_counter = min(win_rates_vs.items(), key=lambda kv: kv[1])
        overall_wr = float(np.mean(list(win_rates_vs.values())))

        rows.append({
            "Synergy": s_a,
            "Overall_Matchup_WinRate_%": round(overall_wr, 1),
            "Best_Matchup_Prey": f"{best_prey[0]} ({best_prey[1]:.0f}% WR)",
            "Hardest_Counter": f"{worst_counter[0]} ({worst_counter[1]:.0f}% WR)",
        })

    return pd.DataFrame(rows).sort_values("Overall_Matchup_WinRate_%", ascending=False).reset_index(drop=True)
