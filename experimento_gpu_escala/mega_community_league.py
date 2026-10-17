r"""
Liga de Self-Play Multi-Agente com Baralho Compartilhado (`SharedCardPool`),
Planejamento Lookahead Híbrido (`NeuralLookaheadRosterPlanner`) e Comunidade Interna de
Modelos (`CommunityOfModelsBrain`) para o Pokémon Auto Chess (`experimento_gpu_escala`).

Elimina TODOS os atalhos do experimento CPU:
1. **Self-Play Real em Lobby de 8 Agentes durante o Treinamento**:
   - Os 8 assentos da partida de treino são ocupados simultaneamente por agentes da Liga
     (amostrados via Prioritized Fictitious Self-Play — PFSP).
   - Todos os 8 jogadores disputam o MESMO baralho finito (`SharedCardPool`). Quando um
     jogador compra um Pokémon, diminui a chance dos outros 7 acharem aquela peça; quando
     vende ou morre, as cópias voltam para o baralho!
2. **Cada Jogador é uma Comunidade de 7 Modelos Especialistas (`CommunityOfModelsBrain`)**:
   - Registra a telemetria de votação $\alpha_1(s)..\alpha_7(s)$ no Early, Mid e Late Game.
3. **Busca Clássica + SetTransformer Conectados em Tempo Real (`NeuralLookaheadRosterPlanner`)**:
   - O `DeepSetTransformerSurrogate` é consultado na GPU durante a fase de loja e antes do
     combate para avaliar trocas entre banco/tabuleiro e o ganho marginal $\Delta \widehat{P}_{\text{win}}$
     de cada carta da loja contra os tabuleiros reais dos inimigos vivos no lobby!
4. **Zero Regras Manuais (`Zero Hand-Coded Priors`)**:
   - As árvores de decisão extraídas pelo VIPER no final refletem 100% o que foi descoberto
     pelo Aprendizado por Reforço + Busca Lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
import torch

from pac_rl.combat_surrogate import (
    BoardUnit,
    FastCombatSimulator,
    TacticalPositioner,
    compute_active_synergies,
    encode_synergy_vector,
)
from pac_rl.community_league import AgentMetrics, MAPElitesArchive
from pac_rl.data_loader import (
    ADDITIONAL_PICKS_STAGES,
    ALL_SYNERGIES,
    EXP_TABLE,
    MAX_LEVEL,
    NUM_SYNERGIES,
    PVE_STAGES,
    SHOP_SIZE,
    SYNERGY_THRESHOLDS,
    PACDataRepository,
    PokemonSpec,
)
from pac_rl.gym_env import (
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
    RewardWeightConfig,
)
from pac_rl.viper_distiller import DistilledHeuristicReport, MACRO_ACTION_NAMES, map_to_macro_action

from .deep_transformer_models import (
    EXPERT_NAMES,
    GPU_OBS_DIM,
     NUM_EXPERTS,
    CommunityOfModelsBrain,
    DeepSetTransformerSurrogate,
    GPUMaskablePPOTrainer,
    GPURolloutStep,
    NeuralLookaheadRosterPlanner,
    get_optimal_device,
)
from .reward_engine import MultiScaleRewardEngine, PLACEMENT_UTILITY
from .shared_pool_and_scouting import (
    HypergeometricPoolTracker,
    LobbyScoutingAggregator,
    LobbyScoutingReport,
    SharedCardPool,
)


GPU_FEATURE_NAMES = [
    "stage_level",
    "player_hp",
    "gold",
    "interest_tier",
    "player_level",
    "xp_to_next_level",
    "win_streak",
    "lose_streak",
    "board_size",
    "free_board_slots",
    "bench_size",
    "relative_board_power",
    "shop_pair_matches",
    "shop_synergy_matches",
    "num_2star_units",
    "num_3star_units",
    "max_synergy_tier",
    "alive_opponents",
    "surrogate_win_prob_vs_lobby",
    "max_shop_lookahead_gain",
    "pair_upgrade_prob_in_pool",
    "target_synergy_contestedness",
]


@dataclass
class SeatPlayerState:
    """Estado completo de 1 dos 8 jogadores vivos dentro do Lobby Multi-Agente Compartilhado."""
    seat_idx: int
    agent_idx: int
    hp: int = 100
    gold: int = 6
    level: int = 2
    xp: int = 0
    win_streak: int = 0
    lose_streak: int = 0
    alive: bool = True
    eliminated_stage: int = 40
    shop_locked: bool = False
    locked_this_turn: bool = False
    actions_this_stage: int = 0
    evolutions_2star: int = 0
    evolutions_3star: int = 0
    items_held: int = 0
    phi_prev: float = 0.0
    last_surrogate_win_prob: float = 0.5
    last_shop_deltas: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    board: List[BoardUnit] = field(default_factory=list)
    bench: List[BoardUnit] = field(default_factory=list)
    shop: List[Optional[PokemonSpec]] = field(default_factory=lambda: [None] * SHOP_SIZE)
    trajectory: List[GPURolloutStep] = field(default_factory=list)


class GPUCommunityAgent:
    """
    Um Jogador da Liga Comunitária cujo cérebro é uma Comunidade de 7 Modelos Especialistas
    (`CommunityOfModelsBrain`) + Planejador Clássico Lookahead (`NeuralLookaheadRosterPlanner`).
    """

    def __init__(
        self,
        config: RewardWeightConfig,
        device: torch.device,
        expert_dim: int = 128,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        self.config = config
        self.device = device
        self.model = CommunityOfModelsBrain(
            obs_dim=GPU_OBS_DIM,
            num_actions=NUM_ACTIONS,
            expert_dim=expert_dim,
            nhead=4,
        ).to(device)
        self.trainer = GPUMaskablePPOTrainer(self.model, device=device, lr=lr)
        self.history: List[AgentMetrics] = []
        self.last_board_names: List[str] = []
        self.record_dps_unit: Tuple[str, float] = ("NONE", 0.0)
        self.record_tank_unit: Tuple[str, float] = ("NONE", 0.0)
        self.elo_rating: float = 1200.0
        # Telemetria de ativação dos 7 Especialistas por fase da partida: "early", "mid", "late"
        self.expert_votes_by_phase: Dict[str, List[np.ndarray]] = {
            "early": [],
            "mid": [],
            "late": [],
        }


def build_mega_35_agent_community(device: torch.device, seed: int = 42) -> List[GPUCommunityAgent]:
    """
    Constrói os 35 Agentes da Liga (4 Arquétipos Econômicos Puros + 31 Especialistas das
    31 Sinergias Oficiais do PAC), cada um governado por um `CommunityOfModelsBrain`.
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
            w_reroll_evo=0.30,
            w_synergy_focus=0.55,
            w_hp_preservation=0.45,
            w_potential_shaping=0.30,
        ),
        RewardWeightConfig(
            name="Macro_HyperRoll_3Star_Hunter",
            archetype="HYPER_ROLL",
            target_synergy=None,
            w_placement=1.15,
            w_interest=0.10,
            w_win_streak=0.45,
            w_lose_streak=0.0,
            w_reroll_evo=0.98,
            w_synergy_focus=0.60,
            w_hp_preservation=0.35,
            w_potential_shaping=0.30,
        ),
        RewardWeightConfig(
            name="Macro_Tempo_WinStreak_Crusher",
            archetype="TEMPO_WIN_STREAK",
            target_synergy=None,
            w_placement=1.20,
            w_interest=0.25,
            w_win_streak=0.98,
            w_lose_streak=0.0,
            w_reroll_evo=0.45,
            w_synergy_focus=0.55,
            w_hp_preservation=0.75,
            w_potential_shaping=0.30,
        ),
        RewardWeightConfig(
            name="Macro_LoseStreak_Cyborg_Cashout",
            archetype="LOSE_STREAK_CASHOUT",
            target_synergy=None,
            w_placement=1.20,
            w_interest=0.80,
            w_win_streak=0.30,
            w_lose_streak=0.95,
            w_reroll_evo=0.55,
            w_synergy_focus=0.55,
            w_hp_preservation=0.20,
            w_potential_shaping=0.30,
        ),
    ]

    hyper_roll_syns = {"NORMAL", "BUG", "BABY", "DARK", "POISON", "GRASS", "SOUND", "WILD"}
    tempo_syns = {"FIRE", "ELECTRIC", "FIGHTING", "FLYING", "FIELD", "MONSTER", "AQUATIC", "FLORA"}
    lose_streak_syns = {"ARTIFICIAL", "FOSSIL", "ROCK", "GOURMET"}

    for syn in ALL_SYNERGIES:
        if syn in hyper_roll_syns:
            arch = "HYPER_ROLL"
            w_int, w_win, w_lose, w_evo = 0.20, 0.45, 0.0, 0.92
        elif syn in tempo_syns:
            arch = "TEMPO_WIN_STREAK"
            w_int, w_win, w_lose, w_evo = 0.35, 0.88, 0.0, 0.50
        elif syn in lose_streak_syns:
            arch = "LOSE_STREAK_CASHOUT"
            w_int, w_win, w_lose, w_evo = 0.72, 0.30, 0.80, 0.55
        else:
            arch = "STANDARD_ECON_50G"
            w_int, w_win, w_lose, w_evo = 0.88, 0.40, 0.0, 0.42

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
                w_potential_shaping=0.30,
            )
        )

    return [
        GPUCommunityAgent(cfg, device=device, seed=seed + i * 19)
        for i, cfg in enumerate(configs)
    ]


class MegaCommunityLeagueTrainer:
    """
    Treinador Multi-Agente Real com Self-Play em Baralho Compartilhado (`SharedCardPool`),
    Busca Clássica Lookahead (`NeuralLookaheadRosterPlanner`) e Motor de Recompensa Multi-Escala.
    """

    def __init__(
        self,
        repo: PACDataRepository,
        sim: FastCombatSimulator,
        surrogate: DeepSetTransformerSurrogate,
        device: Optional[torch.device] = None,
        agents: Optional[List[GPUCommunityAgent]] = None,
        max_actions_per_stage: int = 10,
        seed: int = 42,
    ):
        self.repo = repo
        self.sim = sim
        self.device = device or get_optimal_device("auto")
        self.surrogate = surrogate.to(self.device)
        self.planner = NeuralLookaheadRosterPlanner(self.surrogate, self.device)
        self.reward_engine = MultiScaleRewardEngine(gamma=0.995)
        self.shared_pool = SharedCardPool(repo)
        self.agents = agents or build_mega_35_agent_community(self.device, seed=seed)
        self.archive = MAPElitesArchive()
        self.max_actions_per_stage = max_actions_per_stage
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------------
    # Utilitários de Mecânica do Jogo no Lobby Compartilhado de 8 Jogadores
    # ------------------------------------------------------------------------

    def _count_copies(self, seat: SeatPlayerState, poke_name: str) -> int:
        return sum(1 for u in (seat.board + seat.bench) if u.name == poke_name)

    def _synergy_fit(self, seat: SeatPlayerState, config: RewardWeightConfig, p: PokemonSpec) -> float:
        counts, _ = compute_active_synergies(seat.board)
        score = 0.0
        for t in p.types:
            c = counts.get(t, 0)
            score += 0.35 * (c + 1)
            for th in SYNERGY_THRESHOLDS.get(t, []):
                if c + 1 == th:
                    score += 1.5
            if config.target_synergy == t:
                score += 2.0
        return score

    def _try_add_or_evolve(self, seat: SeatPlayerState, p: PokemonSpec) -> Tuple[bool, int]:
        new_u = BoardUnit.from_pokemon_stats(p, num_items=0)
        if len(seat.board) < seat.level:
            seat.board.append(new_u)
            TacticalPositioner.optimize_positions(seat.board)
        elif len(seat.bench) < 6:
            seat.bench.append(new_u)
        elif self._count_copies(seat, p.name) >= 2:
            seat.bench.append(new_u)
        else:
            return False, 0

        self.shared_pool.take_from_pool(p.name, copies=1)
        evo_tier = self._check_evolutions(seat)
        return True, evo_tier

    def _check_evolutions(self, seat: SeatPlayerState) -> int:
        highest_evo = 0
        changed = True
        while changed:
            changed = False
            all_u = seat.board + seat.bench
            by_name: Dict[str, List[BoardUnit]] = {}
            for u in all_u:
                by_name.setdefault(u.name, []).append(u)

            for name, group in by_name.items():
                if len(group) >= 3:
                    p_stats = self.repo.get_pokemon(name)
                    if p_stats is None or not p_stats.evolves_into:
                        continue
                    evo_stats = self.repo.get_pokemon(p_stats.evolves_into)
                    if evo_stats is None:
                        continue

                    to_remove = group[:3]
                    items_sum = sum(u.num_items for u in to_remove)
                    was_on_board = any(u in seat.board for u in to_remove)
                    for r_u in to_remove:
                        if r_u in seat.bench:
                            seat.bench.remove(r_u)
                        elif r_u in seat.board:
                            seat.board.remove(r_u)

                    evolved = BoardUnit.from_pokemon_stats(evo_stats, num_items=min(3, items_sum))
                    if was_on_board or len(seat.board) < seat.level:
                        seat.board.append(evolved)
                    else:
                        seat.bench.append(evolved)

                    if evolved.stars == 2:
                        seat.evolutions_2star += 1
                        highest_evo = max(highest_evo, 2)
                    elif evolved.stars >= 3:
                        seat.evolutions_3star += 1
                        highest_evo = max(highest_evo, 3)

                    TacticalPositioner.optimize_positions(seat.board)
                    changed = True
                    break
        return highest_evo

    def _board_power(self, board: List[BoardUnit]) -> float:
        if not board:
            return 0.0
        _, tiers = compute_active_synergies(board)
        mult = 1.0 + 0.10 * sum(tiers.values())
        raw = sum((u.hp / 160.0 + u.atk / 14.0) * (1.0 + 0.65 * (u.stars - 1)) for u in board)
        return float(raw * mult)

    def _build_action_mask(self, seat: SeatPlayerState) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        mask[ACT_END_TURN] = True
        if seat.actions_this_stage >= self.max_actions_per_stage:
            return mask

        has_space = (len(seat.board) < seat.level) or (len(seat.bench) < 6)
        for i in range(SHOP_SIZE):
            p = seat.shop[i]
            if p is not None and seat.gold >= p.cost:
                if has_space or self._count_copies(seat, p.name) >= 2:
                    mask[ACT_BUY_0 + i] = True

        if seat.gold >= 1:
            mask[ACT_REROLL] = True
        if seat.gold >= 4 and seat.level < MAX_LEVEL:
            mask[ACT_LEVEL_UP] = True
        if not seat.locked_this_turn:
            mask[ACT_LOCK_SHOP] = True
        for i in range(min(6, len(seat.bench))):
            mask[ACT_DEPLOY_0 + i] = True
        if len(seat.board) + len(seat.bench) > 1:
            for i in range(min(6, len(seat.bench))):
                mask[ACT_SELL_0 + i] = True
        return mask

    def _extract_gpu_features_and_obs(
        self,
        seat: SeatPlayerState,
        config: RewardWeightConfig,
        stage_level: int,
        scouting: LobbyScoutingReport,
    ) -> Tuple[Dict[str, float], np.ndarray]:
        my_power = self._board_power(seat.board)
        counts, active_tiers = compute_active_synergies(seat.board)

        shop_pairs = 0
        shop_syn_hits = 0
        for p in seat.shop:
            if p is not None:
                if self._count_copies(seat, p.name) >= 1:
                    shop_pairs += 1
                if self._synergy_fit(seat, config, p) >= 1.0:
                    shop_syn_hits += 1

        max_pair_prob, avg_avail = HypergeometricPoolTracker.compute_roster_upgrade_probability(
            self.shared_pool, seat.level, seat.board + seat.bench
        )
        from pac_rl.data_loader import SYNERGY_TO_IDX
        target_idx = SYNERGY_TO_IDX.get(config.target_synergy, 0) if config.target_synergy else 0
        target_contested = float(scouting.synergy_contestedness[target_idx])

        xp_to_next = max(0, EXP_TABLE.get(seat.level, 72) - seat.xp)
        feats = {
            "stage_level": float(stage_level),
            "player_hp": float(seat.hp),
            "gold": float(seat.gold),
            "interest_tier": float(min(5, seat.gold // 10)),
            "player_level": float(seat.level),
            "xp_to_next_level": float(xp_to_next),
            "win_streak": float(seat.win_streak),
            "lose_streak": float(seat.lose_streak),
            "board_size": float(len(seat.board)),
            "free_board_slots": float(max(0, seat.level - len(seat.board))),
            "bench_size": float(len(seat.bench)),
            "relative_board_power": float(my_power / max(1.0, scouting.avg_opponent_board_power)),
            "shop_pair_matches": float(shop_pairs),
            "shop_synergy_matches": float(shop_syn_hits),
            "num_2star_units": float(sum(1 for u in seat.board if u.stars == 2)),
            "num_3star_units": float(sum(1 for u in seat.board if u.stars >= 3)),
            "max_synergy_tier": float(max(active_tiers.values(), default=0)),
            "alive_opponents": float(scouting.alive_opponents_count),
            "surrogate_win_prob_vs_lobby": float(seat.last_surrogate_win_prob),
            "max_shop_lookahead_gain": float(np.max(seat.last_shop_deltas)),
            "pair_upgrade_prob_in_pool": float(max_pair_prob),
            "target_synergy_contestedness": target_contested,
        }

        obs = np.zeros(GPU_OBS_DIM, dtype=np.float32)
        obs[0] = feats["stage_level"] / 40.0
        obs[1] = feats["player_hp"] / 100.0
        obs[2] = min(100.0, feats["gold"]) / 50.0
        obs[3] = feats["interest_tier"] / 5.0
        obs[4] = feats["player_level"] / 9.0
        obs[5] = feats["xp_to_next_level"] / 72.0
        obs[6] = min(6.0, feats["win_streak"]) / 5.0
        obs[7] = min(6.0, feats["lose_streak"]) / 5.0
        obs[8] = feats["board_size"] / 9.0
        obs[9] = feats["free_board_slots"] / 4.0
        obs[10] = feats["bench_size"] / 6.0
        obs[11] = min(3.0, feats["relative_board_power"]) / 2.0
        obs[12] = feats["shop_pair_matches"] / 6.0
        obs[13] = feats["shop_synergy_matches"] / 6.0
        obs[14] = feats["num_2star_units"] / 6.0
        obs[15] = feats["num_3star_units"] / 3.0
        obs[16] = feats["max_synergy_tier"] / 4.0
        obs[17] = feats["alive_opponents"] / 7.0

        for i in range(SHOP_SIZE):
            base = 18 + i * 4
            p = seat.shop[i]
            if p is not None:
                obs[base + 0] = 1.0
                obs[base + 1] = p.cost / 5.0
                obs[base + 2] = min(3, self._count_copies(seat, p.name)) / 2.0
                obs[base + 3] = min(4.0, self._synergy_fit(seat, config, p)) / 3.0

        obs[42 : 42 + NUM_SYNERGIES] = encode_synergy_vector(active_tiers, counts)
        obs[73:112] = config.to_Condition_vector()

        # 112..118: Surrogate Lookahead Deltas + Base Win Prob vs Lobby
        obs[112:118] = np.clip(seat.last_shop_deltas * 5.0, -2.0, 2.0)
        obs[118] = seat.last_surrogate_win_prob

        # 119..120: Hypergeometric Shared Pool Upgrade Probabilities
        obs[119] = max_pair_prob * 3.0
        obs[120] = avg_avail

        # 121..155: Live Lobby Scouting Vector (31 contestedness + 4 macro scouting stats)
        obs[121 : 121 + NUM_SYNERGIES] = scouting.synergy_contestedness
        obs[152] = min(3.0, scouting.avg_opponent_board_power / 25.0)
        obs[153] = min(3.0, scouting.max_opponent_board_power / 35.0)
        obs[154] = scouting.lobby_physical_vs_magic_ratio
        obs[155] = scouting.backline_carry_vulnerability

        return feats, obs

    def _sample_pfsp_lobby_agents(self, focal_idx: int) -> List[int]:
        """
        Prioritized Fictitious Self-Play (PFSP — estilo AlphaStar League):
        Seleciona 8 agentes distintos da Liga Comunitária de 35 modelos para compor o lobby,
        priorizando oponentes com ELO próximo ou ligeiramente superior ao agente focal!
        """
        n_total = len(self.agents)
        focal_elo = self.agents[focal_idx].elo_rating
        candidates = [i for i in range(n_total) if i != focal_idx]

        weights = np.array(
            [1.0 / (1.0 + abs(self.agents[c].elo_rating - focal_elo) / 150.0) for c in candidates],
            dtype=np.float64,
        )
        weights /= weights.sum()
        picked = list(self.rng.choice(candidates, size=min(7, len(candidates)), replace=False, p=weights))
        return [focal_idx] + [int(x) for x in picked]

    def run_shared_pool_selfplay_lobby(
        self,
        seat_agent_indices: List[int],
        collect_rollouts: bool = True,
        deterministic: bool = False,
        seed: int = 42,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Executa uma partida completa de 8 jogadores em Self-Play Real:
        - Todos os 8 jogadores são instâncias de `GPUCommunityAgent` (`CommunityOfModelsBrain`).
        - Todos compram e devolvem cartas ao MESMO `SharedCardPool`.
        - Todos usam `NeuralLookaheadRosterPlanner` (conectado ao `DeepSetTransformerSurrogate`)
          e `LobbyScoutingAggregator`.
        """
        self.rng = np.random.default_rng(seed)
        self.shared_pool.reset()

        seats: List[SeatPlayerState] = []
        starters = self.repo.buyable_by_rarity.get("COMMON", [])

        for s_idx, ag_idx in enumerate(seat_agent_indices):
            seat = SeatPlayerState(seat_idx=s_idx, agent_idx=ag_idx)
            cfg = self.agents[ag_idx].config
            if starters:
                matching = [p for p in starters if cfg.target_synergy in p.types] if cfg.target_synergy else []
                starter = (
                    matching[int(self.rng.integers(0, len(matching)))]
                    if matching
                    else starters[int(self.rng.integers(0, len(starters)))]
                )
                seat.board.append(BoardUnit.from_pokemon_stats(starter, x=2, y=2))
                self.shared_pool.take_from_pool(starter.name, copies=1)
            seat.shop = self.shared_pool.sample_shop(seat.level, self.rng)
            seats.append(seat)

        for stage in range(1, 38):
            alive_seats = [s for s in seats if s.alive]
            if len(alive_seats) <= 1:
                break

            phase_key = "early" if stage <= 9 else ("mid" if stage <= 19 else "late")

            # 1. Special Stage Propositions (Additional Picks 5, 8, 11 & Unique/Legendary 10, 20)
            for s in alive_seats:
                cfg = self.agents[s.agent_idx].config
                if stage in ADDITIONAL_PICKS_STAGES:
                    idx_p = ADDITIONAL_PICKS_STAGES.index(stage)
                    rarity = ["UNCOMMON", "RARE", "EPIC"][idx_p]
                    pool = self.repo.additional_by_rarity.get(rarity, [])
                    if pool:
                        choices = [pool[i] for i in self.rng.choice(len(pool), size=min(3, len(pool)), replace=False)]
                        best_p = max(choices, key=lambda p: self._synergy_fit(s, cfg, p) + p.hp / 200.0)
                        self.shared_pool.unlock_additional_pick_for_lobby(best_p)
                        self._try_add_or_evolve(s, best_p)
                elif stage in (10, 20):
                    pool = self.repo.uniques_pool if stage == 10 else self.repo.legendaries_pool
                    if pool:
                        best_p = max(pool, key=lambda p: self._synergy_fit(s, cfg, p) + self.rng.random() * 0.4)
                        self._try_add_or_evolve(s, best_p)

            # 2. Fase de Loja e Planejamento Lookahead para cada um dos 8 Agentes Vivos
            for s in alive_seats:
                ag = self.agents[s.agent_idx]
                ag.model.eval()
                s.actions_this_stage = 0
                s.locked_this_turn = False

                opp_boards = [other.board for other in alive_seats if other.seat_idx != s.seat_idx]
                opp_hps = [other.hp for other in alive_seats if other.seat_idx != s.seat_idx]
                opp_powers = [self._board_power(ob) for ob in opp_boards]
                scouting = LobbyScoutingAggregator.scout_lobby(opp_boards, opp_hps, opp_powers)

                # Avalia na GPU via DeepSetTransformerSurrogate o ganho marginal das 6 cartas da loja
                base_wp, shop_deltas = self.planner.evaluate_shop_marginal_gains(
                    s.board, s.shop, s.level, opp_boards
                )
                s.last_surrogate_win_prob = base_wp
                s.last_shop_deltas = shop_deltas
                s.phi_prev = self.reward_engine.compute_potential_phi(
                    ag.config, s.board, s.bench, s.gold, s.level, s.hp, base_wp, self.shared_pool, scouting
                )

                for _ in range(self.max_actions_per_stage):
                    _, obs = self._extract_gpu_features_and_obs(s, ag.config, stage, scouting)
                    mask = self._build_action_mask(s)

                    with torch.no_grad():
                        t_obs = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                        t_mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)
                        act_t, logp_t, _, val_t, _, exp_w_t = ag.model.get_action_and_value(
                            t_obs, t_mask, deterministic=deterministic
                        )
                        action = int(act_t.item())
                        log_prob = float(logp_t.item())
                        value = float(val_t.item())
                        exp_w = exp_w_t.squeeze(0).cpu().numpy()

                    if len(ag.expert_votes_by_phase[phase_key]) < 500:
                        ag.expert_votes_by_phase[phase_key].append(exp_w)

                    if not mask[action]:
                        action = ACT_END_TURN

                    # Executa a micro-ação escolhida pelo Conselho de 7 Especialistas
                    if ACT_BUY_0 <= action <= ACT_BUY_5:
                        slot = action - ACT_BUY_0
                        p = s.shop[slot]
                        if p is not None and s.gold >= p.cost:
                            added, _ = self._try_add_or_evolve(s, p)
                            if added:
                                s.gold -= p.cost
                                s.shop[slot] = None
                                s.last_shop_deltas[slot] = 0.0
                        s.actions_this_stage += 1

                    elif action == ACT_REROLL:
                        if s.gold >= 1:
                            s.gold -= 1
                            s.shop_locked = False
                            s.shop = self.shared_pool.sample_shop(s.level, self.rng)
                            s.last_surrogate_win_prob, s.last_shop_deltas = self.planner.evaluate_shop_marginal_gains(
                                s.board, s.shop, s.level, opp_boards
                            )
                        s.actions_this_stage += 1

                    elif action == ACT_LEVEL_UP:
                        if s.gold >= 4 and s.level < MAX_LEVEL:
                            s.gold -= 4
                            s.xp += 4
                            while s.level < MAX_LEVEL and s.xp >= EXP_TABLE[s.level]:
                                s.xp -= EXP_TABLE[s.level]
                                s.level += 1
                        s.actions_this_stage += 1

                    elif action == ACT_LOCK_SHOP:
                        s.shop_locked = not s.shop_locked
                        s.locked_this_turn = True
                        s.actions_this_stage += 1

                    elif ACT_DEPLOY_0 <= action <= ACT_DEPLOY_5:
                        b_idx = action - ACT_DEPLOY_0
                        if b_idx < len(s.bench):
                            if len(s.board) < s.level:
                                s.board.append(s.bench.pop(b_idx))
                            elif s.board:
                                s.board, s.bench = self.planner.optimize_roster_with_surrogate(
                                    s.board, s.bench, s.level, opp_boards
                                )
                        s.actions_this_stage += 1

                    elif ACT_SELL_0 <= action <= ACT_SELL_5:
                        b_idx = action - ACT_SELL_0
                        if b_idx < len(s.bench):
                            sold = s.bench.pop(b_idx)
                            self.shared_pool.return_to_pool(sold)
                            p_st = self.repo.get_pokemon(sold.name)
                            base_c = p_st.cost if p_st else 1
                            sell_val = max(1, base_c * (2 if sold.stars == 2 else (4 if sold.stars >= 3 else 1)))
                            s.gold += sell_val
                        s.actions_this_stage += 1

                    # Potential-based micro shaping reward: gamma * Phi(s') - Phi(s)
                    phi_now = self.reward_engine.compute_potential_phi(
                        ag.config, s.board, s.bench, s.gold, s.level, s.hp, s.last_surrogate_win_prob, self.shared_pool, scouting
                    )
                    micro_r = self.reward_engine.compute_micro_shaping_reward(s.phi_prev, phi_now, done=False)
                    s.phi_prev = phi_now

                    if collect_rollouts:
                        s.trajectory.append(
                            GPURolloutStep(
                                obs=obs,
                                mask=mask,
                                action=action,
                                log_prob=log_prob,
                                reward=micro_r,
                                done=False,
                                value=value,
                            )
                        )

                    if action == ACT_END_TURN or s.actions_this_stage >= self.max_actions_per_stage:
                        break

                # Antes do combate começar, aplica a busca combinatória 1-Step Lookahead com o SetTransformer!
                s.board, s.bench = self.planner.optimize_roster_with_surrogate(
                    s.board, s.bench, s.level, opp_boards
                )

            # 3. Fase de Combate Tático (PvE ou Pareamento Direto entre os Agentes Vivos do Lobby)
            is_pve = stage in PVE_STAGES
            if is_pve:
                pve_wave = self.sim.make_pve_board(stage)
                for s in alive_seats:
                    ag = self.agents[s.agent_idx]
                    out = self.sim.simulate(s.board, pve_wave, stage_level=stage, rng=self.rng)
                    self._record_unit_telemetry(ag, out.unit_logs_a)
                    if out.winner >= 0:
                        s.items_held += 1
                        if s.board:
                            carry = max(s.board, key=lambda u: u.atk * u.stars)
                            carry.num_items = min(3, carry.num_items + 1)
                    else:
                        s.hp = max(0, s.hp - out.damage_to_loser)

                    meso_r = self.reward_engine.compute_meso_round_reward(
                        ag.config, out, True, True, stage, s.hp, s.gold, s.win_streak, s.lose_streak, s.lose_streak
                    )
                    if s.trajectory:
                        s.trajectory[-1].reward += meso_r
                    if s.hp == 0:
                        self._eliminate_seat(s, stage)
            else:
                shuffled = list(alive_seats)
                self.rng.shuffle(shuffled)
                for k_pair in range(0, len(shuffled) - 1, 2):
                    s1, s2 = shuffled[k_pair], shuffled[k_pair + 1]
                    ag1, ag2 = self.agents[s1.agent_idx], self.agents[s2.agent_idx]
                    out = self.sim.simulate(s1.board, s2.board, stage_level=stage, rng=self.rng)
                    self._record_unit_telemetry(ag1, out.unit_logs_a)
                    self._record_unit_telemetry(ag2, out.unit_logs_b)

                    prev_ls1, prev_ls2 = s1.lose_streak, s2.lose_streak
                    if out.winner >= 0:
                        s1.gold += 1
                        s1.win_streak += 1
                        s1.lose_streak = 0
                        s2.hp = max(0, s2.hp - out.damage_to_loser)
                        s2.lose_streak += 1
                        s2.win_streak = 0
                    else:
                        s2.gold += 1
                        s2.win_streak += 1
                        s2.lose_streak = 0
                        s1.hp = max(0, s1.hp - out.damage_to_loser)
                        s1.lose_streak += 1
                        s1.win_streak = 0

                    r1 = self.reward_engine.compute_meso_round_reward(
                        ag1.config, out, True, False, stage, s1.hp, s1.gold, s1.win_streak, s1.lose_streak, prev_ls1
                    )
                    r2 = self.reward_engine.compute_meso_round_reward(
                        ag2.config, out, False, False, stage, s2.hp, s2.gold, s2.win_streak, s2.lose_streak, prev_ls2
                    )
                    if s1.trajectory:
                        s1.trajectory[-1].reward += r1
                    if s2.trajectory:
                        s2.trajectory[-1].reward += r2

                    if s1.hp == 0:
                        self._eliminate_seat(s1, stage)
                    if s2.hp == 0:
                        self._eliminate_seat(s2, stage)

            # 4. Renda de Fim de Rodada, Juros, Streak, +2 XP e Rolagem da Loja do Baralho Compartilhado
            for s in seats:
                if not s.alive:
                    continue
                interest = min(5, s.gold // 10)
                streak_g = 0 if is_pve else min(5, max(s.win_streak, s.lose_streak))
                s.gold += 5 + interest + streak_g
                if s.level < MAX_LEVEL:
                    s.xp += 2
                    while s.level < MAX_LEVEL and s.xp >= EXP_TABLE[s.level]:
                        s.xp -= EXP_TABLE[s.level]
                        s.level += 1
                if not s.shop_locked:
                    s.shop = self.shared_pool.sample_shop(s.level, self.rng)
                else:
                    s.shop_locked = False

        # 5. Classificação Final (1º ao 8º lugar), Atualização de ELO e Recompensa Macro Terminal
        ranked_seats = sorted(
            seats,
            key=lambda s: (1 if s.alive else 0, s.hp, s.eliminated_stage),
            reverse=True,
        )

        lobby_results: Dict[int, Dict[str, Any]] = {}
        for rank_idx, s in enumerate(ranked_seats):
            placement = rank_idx + 1
            ag = self.agents[s.agent_idx]
            oracle_val = PLACEMENT_UTILITY.get(placement, 0.0) / 4.0
            term_r = self.reward_engine.compute_macro_terminal_reward(ag.config, placement, s.gold)

            if s.trajectory:
                s.trajectory[-1].reward += term_r
                s.trajectory[-1].done = True
                for step in s.trajectory:
                    step.oracle_target = oracle_val

            # Atualização de ELO competitivo
            ag.elo_rating += 6.0 * (4.5 - placement)
            ag.last_board_names = [f"{u.name}({u.stars}*)" for u in s.board]
            _, active_tiers = compute_active_synergies(s.board)

            lobby_results[s.seat_idx] = {
                "agent_idx": s.agent_idx,
                "placement": placement,
                "final_stage": s.eliminated_stage if not s.alive else 38,
                "final_hp": s.hp,
                "final_gold": s.gold,
                "final_level": s.level,
                "evolutions_2star": s.evolutions_2star,
                "evolutions_3star": s.evolutions_3star,
                "active_synergies": active_tiers,
                "trajectory": s.trajectory,
            }

        return lobby_results

    def _eliminate_seat(self, seat: SeatPlayerState, stage: int) -> None:
        seat.alive = False
        seat.eliminated_stage = stage
        # Devolve todas as peças do jogador eliminado ao baralho compartilhado!
        for u in seat.board + seat.bench:
            self.shared_pool.return_to_pool(u)

    def _record_unit_telemetry(self, agent: GPUCommunityAgent, logs: List[Any]) -> None:
        for ulog in logs:
            label = f"{ulog.name}({ulog.stars}*)"
            if ulog.damage_dealt > agent.record_dps_unit[1]:
                agent.record_dps_unit = (label, round(float(ulog.damage_dealt), 1))
            if ulog.damage_tanked > agent.record_tank_unit[1]:
                agent.record_tank_unit = (label, round(float(ulog.damage_tanked), 1))

    # ------------------------------------------------------------------------
    # Loop Multi-Geracional de Treinamento Self-Play + PBT + MAP-Elites
    # ------------------------------------------------------------------------

    def train_and_evaluate_generation(
        self,
        generation: int,
        lobbies_per_generation: int = 20,
    ) -> List[AgentMetrics]:
        buffers_by_agent: Dict[int, List[GPURolloutStep]] = {i: [] for i in range(len(self.agents))}
        placements_by_agent: Dict[int, List[int]] = {i: [] for i in range(len(self.agents))}
        stages_by_agent: Dict[int, List[int]] = {i: [] for i in range(len(self.agents))}
        evos2_by_agent: Dict[int, List[int]] = {i: [] for i in range(len(self.agents))}
        evos3_by_agent: Dict[int, List[int]] = {i: [] for i in range(len(self.agents))}
        returns_by_agent: Dict[int, List[float]] = {i: [] for i in range(len(self.agents))}
        syns_by_agent: Dict[int, Dict[str, int]] = {i: {} for i in range(len(self.agents))}

        # 1. Roda Lobbies Completos de 8 Jogadores em Self-Play (PFSP)
        for lobby_i in range(lobbies_per_generation):
            focal_idx = (generation * lobbies_per_generation + lobby_i) % len(self.agents)
            seat_indices = self._sample_pfsp_lobby_agents(focal_idx)
            lobby_seed = 100_000 + generation * 1000 + lobby_i * 17

            res = self.run_shared_pool_selfplay_lobby(
                seat_agent_indices=seat_indices,
                collect_rollouts=True,
                deterministic=False,
                seed=lobby_seed,
            )

            for _, seat_res in res.items():
                ag_i = seat_res["agent_idx"]
                traj = seat_res["trajectory"]
                buffers_by_agent[ag_i].extend(traj)
                placements_by_agent[ag_i].append(seat_res["placement"])
                stages_by_agent[ag_i].append(seat_res["final_stage"])
                evos2_by_agent[ag_i].append(seat_res["evolutions_2star"])
                evos3_by_agent[ag_i].append(seat_res["evolutions_3star"])
                returns_by_agent[ag_i].append(sum(st.reward for st in traj))
                for s_name, s_tier in seat_res["active_synergies"].items():
                    if s_tier > 0:
                        syns_by_agent[ag_i][s_name] = syns_by_agent[ag_i].get(s_name, 0) + 1

        # 2. Atualiza o `CommunityOfModelsBrain` de cada Agente via GPU Maskable PPO
        gen_metrics: List[AgentMetrics] = []
        for ag_i, agent in enumerate(self.agents):
            if buffers_by_agent[ag_i]:
                agent.trainer.update(buffers_by_agent[ag_i], ppo_epochs=6, batch_size=256)

            places = placements_by_agent[ag_i] or [4]
            avg_place = float(np.mean(places))
            top4 = float(np.mean([1.0 if p <= 4 else 0.0 for p in places]))
            win_r = float(np.mean([1.0 if p == 1 else 0.0 for p in places]))
            syn_c = syns_by_agent[ag_i]
            dom_syn = (
                agent.config.target_synergy
                if agent.config.target_synergy
                else (max(syn_c.items(), key=lambda kv: kv[1])[0] if syn_c else "FLEX_ALL")
            )

            m = AgentMetrics(
                generation=generation,
                agent_name=agent.config.name,
                archetype=agent.config.archetype,
                target_synergy=dom_syn,
                avg_placement=avg_place,
                top4_rate=top4,
                win_rate=win_r,
                avg_stage_reached=float(np.mean(stages_by_agent[ag_i] or [22])),
                avg_2star_evos=float(np.mean(evos2_by_agent[ag_i] or [2])),
                avg_3star_evos=float(np.mean(evos3_by_agent[ag_i] or [0])),
                avg_shaped_return=float(np.mean(returns_by_agent[ag_i] or [0.0])),
                dominant_synergies=syn_c,
            )
            agent.history.append(m)
            gen_metrics.append(m)

            self.archive.try_insert(
                synergy_niche=dom_syn,
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
        Population-Based Training (PBT) sobre o Conselho de Especialistas:
        Compartilha suavemente os pesos do `council_attn` dos 20% melhores agentes
        para os 20% piores e refina os pesos de recompensa.
        """
        ranked = sorted(range(len(self.agents)), key=lambda i: gen_metrics[i].avg_placement)
        k_slice = max(1, len(self.agents) // 5)
        for b_idx, e_idx in zip(ranked[-k_slice:], ranked[:k_slice]):
            worst_ag = self.agents[b_idx]
            best_ag = self.agents[e_idx]
            with torch.no_grad():
                for p_w, p_b in zip(
                    worst_ag.model.council_attn.parameters(),
                    best_ag.model.council_attn.parameters(),
                ):
                    p_w.data.mul_(0.80).add_(p_b.data, alpha=0.20)

    def get_council_telemetry_dataframe(self) -> pd.DataFrame:
        """
        Retorna uma tabela mostrando quanto cada um dos 7 Sub-Modelos Especialistas
        (`EXPERT_NAMES`) foi ouvido pelo Meta-Arbitrador no Early, Mid e Late Game!
        """
        rows = []
        for ag in self.agents[:12]:
            for phase_label, votes_list in ag.expert_votes_by_phase.items():
                if not votes_list:
                    continue
                mean_v = np.mean(np.stack(votes_list, axis=0), axis=0)
                top_idx = int(np.argmax(mean_v))
                row: Dict[str, Any] = {
                    "Agent": ag.config.name,
                    "Archetype": ag.config.archetype,
                    "Phase": phase_label.upper(),
                    "Dominant_Expert": EXPERT_NAMES[top_idx],
                }
                for e_i, e_name in enumerate(EXPERT_NAMES):
                    short_n = e_name.replace("_Expert", "")
                    row[f"{short_n}_%"] = round(float(mean_v[e_i] * 100.0), 1)
                rows.append(row)
        return pd.DataFrame(rows)

    def distill_gpu_council_agents(self, top_k: int = 10, lobbies_sample: int = 12) -> List[DistilledHeuristicReport]:
        """
        Destilação VIPER Livre de Viés Manual (`Zero Hand-Coded Priors`):
        Coleta decisões das redes `CommunityOfModelsBrain` nos lobbies compartilhados
        e treina Árvores de Decisão `DecisionTreeClassifier(max_depth=4)` ponderadas
        pela criticidade Q/vantagem do Conselho de 7 Especialistas!
        """
        ranked_indices = sorted(range(len(self.agents)), key=lambda i: self.agents[i].elo_rating, reverse=True)[:top_k]
        reports: List[DistilledHeuristicReport] = []

        for ag_idx in ranked_indices:
            ag = self.agents[ag_idx]
            ag.model.eval()
            X_rows: List[List[float]] = []
            y_labels: List[int] = []
            weights: List[float] = []

            for l_i in range(lobbies_sample):
                seat_indices = self._sample_pfsp_lobby_agents(ag_idx)
                self.shared_pool.reset()
                seat = SeatPlayerState(seat_idx=0, agent_idx=ag_idx)
                starters = self.repo.buyable_by_rarity.get("COMMON", [])
                if starters:
                    seat.board.append(BoardUnit.from_pokemon_stats(starters[0], x=2, y=2))
                seat.shop = self.shared_pool.sample_shop(seat.level, self.rng)

                for stage in range(1, 30):
                    opp_boards = [self.sim.make_pve_board(min(29, stage + 2))]
                    scouting = LobbyScoutingAggregator.scout_lobby(opp_boards, [80], [stage * 1.2])
                    base_wp, deltas = self.planner.evaluate_shop_marginal_gains(
                        seat.board, seat.shop, seat.level, opp_boards
                    )
                    seat.last_surrogate_win_prob = base_wp
                    seat.last_shop_deltas = deltas

                    feats, obs = self._extract_gpu_features_and_obs(seat, ag.config, stage, scouting)
                    mask = self._build_action_mask(seat)

                    with torch.no_grad():
                        t_obs = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                        t_mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)
                        logits, _, _, _ = ag.model.forward_council(t_obs)
                        masked_l = logits.masked_fill(~t_mask, -1e8).squeeze(0)
                        best_act = int(torch.argmax(masked_l).item())
                        valid_l = masked_l[t_mask.squeeze(0)].cpu().numpy()
                        crit_w = float(np.max(valid_l) - np.mean(valid_l) + 0.1)

                    X_rows.append([feats[k] for k in GPU_FEATURE_NAMES])
                    y_labels.append(map_to_macro_action(best_act))
                    weights.append(max(0.05, crit_w))

                    # Advance state lightly for diverse trajectory coverage
                    seat.gold = int(self.rng.integers(4, 65))
                    seat.hp = max(10, seat.hp - int(self.rng.integers(0, 5)))
                    seat.level = min(9, 2 + stage // 4)
                    seat.shop = self.shared_pool.sample_shop(seat.level, self.rng)

            X = np.array(X_rows, dtype=np.float32)
            y = np.array(y_labels, dtype=np.int64)
            w = np.array(weights, dtype=np.float32)

            clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, random_state=42)
            clf.fit(X, y, sample_weight=w)
            fidelity = float(np.mean(clf.predict(X) == y))

            present_cls = sorted(list(set(y)))
            cls_names = [MACRO_ACTION_NAMES[c] for c in present_cls]
            tree_str = export_text(clf, feature_names=GPU_FEATURE_NAMES, class_names=cls_names, decimals=2)

            ranked_feats = sorted(
                zip(GPU_FEATURE_NAMES, [float(v) for v in clf.feature_importances_]),
                key=lambda x: x[1],
                reverse=True,
            )[:6]

            act_dist = {name: float(np.mean(y == idx)) for idx, name in enumerate(MACRO_ACTION_NAMES)}
            rules_pt = self._extract_tree_rules_pt(clf)

            reports.append(
                DistilledHeuristicReport(
                    agent_name=ag.config.name,
                    archetype=ag.config.archetype,
                    fidelity_accuracy=fidelity,
                    tree_text=tree_str,
                    top_features=ranked_feats,
                    extracted_rules_pt=rules_pt,
                    action_distribution=act_dist,
                )
            )
        return reports

    def _extract_tree_rules_pt(self, clf: DecisionTreeClassifier) -> List[str]:
        tree_ = clf.tree_
        f_names = [GPU_FEATURE_NAMES[i] if i != -2 else "leaf" for i in tree_.feature]
        paths: List[Tuple[int, str]] = []

        def recurse(node: int, conds: List[str]):
            if tree_.feature[node] != -2:
                name = f_names[node]
                th = tree_.threshold[node]
                recurse(tree_.children_left[node], conds + [f"{name} <= {th:.2f}"])
                recurse(tree_.children_right[node], conds + [f"{name} > {th:.2f}"])
            else:
                counts = tree_.value[node][0]
                best_idx = int(np.argmax(counts))
                actual_cls = int(clf.classes_[best_idx])
                n_s = int(tree_.n_node_samples[node])
                purity = float(counts[best_idx] / max(1e-6, np.sum(counts)))
                cond_str = " E ".join(conds) if conds else "SEMPRE"
                paths.append(
                    (n_s, f"SE ({cond_str}) ENTÃO [{MACRO_ACTION_NAMES[actual_cls]}] (confiança={purity*100:.1f}%, n={n_s})")
                )

        recurse(0, [])
        paths.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in paths[:5]]


def compute_synergy_counter_matrix(
    repo: PACDataRepository,
    sim: FastCombatSimulator,
    duels_per_pair: int = 20,
) -> pd.DataFrame:
    """
    Constrói a Matriz de Counter-Picks 31x31 simulando confrontos diretos entre os melhores
    tabuleiros de cada Sinergia, identificando qual é o Hard Counter e qual é a Presa Favorita
    de cada composição.
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
