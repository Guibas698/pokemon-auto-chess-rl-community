"""
Gymnasium Environment (`PokemonAutoChessGymEnv`) for 8-Player Pokemon Auto Chess (PAC).

Implements:
- Exact PAC economy: 5g base income, min(5, floor(gold/10)) interest, min(5, streak)
  win/lose streak gold on PvP rounds, +1g on PvP win, 1g reroll, 4g for +4 XP,
  and automatic +2 XP per round.
- Evolution system: 3 copies of 1-star -> 2-star (`evolves_into`), 3 copies of 2-star -> 3-star.
- PAC Special Stages:
  - Additional Picks at stages 5 (`UNCOMMON`), 8 (`RARE`), 11 (`EPIC`)
  - Unique & Legendary Propositions at stages 10 (`UNIQUE`) and 20 (`LEGENDARY`)
    tailored to active board synergies.
- Configurable Multi-Objective Reward Vector `RewardWeightConfig` with Potential-Based
  Reward Shaping (`gamma * Phi(s') - Phi(s)`) so every agent in the Community League
  can optimize a distinct persona/heuristic while being evaluated on True Placement (1..8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .data_loader import (
    ADDITIONAL_PICKS_STAGES,
    EXP_TABLE,
    MAX_LEVEL,
    NUM_SYNERGIES,
    PORTAL_CAROUSEL_STAGES,
    PVE_STAGES,
    SHOP_ODDS_BY_LEVEL,
    SHOP_SIZE,
    SYNERGIES,
    SYNERGY_THRESHOLDS,
    SYNERGY_TO_IDX,
    PACDataRepository,
    PokemonStats,
)
from .combat_surrogate import (
    BoardUnit,
    FastCombatSimulator,
    ItemAllocationOracle,
    TacticalPositioner,
    compute_active_synergies,
    encode_synergy_vector,
)


@dataclass
class RewardWeightConfig:
    """
    Defines the reward persona $\mathbf{w}^{(k)}$ of an agent in the Community League.
    """
    name: str = "Balanced_Meta"
    archetype: str = "STANDARD_ECON_50G"  # HYPER_ROLL, TEMPO_WIN_STREAK, STANDARD_ECON_50G, LOSE_STREAK_CASHOUT, SYNERGY_SPECIALIST
    target_synergy: Optional[str] = None
    w_placement: float = 1.0
    w_interest: float = 0.25
    w_win_streak: float = 0.20
    w_lose_streak: float = 0.0
    w_reroll_evo: float = 0.25
    w_synergy_focus: float = 0.30
    w_hp_preservation: float = 0.15
    w_potential_shaping: float = 0.20

    def to_Condition_vector(self) -> np.ndarray:
        """Encodes the reward weights and target synergy into a conditioning vector of shape (39,)."""
        vec = np.zeros(8 + NUM_SYNERGIES, dtype=np.float32)
        vec[0] = self.w_placement
        vec[1] = self.w_interest
        vec[2] = self.w_win_streak
        vec[3] = self.w_lose_streak
        vec[4] = self.w_reroll_evo
        vec[5] = self.w_synergy_focus
        vec[6] = self.w_hp_preservation
        vec[7] = self.w_potential_shaping
        if self.target_synergy and self.target_synergy in SYNERGY_TO_IDX:
            vec[8 + SYNERGY_TO_IDX[self.target_synergy]] = 1.0
        return vec


# Action space indices (Discrete(22))
ACT_BUY_0 = 0
ACT_BUY_5 = 5
ACT_REROLL = 6
ACT_LEVEL_UP = 7
ACT_LOCK_SHOP = 8
ACT_DEPLOY_0 = 9
ACT_DEPLOY_5 = 14
ACT_SELL_0 = 15
ACT_SELL_5 = 20
ACT_END_TURN = 21
NUM_ACTIONS = 22

# Observation space dimension
OBS_DIM = 112


@dataclass
class OpponentState:
    bot_id: str
    name: str
    elo: float
    hp: int = 100
    level: int = 2
    gold: int = 5
    win_streak: int = 0
    lose_streak: int = 0
    alive: bool = True
    eliminated_stage: int = 40
    board: List[BoardUnit] = field(default_factory=list)


class PokemonAutoChessGymEnv(gym.Env):
    """
    Single-player-perspective Gymnasium environment embedded inside a full 8-player PAC lobby.
    Supports Invalid Action Masking via `env.action_masks()` and interpretable state summaries
    via `env.extract_interpretable_features()` for VIPER decision tree distillation.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        repo: Optional[PACDataRepository] = None,
        sim: Optional[FastCombatSimulator] = None,
        reward_config: Optional[RewardWeightConfig] = None,
        max_actions_per_stage: int = 12,
        seed: int = 42,
    ):
        super().__init__()
        self.repo = repo or PACDataRepository()
        self.sim = sim or FastCombatSimulator(self.repo)
        self.reward_config = reward_config or RewardWeightConfig()
        self.max_actions_per_stage = max_actions_per_stage
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        # Episode state
        self.stage_level: int = 1
        self.hp: int = 100
        self.gold: int = 6
        self.level: int = 2
        self.xp: int = 0
        self.win_streak: int = 0
        self.lose_streak: int = 0
        self.shop_locked: bool = False
        self.locked_this_turn: bool = False
        self.actions_this_stage: int = 0
        self.total_rerolls: int = 0
        self.total_evolutions_2star: int = 0
        self.total_evolutions_3star: int = 0
        self.items_held: int = 0
        self.cashout_triggered: bool = False

        self.board: List[BoardUnit] = []
        self.bench: List[BoardUnit] = []  # up to 6 active bench slots
        self.shop: List[Optional[PokemonStats]] = [None] * SHOP_SIZE
        self.additional_pool_Unlocked: List[PokemonStats] = []
        self.opponents: List[OpponentState] = []
        self.prev_potential: float = 0.0

    def set_reward_config(self, config: RewardWeightConfig) -> None:
        self.reward_config = config

    def _unit_power(self, u: BoardUnit, include_items: bool = True) -> float:
        """Heuristic combat value of a single unit on board/bench."""
        rarity_bonus = {
            "COMMON": 1.0,
            "UNCOMMON": 1.35,
            "RARE": 1.75,
            "EPIC": 2.25,
            "ULTRA": 2.90,
            "UNIQUE": 2.40,
            "LEGENDARY": 3.20,
        }.get(u.rarity, 1.2)
        star_mult = {1: 1.0, 2: 1.85, 3: 3.10}.get(u.stars, 1.0)
        stat_score = (u.hp / 160.0) + (u.atk / 15.0) + ((u.defense + u.spe_def) / 14.0) + (u.speed / 65.0)
        syn_bonus = 0.0
        if self.reward_config.target_synergy and self.reward_config.target_synergy in u.types:
            syn_bonus = 1.25
        item_mult = (1.0 + 0.15 * u.num_items) if include_items else 1.0
        return float(stat_score * rarity_bonus * star_mult * item_mult + syn_bonus)

    def compute_board_power(self, units: Optional[List[BoardUnit]] = None) -> float:
        target = self.board if units is None else units
        if not target:
            return 0.0
        raw_power = sum(self._unit_power(u) for u in target)
        _, active_tiers = compute_active_synergies(target)
        syn_mult = 1.0 + 0.10 * sum(active_tiers.values())
        return float(raw_power * syn_mult)

    def _compute_potential(self) -> float:
        """Potential function $\Phi(s)$ for potential-based reward shaping (Ng et al., 1999)."""
        board_p = self.compute_board_power() / 35.0
        econ_p = min(50, self.gold) / 50.0
        level_p = self.level / 9.0
        _, tiers = compute_active_synergies(self.board)
        target_tier = tiers.get(self.reward_config.target_synergy, 0) if self.reward_config.target_synergy else max(tiers.values(), default=0)
        syn_p = target_tier / 4.0
        return float(0.45 * board_p + 0.20 * econ_p + 0.15 * level_p + 0.20 * syn_p)

    def _roll_shop(self) -> None:
        odds = SHOP_ODDS_BY_LEVEL.get(self.level, SHOP_ODDS_BY_LEVEL[9])
        rarities = ["COMMON", "UNCOMMON", "RARE", "EPIC", "ULTRA"]
        for i in range(SHOP_SIZE):
            r_choice = self.rng.choice(rarities, p=odds)
            candidates = list(self.repo.buyable_by_rarity.get(r_choice, []))
            # Add unlocked additional picks of matching rarity
            for add_p in self.additional_pool_Unlocked:
                if add_p.rarity == r_choice:
                    candidates.append(add_p)
            if not candidates:
                candidates = self.repo.buyable_by_rarity["COMMON"]
            picked = candidates[int(self.rng.integers(0, len(candidates)))]
            self.shop[i] = picked

    def _count_copies_owned(self, poke_name: str) -> int:
        return sum(1 for u in (self.board + self.bench) if u.name == poke_name)

    def _count_family_synergy_fit(self, p: PokemonStats) -> float:
        counts, tiers = compute_active_synergies(self.board)
        score = 0.0
        for t in p.types:
            c = counts.get(t, 0)
            score += 0.35 * (c + 1)
            for th in SYNERGY_THRESHOLDS.get(t, []):
                if c + 1 == th:
                    score += 1.5  # completes a synergy threshold!
            if self.reward_config.target_synergy == t:
                score += 2.0
        return score

    def _try_add_or_evolve(self, p: PokemonStats, force_make_room: bool = False) -> Tuple[bool, int]:
        """
        Adds `p` to the player's roster and checks for 3-of-a-kind evolutions
        (chain reactions 1* -> 2* -> 3* just like `count-evolution-handler.ts`).
        Returns `(added_ok, evo_tier_reached)` where `evo_tier_reached` is 0, 2, or 3.
        """
        new_unit = BoardUnit.from_pokemon_stats(p, num_items=0)
        if len(self.board) < self.level:
            self.board.append(new_unit)
            TacticalPositioner.optimize_positions(self.board, in_place=True)
        elif len(self.bench) < 6:
            self.bench.append(new_unit)
        elif self._count_copies_owned(p.name) >= 2:
            # Direct merge on purchase even when bench is full!
            self.bench.append(new_unit)
        elif force_make_room and self.bench:
            # Sell the weakest non-core bench unit so Unique/Legendary/Additional picks are never lost!
            weakest_idx = int(np.argmin([self._unit_power(u, include_items=False) for u in self.bench]))
            sold = self.bench.pop(weakest_idx)
            p_st = self.repo.get_pokemon(sold.name)
            self.gold += max(1, (p_st.cost if p_st else 1) * (2 if sold.stars == 2 else 1))
            self.bench.append(new_unit)
        else:
            return False, 0

        evo_tier = self._check_evolutions()
        ItemAllocationOracle.allocate_items(self.board, self.bench, total_items=self.items_held)
        return True, evo_tier

    def _check_evolutions(self) -> int:
        """Checks all owned units for 3 copies of the same species and evolves them."""
        highest_evo = 0
        changed = True
        while changed:
            changed = False
            all_units = self.board + self.bench
            by_name: Dict[str, List[BoardUnit]] = {}
            for u in all_units:
                by_name.setdefault(u.name, []).append(u)

            for name, group in by_name.items():
                if len(group) >= 3:
                    p_stats = self.repo.get_pokemon(name)
                    if p_stats is None or not p_stats.evolves_into:
                        continue
                    evo_stats = self.repo.get_pokemon(p_stats.evolves_into)
                    if evo_stats is None:
                        continue

                    # Remove 3 copies (prefer removing from bench first, keeping board slots)
                    to_remove = group[:3]
                    preserved_items = sum(u.num_items for u in to_remove)
                    was_on_board = any(u in self.board for u in to_remove)
                    for r_u in to_remove:
                        if r_u in self.bench:
                            self.bench.remove(r_u)
                        elif r_u in self.board:
                            self.board.remove(r_u)

                    evolved_unit = BoardUnit.from_pokemon_stats(
                        evo_stats,
                        num_items=min(3, preserved_items),
                    )
                    if was_on_board or len(self.board) < self.level:
                        self.board.append(evolved_unit)
                    else:
                        self.bench.append(evolved_unit)

                    if evolved_unit.stars == 2:
                        self.total_evolutions_2star += 1
                        highest_evo = max(highest_evo, 2)
                    elif evolved_unit.stars >= 3:
                        self.total_evolutions_3star += 1
                        highest_evo = max(highest_evo, 3)

                    TacticalPositioner.optimize_positions(self.board, in_place=True)
                    changed = True
                    break
        return highest_evo

    def _handle_special_stage_propositions(self) -> None:
        """
        Handles PAC's Additional Picks (stages 5, 8, 11), Portal Carousel Items
        (stages 4, 12, 17, 22, 27, 32), and Unique/Legendary Propositions (stages 10, 20).
        """
        if self.stage_level in PORTAL_CAROUSEL_STAGES:
            self.items_held += 1

        if self.stage_level in ADDITIONAL_PICKS_STAGES:
            idx = ADDITIONAL_PICKS_STAGES.index(self.stage_level)
            rarity = ["UNCOMMON", "RARE", "EPIC"][idx]
            pool = self.repo.additional_by_rarity.get(rarity, [])
            if pool:
                n_choices = min(3, len(pool))
                choices = [pool[i] for i in self.rng.choice(len(pool), size=n_choices, replace=False)]
                best = max(choices, key=lambda p: self._count_family_synergy_fit(p) + (p.hp / 200.0))
                self.additional_pool_Unlocked.append(best)
                self._try_add_or_evolve(best, force_make_room=True)

        elif self.stage_level in (10, 20):
            pool = self.repo.uniques_pool if self.stage_level == 10 else self.repo.legendaries_pool
            if pool:
                scored = sorted(pool, key=lambda p: self._count_family_synergy_fit(p) + self.rng.random() * 0.5, reverse=True)
                best = scored[0]
                self._try_add_or_evolve(best, force_make_room=True)

    def _auto_optimize_roster(self) -> None:
        """
        Ensures `len(self.board) == min(self.level, len(self.board) + len(self.bench))`,
        pivots higher-value / higher-star units from bench onto the board (`ValueDiff` fix),
        positions units on the 6x4 grid, and dynamically reallocates `self.items_held`
        via `ItemAllocationOracle`.
        """
        all_owned = self.board + self.bench
        if not all_owned:
            return
        all_owned.sort(key=lambda u: self._unit_power(u, include_items=False), reverse=True)
        k_board = min(self.level, len(all_owned))
        self.board = all_owned[:k_board]
        self.bench = all_owned[k_board:k_board + 6]
        TacticalPositioner.optimize_positions(self.board, in_place=True)
        ItemAllocationOracle.allocate_items(self.board, self.bench, total_items=self.items_held)

    def _update_opponents_for_stage(self) -> None:
        """Updates the 7 opponent boards for the current `stage_level` from `botv2.json`."""
        bots_by_id = {b.id: b for b in self.repo.bots}
        for opp in self.opponents:
            if not opp.alive:
                continue
            opp.level = min(MAX_LEVEL, max(2, 2 + self.stage_level // 4))
            bot = bots_by_id.get(opp.bot_id)
            if bot is not None:
                entries = self.repo.get_bot_stage_board(bot, self.stage_level)
                units: List[BoardUnit] = []
                for name, x, y, n_items in entries[:opp.level]:
                    p = self.repo.get_pokemon(name)
                    if p is not None:
                        units.append(
                            BoardUnit.from_pokemon_stats(
                                p,
                                x=min(5, max(0, x)),
                                y=min(3, max(0, y - 1)),
                                num_items=n_items,
                            )
                        )
                if units:
                    opp.board = TacticalPositioner.optimize_positions(units)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.stage_level = 1
        self.hp = 100
        self.gold = 6
        self.level = 2
        self.xp = 0
        self.win_streak = 0
        self.lose_streak = 0
        self.shop_locked = False
        self.locked_this_turn = False
        self.actions_this_stage = 0
        self.total_rerolls = 0
        self.total_evolutions_2star = 0
        self.total_evolutions_3star = 0
        self.items_held = 0
        self.cashout_triggered = False

        self.board = []
        self.bench = []
        self.additional_pool_Unlocked = []

        # Starter Pokemon at Stage 0/1 (just like PAC's Portal Carousel starter)
        starters = self.repo.buyable_by_rarity.get("COMMON", [])
        if starters:
            if self.reward_config.target_synergy:
                matching = [p for p in starters if self.reward_config.target_synergy in p.types]
                starter = matching[int(self.rng.integers(0, len(matching)))] if matching else starters[int(self.rng.integers(0, len(starters)))]
            else:
                starter = starters[int(self.rng.integers(0, len(starters)))]
            self.board.append(BoardUnit.from_pokemon_stats(starter, x=2, y=2))

        # Sample 7 opponents stratified across the 726 bots in `botv2.json`
        self.opponents = []
        if self.repo.bots:
            chosen_indices = self.rng.choice(len(self.repo.bots), size=min(7, len(self.repo.bots)), replace=False)
            for idx in chosen_indices:
                b = self.repo.bots[int(idx)]
                self.opponents.append(
                    OpponentState(
                        bot_id=b.id,
                        name=b.name,
                        elo=b.elo,
                        hp=100,
                        level=2,
                        gold=6,
                    )
                )

        self._roll_shop()
        self.prev_potential = self._compute_potential()
        return self._get_obs(), {"stage": self.stage_level, "hp": self.hp, "gold": self.gold}

    def action_masks(self) -> np.ndarray:
        """
        Returns a boolean mask of shape `(22,)` indicating valid actions in the current state.
        Used by `MaskablePPOTrainer` (`logits.masked_fill(~mask, -1e8)`).
        """
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        mask[ACT_END_TURN] = True  # Ending the shop phase is always valid

        if self.actions_this_stage >= self.max_actions_per_stage:
            return mask

        has_space = (len(self.board) < self.level) or (len(self.bench) < 6)
        # Shop buy actions 0..5
        for i in range(SHOP_SIZE):
            p = self.shop[i]
            if p is not None and self.gold >= p.cost:
                can_merge = self._count_copies_owned(p.name) >= 2
                if has_space or can_merge:
                    mask[ACT_BUY_0 + i] = True

        # Reroll (1g)
        if self.gold >= 1:
            mask[ACT_REROLL] = True

        # Buy XP (4g for +4 XP, if level < 9)
        if self.gold >= 4 and self.level < MAX_LEVEL:
            mask[ACT_LEVEL_UP] = True

        # Lock/Unlock shop (at most once per turn to avoid infinite toggling)
        if not self.locked_this_turn:
            mask[ACT_LOCK_SHOP] = True

        # Deploy/swap bench units 0..5
        for i in range(min(6, len(self.bench))):
            mask[ACT_DEPLOY_0 + i] = True

        # Sell bench units 0..5 (only allow selling if we have at least 2 total units)
        if len(self.board) + len(self.bench) > 1:
            for i in range(min(6, len(self.bench))):
                mask[ACT_SELL_0 + i] = True

        return mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        mask = self.action_masks()
        if not mask[action]:
            action = ACT_END_TURN

        step_reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        if ACT_BUY_0 <= action <= ACT_BUY_5:
            slot = action - ACT_BUY_0
            p = self.shop[slot]
            if p is not None and self.gold >= p.cost:
                syn_fit = self._count_family_synergy_fit(p)
                copies_before = self._count_copies_owned(p.name)
                added, evo_tier = self._try_add_or_evolve(p)
                if added:
                    self.gold -= p.cost
                    self.shop[slot] = None
                    # Dense micro-reward for synergy match and pair/evolution progress
                    step_reward += 0.04 * self.reward_config.w_synergy_focus * syn_fit
                    if copies_before == 1:
                        step_reward += 0.08 * self.reward_config.w_reroll_evo
                    if evo_tier == 2:
                        step_reward += 0.45 * self.reward_config.w_reroll_evo
                    elif evo_tier >= 3:
                        step_reward += 1.20 * self.reward_config.w_reroll_evo
            self.actions_this_stage += 1

        elif action == ACT_REROLL:
            if self.gold >= 1:
                self.gold -= 1
                self.total_rerolls += 1
                self.shop_locked = False
                self._roll_shop()
                # Small reward/penalty depending on archetype and current interest threshold
                if self.reward_config.archetype == "HYPER_ROLL" and self.level <= 5:
                    step_reward += 0.03 * self.reward_config.w_reroll_evo
                elif self.gold < 20 and self.reward_config.w_interest > 0.3:
                    step_reward -= 0.03 * self.reward_config.w_interest
            self.actions_this_stage += 1

        elif action == ACT_LEVEL_UP:
            if self.gold >= 4 and self.level < MAX_LEVEL:
                self.gold -= 4
                self.xp += 4
                while self.level < MAX_LEVEL and self.xp >= EXP_TABLE[self.level]:
                    self.xp -= EXP_TABLE[self.level]
                    self.level += 1
                    self._auto_optimize_roster()
                    step_reward += 0.15 * self.reward_config.w_win_streak
            self.actions_this_stage += 1

        elif action == ACT_LOCK_SHOP:
            self.shop_locked = not self.shop_locked
            self.locked_this_turn = True
            self.actions_this_stage += 1

        elif ACT_DEPLOY_0 <= action <= ACT_DEPLOY_5:
            b_idx = action - ACT_DEPLOY_0
            if b_idx < len(self.bench):
                bench_unit = self.bench[b_idx]
                if len(self.board) < self.level:
                    self.board.append(self.bench.pop(b_idx))
                    TacticalPositioner.optimize_positions(self.board)
                elif self.board:
                    # Swap with weakest board unit if bench unit improves board power
                    weakest_idx = int(np.argmin([self._unit_power(u) for u in self.board]))
                    if self._unit_power(bench_unit) > self._unit_power(self.board[weakest_idx]):
                        old_u = self.board[weakest_idx]
                        self.board[weakest_idx] = self.bench[b_idx]
                        self.bench[b_idx] = old_u
                        TacticalPositioner.optimize_positions(self.board)
            self.actions_this_stage += 1

        elif ACT_SELL_0 <= action <= ACT_SELL_5:
            b_idx = action - ACT_SELL_0
            if b_idx < len(self.bench):
                sold = self.bench.pop(b_idx)
                p_stats = self.repo.get_pokemon(sold.name)
                base_cost = p_stats.cost if p_stats else 1
                sell_val = max(1, base_cost * (2 if sold.stars == 2 else (4 if sold.stars >= 3 else 1)))
                old_interest = min(5, self.gold // 10)
                self.gold += sell_val
                new_interest = min(5, self.gold // 10)
                if new_interest > old_interest:
                    step_reward += 0.08 * self.reward_config.w_interest
            self.actions_this_stage += 1

        # If action is END_TURN or max actions reached, execute combat & advance stage!
        if action == ACT_END_TURN or self.actions_this_stage >= self.max_actions_per_stage:
            round_reward, terminated, info = self._resolve_combat_and_advance_stage()
            step_reward += round_reward

        # Potential-based shaping (Ng et al., 1999)
        new_potential = 0.0 if terminated else self._compute_potential()
        shaping = self.reward_config.w_potential_shaping * (0.99 * new_potential - self.prev_potential)
        self.prev_potential = new_potential
        total_reward = float(step_reward + shaping)

        return self._get_obs(), total_reward, terminated, truncated, info

    def _resolve_combat_and_advance_stage(self) -> Tuple[float, bool, Dict[str, Any]]:
        self._auto_optimize_roster()
        self._update_opponents_for_stage()

        is_pve = self.stage_level in PVE_STAGES
        round_reward = 0.0

        # 1. Resolve player's combat
        if is_pve:
            pve_board = self.sim.make_pve_board(self.stage_level)
            outcome = self.sim.simulate(self.board, pve_board, stage_level=self.stage_level, rng=self.rng)
            if outcome.winner >= 0:
                self.items_held += 1
                if self.board:
                    carry = max(self.board, key=lambda u: u.atk * u.stars)
                    carry.num_items = min(3, carry.num_items + 1)
                round_reward += 0.15
            else:
                self.hp = max(0, self.hp - outcome.damage_to_loser)
                round_reward -= 0.10 * self.reward_config.w_hp_preservation * (outcome.damage_to_loser / 10.0)
        else:
            alive_opps = [o for o in self.opponents if o.alive]
            if alive_opps:
                opp = alive_opps[int(self.rng.integers(0, len(alive_opps)))]
                outcome = self.sim.simulate(self.board, opp.board, stage_level=self.stage_level, rng=self.rng)
                if outcome.winner >= 0:
                    self.gold += 1  # +1g immediate PvP win bonus
                    self.win_streak += 1
                    if self.lose_streak >= 3 and not self.cashout_triggered:
                        # Cashout bonus for Lose-Streak / Cyborg / Fossils archetype transitioning to win!
                        self.cashout_triggered = True
                        round_reward += 0.80 * self.reward_config.w_lose_streak
                    self.lose_streak = 0
                    opp.hp = max(0, opp.hp - outcome.damage_to_loser)
                    if opp.hp == 0:
                        opp.alive = False
                        opp.eliminated_stage = self.stage_level
                    round_reward += 0.20 * self.reward_config.w_win_streak * (1.0 + 0.25 * min(5, self.win_streak))
                else:
                    self.hp = max(0, self.hp - outcome.damage_to_loser)
                    self.lose_streak += 1
                    self.win_streak = 0
                    if self.stage_level <= 12 and self.reward_config.w_lose_streak > 0:
                        # Controlled early lose-streak reward if gold >= 10g
                        round_reward += 0.18 * self.reward_config.w_lose_streak * min(5, self.lose_streak) / 5.0
                    else:
                        round_reward -= 0.12 * self.reward_config.w_hp_preservation * (outcome.damage_to_loser / 10.0)

        # 2. Simulate fights between the remaining 7 opponents in the lobby
        alive_opps = [o for o in self.opponents if o.alive]
        if not is_pve and len(alive_opps) >= 2:
            shuffled = list(alive_opps)
            self.rng.shuffle(shuffled)
            for i in range(0, len(shuffled) - 1, 2):
                o1, o2 = shuffled[i], shuffled[i + 1]
                out = self.sim.simulate(o1.board, o2.board, stage_level=self.stage_level, rng=self.rng)
                if out.winner >= 0:
                    o2.hp = max(0, o2.hp - out.damage_to_loser)
                    if o2.hp == 0:
                        o2.alive = False
                        o2.eliminated_stage = self.stage_level
                else:
                    o1.hp = max(0, o1.hp - out.damage_to_loser)
                    if o1.hp == 0:
                        o1.alive = False
                        o1.eliminated_stage = self.stage_level

        # 3. Check termination conditions (player eliminated, all opponents eliminated, or stage 40 complete)
        alive_opps_after = [o for o in self.opponents if o.alive]
        if self.hp <= 0 or len(alive_opps_after) == 0 or self.stage_level >= 40:
            placement = self._compute_final_placement()
            # Placement reward from +3.0 (1st) to -3.0 (8th)
            placement_reward = (4.5 - placement) * (3.0 / 3.5) * self.reward_config.w_placement
            round_reward += placement_reward
            counts, active_tiers = compute_active_synergies(self.board)
            info = {
                "placement": placement,
                "top4": placement <= 4,
                "win": placement == 1,
                "final_stage": self.stage_level,
                "final_hp": self.hp,
                "final_level": self.level,
                "evolutions_2star": self.total_evolutions_2star,
                "evolutions_3star": self.total_evolutions_3star,
                "active_synergies": active_tiers,
                "synergy_counts": counts,
            }
            return round_reward, True, info

        # 4. End-of-round PAC Income (`game-commands.ts:1417-1454`)
        interest_gold = min(5, self.gold // 10)
        streak_gold = 0 if is_pve else min(5, max(self.win_streak, self.lose_streak))
        self.gold += 5 + interest_gold + streak_gold

        # Interest & synergy shaping rewards per round
        round_reward += 0.12 * self.reward_config.w_interest * (interest_gold / 5.0)
        _, active_tiers = compute_active_synergies(self.board)
        if self.reward_config.target_synergy:
            t_tier = active_tiers.get(self.reward_config.target_synergy, 0)
            round_reward += 0.10 * self.reward_config.w_synergy_focus * t_tier
        else:
            round_reward += 0.05 * self.reward_config.w_synergy_focus * max(active_tiers.values(), default=0)

        # Automatic +2 XP per stage
        if self.level < MAX_LEVEL:
            self.xp += 2
            while self.level < MAX_LEVEL and self.xp >= EXP_TABLE[self.level]:
                self.xp -= EXP_TABLE[self.level]
                self.level += 1

        # Advance to next stage
        self.stage_level += 1
        self.actions_this_stage = 0
        self.locked_this_turn = False
        self._handle_special_stage_propositions()
        if not self.shop_locked:
            self._roll_shop()
        else:
            self.shop_locked = False

        return round_reward, False, {"stage": self.stage_level, "hp": self.hp}

    def _compute_final_placement(self) -> int:
        """Computes exact 1..8 rank in the 8-player lobby."""
        better_opponents = 0
        for o in self.opponents:
            if o.alive and self.hp <= 0:
                better_opponents += 1
            elif o.alive and self.hp > 0:
                if o.hp > self.hp:
                    better_opponents += 1
            elif not o.alive and self.hp <= 0:
                if o.eliminated_stage > self.stage_level:
                    better_opponents += 1
        return int(min(8, max(1, 1 + better_opponents)))

    def extract_interpretable_features(self) -> Dict[str, float]:
        """
        Extracts 18 named, human-readable game state features for VIPER Decision Tree
        distillation (`sklearn.tree.DecisionTreeClassifier`).
        """
        my_power = self.compute_board_power()
        alive_opps = [o for o in self.opponents if o.alive]
        avg_opp_power = (
            float(np.mean([self.compute_board_power(o.board) for o in alive_opps]))
            if alive_opps
            else my_power
        )
        counts, active_tiers = compute_active_synergies(self.board)
        shop_pairs = 0
        shop_synergy_hits = 0
        min_shop_cost = 99
        for p in self.shop:
            if p is not None:
                min_shop_cost = min(min_shop_cost, p.cost)
                if self._count_copies_owned(p.name) >= 1:
                    shop_pairs += 1
                if self._count_family_synergy_fit(p) >= 1.0:
                    shop_synergy_hits += 1

        xp_to_next = max(0, EXP_TABLE.get(self.level, 72) - self.xp)
        return {
            "stage_level": float(self.stage_level),
            "player_hp": float(self.hp),
            "gold": float(self.gold),
            "interest_tier": float(min(5, self.gold // 10)),
            "player_level": float(self.level),
            "xp_to_next_level": float(xp_to_next),
            "win_streak": float(self.win_streak),
            "lose_streak": float(self.lose_streak),
            "board_size": float(len(self.board)),
            "free_board_slots": float(max(0, self.level - len(self.board))),
            "bench_size": float(len(self.bench)),
            "relative_board_power": float(my_power / max(1.0, avg_opp_power)),
            "shop_pair_matches": float(shop_pairs),
            "shop_synergy_matches": float(shop_synergy_hits),
            "num_2star_units": float(sum(1 for u in self.board if u.stars == 2)),
            "num_3star_units": float(sum(1 for u in self.board if u.stars >= 3)),
            "max_synergy_tier": float(max(active_tiers.values(), default=0)),
            "alive_opponents": float(len(alive_opps)),
        }

    def _get_obs(self) -> np.ndarray:
        """Builds a fixed-size observation vector of shape `(OBS_DIM=112,)`."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        feats = self.extract_interpretable_features()

        # 0..17: Core scalar features (normalized)
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

        # 18..41: Per-shop-slot features (6 slots * 4 features: present, cost, copies_owned, synergy_fit)
        for i in range(SHOP_SIZE):
            base = 18 + i * 4
            p = self.shop[i]
            if p is not None:
                obs[base + 0] = 1.0
                obs[base + 1] = p.cost / 5.0
                obs[base + 2] = min(3, self._count_copies_owned(p.name)) / 2.0
                obs[base + 3] = min(4.0, self._count_family_synergy_fit(p)) / 3.0

        # 42..72: Active board synergy vector (31 dims)
        counts, active_tiers = compute_active_synergies(self.board)
        obs[42 : 42 + NUM_SYNERGIES] = encode_synergy_vector(active_tiers, counts)

        # 73..111: Agent Reward Conditioning Vector (39 dims: 8 weights + 31 target synergy one-hot)
        obs[73:112] = self.reward_config.to_Condition_vector()

        return obs
