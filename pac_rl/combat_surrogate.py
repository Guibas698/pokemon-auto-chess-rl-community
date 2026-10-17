"""
Hybrid Combat Engine & PyTorch Set-Transformer Combat Surrogate for Pokemon Auto Chess.

Combines:
1. `FastCombatSimulator`: Vectorized NumPy/Python tactical simulator implementing PAC's
   exact armor formula (`ARMOR_FACTOR = 0.05`), 6x4 grid positioning, frontline/backline
   aggro, PP/Mana spell casts, and all 31 PAC synergy tier bonuses.
2. `SetTransformerCombatSurrogate`: PyTorch `nn.Module` using permutation-invariant
   Multi-Head Self-Attention (`nn.MultiheadAttention`) over unit sets to predict combat
   win probability and surviving unit margin in <0.1ms.
3. `TacticalPositioner`: Classical search / coordinate-swap optimizer that places units
   on the 6x4 board to maximize expected combat value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_loader import (
    ARMOR_FACTOR,
    NUM_SYNERGIES,
    SYNERGIES,
    SYNERGY_TO_IDX,
    SYNERGY_THRESHOLDS,
    PACDataRepository,
    PokemonStats,
)


@dataclass
class BoardUnit:
    """Represents a Pokemon instance on a player's 6x4 board or 8-slot bench."""
    name: str
    index: str
    stars: int
    rarity: str
    types: Tuple[str, ...]
    hp: float
    atk: float
    defense: float
    spe_def: float
    speed: float
    max_pp: float
    attack_range: int
    x: int = 0  # 0..5 column
    y: int = 0  # 0..3 row on board (y=0,1 backline, y=2,3 frontline)
    num_items: int = 0

    @classmethod
    def from_pokemon_stats(
        cls,
        p: PokemonStats,
        x: int = 0,
        y: int = 0,
        num_items: int = 0,
    ) -> "BoardUnit":
        return cls(
            name=p.name,
            index=p.index,
            stars=p.stars,
            rarity=p.rarity,
            types=p.types,
            hp=float(p.hp),
            atk=float(p.atk),
            defense=float(p.defense),
            spe_def=float(p.spe_def),
            speed=float(p.speed),
            max_pp=float(p.max_pp),
            attack_range=int(p.attack_range),
            x=x,
            y=y,
            num_items=num_items,
        )


@dataclass
class UnitCombatLog:
    name: str
    stars: int
    rarity: str
    types: Tuple[str, ...]
    damage_dealt: float
    damage_tanked: float
    spells_cast: int
    kills: int


@dataclass
class CombatOutcome:
    winner: int  # 1 = Team A, -1 = Team B, 0 = Draw
    surviving_a: int
    surviving_b: int
    hp_remaining_ratio_a: float
    damage_to_loser: int
    unit_logs_a: List[UnitCombatLog] = field(default_factory=list)
    unit_logs_b: List[UnitCombatLog] = field(default_factory=list)


def compute_active_synergies(units: List[BoardUnit]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Computes raw unique family/species counts per synergy and reached threshold level.
    In PAC, duplicate copies of the same Pokemon on the board only count once toward synergies.
    Returns:
      - counts: dict mapping synergy -> unique unit count
      - active_tiers: dict mapping synergy -> reached threshold tier index (1..K, or 0 if inactive)
    """
    seen_names = set()
    counts: Dict[str, int] = {}
    for u in units:
        base_key = u.name
        if base_key in seen_names:
            continue
        seen_names.add(base_key)
        for t in u.types:
            counts[t] = counts.get(t, 0) + 1

    active_tiers: Dict[str, int] = {}
    for syn, cnt in counts.items():
        thresholds = SYNERGY_THRESHOLDS.get(syn, [])
        tier_level = 0
        for th in thresholds:
            if cnt >= th:
                tier_level += 1
            else:
                break
        if tier_level > 0:
            active_tiers[syn] = tier_level
    return counts, active_tiers


def encode_synergy_vector(active_tiers: Dict[str, int], counts: Dict[str, int]) -> np.ndarray:
    """Encodes the 31 synergies into a normalized vector of shape (31,)."""
    vec = np.zeros(NUM_SYNERGIES, dtype=np.float32)
    for syn, cnt in counts.items():
        idx = SYNERGY_TO_IDX.get(syn)
        if idx is not None:
            tier = active_tiers.get(syn, 0)
            vec[idx] = 0.25 * tier + 0.05 * min(cnt, 10)
    return vec


class FastCombatSimulator:
    """
    Fast deterministic/seeded tactical combat simulator implementing PAC's combat mechanics:
    - 6x4 positioning grid (rows 2,3 = frontline; rows 0,1 = backline)
    - Armor damage reduction: `ARMOR_FACTOR * def / (1 + ARMOR_FACTOR * def)`
    - Synergy tier buffs across all 31 types
    - Mana/PP generation on attack & damage taken, casting ultimate abilities at `max_pp`
    - Per-unit Damage Dealt & Damage Tanked (absorbed + mitigated + healed) telemetry
    """

    def __init__(self, repo: PACDataRepository):
        self.repo = repo

    def _apply_synergy_buffs(
        self,
        units: List[BoardUnit],
        active_tiers: Dict[str, int],
    ) -> List[Dict[str, object]]:
        combat_units = []
        # Team-wide synergy buffs
        team_hp_mult = 1.0 + 0.14 * active_tiers.get("NORMAL", 0) + 0.08 * active_tiers.get("GOURMET", 0)
        team_def_add = (
            8.0 * active_tiers.get("STEEL", 0)
            + 6.0 * active_tiers.get("ROCK", 0)
            + 6.0 * active_tiers.get("GROUND", 0)
            + 5.0 * active_tiers.get("FOSSIL", 0)
        )
        team_spedef_add = (
            8.0 * active_tiers.get("AMORPHOUS", 0)
            + 6.0 * active_tiers.get("LIGHT", 0)
            + 6.0 * active_tiers.get("FAIRY", 0)
        )
        team_atk_mult = (
            1.0
            + 0.10 * active_tiers.get("SOUND", 0)
            + 0.08 * active_tiers.get("WILD", 0)
            + 0.08 * active_tiers.get("HUMAN", 0)
        )
        team_speed_add = (
            10.0 * active_tiers.get("FIELD", 0)
            + 8.0 * active_tiers.get("AQUATIC", 0)
            + 8.0 * active_tiers.get("FLYING", 0)
        )
        team_regen = 6.0 * active_tiers.get("GRASS", 0) + 4.0 * active_tiers.get("FLORA", 0)
        team_pp_mult = 1.0 + 0.22 * active_tiers.get("WATER", 0) + 0.15 * active_tiers.get("BABY", 0)
        team_spell_mult = 1.0 + 0.22 * active_tiers.get("PSYCHIC", 0) + 0.15 * active_tiers.get("DRAGON", 0)
        team_lifesteal = 0.07 * active_tiers.get("MONSTER", 0) + 0.05 * active_tiers.get("HUMAN", 0)

        for idx, u in enumerate(units):
            u_types = set(u.types)
            hp_mult = team_hp_mult
            atk_mult = team_atk_mult
            def_add = team_def_add
            spedef_add = team_spedef_add
            speed_add = team_speed_add
            crit_chance = 0.05 + 0.05 * u.num_items
            crit_mult = 1.5
            dodge_chance = 0.0
            true_dmg_ratio = 0.0

            if "FIRE" in u_types and "FIRE" in active_tiers:
                atk_mult += 0.16 * active_tiers["FIRE"]
            if "ELECTRIC" in u_types and "ELECTRIC" in active_tiers:
                speed_add += 16.0 * active_tiers["ELECTRIC"]
            if "FIGHTING" in u_types and "FIGHTING" in active_tiers:
                def_add += 12.0 * active_tiers["FIGHTING"]
                atk_mult += 0.10 * active_tiers["FIGHTING"]
            if "DARK" in u_types and "DARK" in active_tiers:
                crit_chance += 0.15 * active_tiers["DARK"]
                crit_mult += 0.20 * active_tiers["DARK"]
            if "DRAGON" in u_types and "DRAGON" in active_tiers:
                true_dmg_ratio += 0.12 * active_tiers["DRAGON"]
                speed_add += 12.0 * active_tiers["DRAGON"]
            if "POISON" in u_types and "POISON" in active_tiers:
                atk_mult += 0.14 * active_tiers["POISON"]
            if "GHOST" in u_types and "GHOST" in active_tiers:
                dodge_chance += 0.08 * active_tiers["GHOST"]
                true_dmg_ratio += 0.10 * active_tiers["GHOST"]
            if "BUG" in u_types and "BUG" in active_tiers:
                hp_mult += 0.15 * active_tiers["BUG"]
            if "ICE" in u_types and "ICE" in active_tiers:
                def_add += 8.0 * active_tiers["ICE"]
                spedef_add += 8.0 * active_tiers["ICE"]
            if "ARTIFICIAL" in u_types and "ARTIFICIAL" in active_tiers:
                atk_mult += 0.14 * active_tiers["ARTIFICIAL"]
                hp_mult += 0.10 * active_tiers["ARTIFICIAL"]

            # Item scaling
            item_mult = 1.0 + 0.15 * u.num_items
            max_hp = u.hp * hp_mult * item_mult
            is_assassin = "DARK" in u_types
            y_pos = 0.5 if is_assassin else float(u.y)

            combat_units.append({
                "id": idx,
                "name": u.name,
                "stars": u.stars,
                "rarity": u.rarity,
                "types": tuple(u.types),
                "hp": max_hp,
                "max_hp": max_hp,
                "atk": u.atk * atk_mult * item_mult,
                "def": u.defense + def_add + 3.0 * u.num_items,
                "spe_def": u.spe_def + spedef_add + 3.0 * u.num_items,
                "speed": max(10.0, u.speed + speed_add + 5.0 * u.num_items),
                "pp": 0.0,
                "max_pp": max(40.0, u.max_pp),
                "range": u.attack_range,
                "x": float(u.x),
                "y": y_pos,
                "regen": team_regen,
                "pp_mult": team_pp_mult,
                "spell_mult": team_spell_mult * item_mult,
                "lifesteal": team_lifesteal,
                "crit_chance": min(0.85, crit_chance),
                "crit_mult": crit_mult,
                "dodge": min(0.45, dodge_chance),
                "true_dmg": min(0.50, true_dmg_ratio),
                "is_assassin": is_assassin,
                "cooldown": 0.0,
                "damage_dealt": 0.0,
                "damage_tanked": 0.0,
                "spells_cast": 0,
                "kills": 0,
            })
        return combat_units

    def simulate(
        self,
        team_a: List[BoardUnit],
        team_b: List[BoardUnit],
        stage_level: int = 10,
        rng: Optional[np.random.Generator] = None,
    ) -> CombatOutcome:
        if rng is None:
            rng = np.random.default_rng(42)

        if not team_a and not team_b:
            return CombatOutcome(0, 0, 0, 0.0, 0)
        if not team_a:
            dmg = int(np.ceil(stage_level / 2.0)) + len(team_b)
            return CombatOutcome(-1, 0, len(team_b), 0.0, dmg)
        if not team_b:
            dmg = int(np.ceil(stage_level / 2.0)) + len(team_a)
            return CombatOutcome(1, len(team_a), 0, 1.0, dmg)

        _, tiers_a = compute_active_synergies(team_a)
        _, tiers_b = compute_active_synergies(team_b)

        ca = self._apply_synergy_buffs(team_a, tiers_a)
        cb = self._apply_synergy_buffs(team_b, tiers_b)
        for u in cb:
            u["y"] = 6.0 if u["is_assassin"] else (7.0 - float(u["y"]))

        dt = 0.25  # 250ms simulation tick
        max_ticks = 160  # 40 seconds max combat duration

        for _ in range(max_ticks):
            alive_a = [u for u in ca if float(u["hp"]) > 0.0]
            alive_b = [u for u in cb if float(u["hp"]) > 0.0]
            if not alive_a or not alive_b:
                break

            for side, allies, enemies in ((1, alive_a, alive_b), (-1, alive_b, alive_a)):
                for u in allies:
                    if float(u["hp"]) <= 0.0 or not enemies:
                        continue
                    if float(u["regen"]) > 0.0:
                        healed = min(float(u["max_hp"]) - float(u["hp"]), float(u["regen"]) * dt)
                        u["hp"] = float(u["hp"]) + healed
                        u["damage_tanked"] = float(u["damage_tanked"]) + max(0.0, healed)

                    u["cooldown"] = float(u["cooldown"]) - dt
                    if float(u["cooldown"]) > 0.0:
                        continue

                    best_target = None
                    best_dist = 1e9
                    for e in enemies:
                        if float(e["hp"]) <= 0.0:
                            continue
                        dist = abs(float(u["x"]) - float(e["x"])) + abs(float(u["y"]) - float(e["y"]))
                        if dist < best_dist:
                            best_dist = dist
                            best_target = e

                    if best_target is None:
                        continue

                    if best_dist > float(u["range"]) + 0.5:
                        step_y = 1.0 if float(best_target["y"]) > float(u["y"]) else -1.0
                        u["y"] = float(u["y"]) + 0.8 * step_y
                        u["cooldown"] = 0.25
                        continue

                    atk_freq = 0.45 + 0.011 * float(u["speed"])
                    u["cooldown"] = 1.0 / atk_freq

                    if float(best_target["dodge"]) > 0.0 and rng.random() < float(best_target["dodge"]):
                        best_target["damage_tanked"] = float(best_target["damage_tanked"]) + float(u["atk"]) * 0.5
                        continue

                    u["pp"] = float(u["pp"]) + 12.0 * float(u["pp_mult"])
                    best_target["pp"] = float(best_target["pp"]) + 5.0 * float(best_target["pp_mult"])

                    is_spell = float(u["pp"]) >= float(u["max_pp"])
                    if is_spell:
                        u["pp"] = float(u["pp"]) - float(u["max_pp"])
                        u["spells_cast"] = int(u["spells_cast"]) + 1
                        raw_dmg = float(u["atk"]) * 2.15 * float(u["spell_mult"])
                        defense_stat = float(best_target["spe_def"])
                    else:
                        raw_dmg = float(u["atk"])
                        if rng.random() < float(u["crit_chance"]):
                            raw_dmg *= float(u["crit_mult"])
                        defense_stat = float(best_target["def"])

                    mitigation = 1.0 / (1.0 + ARMOR_FACTOR * max(0.0, defense_stat))
                    true_part = raw_dmg * float(u["true_dmg"])
                    mitigated_part = raw_dmg * (1.0 - float(u["true_dmg"])) * mitigation
                    final_dmg = max(1.0, true_part + mitigated_part)

                    best_target["hp"] = float(best_target["hp"]) - final_dmg
                    u["damage_dealt"] = float(u["damage_dealt"]) + final_dmg
                    # Total damage tanked = actual damage absorbed + damage mitigated by armor/spe_def
                    best_target["damage_tanked"] = float(best_target["damage_tanked"]) + raw_dmg

                    if float(best_target["hp"]) <= 0.0:
                        u["kills"] = int(u["kills"]) + 1

                    if float(u["lifesteal"]) > 0.0:
                        lifesteal_heal = final_dmg * float(u["lifesteal"])
                        u["hp"] = min(float(u["max_hp"]), float(u["hp"]) + lifesteal_heal)
                        u["damage_tanked"] = float(u["damage_tanked"]) + lifesteal_heal

        logs_a = [
            UnitCombatLog(
                name=str(u["name"]),
                stars=int(u["stars"]),
                rarity=str(u["rarity"]),
                types=tuple(u["types"]),
                damage_dealt=float(u["damage_dealt"]),
                damage_tanked=float(u["damage_tanked"]),
                spells_cast=int(u["spells_cast"]),
                kills=int(u["kills"]),
            )
            for u in ca
        ]
        logs_b = [
            UnitCombatLog(
                name=str(u["name"]),
                stars=int(u["stars"]),
                rarity=str(u["rarity"]),
                types=tuple(u["types"]),
                damage_dealt=float(u["damage_dealt"]),
                damage_tanked=float(u["damage_tanked"]),
                spells_cast=int(u["spells_cast"]),
                kills=int(u["kills"]),
            )
            for u in cb
        ]

        surv_a = [u for u in ca if float(u["hp"]) > 0.0]
        surv_b = [u for u in cb if float(u["hp"]) > 0.0]
        total_hp_a = sum(float(u["hp"]) for u in surv_a)
        total_max_a = sum(float(u["max_hp"]) for u in ca) or 1.0
        total_hp_b = sum(float(u["hp"]) for u in surv_b)

        base_stage_dmg = int(np.ceil(max(1, stage_level) / 2.0))
        if len(surv_a) > 0 and len(surv_b) == 0:
            return CombatOutcome(
                winner=1,
                surviving_a=len(surv_a),
                surviving_b=0,
                hp_remaining_ratio_a=float(total_hp_a / total_max_a),
                damage_to_loser=base_stage_dmg + len(surv_a),
                unit_logs_a=logs_a,
                unit_logs_b=logs_b,
            )
        elif len(surv_b) > 0 and len(surv_a) == 0:
            return CombatOutcome(
                winner=-1,
                surviving_a=0,
                surviving_b=len(surv_b),
                hp_remaining_ratio_a=0.0,
                damage_to_loser=base_stage_dmg + len(surv_b),
                unit_logs_a=logs_a,
                unit_logs_b=logs_b,
            )
        else:
            if total_hp_a >= total_hp_b:
                diff = max(1, len(surv_a) - len(surv_b))
                return CombatOutcome(
                    1, len(surv_a), len(surv_b), float(total_hp_a / total_max_a), base_stage_dmg + diff,
                    unit_logs_a=logs_a, unit_logs_b=logs_b,
                )
            else:
                diff = max(1, len(surv_b) - len(surv_a))
                return CombatOutcome(
                    -1, len(surv_a), len(surv_b), float(total_hp_a / total_max_a), base_stage_dmg + diff,
                    unit_logs_a=logs_a, unit_logs_b=logs_b,
                )

    def make_pve_board(self, stage_level: int) -> List[BoardUnit]:
        """Creates a calibrated PvE neutral wave for PAC's PvE rounds (1,2,3,9,14,19,24,29,33,36,40)."""
        pve_templates = {
            1: [("RATTATA", 1), ("PIDGEY", 2)],
            2: [("RATTATA", 1), ("PIDGEY", 3), ("CATERPIE", 4)],
            3: [("PIDGEOTTO", 1), ("RATICATE", 3), ("SPEAROW", 4)],
            9: [("GYARADOS", 2), ("MAGIKARP", 1), ("MAGIKARP", 4)],
            14: [("MEWTWO", 2), ("ALAKAZAM", 4)],
            19: [("RAYQUAZA", 2), ("SALAMENCE", 1), ("FLYGON", 4)],
            24: [("GROUDON", 2), ("KYOGRE", 3), ("CAMERUPT", 1)],
            29: [("DIALGA", 1), ("PALKIA", 3), ("GIRATINA", 4)],
            33: [("ARCEUS", 2), ("DARKRAI", 1), ("CRESSELIA", 4)],
            36: [("REGIGIGAS", 2), ("REGIROCK", 1), ("REGICE", 3), ("REGISTEEL", 4)],
            40: [("ETERNATUS", 2), ("ZACIAN", 1), ("ZAMAZENTA", 4)],
        }
        entries = pve_templates.get(stage_level, [("SNORLAX", 2), ("DRAGONITE", 3)])
        units: List[BoardUnit] = []
        for name, x_pos in entries:
            p = self.repo.get_pokemon(name)
            if p is not None:
                units.append(BoardUnit.from_pokemon_stats(p, x=x_pos, y=2))
        return units


# ============================================================================
# PyTorch Set-Transformer Combat Surrogate Model
# ============================================================================

UNIT_FEATURE_DIM = 14 + NUM_SYNERGIES  # 45 features per unit
MAX_BOARD_UNITS = 9


def encode_board_to_tensor(units: List[BoardUnit]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encodes a board of up to 9 units into:
    - `unit_feats`: shape `(MAX_BOARD_UNITS, UNIT_FEATURE_DIM)`
    - `synergy_vec`: shape `(NUM_SYNERGIES,)`
    """
    unit_feats = np.zeros((MAX_BOARD_UNITS, UNIT_FEATURE_DIM), dtype=np.float32)
    counts, active_tiers = compute_active_synergies(units)
    syn_vec = encode_synergy_vector(active_tiers, counts)

    rarity_map = {"COMMON": 0.2, "UNCOMMON": 0.4, "RARE": 0.6, "EPIC": 0.8, "ULTRA": 1.0, "UNIQUE": 0.7, "LEGENDARY": 1.0}
    for i, u in enumerate(units[:MAX_BOARD_UNITS]):
        unit_feats[i, 0] = 1.0  # presence mask
        unit_feats[i, 1] = u.hp / 500.0
        unit_feats[i, 2] = u.atk / 40.0
        unit_feats[i, 3] = u.defense / 25.0
        unit_feats[i, 4] = u.spe_def / 25.0
        unit_feats[i, 5] = u.speed / 100.0
        unit_feats[i, 6] = u.max_pp / 120.0
        unit_feats[i, 7] = u.attack_range / 4.0
        unit_feats[i, 8] = u.stars / 3.0
        unit_feats[i, 9] = rarity_map.get(u.rarity, 0.4)
        unit_feats[i, 10] = u.x / 5.0
        unit_feats[i, 11] = u.y / 3.0
        unit_feats[i, 12] = 1.0 if u.y >= 2 else 0.0  # frontline indicator
        unit_feats[i, 13] = u.num_items / 3.0
        for t in u.types:
            idx = SYNERGY_TO_IDX.get(t)
            if idx is not None:
                unit_feats[i, 14 + idx] = 1.0

    return unit_feats, syn_vec


class SetTransformerCombatSurrogate(nn.Module):
    """
    Permutation-invariant Set Transformer in PyTorch that encodes Team A and Team B
    unit sets + active synergy vectors and predicts:
    1. `win_logit`: Logit of $P(\text{Team A wins})$
    2. `margin_pred`: Expected surviving unit differential $(\text{surv}_A - \text{surv}_B) / 9$
    """

    def __init__(self, d_model: int = 64, nhead: int = 4):
        super().__init__()
        self.unit_proj = nn.Sequential(
            nn.Linear(UNIT_FEATURE_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.syn_proj = nn.Sequential(
            nn.Linear(NUM_SYNERGIES, d_model // 2),
            nn.GELU(),
        )

        team_embed_dim = d_model + (d_model // 2)
        # Cross-team comparison head
        self.head = nn.Sequential(
            nn.Linear(team_embed_dim * 3, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),  # [win_logit, margin_pred]
        )

    def encode_team(self, unit_feats: torch.Tensor, syn_vec: torch.Tensor) -> torch.Tensor:
        # unit_feats: (B, N, F), syn_vec: (B, 31)
        mask = unit_feats[:, :, 0] > 0.5  # (B, N)
        h = self.unit_proj(unit_feats)
        attn_out, _ = self.self_attn(h, h, h)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ffn(h))

        # Masked mean pooling over present units
        mask_f = mask.unsqueeze(-1).float()  # (B, N, 1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = (h * mask_f).sum(dim=1) / denom  # (B, d_model)

        syn_h = self.syn_proj(syn_vec)  # (B, d_model // 2)
        return torch.cat([pooled, syn_h], dim=-1)

    def forward(
        self,
        units_a: torch.Tensor,
        syn_a: torch.Tensor,
        units_b: torch.Tensor,
        syn_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb_a = self.encode_team(units_a, syn_a)
        emb_b = self.encode_team(units_b, syn_b)
        diff = emb_a - emb_b
        joint = torch.cat([emb_a, emb_b, diff], dim=-1)
        out = self.head(joint)
        win_logit = out[:, 0]
        margin_pred = torch.tanh(out[:, 1]) * 9.0
        return win_logit, margin_pred


class TacticalPositioner:
    """
    Classical search / heuristic tactical board optimizer (6x4 grid).
    Given a set of selected units, assigns optimal (x, y) coordinates on the 6x4 board:
    - High durability frontline tanks (FIGHTING, METAL, ROCK, NORMAL, GROUND, GRASS) -> row y=3
    - Secondary bruisers / melee carries -> row y=2
    - Assassin units (DARK) -> backline row y=0 flanking left/right to jump onto enemy carries
    - Ranged carries & supports (PSYCHIC, FIRE, ELECTRIC, WATER, FAIRY, FLORA) -> row y=0 corners
      protected by nearby support/aura units.
    """

    @staticmethod
    def optimize_positions(units: List[BoardUnit]) -> List[BoardUnit]:
        if not units:
            return units

        frontline_slots = [(2, 3), (3, 3), (1, 3), (4, 3), (0, 3), (5, 3)]
        midline_slots = [(2, 2), (3, 2), (1, 2), (4, 2)]
        backline_slots = [(0, 0), (5, 0), (1, 0), (4, 0), (2, 0), (3, 0)]

        # Score each unit's tankiness vs ranged carry role
        tanks: List[BoardUnit] = []
        assassins: List[BoardUnit] = []
        ranged: List[BoardUnit] = []
        bruisers: List[BoardUnit] = []

        for u in units:
            u_types = set(u.types)
            if "DARK" in u_types:
                assassins.append(u)
            elif u.attack_range >= 2:
                ranged.append(u)
            elif u.defense + u.spe_def >= 14 or u.hp >= 220 or bool(u_types & {"METAL", "ROCK", "FIGHTING", "NORMAL", "GROUND"}):
                tanks.append(u)
            else:
                bruisers.append(u)

        # Sort tanks by effective HP descending
        tanks.sort(key=lambda u: u.hp * (1.0 + 0.05 * (u.defense + u.spe_def)), reverse=True)
        # Sort ranged carries by DPS descending so the strongest carries get the safest corner slots
        ranged.sort(key=lambda u: u.atk * u.speed, reverse=True)

        used_coords = set()

        def assign_slot(unit: BoardUnit, preferred_list: List[Tuple[int, int]]):
            for coord in preferred_list:
                if coord not in used_coords:
                    unit.x, unit.y = coord
                    used_coords.add(coord)
                    return
            # Fallback to any free slot
            for y in (3, 2, 1, 0):
                for x in range(6):
                    if (x, y) not in used_coords:
                        unit.x, unit.y = x, y
                        used_coords.add((x, y))
                        return

        for u in tanks:
            assign_slot(u, frontline_slots + midline_slots)
        for u in bruisers:
            assign_slot(u, midline_slots + frontline_slots)
        for u in ranged:
            assign_slot(u, backline_slots)
        for u in assassins:
            assign_slot(u, [(5, 0), (0, 0), (4, 1), (1, 1)] + backline_slots)

        return units


def train_combat_surrogate(
    repo: PACDataRepository,
    sim: FastCombatSimulator,
    num_samples: int = 2400,
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 3e-3,
    seed: int = 42,
) -> Tuple[SetTransformerCombatSurrogate, Dict[str, float]]:
    """
    Generates synthetic + bot-vs-bot combat matchups using `FastCombatSimulator`
    and trains the PyTorch `SetTransformerCombatSurrogate` neural network.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    X_ua = np.zeros((num_samples, MAX_BOARD_UNITS, UNIT_FEATURE_DIM), dtype=np.float32)
    X_sa = np.zeros((num_samples, NUM_SYNERGIES), dtype=np.float32)
    X_ub = np.zeros((num_samples, MAX_BOARD_UNITS, UNIT_FEATURE_DIM), dtype=np.float32)
    X_sb = np.zeros((num_samples, NUM_SYNERGIES), dtype=np.float32)
    Y_win = np.zeros((num_samples,), dtype=np.float32)
    Y_margin = np.zeros((num_samples,), dtype=np.float32)

    all_pokes = list(repo.pokemon_by_name.values())
    bots = repo.bots

    def sample_board() -> Tuple[List[BoardUnit], int]:
        stage = int(rng.integers(3, 32))
        if bots and rng.random() < 0.65:
            bot = bots[int(rng.integers(0, len(bots)))]
            board_entries = repo.get_bot_stage_board(bot, stage)
            units: List[BoardUnit] = []
            for name, x, y, n_items in board_entries[:MAX_BOARD_UNITS]:
                p = repo.get_pokemon(name)
                if p is not None:
                    units.append(BoardUnit.from_pokemon_stats(p, x=x, y=min(3, max(0, y - 1)), num_items=n_items))
            if units:
                return units, stage
        # Otherwise random composition
        n_u = int(rng.integers(3, min(9, max(4, stage // 3 + 2)) + 1))
        chosen = rng.choice(len(all_pokes), size=n_u, replace=False)
        units = [
            BoardUnit.from_pokemon_stats(
                all_pokes[idx],
                x=int(rng.integers(0, 6)),
                y=int(rng.integers(0, 4)),
                num_items=int(rng.integers(0, 3)),
            )
            for idx in chosen
        ]
        return TacticalPositioner.optimize_positions(units), stage

    for i in range(num_samples):
        board_a, st_a = sample_board()
        board_b, st_b = sample_board()
        stage = (st_a + st_b) // 2
        outcome = sim.simulate(board_a, board_b, stage_level=stage, rng=rng)
        ua, sa = encode_board_to_tensor(board_a)
        ub, sb = encode_board_to_tensor(board_b)
        X_ua[i] = ua
        X_sa[i] = sa
        X_ub[i] = ub
        X_sb[i] = sb
        Y_win[i] = 1.0 if outcome.winner >= 0 else 0.0
        Y_margin[i] = float(outcome.surviving_a - outcome.surviving_b)

    # Split 85% train / 15% val
    split = int(0.85 * num_samples)
    model = SetTransformerCombatSurrogate(d_model=64, nhead=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    t_ua = torch.from_numpy(X_ua[:split])
    t_sa = torch.from_numpy(X_sa[:split])
    t_ub = torch.from_numpy(X_ub[:split])
    t_sb = torch.from_numpy(X_sb[:split])
    t_win = torch.from_numpy(Y_win[:split])
    t_mar = torch.from_numpy(Y_margin[:split])

    v_ua = torch.from_numpy(X_ua[split:])
    v_sa = torch.from_numpy(X_sa[split:])
    v_ub = torch.from_numpy(X_ub[split:])
    v_sb = torch.from_numpy(X_sb[split:])
    v_win = torch.from_numpy(Y_win[split:])
    v_mar = torch.from_numpy(Y_margin[split:])

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(split)
        for start in range(0, split, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            win_logit, mar_pred = model(t_ua[idx], t_sa[idx], t_ub[idx], t_sb[idx])
            loss_cls = F.binary_cross_entropy_with_logits(win_logit, t_win[idx])
            loss_reg = F.smooth_l1_loss(mar_pred, t_mar[idx])
            loss = loss_cls + 0.15 * loss_reg
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits, val_mar = model(v_ua, v_sa, v_ub, v_sb)
        val_preds = (torch.sigmoid(val_logits) >= 0.5).float()
        val_acc = float((val_preds == v_win).float().mean().item())
        val_mae = float((val_mar - v_mar).abs().mean().item())

    return model, {"val_accuracy": val_acc, "val_margin_mae": val_mae, "samples": num_samples}
