import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Exact constants from upstream/app/config/game/
SHOP_SIZE = 6
MAX_LEVEL = 9
BENCH_SIZE = 8
BOARD_WIDTH = 8
BOARD_SIDE_HEIGHT = 3  # 3 active combat rows per player (y=1,2,3)

# Exp required to advance from level L to L+1 (upstream/app/config/game/experience.ts)
EXP_TABLE: Dict[int, int] = {
    1: 0,
    2: 2,
    3: 6,
    4: 10,
    5: 22,
    6: 34,
    7: 52,
    8: 72,
    9: 255,
}

# Shop rarity probabilities per level: [COMMON, UNCOMMON, RARE, EPIC, ULTRA]
# (upstream/app/config/game/shop.ts)
RARITY_PROBA_PER_LEVEL: Dict[int, List[float]] = {
    1: [1.00, 0.00, 0.00, 0.00, 0.00],
    2: [1.00, 0.00, 0.00, 0.00, 0.00],
    3: [0.70, 0.30, 0.00, 0.00, 0.00],
    4: [0.50, 0.40, 0.10, 0.00, 0.00],
    5: [0.36, 0.42, 0.20, 0.02, 0.00],
    6: [0.25, 0.40, 0.30, 0.05, 0.00],
    7: [0.15, 0.35, 0.35, 0.15, 0.00],
    8: [0.11, 0.27, 0.35, 0.22, 0.05],
    9: [0.05, 0.20, 0.35, 0.30, 0.10],
}

SHOP_RARITIES = ["COMMON", "UNCOMMON", "RARE", "EPIC", "ULTRA"]

RARITY_COST: Dict[str, int] = {
    "SPECIAL": 1,
    "COMMON": 1,
    "UNCOMMON": 2,
    "RARE": 3,
    "EPIC": 4,
    "ULTRA": 5,
    "HATCH": 9,
    "UNIQUE": 10,
    "LEGENDARY": 20,
}

# Copies in shared pool by (rarity, nb_stages_in_family)
# (upstream/app/config/game/pools.ts)
POOL_SIZE: Dict[str, Tuple[int, int, int]] = {
    "COMMON": (1, 18, 27),
    "UNCOMMON": (1, 13, 22),
    "RARE": (1, 9, 18),
    "EPIC": (1, 7, 14),
    "ULTRA": (1, 5, 10),
    "UNIQUE": (1, 1, 1),
    "LEGENDARY": (1, 1, 1),
    "SPECIAL": (0, 0, 0),
    "HATCH": (0, 0, 0),
}

# All 31 Synergies and their activation thresholds (upstream/app/config/game/synergies.ts)
SYNERGY_THRESHOLDS: Dict[str, List[int]] = {
    "NORMAL": [3, 5, 7, 9],
    "GRASS": [3, 5, 7, 9],
    "FIRE": [2, 4, 6, 8],
    "WATER": [3, 6, 9],
    "ELECTRIC": [3, 5, 7],
    "FIGHTING": [2, 4, 6, 8],
    "PSYCHIC": [3, 5, 7],
    "DARK": [3, 5, 7],
    "STEEL": [2, 4, 6, 8],
    "GROUND": [2, 4, 6, 8],
    "POISON": [3, 5, 7],
    "DRAGON": [3, 5, 7],
    "FIELD": [3, 6, 9],
    "MONSTER": [2, 4, 6, 8],
    "HUMAN": [2, 4, 6],
    "AQUATIC": [2, 4, 6, 8],
    "BUG": [2, 4, 6, 8],
    "FLYING": [2, 4, 6, 8],
    "FLORA": [3, 4, 5, 6],
    "ROCK": [2, 4, 6],
    "GHOST": [2, 4, 6, 8],
    "FAIRY": [2, 4, 6, 8],
    "ICE": [2, 4, 6, 8],
    "FOSSIL": [2, 4, 6],
    "SOUND": [2, 4, 6],
    "ARTIFICIAL": [2, 4, 6],
    "BABY": [3, 5, 7],
    "LIGHT": [2, 3, 4, 5],
    "WILD": [2, 4, 6, 9],
    "AMORPHOUS": [3, 5, 7],
    "GOURMET": [3, 4, 5],
}

ALL_SYNERGIES: List[str] = list(SYNERGY_THRESHOLDS.keys())
SYNERGIES: List[str] = ALL_SYNERGIES
NUM_SYNERGIES: int = len(ALL_SYNERGIES)
SYNERGY_TO_IDX: Dict[str, int] = {s: i for i, s in enumerate(ALL_SYNERGIES)}
ARMOR_FACTOR: float = 0.05
SHOP_ODDS_BY_LEVEL: Dict[int, List[float]] = RARITY_PROBA_PER_LEVEL

PVE_STAGES = {1, 2, 3, 9, 14, 19, 24, 29, 33, 36, 40}
ADDITIONAL_PICKS_STAGES = [5, 8, 11]
PORTAL_PROPOSITION_STAGES = {0: "COMMON", 10: "UNIQUE", 20: "LEGENDARY"}
PORTAL_CAROUSEL_STAGES = [0, 10, 20]

# Official ShinyItems from upstream/app/types/enum/Item.ts (lines 645-658)
# High-variance 0.5% items excluded by default to eliminate heavy-tailed critic noise in RL
SHINY_ITEMS = frozenset({
    "DYNAMAX_BAND",
    "SHINY_STONE",
    "RARE_CANDY",
    "EVIOLITE",
    "GOLD_MASK",
    "GOLD_BOTTLE_CAP",
    "ABSORB_BULB",
    "SACRED_ASH",
    "STAR_PIECE",
    "REPEAT_BALL",
    "GOLD_BOW",
    "RED_SCALE",
})
DISABLE_SHINY_ITEMS: bool = True

# Native power evaluation tables from upstream/app/core/bot-logic.ts
POWER_AVERAGES = [
    2, 2, 3, 4, 4.5, 5, 6, 7, 8, 9, 11, 11.5, 12, 13, 14, 15, 16, 17, 18, 19, 24,
    25, 25.5, 26, 27, 27.5, 28, 28.5, 29, 29.5, 30, 30.5, 31, 31.5, 32, 32.5, 33,
    33.5, 34, 34.5, 35, 35.5, 36, 36.5, 37, 37.5, 38, 38.5, 39, 39.5, 40
]


@dataclass
class PokemonSpec:
    index: str
    name: str
    category: str       # COMMON, UNCOMMON, RARE, EPIC, ULTRA, UNIQUE, LEGENDARY, HATCH, SPECIAL
    stars: int          # 1, 2, 3, 4
    additional: bool
    types: List[str]
    hp: float
    atk: float
    speed: float
    defense: float
    spe_def: float
    attack_range: int
    max_pp: float
    ability: str
    family: str
    nb_stages: int
    regional: bool
    next_evolution: Optional[str] = None
    unit_power_score: float = 1.0
    species_id: int = 0

    @property
    def rarity(self) -> str:
        return self.category

    @property
    def evolves_into(self) -> Optional[str]:
        return self.next_evolution

    @property
    def cost(self) -> int:
        return RARITY_COST.get(self.category, 1)

    def effective_shop_cost(self, cap_special_to_regular: bool = False) -> int:
        """
        Returns the gold cost to buy this unit in the shop.
        If `cap_special_to_regular=True`, caps UNIQUE/LEGENDARY/HATCH to standard shop range (5g/6g)
        when testing custom shop pools; otherwise returns exact RARITY_COST (1..5 for shop rarities,
        10 for UNIQUE, 20 for LEGENDARY).
        """
        raw = RARITY_COST.get(self.category, 1)
        return min(6, raw) if cap_special_to_regular else raw

    @property
    def is_frontline_tank(self) -> bool:
        """True if unit has melee range and high effective HP/Armor profile."""
        ehp = self.hp * (1.0 + ARMOR_FACTOR * (self.defense + self.spe_def))
        return self.attack_range <= 1 and (ehp >= 210.0 * self.stars or (self.defense + self.spe_def) >= 12.0)

    @property
    def is_backline_carry(self) -> bool:
        """True if unit is ranged or high-DPS offensive carry."""
        dps_proxy = self.atk * (0.45 + 0.011 * self.speed)
        return self.attack_range >= 2 or dps_proxy >= 14.0 * self.stars


PokemonStats = PokemonSpec


@dataclass
class CommunityBotSpec:
    bot_id: str
    name: str
    author: str
    elo: float
    stage_boards: List[List[Tuple[str, int, int, int]]]  # per stage: list of (pkm_name, x, y, nb_items)

    @property
    def id(self) -> str:
        return self.bot_id


class PACDataRepository:
    """
    Loads the authentic Pokemon Auto Chess dataset from upstream/app/models/precomputed/pokemons-data.csv
    and the 726 community bots from upstream/db-commands/botv2.json.
    """

    def __init__(self, upstream_dir: Optional[Path] = None, disable_shiny_items: bool = DISABLE_SHINY_ITEMS):
        if upstream_dir is None:
            upstream_dir = Path(__file__).resolve().parent.parent / "upstream"
        self.upstream_dir = upstream_dir
        self.disable_shiny_items = disable_shiny_items
        self.shiny_items_removed_from_bots: int = 0
        self.pokemons: Dict[str, PokemonSpec] = {}
        self.species_to_id: Dict[str, int] = {}
        self.family_chains: Dict[str, Dict[int, List[str]]] = {}
        self.regular_pool_tier1: Dict[str, List[str]] = {r: [] for r in SHOP_RARITIES}
        self.additional_pool_tier1: Dict[str, List[str]] = {"UNCOMMON": [], "RARE": [], "EPIC": []}
        self.unique_pool: List[str] = []
        self.legendary_pool: List[str] = []
        self.bots: List[CommunityBotSpec] = []

        self._load_pokemons_csv()
        self._load_bots_json()

    def _compute_unit_power(self, cat: str, stars: int, nb_stages: int) -> float:
        # Matches POWER_SCORE_BY_CATEGORY in upstream/app/core/bot-logic.ts
        base_map = {
            ("COMMON", 1): 1.0,
            ("COMMON", 2): 2.25,
            ("COMMON", 3): 4.0,
            ("UNCOMMON", 1): 1.25,
            ("UNCOMMON", 2): 2.75,
            ("UNCOMMON", 3): 5.0,
            ("RARE", 1): 1.5,
            ("RARE", 2): 3.5,
            ("RARE", 3): 6.5,
            ("HATCH", 1): 1.5,
            ("HATCH", 2): 3.0,
            ("HATCH", 3): 5.0,
            ("EPIC", 1): 2.0,
            ("EPIC", 2): 4.5,
            ("EPIC", 3): 7.5,
            ("ULTRA", 1): 3.0,
            ("ULTRA", 2): 6.0,
            ("ULTRA", 3): 10.0,
            ("UNIQUE", 1): 2.5,
            ("UNIQUE", 2): 3.5,
            ("UNIQUE", 3): 4.5,
            ("LEGENDARY", 1): 5.0,
            ("LEGENDARY", 2): 5.5,
            ("LEGENDARY", 3): 7.0,
            ("SPECIAL", 1): 1.5,
            ("SPECIAL", 2): 3.0,
            ("SPECIAL", 3): 5.0,
        }
        return base_map.get((cat, min(3, max(1, stars))), float(stars) * 1.5)

    def _load_pokemons_csv(self) -> None:
        local_data_csv = Path(__file__).resolve().parent.parent / "data" / "pokemons-data.csv"
        csv_path = (
            local_data_csv
            if local_data_csv.exists()
            else self.upstream_dir / "app" / "models" / "precomputed" / "pokemons-data.csv"
        )
        df = pd.read_csv(csv_path, keep_default_na=False)

        for _, row in df.iterrows():
            name = str(row["Name"]).strip()
            cat = str(row["Category"]).strip().upper()
            try:
                stars = int(row["Tier"]) if str(row["Tier"]).strip() != "" else 1
            except ValueError:
                stars = 1
            additional = str(row["Additional pick"]).strip().lower() == "true"
            regional = str(row["Regional"]).strip().lower() == "true"
            variant = str(row["Variant"]).strip().lower() == "true"
            if variant:
                continue

            types = []
            for tcol in ["Type 1", "Type 2", "Type 3", "Type 4"]:
                val = str(row[tcol]).strip().upper()
                if val and val in SYNERGY_THRESHOLDS:
                    types.append(val)

            try:
                nb_stages = int(row["Nb stages"]) if str(row["Nb stages"]).strip() != "" else 1
            except ValueError:
                nb_stages = 1

            spec = PokemonSpec(
                index=str(row["Index"]).strip(),
                name=name,
                category=cat,
                stars=stars,
                additional=additional,
                types=types,
                hp=float(row["HP"] or 60),
                atk=float(row["Attack"] or 5),
                speed=float(row["Speed"] or 50),
                defense=float(row["Defense"] or 2),
                spe_def=float(row["Special Defense"] or 2),
                attack_range=int(row["Attack Range"] or 1),
                max_pp=float(row["Max PP"] or 100),
                ability=str(row["Ability"]).strip(),
                family=str(row["Family"]).strip() or name,
                nb_stages=nb_stages,
                regional=regional,
                unit_power_score=self._compute_unit_power(cat, stars, nb_stages),
                species_id=len(self.pokemons) + 1,
            )
            self.pokemons[name] = spec
            self.species_to_id[name] = spec.species_id
            fam = spec.family
            if fam not in self.family_chains:
                self.family_chains[fam] = {}
            self.family_chains[fam].setdefault(stars, []).append(name)

            if stars == 1 and not regional:
                if cat in SHOP_RARITIES and not additional:
                    self.regular_pool_tier1[cat].append(name)
                elif cat in self.additional_pool_tier1 and additional:
                    self.additional_pool_tier1[cat].append(name)

            if cat == "UNIQUE" and not regional:
                self.unique_pool.append(name)
            elif cat == "LEGENDARY" and not regional:
                self.legendary_pool.append(name)

        # Link evolution chains (Tier 1 -> Tier 2 -> Tier 3)
        for pkm in self.pokemons.values():
            chain = self.family_chains.get(pkm.family, {})
            next_tier = chain.get(pkm.stars + 1, [])
            if next_tier:
                # Prefer non-regional evolution
                non_reg = [c for c in next_tier if not self.pokemons[c].regional]
                pkm.next_evolution = non_reg[0] if non_reg else next_tier[0]

    def _load_bots_json(self) -> None:
        local_bot_path = Path(__file__).resolve().parent.parent / "data" / "botv2.json"
        bot_path = (
            local_bot_path
            if local_bot_path.exists()
            else self.upstream_dir / "db-commands" / "botv2.json"
        )
        with open(bot_path, "r", encoding="utf-8") as f:
            raw_bots = json.load(f)

        for rb in raw_bots:
            steps = rb.get("steps", [])
            if not steps:
                continue
            expanded_boards: List[List[Tuple[str, int, int, int]]] = []
            for st in steps:
                rounds_req = max(1, int(st.get("roundsRequired", 1) or 1))
                board_units: List[Tuple[str, int, int, int]] = []
                for unit in st.get("board", []):
                    uname = str(unit.get("name", ""))
                    if uname in self.pokemons and int(unit.get("y", 1)) > 0:
                        raw_items = [str(it).strip().upper() for it in unit.get("items", [])]
                        if self.disable_shiny_items:
                            filtered_items = [it for it in raw_items if it not in SHINY_ITEMS]
                            self.shiny_items_removed_from_bots += len(raw_items) - len(filtered_items)
                        else:
                            filtered_items = raw_items
                        board_units.append(
                            (uname, int(unit.get("x", 0)), int(unit.get("y", 1)), len(filtered_items))
                        )
                for _ in range(rounds_req):
                    expanded_boards.append(board_units)
                    if len(expanded_boards) > 35:
                        break
                if len(expanded_boards) > 35:
                    break

            if expanded_boards:
                self.bots.append(
                    CommunityBotSpec(
                        bot_id=str(rb.get("id", "")),
                        name=str(rb.get("name", "BOT")),
                        author=str(rb.get("author", "")),
                        elo=float(rb.get("elo", 1000) or 1000),
                        stage_boards=expanded_boards,
                    )
                )

    @property
    def pokemon_by_name(self) -> Dict[str, PokemonSpec]:
        return self.pokemons

    @property
    def buyable_by_rarity(self) -> Dict[str, List[PokemonSpec]]:
        return {
            r: [self.pokemons[name] for name in names if name in self.pokemons]
            for r, names in self.regular_pool_tier1.items()
        }

    @property
    def additional_by_rarity(self) -> Dict[str, List[PokemonSpec]]:
        return {
            r: [self.pokemons[name] for name in names if name in self.pokemons]
            for r, names in self.additional_pool_tier1.items()
        }

    @property
    def uniques_pool(self) -> List[PokemonSpec]:
        return [self.pokemons[name] for name in self.unique_pool if name in self.pokemons]

    @property
    def legendaries_pool(self) -> List[PokemonSpec]:
        return [self.pokemons[name] for name in self.legendary_pool if name in self.pokemons]

    def get_pokemon(self, name: str) -> Optional[PokemonSpec]:
        return self.pokemons.get(name)

    def get_bot_stage_board(self, bot: CommunityBotSpec, stage: int) -> List[Tuple[str, int, int, int]]:
        if not bot.stage_boards:
            return []
        idx = min(max(0, stage - 1), len(bot.stage_boards) - 1)
        return bot.stage_boards[idx]

    def summary(self) -> Dict[str, object]:
        by_rarity: Dict[str, int] = {}
        for p in self.pokemons.values():
            by_rarity[p.category] = by_rarity.get(p.category, 0) + 1
        return {
            "total_pokemon": len(self.pokemons),
            "num_synergies": NUM_SYNERGIES,
            "total_bots_loaded": len(self.bots),
            "shiny_items_disabled": self.disable_shiny_items,
            "shiny_items_removed_from_bots": self.shiny_items_removed_from_bots,
            "by_rarity": by_rarity,
        }


_GLOBAL_REPO: Optional[PACDataRepository] = None


def get_pac_repository() -> PACDataRepository:
    global _GLOBAL_REPO
    if _GLOBAL_REPO is None:
        _GLOBAL_REPO = PACDataRepository()
    return _GLOBAL_REPO
