r"""
Baralho Compartilhado Finito (`SharedCardPool`), Calculadora Clássica Hipergeométrica de
Probabilidades de Loja (`HypergeometricPoolTracker`) e Sistema de Scouting de Oponentes
(`LobbyScoutingAggregator`) para o Pokémon Auto Chess.

Elimina os atalhos de loja infinita:
- Todos os 8 jogadores do lobby compram e devolvem cartas do MESMO `SharedCardPool`,
  seguindo exatamente `POOL_SIZE` de `upstream/app/config/game/pools.ts`.
- Calcula matematicamente a probabilidade exata de achar a cópia que falta para evoluir
  qualquer Pokémon ($1\star \to 2\star \to 3\star$) dado o que os outros 7 jogadores já compraram.
- Faz Scouting em tempo real dos 7 oponentes vivos para alimentar o especialista
  `LobbyScout_Counter_Expert` da Comunidade de Modelos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from pac_rl.combat_surrogate import BoardUnit, compute_active_synergies
from pac_rl.data_loader import (
    ALL_SYNERGIES,
    NUM_SYNERGIES,
    POOL_SIZE,
    RARITY_PROBA_PER_LEVEL,
    SHOP_RARITIES,
    SHOP_SIZE,
    SYNERGY_TO_IDX,
    PACDataRepository,
    PokemonSpec,
)


class SharedCardPool:
    """
    Implementa o baralho compartilhado finito de um lobby de 8 jogadores do PAC:
      - COMMON: 27 cópias (famílias de 3 estágios) ou 18 cópias (famílias de 2 estágios)
      - UNCOMMON: 22 cópias (3 estágios) ou 13 cópias (2 estágios)
      - RARE: 18 cópias (3 estágios) ou 9 cópias (2 estágios)
      - EPIC: 14 cópias (3 estágios) ou 7 cópias (2 estágios)
      - ULTRA: 10 cópias (3 estágios) ou 5 cópias (2 estágios)
    Todas as compras são descontadas da forma base (Tier 1 da família) e devolvidas
    ao pool quando um jogador vende a unidade ou é eliminado do lobby.
    """

    def __init__(self, repo: PACDataRepository):
        self.repo = repo
        self.initial_counts: Dict[str, Dict[str, int]] = {r: {} for r in SHOP_RARITIES}
        self.remaining_counts: Dict[str, Dict[str, int]] = {r: {} for r in SHOP_RARITIES}
        self.base_form_of: Dict[str, str] = {}
        self._build_initial_template()
        self.reset()

    def _build_initial_template(self) -> None:
        # Map every evolution back to its Tier-1 base species in the pool
        for fam, chain in self.repo.family_chains.items():
            tier1_list = chain.get(1, [])
            if not tier1_list:
                continue
            base_name = tier1_list[0]
            for _, names in chain.items():
                for n in names:
                    self.base_form_of[n] = base_name

        for rarity in SHOP_RARITIES:
            for base_name in self.repo.regular_pool_tier1.get(rarity, []):
                spec = self.repo.pokemons[base_name]
                pool_tuple = POOL_SIZE.get(rarity, (1, 9, 18))
                idx = min(2, max(0, spec.nb_stages - 1))
                count = pool_tuple[idx]
                self.initial_counts[rarity][base_name] = count

    def reset(self) -> None:
        self.remaining_counts = {
            r: dict(counts) for r, counts in self.initial_counts.items()
        }

    def unlock_additional_pick_for_lobby(self, p: PokemonSpec) -> None:
        """Adiciona um Pokémon de Additional Pick (Estágios 5, 8, 11) ao pool compartilhado."""
        rarity = p.category
        if rarity not in self.remaining_counts:
            return
        base_name = self.base_form_of.get(p.name, p.name)
        if base_name not in self.remaining_counts[rarity]:
            pool_tuple = POOL_SIZE.get(rarity, (1, 9, 18))
            idx = min(2, max(0, p.nb_stages - 1))
            cnt = pool_tuple[idx]
            self.initial_counts[rarity][base_name] = cnt
            self.remaining_counts[rarity][base_name] = cnt

    def sample_shop(self, player_level: int, rng: np.random.Generator) -> List[Optional[PokemonSpec]]:
        """
        Sorteia 6 cartas ponderadas pelo número exato de cópias restantes no baralho
        compartilhado para o nível atual do jogador (`RARITY_PROBA_PER_LEVEL`).
        """
        odds = RARITY_PROBA_PER_LEVEL.get(player_level, RARITY_PROBA_PER_LEVEL[9])
        shop: List[Optional[PokemonSpec]] = []

        for _ in range(SHOP_SIZE):
            r_choice = str(rng.choice(SHOP_RARITIES, p=odds))
            pool_dict = self.remaining_counts.get(r_choice, {})
            names = [k for k, v in pool_dict.items() if v > 0]
            if not names:
                # Fallback to COMMON if rarity pool is completely exhausted
                pool_dict = self.remaining_counts["COMMON"]
                names = [k for k, v in pool_dict.items() if v > 0]
            if not names:
                shop.append(None)
                continue

            weights = np.array([pool_dict[k] for k in names], dtype=np.float64)
            weights /= weights.sum()
            picked_name = str(rng.choice(names, p=weights))
            shop.append(self.repo.pokemons[picked_name])

        return shop

    def take_from_pool(self, pokemon_name: str, copies: int = 1) -> None:
        base_name = self.base_form_of.get(pokemon_name, pokemon_name)
        spec = self.repo.pokemons.get(base_name)
        if spec is None or spec.category not in self.remaining_counts:
            return
        curr = self.remaining_counts[spec.category].get(base_name, 0)
        self.remaining_counts[spec.category][base_name] = max(0, curr - copies)

    def return_to_pool(self, unit: BoardUnit) -> None:
        """
        Quando uma unidade é vendida ou seu dono é eliminado, devolve as cópias equivalentes
        (1 para 1*, 3 para 2*, 9 para 3*) ao pool compartilhado.
        """
        base_name = self.base_form_of.get(unit.name, unit.name)
        spec = self.repo.pokemons.get(base_name)
        if spec is None or spec.category not in self.remaining_counts:
            return
        copies = 1 if unit.stars == 1 else (3 if unit.stars == 2 else 9)
        max_cap = self.initial_counts[spec.category].get(base_name, 18)
        curr = self.remaining_counts[spec.category].get(base_name, 0)
        self.remaining_counts[spec.category][base_name] = min(max_cap, curr + copies)

    def get_remaining_ratio(self, pokemon_name: str) -> float:
        base_name = self.base_form_of.get(pokemon_name, pokemon_name)
        spec = self.repo.pokemons.get(base_name)
        if spec is None or spec.category not in self.remaining_counts:
            return 0.5
        init_c = max(1, self.initial_counts[spec.category].get(base_name, 10))
        rem_c = self.remaining_counts[spec.category].get(base_name, 0)
        return float(rem_c / init_c)


class HypergeometricPoolTracker:
    """
    Calculadora Clássica de Probabilidade de Loja:
    Computa a probabilidade exata $P(\text{achar pelo menos 1 cópia em 1 Reroll de 6 slots})$
    para os pares no banco/tabuleiro do jogador, guiando matematicamente a decisão entre
    rodar a loja (Hyper-Roll / 3*) ou subir de nível (Fast-8/9).
    """

    @staticmethod
    def prob_hit_in_single_reroll(
        pool: SharedCardPool,
        player_level: int,
        target_pokemon_name: str,
    ) -> float:
        base_name = pool.base_form_of.get(target_pokemon_name, target_pokemon_name)
        spec = pool.repo.pokemons.get(base_name)
        if spec is None or spec.category not in SHOP_RARITIES:
            return 0.0

        rarity = spec.category
        r_idx = SHOP_RARITIES.index(rarity)
        p_rarity = RARITY_PROBA_PER_LEVEL.get(player_level, RARITY_PROBA_PER_LEVEL[9])[r_idx]
        if p_rarity <= 0.0:
            return 0.0

        pool_dict = pool.remaining_counts.get(rarity, {})
        total_in_tier = sum(pool_dict.values())
        target_rem = pool_dict.get(base_name, 0)
        if total_in_tier <= 0 or target_rem <= 0:
            return 0.0

        p_single_slot = p_rarity * (target_rem / total_in_tier)
        # Probability of hitting at least 1 copy across 6 shop slots
        return float(1.0 - (1.0 - p_single_slot) ** SHOP_SIZE)

    @staticmethod
    def compute_roster_upgrade_probability(
        pool: SharedCardPool,
        player_level: int,
        units: List[BoardUnit],
    ) -> Tuple[float, float]:
        """
        Retorna:
          - `max_pair_hit_prob`: Maior probabilidade de fechar evolução 2*/3* em 1 reroll
          - `avg_pool_availability`: Disponibilidade média no baralho das unidades do time
        """
        if not units:
            return 0.0, 1.0
        by_name: Dict[str, int] = {}
        for u in units:
            by_name[u.name] = by_name.get(u.name, 0) + 1

        pair_probs = [
            HypergeometricPoolTracker.prob_hit_in_single_reroll(pool, player_level, name)
            for name, cnt in by_name.items()
            if cnt >= 2
        ]
        avail = [pool.get_remaining_ratio(u.name) for u in units]
        max_p = max(pair_probs) if pair_probs else 0.0
        return float(max_p), float(np.mean(avail))


@dataclass
class LobbyScoutingReport:
    """Relatório de Scouting dos outros 7 jogadores do lobby em tempo real."""
    synergy_contestedness: np.ndarray     # shape (31,): quantos oponentes disputam cada sinergia
    avg_opponent_board_power: float
    max_opponent_board_power: float
    avg_opponent_hp: float
    alive_opponents_count: int
    lobby_physical_vs_magic_ratio: float  # >0.5 lobby físico (pede DEF/STEEL/ROCK), <0.5 lobby mágico (pede SpeDef)
    backline_carry_vulnerability: float   # fração de inimigos dependentes de backline frágil (pede DARK/Assassins)


class LobbyScoutingAggregator:
    """
    Inspeciona todos os oponentes vivos no lobby de 8 jogadores e sintetiza o
    `LobbyScoutingReport` para o especialista `LobbyScout_Counter_Expert`.
    """

    @staticmethod
    def scout_lobby(
        opponent_boards: List[List[BoardUnit]],
        opponent_hps: List[int],
        opponent_powers: List[float],
    ) -> LobbyScoutingReport:
        contested = np.zeros(NUM_SYNERGIES, dtype=np.float32)
        if not opponent_boards:
            return LobbyScoutingReport(
                synergy_contestedness=contested,
                avg_opponent_board_power=10.0,
                max_opponent_board_power=12.0,
                avg_opponent_hp=100.0,
                alive_opponents_count=0,
                lobby_physical_vs_magic_ratio=0.5,
                backline_carry_vulnerability=0.5,
            )

        ranged_units = 0
        total_units = 0
        magic_syns = {"PSYCHIC", "WATER", "FIRE", "DRAGON", "FAIRY", "LIGHT", "GHOST", "AMORPHOUS"}
        magic_score = 0.0
        phys_score = 0.0

        for board in opponent_boards:
            _, active_tiers = compute_active_synergies(board)
            for syn, tier in active_tiers.items():
                if tier >= 1 and syn in SYNERGY_TO_IDX:
                    contested[SYNERGY_TO_IDX[syn]] += 0.25 * tier
                if syn in magic_syns:
                    magic_score += tier
                else:
                    phys_score += tier

            for u in board:
                total_units += 1
                if u.attack_range >= 2:
                    ranged_units += 1

        contested = np.clip(contested / max(1, len(opponent_boards)), 0.0, 1.0)
        phys_ratio = float(phys_score / max(1.0, phys_score + magic_score))
        backline_vuln = float(ranged_units / max(1, total_units))

        return LobbyScoutingReport(
            synergy_contestedness=contested,
            avg_opponent_board_power=float(np.mean(opponent_powers)) if opponent_powers else 10.0,
            max_opponent_board_power=float(np.max(opponent_powers)) if opponent_powers else 12.0,
            avg_opponent_hp=float(np.mean(opponent_hps)) if opponent_hps else 50.0,
            alive_opponents_count=len(opponent_boards),
            lobby_physical_vs_magic_ratio=phys_ratio,
            backline_carry_vulnerability=backline_vuln,
        )
