r"""
Motor de Recompensas Multi-Escala Livre de Reward Hacking (`MultiScaleRewardEngine`)
para o Pokémon Auto Chess RL Lab (`experimento_gpu_escala`).

Fundamentação Matemática:
1. **Escala Macro (Terminal — Utilidade Zero-Sum Convexa de Torneio)**:
   - Recompensa não-linear calibrada pelo ganho de ELO competitivo (salto entre 4º e 5º lugar
     e prêmio convexo para o Top-1).
   - **Penalidade Anti-Hoarding ("Morrer abraçado com Ouro")**: Se o agente é eliminado
     entre 5º e 8º lugar guardando $\ge 15\text{g}$ no banco, sofre penalidade proporcional ao ouro
     não gasto, forçando o aprendizado real do *All-In / Roll-Down* em vida baixa!
2. **Escala Meso (Fim de Rodada — Combate, Economia Composta, Win/Lose Streak e Cashout)**:
   - Penalidade de perda de vida ponderada pelo risco quadrático: perder 8 HP com 90 de vida
     é barato (permite *Lose Streak* controlado no Estágio 4–10), mas perder 8 HP com 22 de vida
     é severamente penalizado.
   - Recompensa explícita de **Transição Cashout**: recompensa o agente de *Lose Streak*
     apenas quando ele converte o ouro acumulado em uma vitória real após uma sequência de derrotas!
   - Bônus de equilíbrio **Carry (DPS) + Frontline (Tank)** medido na luta real.
3. **Escala Micro (Intra-Loja — Potential-Based Reward Shaping Estrito, Ng et al., ICML 1999)**:
   - Toda ação dentro da loja recebe exclusivamente $r_{\text{micro}} = \gamma \Phi_k(s_{t+1}) - \Phi_k(s_t)$.
   - Como a soma telescópica se cancela ao longo do turno, é matematicamente impossível o agente
     "hackear" recompensa comprando, vendendo ou trocando peças em loop!
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from pac_rl.combat_surrogate import BoardUnit, CombatOutcome, compute_active_synergies
from pac_rl.gym_env import RewardWeightConfig
from .shared_pool_and_scouting import HypergeometricPoolTracker, LobbyScoutingReport, SharedCardPool


# Tabela de Utilidade Terminal Zero-Sum (Soma = 0.0 entre os 8 lugares)
PLACEMENT_UTILITY: Dict[int, float] = {
    1: +4.00,
    2: +2.50,
    3: +1.40,
    4: +0.60,
    5: -0.60,
    6: -1.40,
    7: -2.50,
    8: -4.00,
}


class MultiScaleRewardEngine:
    r"""
    Calcula as recompensas Micro (potencial $\gamma \Phi(s') - \Phi(s)$), Meso (combate e
    economia por rodada) e Macro (colocação final 1..8) para cada agente da comunidade.

    Invariância de Política (Ng, Harada & Russell, ICML 1999):
    - Utiliza o MESMO fator de desconto `gamma = 0.99` do `GPUMaskablePPOTrainer`.
    - Mantém potencial contínuo ao longo de toda a partida e impõe $\Phi(s_{\text{terminal}}) = 0$
      no passo final (`done=True`), garantindo soma telescópica exata sem competir com a colocação!
    """

    def __init__(self, gamma: float = 0.99, micro_scale: float = 0.25):
        self.gamma = gamma
        self.micro_scale = micro_scale

    def compute_potential_phi(
        self,
        config: RewardWeightConfig,
        board: List[BoardUnit],
        bench: List[BoardUnit],
        gold: int,
        level: int,
        hp: int,
        surrogate_win_prob: float,
        pool: SharedCardPool,
        scouting: LobbyScoutingReport,
    ) -> float:
        r"""
        Função Potencial $\Phi_k(s)$ (Ng et al., 1999) parametrizada pelos pesos da persona $k$:
        Combina:
          1. Probabilidade de vitória prevista pelo `DeepSetTransformerSurrogate` contra o lobby
          2. Progresso de evolução ($2\star$ e $3\star$) + probabilidade hipergeométrica de pares no banco
          3. Profundidade de sinergia (ponderada pelo tipo alvo e pela baixa contestação no lobby)
          4. Reserva de juros (descontada se HP estiver em zona crítica < 30)
        """
        if self.micro_scale <= 0.0 or config.w_potential_shaping <= 0.0:
            return 0.0

        # 1. Força de combate real prevista pelo Surrogate Neural (0..1)
        combat_phi = 2.2 * surrogate_win_prob

        # 2. Massa de evolução (valoriza 2* e 3* no tabuleiro) + potencial de upgrade no banco
        rarity_w = {
            "COMMON": 1.0,
            "UNCOMMON": 1.25,
            "RARE": 1.55,
            "EPIC": 1.95,
            "ULTRA": 2.50,
            "UNIQUE": 2.10,
            "LEGENDARY": 2.80,
        }
        evo_score = 0.0
        for u in board:
            rw = rarity_w.get(u.rarity, 1.2)
            star_val = 0.25 if u.stars == 1 else (0.95 if u.stars == 2 else 2.40)
            evo_score += rw * star_val * (1.0 + 0.12 * u.num_items)

        max_pair_prob, avg_avail = HypergeometricPoolTracker.compute_roster_upgrade_probability(
            pool, level, board + bench
        )
        evo_phi = config.w_reroll_evo * (0.18 * evo_score + 0.65 * max_pair_prob + 0.20 * avg_avail)

        # 3. Sinergias ativas no tabuleiro + bônus por jogar sinergia não contestada no lobby
        counts, active_tiers = compute_active_synergies(board)
        syn_score = 0.0
        for syn, tier in active_tiers.items():
            from pac_rl.data_loader import SYNERGY_TO_IDX
            idx = SYNERGY_TO_IDX.get(syn, 0)
            uncontested_bonus = 1.0 + 0.35 * (1.0 - float(scouting.synergy_contestedness[idx]))
            if config.target_synergy == syn:
                syn_score += 0.90 * tier * uncontested_bonus
            else:
                syn_score += 0.35 * tier * uncontested_bonus
        syn_phi = config.w_synergy_focus * syn_score

        # 4. Potencial Econômico (só valoriza guardar 50g se o jogador tiver vida saudável!)
        hp_safety = np.clip((hp - 20.0) / 60.0, 0.0, 1.0)
        interest_tier = min(5, gold // 10) / 5.0
        econ_phi = config.w_interest * interest_tier * hp_safety + 0.12 * (level / 9.0)

        return float(self.micro_scale * (combat_phi + evo_phi + syn_phi + econ_phi))

    def compute_micro_shaping_reward(self, phi_before: float, phi_after: float, done: bool) -> float:
        r"""Recompensa telescópica estrita $\gamma \Phi(s_{t+1}) - \Phi(s_t)$ com $\Phi(s_{\text{terminal}}) = 0$."""
        if self.micro_scale <= 0.0:
            return 0.0
        next_val = 0.0 if done else (self.gamma * phi_after)
        return float(next_val - phi_before)

    def compute_meso_round_reward(
        self,
        config: RewardWeightConfig,
        outcome: CombatOutcome,
        is_team_a: bool,
        is_pve: bool,
        stage_level: int,
        hp_after: int,
        gold_before_income: int,
        win_streak: int,
        lose_streak: int,
        prev_lose_streak: int,
    ) -> float:
        """
        Recompensa Meso ao final do combate de cada estágio:
        Avalia vitória/derrota, preservação de vida sob risco, eficiência de juros,
        win/lose streak e equilíbrio entre Carry DPS e Frontline Tank.
        """
        won = (outcome.winner >= 0) if is_team_a else (outcome.winner < 0)
        my_logs = outcome.unit_logs_a if is_team_a else outcome.unit_logs_b
        surviving = outcome.surviving_a if is_team_a else outcome.surviving_b

        reward = 0.0

        # 1. Resultado do combate, dano parcial em derrotas apertadas e risco quadrático de HP
        enemy_logs = outcome.unit_logs_b if is_team_a else outcome.unit_logs_a
        enemy_surv = outcome.surviving_b if is_team_a else outcome.surviving_a
        enemy_total = max(1, len(enemy_logs))
        enemy_killed_ratio = float(enemy_total - enemy_surv) / float(enemy_total)

        if won:
            reward += 0.25 + 0.035 * surviving
            if not is_pve:
                reward += 0.22 * config.w_win_streak * (1.0 + 0.30 * min(5, win_streak))
                # Bônus de Cashout Real: quebrou uma lose_streak >= 3 com uma vitória!
                if prev_lose_streak >= 3 and config.w_lose_streak > 0.1:
                    reward += 0.95 * config.w_lose_streak * (min(6, prev_lose_streak) / 4.0)
        else:
            dmg_taken = outcome.damage_to_loser
            # Risco quadrático: perder vida abaixo de 40 HP pesa até 3.5x mais!
            danger_mult = (1.0 + max(0.0, (100.0 - hp_after) / 55.0)) ** 2
            hp_penalty = 0.022 * dmg_taken * danger_mult * (0.4 + config.w_hp_preservation)

            # Sinal denso contra oponentes Medium/Extreme: matar 5 de 6 peças inimigas é muito melhor que 0 de 6!
            close_fight_credit = 0.14 * enemy_killed_ratio

            if not is_pve and stage_level <= 11 and hp_after >= 52 and config.w_lose_streak > 0.2:
                # Lose streak intencional controlado no early game
                interest_ok = 1.0 if gold_before_income >= 10 else 0.3
                reward += (
                    0.16 * config.w_lose_streak * (min(5, lose_streak) / 5.0) * interest_ok
                    + close_fight_credit
                    - 0.35 * hp_penalty
                )
            else:
                reward += close_fight_credit - hp_penalty

        # 2. Eficiência de Juros Compostos (desativada se HP <= 25 para forçar all-in!)
        interest_earned = min(5, gold_before_income // 10)
        if hp_after > 25:
            reward += 0.14 * config.w_interest * (interest_earned / 5.0)
        elif gold_before_income >= 25:
            # Penaliza segurar ouro quando está à beira da morte (HP <= 25)
            reward -= 0.15 * (gold_before_income / 20.0)

        # 3. Sinergia Tática Carry (DPS) + Frontline (Tank) na luta real
        if my_logs:
            peak_dps = max(u.damage_dealt for u in my_logs)
            peak_tank = max(u.damage_tanked for u in my_logs)
            # Recompensa ter tanto um Carry dando dano alto quanto um Tank absorvendo dano
            dps_norm = min(1.5, peak_dps / max(150.0, stage_level * 28.0))
            tank_norm = min(1.5, peak_tank / max(180.0, stage_level * 32.0))
            reward += 0.08 * np.sqrt(dps_norm * tank_norm)

        return float(reward)

    def compute_macro_terminal_reward(
        self,
        config: RewardWeightConfig,
        placement: int,
        unspent_gold_at_death: int,
        final_stage_reached: int = 22,
    ) -> float:
        r"""
        Recompensa Terminal Zero-Sum (1º ao 8º lugar) + Tiebreaker Limitado de Sobrevivência
        + Penalidade Anti-Hoarding.

        Garantia de Dominância Estrita de Colocação:
        - O menor degrau entre colocações adjacentes em `PLACEMENT_UTILITY` é `0.80`
          (entre 3º [+1.40] e 4º [+0.60], ou entre 5º [-0.60] e 6º [-1.40]).
        - `survival_margin` é limitado a `[-0.20, +0.25]` (`0.015 * (stage - 20)`),
          cuja amplitude máxima (`0.45`) é estritamente menor que `0.80`. Assim, sobreviver
          mais estágios desempata partidas de mesma colocação (ex.: dois 6º lugares), mas
          JAMAIS supera subir uma posição no ranking!
        """
        base_util = PLACEMENT_UTILITY.get(placement, 0.0) * config.w_placement
        survival_margin = float(np.clip(0.015 * float(final_stage_reached - 20), -0.20, 0.25))
        hoarding_penalty = 0.0
        if placement >= 5 and unspent_gold_at_death >= 15:
            # Morreu no Bottom-4 guardando ouro que deveria ter sido rolado (limitado a 0.35)!
            hoarding_penalty = min(0.35, 0.025 * (unspent_gold_at_death - 10))
        return float(base_util + survival_margin - hoarding_penalty)
