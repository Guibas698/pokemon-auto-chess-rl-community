r"""
Arquitetura Híbrida Estado da Arte para GPU (`experimento_gpu_escala/deep_transformer_models.py`):
1. `DeepSetTransformerSurrogate`: Set-Transformer de 4 camadas (`d_model=256, nhead=8`)
   com Cross-Attention entre os dois tabuleiros, prevendo vitória, saldo de sobreviventes,
   pico de Dano (Carry) e pico de Tank (Frontline).
2. `NeuralLookaheadRosterPlanner` (Busca Clássica + Rede Neural Conectada em Tempo Real):
   - Avalia em lote na GPU os 6 slots da loja (`evaluate_shop_marginal_gains`) para medir
     exatamente quanto cada compra aumenta a chance de vencer os oponentes reais do lobby.
   - Executa **Busca Combinatória 1-Step Swap (`optimize_roster_with_surrogate`)** entre o
     tabuleiro e o banco guiada pelo `DeepSetTransformerSurrogate`!
3. `CommunityOfModelsBrain` ("Uma Comunidade de Modelos para Ser Um Player" — Zero Regras Manuais):
   - Cada jogador é governado por um **Conselho de 7 Sub-Modelos Especialistas**
     (`Economist`, `WinStreak_Tempo`, `LoseStreak_Cashout`, `HyperRoll_Evolution`,
     `Synergy_Vertical`, `LobbyScout_Counter`, `LateGame_Cap`) coordenados por um
     **`MetaArbitratorTransformer`** que aprende pesos de confiança dinâmicos $\alpha_1(s)..\alpha_7(s)$.
   - **Zero `compute_Domain_prior_logits` manual**: 100% das decisões emergem do aprendizado
     por reforço + busca lookahead, garantindo que as árvores VIPER descubram heurísticas puras!
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from pac_rl.combat_surrogate import (
    MAX_BOARD_UNITS,
    UNIT_FEATURE_DIM,
    BoardUnit,
    FastCombatSimulator,
    TacticalPositioner,
    encode_board_to_tensor,
)
from pac_rl.data_loader import NUM_SYNERGIES, PACDataRepository, PokemonSpec
from pac_rl.gym_env import NUM_ACTIONS


# Dimensão expandida da observação para a Comunidade de Modelos (156 features)
GPU_OBS_DIM = 156

EXPERT_NAMES = [
    "Economist_Compound_Expert",
    "WinStreak_Tempo_Expert",
    "LoseStreak_Cashout_Expert",
    "HyperRoll_Evolution_Expert",
    "Synergy_Vertical_Expert",
    "LobbyScout_Counter_Expert",
    "LateGame_Cap_Expert",
]
NUM_EXPERTS = len(EXPERT_NAMES)


def get_optimal_device(preferred: str = "auto") -> torch.device:
    if preferred == "cuda" or (preferred == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


class PreNormAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, dropout: float = 0.08):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class CrossTeamAttentionBlock(nn.Module):
    """Cross-Attention permitindo que cada unidade do Time A inspecione os alvos do Time B."""

    def __init__(self, d_model: int = 256, nhead: int = 8, dropout: float = 0.08):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, team_q: torch.Tensor, team_kv: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(team_q)
        kv = self.norm_kv(team_kv)
        cross_out, _ = self.cross_attn(q, kv, kv)
        h = team_q + cross_out
        h = h + self.ffn(self.norm_ffn(h))
        return h


class DeepSetTransformerSurrogate(nn.Module):
    """
    Set-Transformer Profundo (GPU Scale) para previsão tática de combate:
    Prevê:
      1. `win_logit`: Logit de $P(\text{Time A vence})$
      2. `margin_pred`: Diferencial esperado de unidades sobreviventes
      3. `peak_dps_pred`: Previsão do recorde de dano do Carry na luta
      4. `peak_tank_pred`: Previsão do recorde de absorção do Tank na luta
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 4, dropout: float = 0.08):
        super().__init__()
        self.unit_embed = nn.Sequential(
            nn.Linear(UNIT_FEATURE_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.self_layers = nn.ModuleList(
            [PreNormAttentionBlock(d_model=d_model, nhead=nhead, dropout=dropout) for _ in range(num_layers)]
        )
        self.cross_block = CrossTeamAttentionBlock(d_model=d_model, nhead=nhead, dropout=dropout)

        self.syn_embed = nn.Sequential(
            nn.Linear(NUM_SYNERGIES, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
        )

        team_dim = d_model + (d_model // 2)
        self.prediction_head = nn.Sequential(
            nn.Linear(team_dim * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 4),
        )

    def forward(
        self,
        units_a: torch.Tensor,
        syn_a: torch.Tensor,
        units_b: torch.Tensor,
        syn_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_a = (units_a[:, :, 0] > 0.5).unsqueeze(-1).float()
        mask_b = (units_b[:, :, 0] > 0.5).unsqueeze(-1).float()

        ha = self.unit_embed(units_a)
        hb = self.unit_embed(units_b)

        for layer in self.self_layers:
            ha = layer(ha)
            hb = layer(hb)

        ha_cross = self.cross_block(ha, hb)
        hb_cross = self.cross_block(hb, ha)

        pool_a = (ha_cross * mask_a).sum(dim=1) / mask_a.sum(dim=1).clamp(min=1.0)
        pool_b = (hb_cross * mask_b).sum(dim=1) / mask_b.sum(dim=1).clamp(min=1.0)

        emb_a = torch.cat([pool_a, self.syn_embed(syn_a)], dim=-1)
        emb_b = torch.cat([pool_b, self.syn_embed(syn_b)], dim=-1)

        joint = torch.cat([emb_a, emb_b, emb_a - emb_b], dim=-1)
        out = self.prediction_head(joint)

        win_logit = out[:, 0]
        margin_pred = torch.tanh(out[:, 1]) * 9.0
        peak_dps_pred = F.softplus(out[:, 2]) * 500.0
        peak_tank_pred = F.softplus(out[:, 3]) * 500.0
        return win_logit, margin_pred, peak_dps_pred, peak_tank_pred


# ============================================================================
# Planejador Híbrido Clássico + Neural Lookahead (Conectado ao Surrogate!)
# ============================================================================

class NeuralLookaheadRosterPlanner:
    """
    Integra Busca Clássica Combinatória (1-Step Lookahead & Swap Hill-Climbing)
    com o `DeepSetTransformerSurrogate` rodando em lote na GPU!
    """

    def __init__(self, surrogate: DeepSetTransformerSurrogate, device: torch.device):
        self.surrogate = surrogate
        self.device = device

    @torch.no_grad()
    def predict_win_prob_vs_lobby(
        self,
        candidate_boards: List[List[BoardUnit]],
        opponent_boards: List[List[BoardUnit]],
    ) -> np.ndarray:
        """
        Avalia em um ÚNICO forward pass em lote na GPU a probabilidade média de vitória
        de cada `candidate_board` contra os tabuleiros reais dos oponentes vivos no lobby.
        """
        if not candidate_boards:
            return np.zeros(0, dtype=np.float32)

        valid_opps = [b for b in opponent_boards if b]
        if not valid_opps:
            return np.full(len(candidate_boards), 0.75, dtype=np.float32)

        # Sample up to 3 representative opponents from the lobby for fast batched evaluation
        sampled_opps = valid_opps[:3]
        n_cand = len(candidate_boards)
        n_opps = len(sampled_opps)
        total_pairs = n_cand * n_opps

        ua_batch = np.zeros((total_pairs, MAX_BOARD_UNITS, UNIT_FEATURE_DIM), dtype=np.float32)
        sa_batch = np.zeros((total_pairs, NUM_SYNERGIES), dtype=np.float32)
        ub_batch = np.zeros((total_pairs, MAX_BOARD_UNITS, UNIT_FEATURE_DIM), dtype=np.float32)
        sb_batch = np.zeros((total_pairs, NUM_SYNERGIES), dtype=np.float32)

        opp_encoded = [encode_board_to_tensor(ob) for ob in sampled_opps]

        k = 0
        for c_board in candidate_boards:
            ua, sa = encode_board_to_tensor(c_board)
            for ub, sb in opp_encoded:
                ua_batch[k] = ua
                sa_batch[k] = sa
                ub_batch[k] = ub
                sb_batch[k] = sb
                k += 1

        self.surrogate.eval()
        t_ua = torch.from_numpy(ua_batch).to(self.device)
        t_sa = torch.from_numpy(sa_batch).to(self.device)
        t_ub = torch.from_numpy(ub_batch).to(self.device)
        t_sb = torch.from_numpy(sb_batch).to(self.device)

        win_logits, _, _, _ = self.surrogate(t_ua, t_sa, t_ub, t_sb)
        win_probs = torch.sigmoid(win_logits).view(n_cand, n_opps).mean(dim=1).cpu().numpy()
        return win_probs.astype(np.float32)

    @torch.no_grad()
    def evaluate_shop_marginal_gains(
        self,
        board: List[BoardUnit],
        shop: List[Optional[PokemonSpec]],
        player_level: int,
        opponent_boards: List[List[BoardUnit]],
    ) -> Tuple[float, np.ndarray]:
        r"""
        Para o estado atual, usa o `DeepSetTransformerSurrogate` na GPU para calcular:
        1. `base_win_prob`: Chance de vitória atual do tabuleiro contra o lobby
        2. `marginal_gains[0..5]`: $\Delta \widehat{P}_{\text{win}}$ se comprarmos/colocarmos a carta `shop[i]`!
        """
        candidates: List[List[BoardUnit]] = [board]
        valid_slot_indices: List[int] = []

        for slot_i, p in enumerate(shop):
            if p is None:
                continue
            new_u = BoardUnit.from_pokemon_stats(p, x=2, y=2)
            if len(board) < player_level:
                hyp_board = TacticalPositioner.optimize_positions(list(board) + [new_u])
            elif board:
                # Simulate replacing the lowest-HP/lowest-star unit on board
                weak_idx = min(range(len(board)), key=lambda idx: board[idx].hp * board[idx].stars)
                hyp_board = list(board)
                hyp_board[weak_idx] = new_u
                hyp_board = TacticalPositioner.optimize_positions(hyp_board)
            else:
                hyp_board = [new_u]
            candidates.append(hyp_board)
            valid_slot_indices.append(slot_i)

        probs = self.predict_win_prob_vs_lobby(candidates, opponent_boards)
        base_prob = float(probs[0]) if len(probs) > 0 else 0.5
        deltas = np.zeros(6, dtype=np.float32)
        for idx_in_probs, slot_i in enumerate(valid_slot_indices, start=1):
            deltas[slot_i] = float(probs[idx_in_probs] - base_prob)

        return base_prob, deltas

    @torch.no_grad()
    def optimize_roster_with_surrogate(
        self,
        board: List[BoardUnit],
        bench: List[BoardUnit],
        player_level: int,
        opponent_boards: List[List[BoardUnit]],
    ) -> Tuple[List[BoardUnit], List[BoardUnit]]:
        r"""
        Algoritmo Clássico de Busca Local (1-Step Swap Hill-Climbing) guiado pelo
        `DeepSetTransformerSurrogate`: testa todas as trocas unitárias entre `board` e `bench`
        em um único batch na GPU e aplica a troca se ela aumentar a chance real de vitória!
        """
        while len(board) < player_level and bench:
            board.append(bench.pop(0))

        board = TacticalPositioner.optimize_positions(board)
        if not bench or not board:
            return board, bench

        candidates: List[List[BoardUnit]] = [list(board)]
        swap_pairs: List[Tuple[int, int]] = []

        for b_idx in range(len(board)):
            for bench_idx in range(min(4, len(bench))):
                cand = list(board)
                cand[b_idx] = bench[bench_idx]
                candidates.append(TacticalPositioner.optimize_positions(cand))
                swap_pairs.append((b_idx, bench_idx))

        probs = self.predict_win_prob_vs_lobby(candidates, opponent_boards)
        best_cand_idx = int(np.argmax(probs))
        if best_cand_idx > 0 and probs[best_cand_idx] > probs[0] + 0.01:
            b_i, bench_i = swap_pairs[best_cand_idx - 1]
            old_board_u = board[b_i]
            board[b_i] = bench[bench_i]
            bench[bench_i] = old_board_u
            board = TacticalPositioner.optimize_positions(board)

        return board, bench


# ============================================================================
# Comunidade Interna de Modelos: 7 Peritos Especialistas + Meta-Arbitrador
# ============================================================================

class SubAgentExpertModule(nn.Module):
    r"""
    Um Sub-Modelo Especialista dentro do Conselho/Comunidade que compõe 1 jogador.
    Possui seu próprio extrator de representações focado nas features do seu domínio,
    produzindo um vetor latente de argumento `h_expert` e sua proposta de logits de ação.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.action_proposal = nn.Linear(hidden_dim, num_actions)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        logits = self.action_proposal(h)
        return h, logits


class CommunityOfModelsBrain(nn.Module):
    r"""
    "Uma Comunidade de Modelos para Ser Um Player" (Hierarchical Mixture-of-Experts):
    - 7 Sub-Modelos Especialistas (`EXPERT_NAMES`) analisam simultaneamente a partida:
      1. `Economist_Compound_Expert` (Juros, Breakpoints 10g..50g, Curva de XP)
      2. `WinStreak_Tempo_Expert` (Pressão de Tabuleiro, Nível Agressivo, Win Streak)
      3. `LoseStreak_Cashout_Expert` (Economia Reversa Estágios 1..11 e Gatilho de All-In Cashout)
      4. `HyperRoll_Evolution_Expert` (Probabilidade Hipergeométrica de Pares e Evoluções 3*)
      5. `Synergy_Vertical_Expert` (Ativação de Limiares das 31 Sinergias)
      6. `LobbyScout_Counter_Expert` (Scouting de Cartas Contestadas, Counters e Lookahead Surrogate)
      7. `LateGame_Cap_Expert` (Transição Nível 8/9, Uniques Estágio 10 e Legendaries Estágio 20)
    - `MetaArbitratorTransformer`: Faz Self-Attention entre as propostas dos 7 especialistas
      condicionado pela Persona $\mathbf{w}^{(k)}$ do agente e gera os pesos de voto
      $\boldsymbol{\alpha}(s) \in \Delta^6$ ($\sum_{m=1}^7 \alpha_m(s) = 1.0$).
    - **ZERO Regras Hardcoded (`compute_Domain_prior_logits` removido!)**: Tudo é aprendido
      por Reforço + Busca Lookahead.
    """

    def __init__(
        self,
        obs_dim: int = GPU_OBS_DIM,
        num_actions: int = NUM_ACTIONS,
        expert_dim: int = 128,
        nhead: int = 4,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.expert_dim = expert_dim

        # Os 7 Sub-Agentes Especialistas
        self.experts = nn.ModuleList(
            [SubAgentExpertModule(obs_dim, expert_dim, num_actions) for _ in range(NUM_EXPERTS)]
        )

        # Codificador da Persona / Objetivo do Jogador (slice obs[:, 73:112])
        self.persona_proj = nn.Sequential(
            nn.Linear(39, expert_dim),
            nn.LayerNorm(expert_dim),
            nn.GELU(),
        )

        # Meta-Arbitrador Transformer que coloca os 7 Especialistas para debater entre si
        self.council_attn = nn.MultiheadAttention(
            embed_dim=expert_dim,
            num_heads=nhead,
            batch_first=True,
        )
        self.council_norm = nn.LayerNorm(expert_dim)

        # Gating Network: decide o peso de voto alpha_m(s) de cada um dos 7 Especialistas
        self.gating_head = nn.Sequential(
            nn.Linear(expert_dim * 2, expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, 1),
        )

        # Residual Meta-Policy & Twin Value Heads (Shaped Persona Critic + Oracle Placement Critic)
        self.meta_residual_actor = nn.Linear(expert_dim, num_actions)
        self.critic_shaped = nn.Sequential(
            nn.Linear(expert_dim * 2, expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, 1),
        )
        self.critic_oracle = nn.Sequential(
            nn.Linear(expert_dim * 2, expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, 1),
        )

    def forward_council(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Executa os 7 Especialistas e o Meta-Arbitrador Transformer:
        Retorna:
          - `final_logits`: shape `(B, NUM_ACTIONS)`
          - `expert_weights`: shape `(B, NUM_EXPERTS)` com $\sum_m \alpha_m = 1$
          - `v_shaped`: shape `(B,)`
          - `v_oracle`: shape `(B,)`
        """
        expert_hs = []
        expert_logits = []
        for exp in self.experts:
            h_m, l_m = exp(obs)
            expert_hs.append(h_m)
            expert_logits.append(l_m)

        # Stack across the 7 experts: H has shape (B, 7, D), L has shape (B, 7, A)
        H = torch.stack(expert_hs, dim=1)
        L = torch.stack(expert_logits, dim=1)

        # Self-attention debate among the 7 experts
        attn_out, _ = self.council_attn(H, H, H)
        H_debated = self.council_norm(H + attn_out)  # (B, 7, D)

        # Condition gating on the player's persona vector (obs[:, 73:112])
        persona_emb = self.persona_proj(obs[:, 73:112])  # (B, D)
        persona_exp = persona_emb.unsqueeze(1).expand_as(H_debated)  # (B, 7, D)

        gate_scores = self.gating_head(torch.cat([H_debated, persona_exp], dim=-1)).squeeze(-1)  # (B, 7)
        expert_weights = F.softmax(gate_scores, dim=-1)  # (B, 7)

        # Weighted voting of expert action logits + meta residual refinement
        weighted_logits = (L * expert_weights.unsqueeze(-1)).sum(dim=1)  # (B, A)
        pooled_council = (H_debated * expert_weights.unsqueeze(-1)).sum(dim=1)  # (B, D)

        final_logits = weighted_logits + 0.25 * self.meta_residual_actor(pooled_council)
        joint_critic_in = torch.cat([pooled_council, persona_emb], dim=-1)
        v_shaped = self.critic_shaped(joint_critic_in).squeeze(-1)
        v_oracle = self.critic_oracle(joint_critic_in).squeeze(-1)

        return final_logits, expert_weights, v_shaped, v_oracle

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_logits, expert_weights, v_shaped, v_oracle = self.forward_council(obs)
        masked_logits = raw_logits.masked_fill(~action_mask, -1e8)
        dist = Categorical(logits=masked_logits)

        if action is None:
            action = torch.argmax(masked_logits, dim=-1) if deterministic else dist.sample()

        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            v_shaped,
            v_oracle,
            expert_weights,
        )


@dataclass
class GPURolloutStep:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    log_prob: float
    reward: float
    done: bool
    value: float
    oracle_target: float = 0.0


class GPUMaskablePPOTrainer:
    """
    Treinador Maskable PPO Acelerado por GPU para o `CommunityOfModelsBrain`:
    - Suporta CUDA Automatic Mixed Precision (`torch.amp`)
    - Treina simultaneamente a Política do Conselho, o Crítico Moldado (`v_shaped`) e o
      Crítico Oráculo de Colocação Real (`v_oracle`)
    - Inclui regularização de diversidade de especialistas para evitar colapso prematuro de gating.
    """

    def __init__(
        self,
        model: CommunityOfModelsBrain,
        device: torch.device,
        lr: float = 1e-3,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.02,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.025,
        use_amp: bool = True,
    ):
        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.use_amp = use_amp and (device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def update(self, buffer: List[GPURolloutStep], ppo_epochs: int = 6, batch_size: int = 256) -> Dict[str, float]:
        if not buffer:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        n = len(buffer)
        obs_arr = np.array([s.obs for s in buffer], dtype=np.float32)
        mask_arr = np.array([s.mask for s in buffer], dtype=bool)
        act_arr = np.array([s.action for s in buffer], dtype=np.int64)
        old_logp_arr = np.array([s.log_prob for s in buffer], dtype=np.float32)
        rew_arr = np.array([s.reward for s in buffer], dtype=np.float32)
        done_arr = np.array([s.done for s in buffer], dtype=np.float32)
        val_arr = np.array([s.value for s in buffer], dtype=np.float32)
        ora_arr = np.array([s.oracle_target for s in buffer], dtype=np.float32)

        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_non_terminal = 1.0 - done_arr[t]
            next_val = val_arr[t + 1] if t + 1 < n else 0.0
            delta = rew_arr[t] + self.gamma * next_val * next_non_terminal - val_arr[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + val_arr
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        t_obs = torch.from_numpy(obs_arr).to(self.device)
        t_mask = torch.from_numpy(mask_arr).to(self.device)
        t_act = torch.from_numpy(act_arr).to(self.device)
        t_old_logp = torch.from_numpy(old_logp_arr).to(self.device)
        t_adv = torch.from_numpy(advantages).to(self.device)
        t_ret = torch.from_numpy(returns).to(self.device)
        t_ora = torch.from_numpy(ora_arr).to(self.device)

        total_pg, total_vf, total_ent, total_kl, updates = 0.0, 0.0, 0.0, 0.0, 0
        self.model.train()

        for _ in range(ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_kl = 0.0
            epoch_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                self.optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    _, new_logp, entropy, new_val, new_ora, exp_w = self.model.get_action_and_value(
                        t_obs[idx], t_mask[idx], action=t_act[idx]
                    )
                    logratio = new_logp - t_old_logp[idx]
                    ratio = logratio.exp()
                    approx_kl = ((ratio - 1.0) - logratio).mean()

                    mb_adv = t_adv[idx]
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss = 0.5 * F.mse_loss(new_val, t_ret[idx]) + 0.25 * F.mse_loss(new_ora, t_ora[idx])
                    ent_loss = entropy.mean()

                    # Load-balancing entropy across the 7 council experts so no expert dies early
                    mean_exp_w = exp_w.mean(dim=0)
                    council_balance_ent = -(mean_exp_w * torch.log(mean_exp_w + 1e-8)).sum()

                    loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent_loss - 0.01 * council_balance_ent

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_pg += float(pg_loss.item())
                total_vf += float(v_loss.item())
                total_ent += float(ent_loss.item())
                total_kl += float(approx_kl.item())
                epoch_kl += float(approx_kl.item())
                updates += 1
                epoch_batches += 1

            if epoch_batches > 0 and (epoch_kl / epoch_batches) > self.target_kl:
                break

        denom = max(1, updates)
        return {
            "policy_loss": total_pg / denom,
            "value_loss": total_vf / denom,
            "entropy": total_ent / denom,
            "approx_kl": total_kl / denom,
        }


def train_deep_gpu_surrogate(
    repo: PACDataRepository,
    sim: FastCombatSimulator,
    device: torch.device,
    num_samples: int = 25000,
    epochs: int = 25,
    batch_size: int = 512,
    lr: float = 1.5e-3,
    seed: int = 42,
) -> Tuple[DeepSetTransformerSurrogate, Dict[str, float]]:
    """
    Treina o `DeepSetTransformerSurrogate` em larga escala na GPU (prevendo Vitória,
    Margem de Sobreviventes, Pico de Dano do Carry e Pico de Absorção do Tank).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    X_ua = np.zeros((num_samples, MAX_BOARD_UNITS, UNIT_FEATURE_DIM), dtype=np.float32)
    X_sa = np.zeros((num_samples, NUM_SYNERGIES), dtype=np.float32)
    X_ub = np.zeros((num_samples, MAX_BOARD_UNITS, UNIT_FEATURE_DIM), dtype=np.float32)
    X_sb = np.zeros((num_samples, NUM_SYNERGIES), dtype=np.float32)
    Y_win = np.zeros((num_samples,), dtype=np.float32)
    Y_margin = np.zeros((num_samples,), dtype=np.float32)
    Y_dps = np.zeros((num_samples,), dtype=np.float32)
    Y_tank = np.zeros((num_samples,), dtype=np.float32)

    all_pokes = list(repo.pokemon_by_name.values())
    bots = repo.bots

    def sample_board() -> Tuple[List[BoardUnit], int]:
        stage = int(rng.integers(3, 35))
        if bots and rng.random() < 0.70:
            bot = bots[int(rng.integers(0, len(bots)))]
            entries = repo.get_bot_stage_board(bot, stage)
            units = []
            for name, x, y, n_items in entries[:MAX_BOARD_UNITS]:
                p = repo.get_pokemon(name)
                if p is not None:
                    units.append(BoardUnit.from_pokemon_stats(p, x=x, y=min(3, max(0, y - 1)), num_items=n_items))
            if units:
                return units, stage
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
        ba, sa_st = sample_board()
        bb, sb_st = sample_board()
        outcome = sim.simulate(ba, bb, stage_level=(sa_st + sb_st) // 2, rng=rng)
        ua, sa = encode_board_to_tensor(ba)
        ub, sb = encode_board_to_tensor(bb)
        X_ua[i] = ua
        X_sa[i] = sa
        X_ub[i] = ub
        X_sb[i] = sb
        Y_win[i] = 1.0 if outcome.winner >= 0 else 0.0
        Y_margin[i] = float(outcome.surviving_a - outcome.surviving_b)
        Y_dps[i] = max((u.damage_dealt for u in outcome.unit_logs_a), default=0.0)
        Y_tank[i] = max((u.damage_tanked for u in outcome.unit_logs_a), default=0.0)

    split = int(0.88 * num_samples)
    model = DeepSetTransformerSurrogate(d_model=256, nhead=8, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    t_ua = torch.from_numpy(X_ua[:split]).to(device)
    t_sa = torch.from_numpy(X_sa[:split]).to(device)
    t_ub = torch.from_numpy(X_ub[:split]).to(device)
    t_sb = torch.from_numpy(X_sb[:split]).to(device)
    t_win = torch.from_numpy(Y_win[:split]).to(device)
    t_mar = torch.from_numpy(Y_margin[:split]).to(device)
    t_dps = torch.from_numpy(Y_dps[:split]).to(device)
    t_tnk = torch.from_numpy(Y_tank[:split]).to(device)

    v_ua = torch.from_numpy(X_ua[split:]).to(device)
    v_sa = torch.from_numpy(X_sa[split:]).to(device)
    v_ub = torch.from_numpy(X_ub[split:]).to(device)
    v_sb = torch.from_numpy(X_sb[split:]).to(device)
    v_win = torch.from_numpy(Y_win[split:]).to(device)
    v_mar = torch.from_numpy(Y_margin[split:]).to(device)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(split, device=device)
        for start in range(0, split, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                w_log, m_pred, dps_p, tnk_p = model(t_ua[idx], t_sa[idx], t_ub[idx], t_sb[idx])
                loss_cls = F.binary_cross_entropy_with_logits(w_log, t_win[idx])
                loss_mar = F.smooth_l1_loss(m_pred, t_mar[idx])
                loss_aux = 0.001 * (F.smooth_l1_loss(dps_p, t_dps[idx]) + F.smooth_l1_loss(tnk_p, t_tnk[idx]))
                loss = loss_cls + 0.15 * loss_mar + loss_aux
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        val_logits, val_mar, _, _ = model(v_ua, v_sa, v_ub, v_sb)
        val_preds = (torch.sigmoid(val_logits) >= 0.5).float()
        val_acc = float((val_preds == v_win).float().mean().item())
        val_mae = float((val_mar - v_mar).abs().mean().item())

    return model, {"val_accuracy": val_acc, "val_margin_mae": val_mae, "samples": num_samples}
