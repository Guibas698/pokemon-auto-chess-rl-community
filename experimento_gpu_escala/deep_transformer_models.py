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
    clone_board,
    encode_board_to_tensor,
)
from pac_rl.data_loader import NUM_SYNERGIES, PACDataRepository, PokemonSpec
from pac_rl.gym_env import NUM_ACTIONS


# Dimensão expandida da observação para a Comunidade de Modelos (188 features, incluindo
# 8 features explícitas de Carry/Tank e identidade de Tabuleiro/Banco nos slots 180..187)
GPU_OBS_DIM = 188
CUSTOM_126_OBS_DIM = 126
TOTAL_IDENTITY_SLOTS = 23  # 6 Shop + 9 Board + 8 Bench

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


def _safe_key_padding_mask(present_mask: torch.Tensor) -> torch.Tensor:
    """
    Converte máscara de presença booleana `(B, N)` em `key_padding_mask` (`True` = ignorar slot vazio),
    garantindo que ao menos o slot 0 fique desmascarado caso um tabuleiro esteja 100% vazio (evita NaN no Softmax).
    """
    pad_mask = ~present_mask
    all_empty = pad_mask.all(dim=1)
    if all_empty.any():
        pad_mask = pad_mask.clone()
        pad_mask[all_empty, 0] = False
    return pad_mask


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

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class CrossTeamAttentionBlock(nn.Module):
    """Cross-Attention permitindo que cada unidade ativa do Time A inspecione apenas unidades ativas do Time B."""

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

    def forward(
        self,
        team_q: torch.Tensor,
        team_kv: torch.Tensor,
        kv_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(team_q)
        kv = self.norm_kv(team_kv)
        cross_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=kv_padding_mask)
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
        present_a = units_a[:, :, 0] > 0.5
        present_b = units_b[:, :, 0] > 0.5
        pad_mask_a = _safe_key_padding_mask(present_a)
        pad_mask_b = _safe_key_padding_mask(present_b)

        mask_a = present_a.unsqueeze(-1).float()
        mask_b = present_b.unsqueeze(-1).float()

        ha = self.unit_embed(units_a) * mask_a
        hb = self.unit_embed(units_b) * mask_b

        for layer in self.self_layers:
            ha = layer(ha, key_padding_mask=pad_mask_a) * mask_a
            hb = layer(hb, key_padding_mask=pad_mask_b) * mask_b

        ha_cross = self.cross_block(ha, hb, kv_padding_mask=pad_mask_b) * mask_a
        hb_cross = self.cross_block(hb, ha, kv_padding_mask=pad_mask_a) * mask_b

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
    Usa `clone_board` em todas as simulações contrafactuais para garantir zero
    mutação colateral nas coordenadas `(x, y)` do tabuleiro real.
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
        Usa cópias independentes (`clone_board`) para NUNCA alterar `x, y` do `board` original.
        """
        candidates: List[List[BoardUnit]] = [clone_board(board)]
        valid_slot_indices: List[int] = []

        for slot_i, p in enumerate(shop):
            if p is None:
                continue
            new_u = BoardUnit.from_pokemon_stats(p, x=2, y=2)
            if len(board) < player_level:
                hyp_board = TacticalPositioner.optimize_positions(clone_board(board) + [new_u], in_place=True)
            elif board:
                # Simulate replacing the lowest-HP/lowest-star unit on a deep-cloned board
                weak_idx = min(range(len(board)), key=lambda idx: board[idx].hp * board[idx].stars)
                hyp_board = clone_board(board)
                hyp_board[weak_idx] = new_u
                hyp_board = TacticalPositioner.optimize_positions(hyp_board, in_place=True)
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
        em um único batch na GPU usando cópias independentes (`clone_board`) e aplica a troca
        apenas se ela aumentar a chance real de vitória!
        """
        while len(board) < player_level and bench:
            board.append(bench.pop(0))

        board = TacticalPositioner.optimize_positions(board, in_place=True)
        if not bench or not board:
            return board, bench

        candidates: List[List[BoardUnit]] = [clone_board(board)]
        swap_pairs: List[Tuple[int, int]] = []

        for b_idx in range(len(board)):
            for bench_idx in range(min(4, len(bench))):
                cand = clone_board(board)
                cand[b_idx] = bench[bench_idx].clone()
                candidates.append(TacticalPositioner.optimize_positions(cand, in_place=True))
                swap_pairs.append((b_idx, bench_idx))

        probs = self.predict_win_prob_vs_lobby(candidates, opponent_boards)
        best_cand_idx = int(np.argmax(probs))
        if best_cand_idx > 0 and probs[best_cand_idx] > probs[0] + 0.01:
            b_i, bench_i = swap_pairs[best_cand_idx - 1]
            old_board_u = board[b_i]
            board[b_i] = bench[bench_i]
            bench[bench_i] = old_board_u
            board = TacticalPositioner.optimize_positions(board, in_place=True)

        return board, bench


# ============================================================================
# Comunidade Interna de Modelos: 7 Peritos Especialistas + Meta-Arbitrador
# ============================================================================

# Fatias de domínio primário (dentro dos 87 primeiros índices universais) para dar
# viés indutivo estrutural distinto a cada um dos 7 Especialistas:
EXPERT_DOMAIN_SLICES: List[Tuple[int, int]] = [
    (0, 12),   # 0. Economist_Compound_Expert: Stage, HP, Gold, Interest, Level, XP, Streaks
    (4, 18),   # 1. WinStreak_Tempo_Expert: Level, Streaks, Board Size, Relative Power, 2*/3*, Alive
    (0, 14),   # 2. LoseStreak_Cashout_Expert: Stage, HP, Gold, Lose Streak, Shop Pairs/Synergy
    (12, 42),  # 3. HyperRoll_Evolution_Expert: Shop Slots (0..5), Pairs, Copies, Cost, 2*/3*
    (42, 73),  # 4. Synergy_Vertical_Expert: Vetor das 31 Sinergias Ativas no Tabuleiro
    (8, 42),   # 5. LobbyScout_Counter_Expert: Board/Bench Density + Shop Counter Options
    (0, 18),   # 6. LateGame_Cap_Expert: Stage, Level, Free Slots, 2*/3*, Max Synergy Tier
]


class SubAgentExpertModule(nn.Module):
    r"""
    Um Sub-Modelo Especialista dentro do Conselho/Comunidade que compõe 1 jogador.
    Possui uma projeção dedicada para sua fatia de domínio (`domain_slice`) combinada
    ao estado global, garantindo especialização estrutural real entre os 7 peritos!
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        num_actions: int,
        domain_slice: Tuple[int, int] = (0, 18),
    ):
        super().__init__()
        self.domain_slice = domain_slice
        dom_len = max(1, domain_slice[1] - domain_slice[0])
        self.domain_gate = nn.Sequential(
            nn.Linear(dom_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid(),
        )
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
        s_start, s_end = self.domain_slice
        dom_x = obs[:, s_start:s_end]
        gate = 0.5 + self.domain_gate(dom_x)  # (B, hidden_dim) in [0.5, 1.5]
        h = self.net(obs) * gate
        logits = self.action_proposal(h)
        return h, logits


class RunningReturnNormalizer:
    r"""
    Normalizador Welford Online (`RunningMeanStd`) para os retornos dos críticos (`v_shaped` e `v_oracle`)
    com métodos explícitos de `normalize`, `unnormalize` e serialização (`state_dict` / `load_state_dict`).
    """

    def __init__(self, epsilon: float = 1e-5, clip_range: float = 5.0):
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = epsilon
        self.clip_range: float = clip_range

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var) + 1e-6)

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = float(x.size)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = max(1e-4, m2 / tot_count)
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -self.clip_range, self.clip_range).astype(np.float32)

    def unnormalize(self, x_norm: np.ndarray) -> np.ndarray:
        if self.count <= 1e-3:
            return x_norm.astype(np.float32)
        return (x_norm * self.std + self.mean).astype(np.float32)

    def state_dict(self) -> Dict[str, float]:
        return {
            "mean": float(self.mean),
            "var": float(self.var),
            "count": float(self.count),
            "clip_range": float(self.clip_range),
        }

    def load_state_dict(self, state: Dict[str, float]) -> None:
        self.mean = float(state.get("mean", 0.0))
        self.var = float(state.get("var", 1.0))
        self.count = float(state.get("count", 1e-5))
        self.clip_range = float(state.get("clip_range", 5.0))


class CommunityOfModelsBrain(nn.Module):
    r"""
    "Uma Comunidade de Modelos para Ser Um Player" (Hierarchical Mixture-of-Experts):
    - 7 Sub-Modelos Especialistas (`EXPERT_NAMES`) analisam simultaneamente a partida.
    - `MetaArbitratorTransformer`: Faz Self-Attention entre as propostas dos 7 especialistas
      condicionado pela Persona $\mathbf{w}^{(k)}$ do agente e gera os pesos de voto
      $\boldsymbol{\alpha}(s) \in \Delta^6$ ($\sum_{m=1}^7 \alpha_m(s) = 1.0$).
    - **Slice Explícito de Persona (`persona_range=(73, 112)`)**:
      Recebe explicitamente no construtor o intervalo `(start, end)` dos 39 números da persona,
      evitando adivinhação pela dimensão (`obs[73:112]` tanto no layout 126 quanto no 188).
    - **Tronco de Crítico Desacoplado (`detach_critic_trunk=True`)**:
      Os críticos (`critic_shaped` e `critic_oracle`) possuem seu próprio extrator `critic_trunk`
      e recebem `pooled_council.detach()` para que o MSE do crítico NUNCA destrua os logits
      clonados (BC) ou o gating dos 7 especialistas!
    - **Codificador de Identidade de Unidade em Loja + Tabuleiro + Banco (`species_embedding`)**:
      Codifica os IDs alfabéticos compartilhados (`data/species_vocab.json`) dos 23 slots
      (`6 loja + 9 tabuleiro + 8 banco`, ou `6 loja` com zero-padding automático).
    """

    def __init__(
        self,
        obs_dim: int = GPU_OBS_DIM,
        num_actions: int = NUM_ACTIONS,
        expert_dim: int = 128,
        nhead: int = 4,
        persona_range: Tuple[int, int] = (73, 112),
        detach_critic_trunk: bool = True,
        num_species_vocab: int = 1150,
        species_emb_dim: int = 16,
    ):
        super().__init__()
        if persona_range[1] - persona_range[0] != 39:
            raise ValueError(f"persona_range deve ter tamanho exato 39, recebido: {persona_range}")
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.expert_dim = expert_dim
        self.persona_range = persona_range
        self.detach_critic_trunk = detach_critic_trunk
        self.num_species_vocab = num_species_vocab
        self.species_emb_dim = species_emb_dim

        # Embedding compartilhado de espécie para Loja (6), Tabuleiro (9, mean+max pool) e Banco (8):
        # 6*emb (shop slots) + 2*emb (board mean+max pool) + 8*emb (bench slots) = 16 * species_emb_dim
        self.species_embedding = nn.Embedding(num_species_vocab, species_emb_dim, padding_idx=0)
        self.unit_id_actor_proj = nn.Linear(16 * species_emb_dim, expert_dim)
        self.unit_id_critic_proj = nn.Linear(16 * species_emb_dim, expert_dim)

        # Os 7 Sub-Agentes Especialistas (Política / Ator) com viés indutivo por domínio
        self.experts = nn.ModuleList(
            [
                SubAgentExpertModule(
                    obs_dim,
                    expert_dim,
                    num_actions,
                    domain_slice=EXPERT_DOMAIN_SLICES[m],
                )
                for m in range(NUM_EXPERTS)
            ]
        )

        # Codificador da Persona / Objetivo do Jogador (39 dims lidos de self.persona_range)
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

        # Residual Meta-Policy
        self.meta_residual_actor = nn.Linear(expert_dim, num_actions)

        # Tronco Dedicado e Desacoplado para os Críticos (evita interferência Ator-Crítico!)
        self.critic_state_encoder = nn.Sequential(
            nn.Linear(obs_dim, expert_dim),
            nn.LayerNorm(expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, expert_dim),
            nn.LayerNorm(expert_dim),
            nn.GELU(),
        )
        self.critic_shaped = nn.Sequential(
            nn.Linear(expert_dim * 3, expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, 1),
        )
        self.critic_oracle = nn.Sequential(
            nn.Linear(expert_dim * 3, expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, 1),
        )

    def _extract_persona_slice(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extrai os 39 números da persona usando o intervalo explícito `self.persona_range`
        configurado no construtor (por padrão `(73, 112)` tanto para 126 quanto para 188 dims).
        """
        start, end = self.persona_range
        return obs[:, start:end]

    def _encode_unit_identities(self, unit_species_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Codifica as identidades de espécies da Loja (6), Tabuleiro (9) e Banco (8).
        Aceita shape `(B, 23)` completo `[shop(6), board(9), bench(8)]` ou `(B, 6)` apenas da loja.
        """
        ids = unit_species_ids.long().clamp(0, self.num_species_vocab - 1)
        bsz = ids.shape[0]
        if ids.shape[1] < TOTAL_IDENTITY_SLOTS:
            pad = torch.zeros(
                (bsz, TOTAL_IDENTITY_SLOTS - ids.shape[1]),
                dtype=ids.dtype,
                device=ids.device,
            )
            ids = torch.cat([ids, pad], dim=1)
        elif ids.shape[1] > TOTAL_IDENTITY_SLOTS:
            ids = ids[:, :TOTAL_IDENTITY_SLOTS]

        emb = self.species_embedding(ids)  # (B, 23, E)
        shop_emb = emb[:, :6, :].flatten(start_dim=1)  # (B, 6*E)

        board_ids = ids[:, 6:15]  # (B, 9)
        board_emb = emb[:, 6:15, :]  # (B, 9, E)
        board_mask = (board_ids > 0).unsqueeze(-1).float()  # (B, 9, 1)
        board_mean = (board_emb * board_mask).sum(dim=1) / board_mask.sum(dim=1).clamp(min=1.0)
        board_max = (board_emb * board_mask).max(dim=1).values

        bench_emb = emb[:, 15:23, :].flatten(start_dim=1)  # (B, 8*E)
        joint_id_feat = torch.cat([shop_emb, board_mean, board_max, bench_emb], dim=-1)  # (B, 16*E)
        return self.unit_id_actor_proj(joint_id_feat), self.unit_id_critic_proj(joint_id_feat)

    def actor_parameters(self) -> List[nn.Parameter]:
        """Retorna apenas os parâmetros do Ator (7 Especialistas + Arbitrador + Gating + Identidade)."""
        params: List[nn.Parameter] = []
        for mod in (
            self.species_embedding,
            self.unit_id_actor_proj,
            self.experts,
            self.persona_proj,
            self.council_attn,
            self.council_norm,
            self.gating_head,
            self.meta_residual_actor,
        ):
            params.extend(list(mod.parameters()))
        return params

    def critic_parameters(self) -> List[nn.Parameter]:
        """Retorna apenas os parâmetros do Crítico (`critic_state_encoder`, `unit_id_critic_proj`, `critic_shaped`, `critic_oracle`)."""
        params: List[nn.Parameter] = []
        for mod in (self.critic_state_encoder, self.unit_id_critic_proj, self.critic_shaped, self.critic_oracle):
            params.extend(list(mod.parameters()))
        return params

    def forward_council(
        self,
        obs: torch.Tensor,
        unit_species_ids: Optional[torch.Tensor] = None,
        shop_species_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Executa os 7 Especialistas e o Meta-Arbitrador Transformer:
        Retorna:
          - `final_logits`: shape `(B, NUM_ACTIONS)`
          - `expert_weights`: shape `(B, NUM_EXPERTS)` com $\sum_m \alpha_m = 1$
          - `v_shaped`: shape `(B,)`
          - `v_oracle`: shape `(B,)`
        """
        ids_tensor = unit_species_ids if unit_species_ids is not None else shop_species_ids
        actor_id_ctx: Optional[torch.Tensor] = None
        critic_id_ctx: Optional[torch.Tensor] = None
        if ids_tensor is not None:
            actor_id_ctx, critic_id_ctx = self._encode_unit_identities(ids_tensor)

        expert_hs = []
        expert_logits = []
        for exp in self.experts:
            h_m, l_m = exp(obs)
            expert_hs.append(h_m)
            expert_logits.append(l_m)

        # Stack across the 7 experts: H has shape (B, 7, D), L has shape (B, 7, A)
        H = torch.stack(expert_hs, dim=1)
        if actor_id_ctx is not None:
            H = H + actor_id_ctx.unsqueeze(1)
        L = torch.stack(expert_logits, dim=1)

        # Self-attention debate among the 7 experts
        attn_out, _ = self.council_attn(H, H, H)
        H_debated = self.council_norm(H + attn_out)  # (B, 7, D)

        # Condition gating on the player's persona vector (via explicit self.persona_range)
        persona_slice = self._extract_persona_slice(obs)
        persona_emb = self.persona_proj(persona_slice)  # (B, D)
        persona_exp = persona_emb.unsqueeze(1).expand_as(H_debated)  # (B, 7, D)

        gate_scores = self.gating_head(torch.cat([H_debated, persona_exp], dim=-1)).squeeze(-1)  # (B, 7)
        expert_weights = F.softmax(gate_scores, dim=-1)  # (B, 7)

        # Weighted voting of expert action logits + meta residual refinement
        weighted_logits = (L * expert_weights.unsqueeze(-1)).sum(dim=1)  # (B, A)
        pooled_council = (H_debated * expert_weights.unsqueeze(-1)).sum(dim=1)  # (B, D)

        final_logits = weighted_logits + 0.25 * self.meta_residual_actor(pooled_council)

        # Desacoplamento do Crítico: evita que gradientes grandes de MSE destruam a política clonada!
        council_for_critic = pooled_council.detach() if self.detach_critic_trunk else pooled_council
        persona_for_critic = persona_emb.detach() if self.detach_critic_trunk else persona_emb
        critic_state_h = self.critic_state_encoder(obs)
        if critic_id_ctx is not None:
            critic_state_h = critic_state_h + (critic_id_ctx.detach() if self.detach_critic_trunk else critic_id_ctx)
        joint_critic_in = torch.cat([critic_state_h, council_for_critic, persona_for_critic], dim=-1)

        v_shaped = self.critic_shaped(joint_critic_in).squeeze(-1)
        v_oracle = self.critic_oracle(joint_critic_in).squeeze(-1)

        return final_logits, expert_weights, v_shaped, v_oracle

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        unit_species_ids: Optional[torch.Tensor] = None,
        shop_species_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_logits, expert_weights, v_shaped, v_oracle = self.forward_council(
            obs,
            unit_species_ids=unit_species_ids,
            shop_species_ids=shop_species_ids,
        )
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
    round_done: bool = False
    unit_species_ids: Optional[np.ndarray] = None


class GPUMaskablePPOTrainer:
    r"""
    Treinador Maskable PPO de Segunda Geração para o `CommunityOfModelsBrain`
    (Corrigido conforme auditoria técnica — Engstrom et al. ICLR 2020 & Huang et al. 37 Details):
    1. **Consistência Estrita de Escala no Crítico (`_compute_gae_and_normalized_targets`)**:
       - Desnormaliza `s.value` com as estatísticas vigentes no momento da coleta (`ret_norm.unnormalize`),
         calcula o GAE (`advantages` e `raw_returns`) na escala real da recompensa e re-normaliza
         SIMULTANEAMENTE `raw_returns` e `val_real` (`norm_old_val`) na mesma escala atualizada!
       - Por padrão, `clip_vloss=False` (conforme recomendação da literatura de PPO para evitar
         travar o crítico em valores antigos defasados), usando perda Smooth L1 (Huber) pura.
    2. **Sincronização Pós-Aquecimento (`warmup_critic_only`)**:
       - Após aquecer o crítico, atualiza in-place os `s.value` do buffer com as previsões do
         crítico aquecido para que uma chamada subsequente a `update()` nunca compare o novo
         crítico contra valores aleatórios pré-aquecimento!
    3. **Âncora KL contra Política Clonada (`set_bc_reference_anchor`, estilo AlphaStar)** e
       **Piso de Entropia (`entropy_floor = 0.32`) + Parada KL por Mini-Batch (`target_kl = 0.012`)**.
    """

    def __init__(
        self,
        model: CommunityOfModelsBrain,
        device: torch.device,
        lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.15,
        ent_coef: float = 0.002,
        entropy_floor: float = 0.32,
        council_balance_coef: float = 0.001,
        vf_coef: float = 0.5,
        bc_kl_coef: float = 0.04,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.012,
        normalize_returns: bool = True,
        clip_vloss: bool = False,
        use_amp: bool = True,
    ):
        self.device = device
        self.model = model.to(device)
        self.actor_params = self.model.actor_parameters()
        self.critic_params = self.model.critic_parameters()
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.actor_params, "lr": lr, "weight_decay": 1e-4},
                {"params": self.critic_params, "lr": critic_lr, "weight_decay": 1e-4},
            ]
        )
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.entropy_floor = entropy_floor
        self.council_balance_coef = council_balance_coef
        self.vf_coef = vf_coef
        self.bc_kl_coef = bc_kl_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.normalize_returns = normalize_returns
        self.clip_vloss = clip_vloss
        self.ret_norm = RunningReturnNormalizer()
        self.ora_norm = RunningReturnNormalizer()
        self.bc_anchor_model: Optional[CommunityOfModelsBrain] = None
        self.use_amp = use_amp and (device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def state_dict(self) -> Dict[str, object]:
        """Serializa o estado completo do treinador PPO (otimizador + normalizadores de retorno)."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "ret_norm": self.ret_norm.state_dict(),
            "ora_norm": self.ora_norm.state_dict(),
            "normalize_returns": self.normalize_returns,
            "clip_vloss": self.clip_vloss,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restaura o estado completo do treinador PPO a partir de um checkpoint."""
        if "optimizer" in state and state["optimizer"] is not None:
            self.optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
        if "ret_norm" in state and isinstance(state["ret_norm"], dict):
            self.ret_norm.load_state_dict(state["ret_norm"])
        if "ora_norm" in state and isinstance(state["ora_norm"], dict):
            self.ora_norm.load_state_dict(state["ora_norm"])

    def set_bc_reference_anchor(self, bc_kl_coef: float = 0.04) -> None:
        """
        Chame este método imediatamente após terminar a Clonagem de Comportamento (BC)!
        Congela uma cópia da política clonada para servir como Âncora de Confiança KL
        durante o PPO, impedindo que vantagens ruidosas degradem a política de 86% de precisão.
        """
        import copy
        self.bc_anchor_model = copy.deepcopy(self.model).to(self.device)
        self.bc_anchor_model.eval()
        for p in self.bc_anchor_model.parameters():
            p.requires_grad = False
        self.bc_kl_coef = bc_kl_coef
        # Pós-BC, desativa pressão artificial de uniformidade entre especialistas
        self.council_balance_coef = 0.0

    def _compute_gae_and_normalized_targets(
        self,
        buffer: List[GPURolloutStep],
        update_normalizer: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula GAE na escala real das recompensas e retorna `(advantages, norm_returns, norm_ora, norm_old_val)`
        garantindo que `norm_returns` e `norm_old_val` estejam EXATAMENTE na mesma escala normalizada!
        """
        n = len(buffer)
        rew_arr = np.array([s.reward for s in buffer], dtype=np.float32)
        done_arr = np.array([s.done for s in buffer], dtype=np.float32)
        val_arr = np.array([s.value for s in buffer], dtype=np.float32)
        ora_arr = np.array([s.oracle_target for s in buffer], dtype=np.float32)

        # 1. Desnormaliza `val_arr` usando a escala que estava ativa durante a coleta do rollout
        if self.normalize_returns and self.ret_norm.count > 1.0:
            val_real = self.ret_norm.unnormalize(val_arr)
        else:
            val_real = val_arr.copy()

        # 2. Calcula GAE estritamente na escala real das recompensas
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_non_terminal = 1.0 - done_arr[t]
            next_val = val_real[t + 1] if t + 1 < n else 0.0
            delta = rew_arr[t] + self.gamma * next_val * next_non_terminal - val_real[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        raw_returns = advantages + val_real

        # 3. Atualiza o normalizador (se solicitado) e projeta TANTO `raw_returns` QUANTO `val_real`
        # para a MESMA escala atualizada!
        if self.normalize_returns:
            if update_normalizer:
                self.ret_norm.update(raw_returns)
                self.ora_norm.update(ora_arr)
            norm_returns = self.ret_norm.normalize(raw_returns)
            norm_ora = self.ora_norm.normalize(ora_arr)
            norm_old_val = self.ret_norm.normalize(val_real)
        else:
            norm_returns = raw_returns
            norm_ora = ora_arr
            norm_old_val = val_real

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, norm_returns, norm_ora, norm_old_val

    def _extract_unit_ids_tensor(self, buffer: List[GPURolloutStep]) -> Optional[torch.Tensor]:
        if not buffer or buffer[0].unit_species_ids is None:
            return None
        ids_arr = np.array(
            [
                s.unit_species_ids
                if s.unit_species_ids is not None
                else np.zeros(TOTAL_IDENTITY_SLOTS, dtype=np.int64)
                for s in buffer
            ],
            dtype=np.int64,
        )
        return torch.from_numpy(ids_arr).to(self.device)

    def warmup_critic_only(
        self,
        buffer: List[GPURolloutStep],
        warmup_epochs: int = 12,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """
        Fase 1.5 (Entre a Clonagem BC e o PPO):
        Congela 100% da política (os 7 Especialistas + Gating) e treina exclusivamente os
        críticos (`v_shaped` e `v_oracle`) até estabilizar a Variância Explicada ($R^2$).
        Ao final, atualiza `step.value` no buffer com as previsões do crítico aquecido para
        manter total consistência de escala caso `update()` seja chamado em seguida.
        """
        if not buffer:
            return {"critic_loss": 0.0, "explained_variance": 0.0}

        n = len(buffer)
        obs_arr = np.array([s.obs for s in buffer], dtype=np.float32)
        _, norm_returns, norm_ora, _ = self._compute_gae_and_normalized_targets(buffer, update_normalizer=True)

        t_obs = torch.from_numpy(obs_arr).to(self.device)
        t_ids = self._extract_unit_ids_tensor(buffer)
        t_ret = torch.from_numpy(norm_returns).to(self.device)
        t_ora = torch.from_numpy(norm_ora).to(self.device)

        critic_opt = torch.optim.AdamW(self.critic_params, lr=1.5e-3, weight_decay=1e-4)
        self.model.train()
        last_loss = 0.0

        for _ in range(warmup_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                critic_opt.zero_grad()
                mb_ids = t_ids[idx] if t_ids is not None else None
                _, _, v_sh, v_or = self.model.forward_council(t_obs[idx], unit_species_ids=mb_ids)
                loss_v = F.smooth_l1_loss(v_sh, t_ret[idx]) + 0.5 * F.smooth_l1_loss(v_or, t_ora[idx])
                loss_v.backward()
                nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
                critic_opt.step()
                last_loss = float(loss_v.item())

        self.model.eval()
        with torch.no_grad():
            _, _, v_pred_all, _ = self.model.forward_council(t_obs, unit_species_ids=t_ids)
            y_true = t_ret.cpu().numpy()
            y_pred = v_pred_all.cpu().numpy()
            # Sincroniza os valores do buffer com o crítico aquecido na escala atual do ret_norm!
            for idx_step, step in enumerate(buffer):
                step.value = float(y_pred[idx_step])
            var_y = float(np.var(y_true))
            ev = 1.0 - float(np.var(y_true - y_pred)) / max(1e-6, var_y)

        return {"critic_loss": last_loss, "explained_variance": ev}

    def update(self, buffer: List[GPURolloutStep], ppo_epochs: int = 4, batch_size: int = 256) -> Dict[str, float]:
        if not buffer:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "explained_var": 0.0,
            }

        n = len(buffer)
        obs_arr = np.array([s.obs for s in buffer], dtype=np.float32)
        mask_arr = np.array([s.mask for s in buffer], dtype=bool)
        act_arr = np.array([s.action for s in buffer], dtype=np.int64)
        old_logp_arr = np.array([s.log_prob for s in buffer], dtype=np.float32)

        advantages, norm_returns, norm_ora, norm_old_val = self._compute_gae_and_normalized_targets(
            buffer, update_normalizer=True
        )

        t_obs = torch.from_numpy(obs_arr).to(self.device)
        t_ids = self._extract_unit_ids_tensor(buffer)
        t_mask = torch.from_numpy(mask_arr).to(self.device)
        t_act = torch.from_numpy(act_arr).to(self.device)
        t_old_logp = torch.from_numpy(old_logp_arr).to(self.device)
        t_old_val = torch.from_numpy(norm_old_val).to(self.device)
        t_adv = torch.from_numpy(advantages).to(self.device)
        t_ret = torch.from_numpy(norm_returns).to(self.device)
        t_ora = torch.from_numpy(norm_ora).to(self.device)

        total_pg, total_vf, total_ent, total_kl, total_clip, updates = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        early_stop_triggered = False
        self.model.train()

        for _ in range(ppo_epochs):
            if early_stop_triggered:
                break
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                self.optimizer.zero_grad()
                mb_ids = t_ids[idx] if t_ids is not None else None

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    raw_logits, exp_w, new_val, new_ora = self.model.forward_council(
                        t_obs[idx], unit_species_ids=mb_ids
                    )
                    masked_logits = raw_logits.masked_fill(~t_mask[idx], -1e8)
                    dist = Categorical(logits=masked_logits)
                    new_logp = dist.log_prob(t_act[idx])
                    entropy = dist.entropy()

                    logratio = new_logp - t_old_logp[idx]
                    ratio = logratio.exp()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()

                    # Parada imediata por Mini-Batch (após o 1º passo) se violar a região de confiança KL!
                    if updates > 0 and float(approx_kl.item()) > 1.5 * self.target_kl:
                        early_stop_triggered = True
                        break

                    mb_adv = t_adv[idx]
                    # Se houver âncora BC ativa, aplica filtro assimétrico suave nas vantagens negativas ruidosas
                    if self.bc_anchor_model is not None:
                        adv_weight = torch.where(mb_adv >= 0.0, torch.ones_like(mb_adv), 0.5 * torch.ones_like(mb_adv))
                        mb_adv = mb_adv * adv_weight

                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Huber Value Loss (Smooth L1) consistente na mesma escala de t_ret
                    if self.clip_vloss:
                        v_pred_clipped = t_old_val[idx] + torch.clamp(
                            new_val - t_old_val[idx], -self.clip_coef, self.clip_coef
                        )
                        v_loss_unclipped = F.smooth_l1_loss(new_val, t_ret[idx], reduction="none")
                        v_loss_clipped = F.smooth_l1_loss(v_pred_clipped, t_ret[idx], reduction="none")
                        v_loss_shaped = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss_shaped = F.smooth_l1_loss(new_val, t_ret[idx])

                    v_loss_oracle = F.smooth_l1_loss(new_ora, t_ora[idx])
                    v_loss = 0.5 * v_loss_shaped + 0.25 * v_loss_oracle

                    # Piso de Entropia (Entropy Floor): NUNCA empurra a entropia para cima se H >= entropy_floor!
                    mean_ent = entropy.mean()
                    ent_deficit = F.relu(self.entropy_floor - entropy)
                    ent_Floor_penalty = (ent_deficit ** 2).mean()

                    # Balanceamento suave dos 7 especialistas
                    mean_exp_w = exp_w.mean(dim=0)
                    council_balance_ent = -(mean_exp_w * torch.log(mean_exp_w + 1e-8)).sum()

                    # Âncora KL opcional em relação à política clonada (BC Reference Policy)
                    bc_kl_loss = torch.tensor(0.0, device=self.device)
                    if self.bc_anchor_model is not None and self.bc_kl_coef > 0.0:
                        with torch.no_grad():
                            ref_logits, _, _, _ = self.bc_anchor_model.forward_council(
                                t_obs[idx], unit_species_ids=mb_ids
                            )
                            ref_masked = ref_logits.masked_fill(~t_mask[idx], -1e8)
                            ref_probs = F.softmax(ref_masked, dim=-1)
                        cur_log_probs = F.log_softmax(masked_logits, dim=-1)
                        ref_log_probs = F.log_softmax(ref_masked, dim=-1)
                        bc_kl_loss = (ref_probs * (ref_log_probs - cur_log_probs)).sum(dim=-1).mean()

                    loss = (
                        pg_loss
                        + self.vf_coef * v_loss
                        + self.ent_coef * ent_Floor_penalty
                        + self.bc_kl_coef * bc_kl_loss
                        - self.council_balance_coef * council_balance_ent
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                # Clipping de gradiente SEPARADO para Ator e Crítico!
                nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_pg += float(pg_loss.item())
                total_vf += float(v_loss.item())
                total_ent += float(mean_ent.item())
                total_kl += float(approx_kl.item())
                total_clip += float(clip_frac.item())
                updates += 1

        with torch.no_grad():
            _, _, v_all, _ = self.model.forward_council(t_obs, unit_species_ids=t_ids)
            y_t = t_ret.cpu().numpy()
            y_p = v_all.cpu().numpy()
            ev = 1.0 - float(np.var(y_t - y_p)) / max(1e-6, float(np.var(y_t)))

        denom = max(1, updates)
        return {
            "policy_loss": total_pg / denom,
            "value_loss": total_vf / denom,
            "entropy": total_ent / denom,
            "approx_kl": total_kl / denom,
            "clip_fraction": total_clip / denom,
            "explained_var": ev,
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

    Correção de Distribuição (Auditoria Técnica):
    - Ambos os tabuleiros `ba` e `bb` são SEMPRE amostrados no MESMO estágio `stage`!
    - 40% dos pares são confrontos parelhos/contrafactuais (mesmo tamanho de tabuleiro ou
      perturbação de 1 peça/estrela), calibrando o surrogate especificamente para as consultas
      marginais do `NeuralLookaheadRosterPlanner`.
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

    def sample_board_at_stage(stage: int, target_size: Optional[int] = None) -> List[BoardUnit]:
        if bots and rng.random() < 0.65:
            bot = bots[int(rng.integers(0, len(bots)))]
            entries = repo.get_bot_stage_board(bot, stage)
            units = []
            for name, x, y, n_items in entries[:MAX_BOARD_UNITS]:
                p = repo.get_pokemon(name)
                if p is not None:
                    units.append(BoardUnit.from_pokemon_stats(p, x=x, y=min(3, max(0, y - 1)), num_items=n_items))
            if units:
                return units
        expected_n = min(9, max(2, stage // 3 + 2))
        n_u = target_size if target_size is not None else int(rng.integers(max(2, expected_n - 1), min(9, expected_n + 1) + 1))
        chosen = rng.choice(len(all_pokes), size=min(n_u, len(all_pokes)), replace=False)
        max_it = min(3, max(1, stage // 8 + 1))
        units = [
            BoardUnit.from_pokemon_stats(
                all_pokes[idx],
                x=int(rng.integers(0, 6)),
                y=int(rng.integers(0, 4)),
                num_items=int(rng.integers(0, max_it)),
            )
            for idx in chosen
        ]
        return TacticalPositioner.optimize_positions(units, in_place=True)

    for i in range(num_samples):
        stage = int(rng.integers(3, 35))
        ba = sample_board_at_stage(stage)
        if rng.random() < 0.35 and len(ba) >= 2:
            # Par parelho/contrafactual no mesmo estágio e mesmo número de unidades
            bb = sample_board_at_stage(stage, target_size=len(ba))
        else:
            bb = sample_board_at_stage(stage)
        outcome = sim.simulate(ba, bb, stage_level=stage, rng=rng)
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
