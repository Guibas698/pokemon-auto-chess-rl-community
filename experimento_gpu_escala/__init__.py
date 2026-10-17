r"""
Experimento em Escala para GPU (`experimento_gpu_escala`) — Pokémon Auto Chess RL Lab.

Projetado para rodar em máquinas com GPU NVIDIA (CUDA + Automatic Mixed Precision FP16/BF16):
- `DeepSetTransformerSurrogate`: Set-Transformer profundo (`d_model=256, nhead=8, 4 camadas`)
  com Cross-Attention entre os dois tabuleiros e cabeças auxiliares de previsão de Dano (DPS)
  e Tank (Dano Absorvido/Mitigado) por unidade.
- `DeepEntityTransformerActorCritic`: Actor-Critic Residual profundo (`d_model=256, 4 blocos`)
  com condicionamento FiLM multi-camada + Busca Clássica Lookahead de 1 passo na GPU.
- `MegaCommunityLeagueTrainer`: Comunidade completa com **35 Agentes RL Especialistas**
  (todas as 31 Sinergias do PAC + 4 Arquétipos Econômicos Puros), Prioritized Fictitious
  Self-Play (PFSP), Population-Based Training (PBT), Matriz MAP-Elites completa (31x4 = 124 células),
  Matriz de Counter-Picks 31x31 e Destilação VIPER de alta profundidade (`max_depth=5`).
"""
