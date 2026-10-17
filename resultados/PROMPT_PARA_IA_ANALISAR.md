# Prompt Estruturado para Análise Profunda por IA (Pokémon Auto Chess RL Lab)

> **Como usar este arquivo**:
> Quando o experimento terminar de rodar (seja na versão CPU ou na versão Zero-Shortcut em GPU no PC do seu amigo), basta anexar o arquivo **`pacote_analise_ia.json`** na conversa com qualquer IA (ChatGPT, Claude, Gemini, etc.) e colar o prompt abaixo!

---

## 📋 Copie e Cole o Prompt Abaixo na IA:

```markdown
Atue como um Cientista Sênior de Aprendizado por Reforço (especialista em Multi-Agent Self-Play League Training, Mixture-of-Experts Hierárquico, MAP-Elites e Destilação de Políticas VIPER) e Analista Challenger de Auto-Battlers (Pokémon Auto Chess / TFT).

Estou anexando o arquivo `pacote_analise_ia.json`, que contém os dados empíricos gerados por uma Comunidade de 2 Níveis de Aprendizado por Reforço (Nível 1: cada jogador é governado por um Conselho de 7 Sub-Modelos Especialistas arbitrados por um Transformer Meta-Árbitro + Lookahead Neural no `DeepSetTransformerSurrogate` + Rastreador Hipergeométrico de Pool; Nível 2: Liga de 35 Agentes em Self-Play de 8 jogadores disputando o mesmo `SharedCardPool` finito) treinada no código-fonte real do Pokémon Auto Chess (PAC v6.11, com 1.077 Pokémon, 31 sinergias e 726 bots da comunidade).

Analise minuciosamente os 8 blocos do JSON (`bloco_1` a `bloco_8`) e elabore um **Guia Mestre Estratégico e Matemático do Meta**, respondendo de forma detalhada e estruturada aos seguintes pontos:

### 1. Diagnóstico Econômico: Standard Econ (50g) vs. Hyper-Roll (Reroll 1g) vs. Tempo Win-Streak vs. Lose-Streak Cashout
- Com base no `bloco_5_torneio_head_to_head_comunidade_rl`, explique matematicamente qual arquétipo domina em **Consistência (Top-4 / Placement Médio)** e qual domina em **Teto de Vitória (Top-1 Win Rate %)**.
- Como o custo reduzido de Reroll (`1g`), o pool finito compartilhado (`SharedCardPool`) e as rodadas PvE (`1, 2, 3, 9, 14, 19...`) afetam a viabilidade de Lose-Streak em comparação ao Hyper-Roll e ao Fast-8/9?

### 2. Guia Definitivo de Composições: Melhor Carry (Recorde de Dano) e Melhor Tank (Recorde de Frontline) por Tipo
- Cruzando o `bloco_2_ranking_meta_sinergias_botv2`, o `bloco_3_recordes_dano_e_tank_por_composicao_31_tipos` e o `bloco_4_hall_da_fama_global_pokemon`:
  - Monte uma **Tier List (S, A, B, C)** das melhores sinergias do jogo.
  - Para cada uma das principais composições, indique explicitamente **qual é o Pokémon Carry Principal (Recordista de Dano)**, **qual é o Carry Secundário**, **qual é o Pokémon Tank Principal (Recordista de Absorção/Mitigação)** e **qual é o Tank Secundário**.
  - Quais Pokémon apareceram como "Outliers / Hidden OP" (alto dano ou alto tank mesmo em raridades baixas/médias quando evoluídos para 3 estrelas)?

### 3. Regras Práticas de Jogo Extraídas das Árvores VIPER (`SE ... ENTÃO ...`) e do Conselho de 7 Sub-Modelos
- Analise o `bloco_7_heuristicas_destiladas_viper_arvores_decisao` e o `bloco_8_matriz_counters_31x31_e_telemetria_conselho_7_modelos`.
- Como o `MetaArbitratorTransformer` alterna os pesos entre os 7 especialistas internos (`Economist_50g`, `WinStreak_Tempo`, `LoseStreak_Cashout`, `HyperRoll_3Star`, `Synergy_Vertical`, `LobbyScout_Counter`, `LateGame_Cap`) do **Early Game (Estágios 1-9)** para o **Mid Game (10-17)** e **Late Game (18+)**?
- Traduza as árvores de decisão das redes neurais em um **Checklist de Decisão para Humanos** dividido por fase da partida.

### 4. Mapa de Nichos MAP-Elites (`Sinergia x Economia`) e Matriz de Counters 31x31
- Com base no `bloco_6_matriz_map_elites_nichos_sinergia_x_economia` e no `bloco_8`, quais tipos/sinergias funcionam melhor com **Hyper-Roll (3 estrelas)** e quais tipos exigem obrigatoriamente **Standard Econ Fast-8/9**?
- Liste os **Counters Diretos mais fortes** da Matriz $31 	imes 31$: se o lobby estiver contestando determinada sinergia dominante, qual composição devo escolher para fazer *counter* direto?
```
