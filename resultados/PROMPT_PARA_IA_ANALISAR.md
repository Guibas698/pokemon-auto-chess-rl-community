# Prompt Estruturado para Análise Profunda por IA (Pokémon Auto Chess RL Lab)

> **Como usar este arquivo**:
> Quando o experimento terminar de rodar (seja na versão CPU ou na versão massiva em GPU no PC do seu amigo), basta anexar o arquivo **`pacote_analise_ia.json`** na conversa com qualquer IA (ChatGPT, Claude, Gemini, etc.) e colar o prompt abaixo!

---

## 📋 Copie e Cole o Prompt Abaixo na IA:

```markdown
Atue como um Cientista Sênior de Aprendizado por Reforço (especialista em Multi-Agent League Training, MAP-Elites e Destilação de Políticas VIPER) e Analista Challenger de Auto-Battlers (Pokémon Auto Chess / TFT).

Estou anexando o arquivo `pacote_analise_ia.json`, que contém os dados empíricos gerados por uma Comunidade de Modelos de Aprendizado por Reforço (PyTorch Maskable PPO + Set-Transformer Combat Surrogate + Árvores de Decisão VIPER) treinada no código-fonte real do Pokémon Auto Chess (PAC v6.11, com 1.077 Pokémon, 31 sinergias e 726 bots da comunidade).

Analise minuciosamente os 7 blocos do JSON (`bloco_1` a `bloco_7`) e elabore um **Guia Mestre Estratégico e Matemático do Meta**, respondendo de forma detalhada e estruturada aos seguintes pontos:

### 1. Diagnóstico Econômico: Standard Econ (50g) vs. Hyper-Roll (Reroll 1g) vs. Tempo Win-Streak vs. Lose-Streak Cashout
- Com base no `bloco_5_torneio_head_to_head_comunidade_rl`, explique matematicamente qual arquétipo domina em **Consistência (Top-4 / Placement Médio)** e qual domina em **Teto de Vitória (Top-1 Win Rate %)**.
- Como o custo reduzido de Reroll (`1g`) e as rodadas PvE (`1, 2, 3, 9, 14, 19...`) afetam a viabilidade de Lose-Streak em comparação ao Hyper-Roll e ao Fast-8/9?

### 2. Guia Definitivo de Composições: Melhor Carry (Recorde de Dano) e Melhor Tank (Recorde de Frontline) por Tipo
- Cruzando o `bloco_2_ranking_meta_sinergias_botv2`, o `bloco_3_recordes_dano_e_tank_por_composicao_31_tipos` e o `bloco_4_hall_da_fama_global_pokemon`:
  - Monte uma **Tier List (S, A, B, C)** das melhores sinergias do jogo.
  - Para cada uma das principais composições, indique explicitamente **qual é o Pokémon Carry Principal (Recordista de Dano)**, **qual é o Carry Secundário**, **qual é o Pokémon Tank Principal (Recordista de Absorção/Mitigação)** e **qual é o Tank Secundário**.
  - Quais Pokémon apareceram como " Outliers / Hidden OP" (alto dano ou alto tank mesmo em raridades baixas/médias quando evoluídos para 3 estrelas)?

### 3. Regras Práticas de Jogo Extraídas das Árvores VIPER (`SE ... ENTÃO ...`)
- Analise o `bloco_7_heuristicas_destiladas_viper_arvores_decisao`.
- Traduza as árvores de decisão das redes neurais em um **Checklist de Decisão para Humanos** dividido por fase da partida:
  - **Early Game (Estágios 1 a 9)**: Quando guardar ouro vs. quando rodar a loja ou subir de nível?
  - **Mid Game (Estágios 10 a 19 — Uniques e Additional Picks)**: Quais gatilhos de `player_hp`, `gold`, `bench_size` e `shop_synergy_matches` fazem os melhores agentes darem all-in ou subirem para o nível 7/8?
  - **Late Game (Estágios 20+ — Legendaries)**: Como converter vantagem econômica em Top-1?

### 4. Mapa de Nichos MAP-Elites (`Sinergia x Economia`)
- Com base no `bloco_6_matriz_map_elites_nichos_sinergia_x_economia`, quais tipos/sinergias funcionam melhor com **Hyper-Roll (3 estrelas)** e quais tipos exigem obrigatoriamente **Standard Econ Fast-8/9**?
- Aponte eventuais relações de *Counter* (Pedra-Papel-Tesoura) entre composições de Explosão/Crítico (`DARK`, `FIRE`, `ELECTRIC`), Sustentação (`FLORA`, `GRASS`, `GOURMET`, `NORMAL`) e Mitigação/Controle (`FIGHTING`, `STEEL`, `WATER`, `ICE`, `DRAGON`).
```
