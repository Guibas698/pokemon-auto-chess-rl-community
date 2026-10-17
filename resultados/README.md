# 📊 Resultados Empíricos — Pokémon Auto Chess RL Community & VIPER Lab

Esta pasta armazena todos os relatórios gerados pelos experimentos de **Aprendizado por Reforço Comunitário (Maskable PPO + PBT + MAP-Elites)**, **Simulação Tática de Combate** e **Destilação de Políticas em Árvores de Decisão (VIPER)** sobre os dados oficiais do **Pokémon Auto Chess (PAC v6.11)**.

---

## 📁 Arquivos Disponíveis Nesta Pasta

| Arquivo | Descrição |
| :--- | :--- |
| 📄 **[`relatorio_experimento_pac_rl.pdf`](./relatorio_experimento_pac_rl.pdf)** | **Relatório Executivo em PDF (Multi-Página)** com gráficos vetoriais, evolução geracional, tabelas de recordes de Dano/Tank por composição, Hall da Fama e regras VIPER. |
| 🤖 **[`pacote_analise_ia.json`](./pacote_analise_ia.json)** | **Pacote de 7 Blocos Semânticos em JSON** otimizado para leitura e análise profunda por LLMs (ChatGPT, Claude, Gemini). |
| 💬 **[`PROMPT_PARA_IA_ANALISAR.md`](./PROMPT_PARA_IA_ANALISAR.md)** | **Prompt Mestre Pronto** para copiar e colar junto com o `pacote_analise_ia.json` após rodar os experimentos (inclusive o experimento massivo em GPU). |
| 📈 **[`recordes_dano_e_tank_por_composicao.csv`](./recordes_dano_e_tank_por_composicao.csv)** | Tabela completa das **31 Sinergias** com o **Melhor Carry (Recorde de Dano)**, **2º Carry**, **Melhor Tank (Recorde de Frontline)** e **2º Tank**. |
| ⚔️ **[`hall_da_fama_top_dps.csv`](./hall_da_fama_top_dps.csv)** | Ranking global dos 20 Pokémon com maior recorde absoluto e média de dano por combate. |
| 🛡️ **[`hall_da_fama_top_tank.csv`](./hall_da_fama_top_tank.csv)** | Ranking global dos 20 Pokémon com maior recorde absoluto e média de dano absorvido/mitigado/curado. |
| 🗂️ **[`results_report_completo.json`](./results_report_completo.json)** | Dump bruto completo de todas as métricas, matrizes e árvores de decisão (`export_text`). |

---

## 🏆 1. Torneio Direto Head-to-Head da Comunidade RL (Todos os 8 Agentes no Mesmo Lobby)

| Pos. | Agente RL (`CommunityAgent`) | Arquétipo Econômico | Sinergia | Placement Médio | Top-4 (%) | Vitória Top-1 (%) | Ouro Médio | Nível Final | Recordista Dano (Carry) | Recordista Tank (Front) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **1º** | `Water_Mana_Control_Specialist` | `STANDARD_ECON_50G` | `WATER` | **3,33** | **76,7%** | 13,3% | 79,2g | **9,00** | `PRIMAL_KYOGRE(4*)` (1017) | `PALKIA(3*)` (1010) |
| **2º** | `HyperRoll_3Star_Collector` | `HYPER_ROLL` | `NORMAL` | **3,40** | **76,7%** | **26,7%** | 53,0g | 7,93 | `MELOETTA(3*)` (1337) | `REGIGIGAS(3*)` (1584) |
| **3º** | `Econ_50g_Fast8_Compounder` | `STANDARD_ECON_50G` | `FLEX` | **3,53** | **66,7%** | 16,7% | 56,3g | 8,30 | `MEWTWO(3*)` (1795) | `TORTERRA(3*)` (2110) |
| **4º** | `Fighting_Steel_Bastion_Specialist` | `STANDARD_ECON_50G` | `FIGHTING` | **3,80** | **66,7%** | 13,3% | 12,2g | 8,37 | `GALLADE(3*)` (1247) | `IRON_VALIANT(3*)` (2609) |
| **5º** | `Dark_Assassin_Crit_Specialist` | `HYPER_ROLL` | `DARK` | **4,67** | 46,7% | 16,7% | 101,4g | 6,93 | `SHADOW_LUGIA(3*)` (1195) | `GUZZLORD(3*)` (2212) |
| **6º** | `Tempo_WinStreak_Snowballer` | `TEMPO_WIN_STREAK` | `FLEX` | **5,53** | 23,3% | 3,3% | 42,6g | 7,77 | `GARCHOMP(3*)` (1249) | `REGIDRAGO(3*)` (1541) |
| **7º** | `Fire_Blaze_Aggro_Specialist` | `TEMPO_WIN_STREAK` | `FIRE` | **5,80** | 23,3% | 3,3% | 42,7g | 7,67 | `HO_OH(3*)` (1166) | `PRIMAL_GROUDON(4*)` (1470) |
| **8º** | `LoseStreak_Cashout_Pivot` | `LOSE_STREAK_CASHOUT` | `ARTIFICIAL` | **5,93** | 20,0% | 6,7% | 50,1g | 6,67 | `MEWTWO(3*)` (1795) | `SILVALLY(3*)` (1579) |

---

## ⚔️🛡️ 2. Recordes de Dano (Melhor Carry) e Recordes de Tank (Melhor Frontline) para Todas as 31 Composições

Abaixo estão os Pokémon que atingiram o **Maior Recorde de Dano em 1 Luta** e o **Maior Recorde de Dano Absorvido/Mitigado/Curado (Tank)** dentro de cada uma das **31 sinergias oficiais** do Pokémon Auto Chess:

| Sinergia / Composição | 🔥 Melhor Carry (Recorde de Dano) | Pico de Dano | Dano Médio | 2º Melhor Carry (Suporte DPS) | 🛡️ Melhor Tank (Recorde Frontline) | Pico de Tank | Tank Médio | 2º Melhor Tank (Suporte Front) |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :--- |
| **`MONSTER`** | `MEWTWO(3*)` | **1794,6** | 657,0 | `GARCHOMP(3*)` (1249) | `TORTERRA(3*)` | **2109,9** | 1165,5 | `PRIMAL_GROUDON(4*)` (1906) |
| **`ARTIFICIAL`** | `MEWTWO(3*)` | **1794,6** | 663,7 | `IRON_VALIANT(3*)` (1186) | `SILVALLY(3*)` | **1579,4** | 750,2 | `IRON_VALIANT(3*)` (1638) |
| **`PSYCHIC`** | `MEWTWO(3*)` | **1763,3** | 701,0 | `MEW(3*)` (1368) | `SOLGALEO(3*)` | **1262,6** | 977,6 | `GALLADE(3*)` (1504) |
| **`AMORPHOUS`** | `MEW(3*)` | **1367,8** | 697,7 | `NIHILEGO(3*)` (1115) | `MELMETAL(3*)` | **1281,5** | 886,6 | `EELEKTROSS(3*)` (987) |
| **`SOUND`** | `MELOETTA(3*)` | **1336,5** | 532,5 | `PIROUETTE_MELOETTA(3*)` (1043) | `MARACTUS(3*)` | **1655,4** | 869,4 | `PIROUETTE_MELOETTA(3*)` (1313) |
| **`NORMAL`** | `MELOETTA(3*)` | **1336,5** | 527,4 | `PIROUETTE_MELOETTA(3*)` (1043) | `REGIGIGAS(3*)` | **1584,1** | 794,2 | `CHANSEY(2*)` (1532) |
| **`AQUATIC`** | `KELDEO(3*)` | **1309,1** | 429,8 | `SHADOW_LUGIA(3*)` (1195) | `WALREIN(3*)` | **1303,6** | 765,1 | `LUGIA(3*)` (1120) |
| **`ICE`** | `KYUREM(3*)` | **1261,6** | 520,6 | `TORNADUS(3*)` (1026) | `REGICE(3*)` | **1463,7** | 940,1 | `ABOMASNOW(2*)` (1675) |
| **`POISON`** | `PECHARUNT(3*)` | **1254,1** | 661,5 | `NIHILEGO(3*)` (1134) | `VENUSAUR(3*)` | **1460,6** | 608,2 | `GLIMMORA(2*)` (1484) |
| **`GOURMET`** | `PECHARUNT(3*)` | **1254,1** | 548,7 | `HYDRAPPLE(4*)` (1186) | `PECHARUNT(3*)` | **2247,9** | 515,6 | `CHANSEY(2*)` (1532) |
| **`GHOST`** | `PECHARUNT(3*)` | **1254,1** | 540,2 | `ORIGIN_GIRATINA(3*)` (871) | `PECHARUNT(3*)` | **2247,9** | 505,2 | `SPECTRIER(3*)` (1441) |
| **`DRAGON`** | `GARCHOMP(3*)` | **1249,0** | 581,8 | `MEGA_RAYQUAZA(4*)` (1201) | `REGIDRAGO(3*)` | **1541,3** | 1051,0 | `DIALGA(3*)` (1459) |
| **`HUMAN`** | `GALLADE(3*)` | **1247,2** | 530,8 | `INFERNAPE(3*)` (1124) | `LILIGANT(2*)` | **1675,0** | 598,6 | `REGIROCK(3*)` (1347) |
| **`FIGHTING`** | `GALLADE(3*)` | **1247,2** | 530,8 | `KELDEO(3*)` (1309) | `IRON_VALIANT(3*)` | **2608,9** | 697,6 | `BRELOOM(2*)` (2099) |
| **`ELECTRIC`** | `THUNDURUS(3*)` | **1220,9** | 721,3 | `ZEKROM(3*)` (1038) | `ZAPDOS(3*)` | **1077,8** | 520,7 | `IRON_HANDS(3*)` (1018) |
| **`FLYING`** | `MEGA_RAYQUAZA(4*)` | **1200,7** | 595,8 | `HO_OH(3*)` (1166) | `SHAYMIN_SKY(4*)` | **1843,3** | 580,3 | `BEAUTIFLY(3*)` (1251) |
| **`DARK`** | `SHADOW_LUGIA(3*)` | **1195,4** | 472,2 | `HYDREIGON(3*)` (990) | `GUZZLORD(3*)` | **2211,6** | 684,2 | `KROOKODILE(3*)` (1265) |
| **`FIRE`** | `HO_OH(3*)` | **1166,1** | 468,5 | `RESHIRAM(3*)` (983) | `PRIMAL_GROUDON(4*)` | **1469,5** | 1049,8 | `INFERNAPE(3*)` (1489) |
| **`FAIRY`** | `TINKATON(3*)` | **1161,2** | 574,0 | `IRON_VALIANT(3*)` (1186) | `IRON_VALIANT(3*)` | **2608,9** | 823,6 | `CHANSEY(2*)` (1532) |
| **`STEEL`** | `TINKATON(3*)` | **1161,2** | 574,0 | `AEGISLASH(3*)` (977) | `STAKATAKA(3*)` | **1883,8** | 1072,7 | `DIALGA(3*)` (1459) |
| **`ROCK`** | `NIHILEGO(3*)` | **1115,5** | 465,0 | `SUDOWOODO(2*)` (1124) | `STAKATAKA(3*)` | **1883,8** | 1118,3 | `SUDOWOODO(2*)` (1768) |
| **`LIGHT`** | `ULTRA_NECROZMA(4*)` | **1106,3** | 590,3 | `HO_OH(3*)` (1141) | `LURANTIS(2*)` | **1682,4** | 911,9 | `SOLGALEO(3*)` (1263) |
| **`FOSSIL`** | `ROARING_MOON(3*)` | **1094,4** | 557,4 | `REGIROCK(3*)` (759) | `REGIDRAGO(3*)` | **1541,3** | 1239,5 | `REGICE(3*)` (1464) |
| **`GROUND`** | `ZYGARDE_50(3*)` | **1087,2** | 519,8 | `LANDORUS(3*)` (879) | `PRIMAL_GROUDON(4*)` | **1905,8** | 1250,7 | `TORTERRA(3*)` (1704) |
| **`WATER`** | `PRIMAL_KYOGRE(4*)` | **1017,1** | 422,9 | `MANAPHY(3*)` (917) | `PALKIA(3*)` | **1009,8** | 650,9 | `SAMUROTT(3*)` (933) |
| **`FLORA`** | `SHAYMIN_SKY(4*)` | **987,7** | 557,1 | `XERNEAS(3*)` (1001) | `OGERPON_TEAL(3*)` | **2180,7** | 1633,0 | `MARACTUS(3*)` (1655) |
| **`GRASS`** | `SHAYMIN_SKY(4*)` | **987,7** | 557,1 | `VIRIZION(3*)` (865) | `OGERPON_TEAL(3*)` | **2180,7** | 1633,0 | `TORTERRA(3*)` (2110) |
| **`FIELD`** | `SERPERIOR(3*)` | **892,4** | 625,7 | `HEATMOR(3*)` (881) | `SAMUROTT(3*)` | **1023,5** | 759,5 | `SUICUNE(3*)` (1025) |
| **`BUG`** | `GENESECT(3*)` | **847,3** | 508,5 | `BEAUTIFLY(3*)` (673) | `HERACROSS(3*)` | **1929,3** | 645,5 | `BEAUTIFLY(3*)` (1251) |
| **`WILD`** | `PRIMEAPE(2*)` | **749,2** | 269,5 | `SUICUNE(3*)` (667) | `EMBOAR(3*)` | **1216,4** | 559,0 | `RAIKOU(3*)` (1056) |
| **`BABY`** | `CHINGLING(2*)` | **381,7** | 150,2 | `HAPPINY(1*)` (140) | `HAPPINY(1*)` | **616,9** | 484,0 | `CHINGLING(2*)` (472) |

---

## 🧠 3. Regras Heurísticas Extraídas via Destilação VIPER (`scikit-learn`)

As políticas neurais PyTorch foram destiladas em Árvores de Decisão verificáveis (`DecisionTreeClassifier`) com fidelidade de até **86,6%**:

* **`Dark_Assassin_Crit_Specialist` (`HYPER_ROLL` — Fidelidade VIPER: `86,6%`)**:
  * `SE (bench_size <= 5 e gold >= 1 e shop_synergy_matches >= 1) ENTÃO [COMPRAR_UNIDADE_SINERGIA_OU_PAR]` (`99,8%` confiança)
  * `SE (bench_size == 6 e relative_board_power > 1.96 e gold > 51g) ENTÃO [REROLL_LOJA_1G]` (`65,4%` confiança)
* **`Water_Mana_Control_Specialist` (`STANDARD_ECON_50G` — Fidelidade VIPER: `80,4%`)**:
  * `SE (bench_size <= 5 e shop_synergy_matches >= 3 e gold >= 1) ENTÃO [COMPRAR_UNIDADE_SINERGIA_OU_PAR]` (`93,2%` confiança)
  * `SE (bench_size == 6 e free_board_slots == 0 e gold <= 53g) ENTÃO [OTIMIZAR_OU_VENDER_BANCO]` (`79,8%` confiança — vende unidades excedentes do banco para atingir os degraus de juros `10g..50g`).
* **`Tempo_WinStreak_Snowballer` (`TEMPO_WIN_STREAK` — Fidelidade VIPER: `79,9%`)**:
  * `SE (player_hp > 63 e bench_size == 6 e gold > 13g) ENTÃO [COMPRAR_XP_SUBIR_NIVEL_4G]` (sobe de nível já com `14g+` para colocar 1 unidade extra em campo e preservar o *win streak*).
* **`HyperRoll_3Star_Collector` (`HYPER_ROLL` — Fidelidade VIPER: `79,7%`)**:
  * `SE (bench_size == 6 e gold > 11g e player_hp > 60) ENTÃO [REROLL_LOJA_1G]` (`81,9%` confiança — com o banco cheio de pares aguardando evolução $3\star$, roda a loja de `1g` agressivamente).
