r"""
Gerador de Relatório PDF Multi-Página (`gerar_pdf_resultados`) e Pacote Estruturado para
Análise por IA (`gerar_pacote_analise_ia`) para o Pokémon Auto Chess RL Lab.

Funciona tanto para o experimento local em CPU (`pac_rl/results_report.json`) quanto para
o experimento massivo em GPU (`experimento_gpu_escala/`) que rodará no computador com placa
de vídeo dedicada.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def _draw_styled_table(
    ax: plt.Axes,
    df: pd.DataFrame,
    col_widths: Optional[List[float]] = None,
    font_size: int = 8,
    header_color: str = "#1e293b",
    row_colors: Tuple[str, str] = ("#f8fafc", "#e2e8f0"),
) -> None:
    ax.axis("off")
    if df.empty:
        ax.text(0.5, 0.5, "Sem dados disponíveis", ha="center", va="center", fontsize=11)
        return

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[(row - 1) % 2])


def gerar_pdf_resultados(
    results_json_path: str | Path,
    output_pdf_path: str | Path,
    titulo_experimento: str = "POKÉMON AUTO CHESS (PAC v6.11) — RELATÓRIO DA COMUNIDADE RL & HEURÍSTICAS VIPER",
) -> Path:
    """
    Gera um relatório executivo e técnico completo em PDF vetorial multi-página a partir
    do arquivo `results_report.json` gerado pelo pipeline de treinamento.

    Inclui:
      1. Capa Executiva, Acurácia do Surrogate Set-Transformer e Evolução Geracional
      2. Meta Empírico das Sinergias (Win Rate % e ELO Médio)
      3. Recordes de Dano (Best Carry) e Recordes de Tank (Best Frontline) por Composição
      4. Hall da Fama Global: Top Pokémon Recordistas de Dano e de Tank
      5. Torneio Direto Head-to-Head entre os Agentes da Comunidade RL
      6. Matriz MAP-Elites (Nicho de Sinergia x Arquétipo Econômico)
      7. Destilação de Políticas Neurais em Árvores de Decisão VIPER (Regras SE ... ENTÃO ...)
    """
    json_path = Path(results_json_path)
    pdf_path = Path(output_pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))

    summary = data.get("data_summary", {})
    surr = data.get("combat_surrogate_metrics", {})
    bot_meta = pd.DataFrame(data.get("botv2_meta_top15", []))
    comp_records = pd.DataFrame(data.get("composition_damage_tank_records", []))
    top_dps_global = pd.DataFrame(data.get("global_top_dps_pokemon", []))
    top_tank_global = pd.DataFrame(data.get("global_top_tank_pokemon", []))
    h2h_df = pd.DataFrame(data.get("head_to_head_tournament", []))
    archive_df = pd.DataFrame(data.get("map_elites_archive", []))
    gen_df = pd.DataFrame(data.get("generation_progression", []))
    viper_list = data.get("viper_distilled_heuristics", [])

    with PdfPages(pdf_path) as pdf:
        # =====================================================================
        # PÁGINA 1: CAPA EXECUTIVA + SURROGATE + EVOLUÇÃO GERACIONAL
        # =====================================================================
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 Paisagem
        fig.patch.set_facecolor("#ffffff")
        fig.suptitle(titulo_experimento, fontsize=14, fontweight="bold", color="#0f172a", y=0.96)

        ax_info = fig.add_axes([0.05, 0.62, 0.90, 0.27])
        ax_info.axis("off")
        val_acc = surr.get("val_accuracy", 0.0) * 100.0
        val_mae = surr.get("val_margin_mae", 0.0)
        n_samp = surr.get("samples", 0)

        resumo_texto = (
            f"1. DADOS OFICIAIS DO REPOSITÓRIO (PAC v6.11):\n"
            f"   • Pokémon Jogáveis Carregados: {summary.get('total_pokemon', 1077)} | "
            f"Sinergias Oficiais: {summary.get('num_synergies', 31)} | "
            f"Bots Comunitários (`botv2.json`): {summary.get('total_bots_loaded', 726)}\n"
            f"   • Regras Econômicas Exatas: Reroll = 1g | Level Up = 4g (+4 XP) | Juros = min(5, floor(gold/10)) | "
            f"Streak = min(5, streak) | +1g por vitória PvP.\n\n"
            f"2. MODELO SURROGATE DE COMBATE (`SetTransformerCombatSurrogate` em PyTorch `nn.MultiheadAttention`):\n"
            f"   • Acurácia de Validação na Previsão de Vitória: {val_acc:.1f}% | "
            f"Erro Médio Absoluto (Unidades Sobreviventes): {val_mae:.2f} unidades (amostras={n_samp})\n\n"
            f"3. ARQUITETURA DA COMUNIDADE DE AGENTES RL:\n"
            f"   • Políticas `RewardConditionedMaskedActorCritic` (FiLM Conditioning + Invalid Action Masking) treinadas com Maskable PPO + GAE.\n"
            f"   • Seleção Evolucionária por Population-Based Training (PBT) + Arquivo Qualidade-Diversidade MAP-Elites + Destilação VIPER."
        )
        ax_info.text(
            0.01,
            0.95,
            resumo_texto,
            va="top",
            ha="left",
            fontsize=10,
            family="sans-serif",
            color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f1f5f9", edgecolor="#94a3b8"),
        )

        # Gráfico inferior: Evolução Geracional ou Desempenho dos Agentes
        ax_gen = fig.add_axes([0.07, 0.09, 0.40, 0.45])
        if not gen_df.empty and "Generation" in gen_df.columns:
            grp = gen_df.groupby("Generation")["Avg_Placement"].agg(["mean", "min"]).reset_index()
            ax_gen.plot(grp["Generation"], grp["mean"], marker="o", linewidth=2.2, color="#2563eb", label="Média da Comunidade")
            ax_gen.plot(grp["Generation"], grp["min"], marker="s", linewidth=2.2, color="#16a34a", label="Melhor Agente (Elite)")
            ax_gen.invert_yaxis()
            ax_gen.set_title("Evolução do Placement por Geração (Menor = Melhor)", fontsize=10, fontweight="bold")
            ax_gen.set_xlabel("Geração PBT / PPO")
            ax_gen.set_ylabel("Placement Médio (1..8)")
            ax_gen.grid(True, linestyle="--", alpha=0.4)
            ax_gen.legend(fontsize=8)

        ax_h2h_bar = fig.add_axes([0.54, 0.09, 0.41, 0.45])
        if not h2h_df.empty:
            agents_short = [str(n)[:22] for n in h2h_df["Agent_Name"]]
            y_pos = np.arange(len(agents_short))
            ax_h2h_bar.barh(y_pos, h2h_df["Top4_Rate_%"], color="#0284c7", alpha=0.85, label="Top-4 (%)")
            ax_h2h_bar.barh(y_pos, h2h_df["Win_Rate_%"], color="#f59e0b", alpha=0.95, label="Top-1 Vitória (%)")
            ax_h2h_bar.set_yticks(y_pos)
            ax_h2h_bar.set_yticklabels(agents_short, fontsize=8)
            ax_h2h_bar.invert_yaxis()
            ax_h2h_bar.set_xlabel("Taxa (%) em 30 Lobbies Head-to-Head")
            ax_h2h_bar.set_title("Torneio Direto entre Arquétipos RL (Top-4 e Top-1)", fontsize=10, fontweight="bold")
            ax_h2h_bar.grid(True, axis="x", linestyle="--", alpha=0.4)
            ax_h2h_bar.legend(fontsize=8, loc="lower right")

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # PÁGINA 2: RECORDES DE DANO (CARRY) E DE TANK POR COMPOSIÇÃO (31 TIPOS)
        # =====================================================================
        if not comp_records.empty:
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.suptitle(
                "RECORDES DE DANO (MELHOR CARRY) E DE TANK (MELHOR FRONTLINE) POR COMPOSIÇÃO / SINERGIA",
                fontsize=12.5,
                fontweight="bold",
                color="#0f172a",
                y=0.97,
            )
            ax_tbl = fig.add_axes([0.02, 0.03, 0.96, 0.90])
            display_cols = [
                "Synergy",
                "Best_Carry_DPS",
                "Max_Dmg_Record",
                "Avg_Dmg_Carry",
                "Secondary_Carry",
                "Best_Frontline_Tank",
                "Max_Tank_Record",
                "Avg_Tank_Absorbed",
                "Secondary_Tank",
            ]
            sub_df = comp_records[[c for c in display_cols if c in comp_records.columns]].head(24).copy()
            sub_df.columns = [
                "Sinergia",
                "Recordista Dano (Carry)",
                "Recorde Dano",
                "Dano Médio",
                "2º Melhor Carry",
                "Recordista Tank (Front)",
                "Recorde Tank",
                "Tank Médio",
                "2º Melhor Tank",
            ][: len(sub_df.columns)]
            _draw_styled_table(ax_tbl, sub_df, font_size=7)
            pdf.savefig(fig)
            plt.close(fig)

            # Se houver mais sinergias além das 24 primeiras, adiciona página complementar
            if len(comp_records) > 24:
                fig = plt.figure(figsize=(11.69, 8.27))
                fig.suptitle(
                    "RECORDES DE DANO E DE TANK POR COMPOSIÇÃO (CONTINUAÇÃO — TODAS AS 31 SINERGIAS)",
                    fontsize=12.5,
                    fontweight="bold",
                    color="#0f172a",
                    y=0.96,
                )
                ax_tbl2 = fig.add_axes([0.02, 0.45, 0.96, 0.47])
                sub_df2 = comp_records[[c for c in display_cols if c in comp_records.columns]].iloc[24:].copy()
                sub_df2.columns = sub_df.columns
                _draw_styled_table(ax_tbl2, sub_df2, font_size=7.5)

                # Gráfico comparativo dos Top 12 Carries vs Top 12 Tanks por Composição
                ax_c1 = fig.add_axes([0.06, 0.06, 0.41, 0.35])
                top10_c = comp_records.head(10)
                labels_c = [f"{r['Synergy']}: {r['Best_Carry_DPS']}" for _, r in top10_c.iterrows()]
                ax_c1.barh(np.arange(len(labels_c)), top10_c["Max_Dmg_Record"], color="#dc2626", alpha=0.85)
                ax_c1.set_yticks(np.arange(len(labels_c)))
                ax_c1.set_yticklabels(labels_c, fontsize=7.5)
                ax_c1.invert_yaxis()
                ax_c1.set_title("Top 10 Recordes de Dano por Sinergia", fontsize=9.5, fontweight="bold")
                ax_c1.set_xlabel("Pico de Dano em 1 Combate")

                ax_c2 = fig.add_axes([0.55, 0.06, 0.41, 0.35])
                top10_t = comp_records.sort_values("Max_Tank_Record", ascending=False).head(10)
                labels_t = [f"{r['Synergy']}: {r['Best_Frontline_Tank']}" for _, r in top10_t.iterrows()]
                ax_c2.barh(np.arange(len(labels_t)), top10_t["Max_Tank_Record"], color="#0d9488", alpha=0.85)
                ax_c2.set_yticks(np.arange(len(labels_t)))
                ax_c2.set_yticklabels(labels_t, fontsize=7.5)
                ax_c2.invert_yaxis()
                ax_c2.set_title("Top 10 Recordes de Tank (Dano Absorvido/Mitigado)", fontsize=9.5, fontweight="bold")
                ax_c2.set_xlabel("Pico de Dano Absorvido/Mitigado em 1 Combate")

                pdf.savefig(fig)
                plt.close(fig)

        # =====================================================================
        # PÁGINA 3: HALL DA FAMA GLOBAL (TOP 15 DPS & TOP 15 TANKS DO JOGO)
        # =====================================================================
        if not top_dps_global.empty or not top_tank_global.empty:
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.suptitle(
                "HALL DA FAMA GLOBAL — TOP 15 POKÉMON RECORDISTAS DE DANO (DPS) E DE TANK NO PAC",
                fontsize=12.5,
                fontweight="bold",
                color="#0f172a",
                y=0.96,
            )
            ax_dps = fig.add_axes([0.02, 0.50, 0.96, 0.42])
            ax_dps.set_title("Top 15 Pokémon com Maior Recorde Absoluto de Dano Causado", fontsize=10, fontweight="bold")
            _draw_styled_table(ax_dps, top_dps_global.head(12), font_size=7.5, header_color="#991b1b")

            ax_tnk = fig.add_axes([0.02, 0.04, 0.96, 0.42])
            ax_tnk.set_title("Top 15 Pokémon com Maior Recorde Absoluto de Dano Tankado / Mitigado", fontsize=10, fontweight="bold")
            _draw_styled_table(ax_tnk, top_tank_global.head(12), font_size=7.5, header_color="#0f766e")

            pdf.savefig(fig)
            plt.close(fig)

        # =====================================================================
        # PÁGINA 4: TORNEIO HEAD-TO-HEAD + ARQUIVO MAP-ELITES + META BOTV2
        # =====================================================================
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle(
            "TORNEIO DIRETO ENTRE OS AGENTES RL (COM RECORDISTAS) & MATRIZ MAP-ELITES",
            fontsize=12.5,
            fontweight="bold",
            color="#0f172a",
            y=0.96,
        )
        ax_t1 = fig.add_axes([0.02, 0.52, 0.96, 0.39])
        _draw_styled_table(ax_t1, h2h_df, font_size=7.5, header_color="#1e3a8a")

        ax_t2 = fig.add_axes([0.02, 0.04, 0.96, 0.44])
        _draw_styled_table(ax_t2, archive_df.head(14), font_size=7.0, header_color="#312e81")

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # PÁGINA 5: HEURÍSTICAS EXTRAÍDAS VIA DESTILAÇÃO VIPER (ÁRVORES DE DECISÃO)
        # =====================================================================
        if viper_list:
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.suptitle(
                "DESTILAÇÃO DE POLÍTICAS NEURAIS EM REGRAS HUMANAS (VIPER / DECISION TREES)",
                fontsize=12.5,
                fontweight="bold",
                color="#0f172a",
                y=0.96,
            )
            ax_v = fig.add_axes([0.04, 0.04, 0.92, 0.88])
            ax_v.axis("off")

            lines_text = []
            for rep in viper_list:
                ag_name = rep.get("agent_name", "")
                arch = rep.get("archetype", "")
                fid = rep.get("fidelity_accuracy", 0.0) * 100.0
                top_f = ", ".join(f"{k} ({v*100:.1f}%)" for k, v in rep.get("top_features", []) if v > 0.01)
                lines_text.append(f"■ {ag_name} [{arch}] — Fidelidade da Árvore vs Rede Neural PyTorch: {fid:.1f}%")
                lines_text.append(f"   Variáveis Decisivas: {top_f}")
                for r in rep.get("extracted_rules_pt", [])[:3]:
                    lines_text.append(f"   • {r}")
                lines_text.append("")

            ax_v.text(
                0.01,
                0.99,
                "\n".join(lines_text[:46]),
                va="top",
                ha="left",
                fontsize=8.2,
                family="monospace",
                color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1"),
            )
            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path


def gerar_pacote_analise_ia(
    results_json_path: str | Path,
    output_dir: str | Path,
) -> Tuple[Path, Path]:
    """
    Transforma os resultados brutos das simulações e do treinamento RL em:
    1. `pacote_analise_ia.json`: Dividido em 7 blocos semânticos compactos e pré-digeridos
       com estatísticas de conversão, sinergias dominantes, recordistas de dano/tank por
       composição e regras VIPER.
    2. `PROMPT_PARA_IA_ANALISAR.md`: Prompt estruturado pronto para colar em qualquer LLM
       (ChatGPT, Claude, Gemini) junto com o JSON após rodar os testes no PC com GPU.
    """
    json_path = Path(results_json_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = json.loads(json_path.read_text(encoding="utf-8"))

    # Construir blocos estruturados para análise de IA
    h2h = raw.get("head_to_head_tournament", [])
    comp_rec = raw.get("composition_damage_tank_records", [])
    viper_rules = raw.get("viper_distilled_heuristics", [])

    # Pré-calcular diagnósticos automáticos para enriquecer o JSON
    melhor_consistencia = h2h[0] if h2h else {}
    melhor_winrate = max(h2h, key=lambda x: x.get("Win_Rate_%", 0.0)) if h2h else {}

    pacote_ia = {
        "instrucoes_do_pacote": (
            "Este arquivo contém os resultados estruturados em 7 blocos do laboratório de "
            "Aprendizado por Reforço Comunitário (Maskable PPO + PBT + MAP-Elites + VIPER) "
            "aplicado ao jogo Pokémon Auto Chess (PAC v6.11)."
        ),
        "bloco_1_resumo_executivo_e_regras_do_jogo": {
            "regras_oficiais_pac": {
                "reroll_custo_ouro": 1,
                "xp_custo_ouro": 4,
                "xp_ganho_por_compra": 4,
                "xp_passivo_por_rodada": 2,
                "renda_base_por_rodada": 5,
                "juros_maximos": "min(5, floor(gold / 10)) em 10g..50g",
                "bonus_streak_pvp": "min(5, streak) + 1g imediato por vitória PvP",
                "evolucao": "3 cópias 1* -> 2*, 3 cópias 2* (9 total) -> 3*",
                "estagios_additional_picks": [5, 8, 11],
                "estagios_unique_e_legendary": [10, 20],
            },
            "dataset_carregado": raw.get("data_summary", {}),
            "acuracia_surrogate_neural_combate": raw.get("combat_surrogate_metrics", {}),
            "destaques_rapidos": {
                "agente_mais_consistente_top4": melhor_consistencia.get("Agent_Name"),
                "placement_medio_lider": melhor_consistencia.get("Avg_Placement"),
                "agente_maior_taxa_top1_vitoria": melhor_winrate.get("Agent_Name"),
                "win_rate_top1_lider_%": melhor_winrate.get("Win_Rate_%"),
            },
        },
        "bloco_2_ranking_meta_sinergias_botv2": raw.get("botv2_meta_top15", []),
        "bloco_3_recordes_dano_e_tank_por_composicao_31_tipos": comp_rec,
        "bloco_4_hall_da_fama_global_pokemon": {
            "top_20_recordistas_dano_dps": raw.get("global_top_dps_pokemon", []),
            "top_20_recordistas_tank_frontline": raw.get("global_top_tank_pokemon", []),
        },
        "bloco_5_torneio_head_to_head_comunidade_rl": h2h,
        "bloco_6_matriz_map_elites_nichos_sinergia_x_economia": raw.get("map_elites_archive", []),
        "bloco_7_heuristicas_destiladas_viper_arvores_decisao": [
            {
                "agente": r.get("agent_name"),
                "arquetipo_economico": r.get("archetype"),
                "fidelidade_arvore_vs_rede_neural": r.get("fidelity_accuracy"),
                "variaveis_mais_importantes": r.get("top_features"),
                "regras_se_entao_portugues": r.get("extracted_rules_pt"),
                "distribuicao_acoes_macro": r.get("action_distribution"),
            }
            for r in viper_rules
        ],
    }

    ai_json_path = out_dir / "pacote_analise_ia.json"
    ai_json_path.write_text(json.dumps(pacote_ia, indent=2, ensure_ascii=False), encoding="utf-8")

    prompt_md = f"""# Prompt Estruturado para Análise Profunda por IA (Pokémon Auto Chess RL Lab)

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
"""

    prompt_path = out_dir / "PROMPT_PARA_IA_ANALISAR.md"
    prompt_path.write_text(prompt_md, encoding="utf-8")

    return ai_json_path, prompt_path
