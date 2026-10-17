"""
Main End-to-End Execution Pipeline for the Pokemon Auto Chess (PAC) Python RL Community Lab.

Uses:
- `numpy` & `pandas`: Data ingestion of 1,180 Pokemon (`pokemons-data.csv`) & 726 bots (`botv2.json`),
  vectorized combat simulation, and tournament analytics.
- `gymnasium`: 8-player `PokemonAutoChessGymEnv` with invalid action masking and potential-based shaping.
- `torch` (PyTorch):
  1. `SetTransformerCombatSurrogate` (`nn.MultiheadAttention`) combat outcome predictor
  2. `RewardConditionedMaskedActorCritic` + `MaskablePPOTrainer` with FiLM reward conditioning
  3. Population-Based Training (PBT) + MAP-Elites Quality-Diversity archive across 8 agent personas.
- `scikit-learn` (`DecisionTreeClassifier`, `export_text`): VIPER policy distillation into
  human-readable Portuguese `SE ... ENTÃO ...` playing heuristics.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch

from pac_rl.data_loader import PACDataRepository
from pac_rl.combat_surrogate import FastCombatSimulator, train_combat_surrogate
from pac_rl.community_league import (
    CommunityLeagueTrainer,
    analyze_botv2_meta,
    analyze_composition_damage_and_tank_records,
)
from pac_rl.viper_distiller import VIPERPolicyDistiller
from pac_rl.analytics_exporter import gerar_pacote_analise_ia, gerar_pdf_resultados


def main() -> None:
    t0 = time.time()
    print("=" * 88)
    print(" POKEMON AUTO CHESS (PAC v6.11) - PYTHON RL COMMUNITY & VIPER HEURISTIC LAB")
    print(f" PyTorch {torch.__version__} | NumPy {np.__version__} | Pandas {pd.__version__}")
    print("=" * 88)

    # 1. Load PAC Data Repository
    repo = PACDataRepository()
    summary = repo.summary()
    print("\n[1/7] Dados Reais Carregados do Repositório (`pokemons-data.csv` & `botv2.json`):")
    print(f"  - Total de Pokémon carregados : {summary['total_pokemon']}")
    print(f"  - Sinergias / Tipos oficiais  : {summary['num_synergies']}")
    print(f"  - Bots comunitários (`botv2`) : {summary['total_bots_loaded']}")
    print(f"  - Distribuicao por Raridade   : {summary['by_rarity']}")

    sim = FastCombatSimulator(repo)

    # 2. Analyze existing 726 Community Bots Meta + Composition Damage & Tank Records
    print("\n[2/7] Analisando Meta Empirico dos 726 Bots (`botv2.json`) no Estagio 25...")
    bot_meta_df = analyze_botv2_meta(repo, sim, num_duels=700)
    print("\nTop 12 Sinergias no Banco de Bots (`botv2.json` - Estagio 25):")
    print(bot_meta_df.head(12).to_string(index=False))

    print("\n[3/7] Calculando Recordes de Dano (Best Carry) e Recordes de Tank por Composicao (31 Sinergias)...")
    comp_records_df, top_dps_global_df, top_tank_global_df = analyze_composition_damage_and_tank_records(
        repo, sim, num_duels=1500
    )
    print("\nRecordes de Dano e Tank por Sinergia (Top 15):")
    print(comp_records_df.head(15).to_string(index=False))

    # 3. Train PyTorch Set-Transformer Combat Surrogate
    print("\n[4/7] Treinando `SetTransformerCombatSurrogate` (PyTorch `nn.MultiheadAttention`)...")
    surrogate_model, surr_metrics = train_combat_surrogate(
        repo=repo,
        sim=sim,
        num_samples=2000,
        epochs=12,
        batch_size=128,
        lr=3e-3,
        seed=42,
    )
    print(
        f"  -> Surrogate Val Accuracy: {surr_metrics['val_accuracy']*100:.1f}% | "
        f"Surviving Units MAE: {surr_metrics['val_margin_mae']:.2f} unidades "
        f"(n={surr_metrics['samples']} combates)"
    )

    # 4. Multi-Generation Maskable PPO + PBT Community League Training
    print("\n[5/7] Treinando Comunidade de 8 Agentes RL (Maskable PPO + PBT + MAP-Elites)...")
    league = CommunityLeagueTrainer(repo=repo, sim=sim, seed=42)
    num_generations = 4
    gen_records = []

    for gen in range(1, num_generations + 1):
        metrics_list = league.train_and_evaluate_generation(
            generation=gen,
            train_episodes_per_agent=8,
            eval_episodes_per_agent=8,
        )
        avg_lobby_place = np.mean([m.avg_placement for m in metrics_list])
        best_m = min(metrics_list, key=lambda m: m.avg_placement)
        print(
            f"  [Geracao {gen}/{num_generations}] Media Placement da Comunidade vs Bots: {avg_lobby_place:.2f} | "
            f"Melhor Agente: {best_m.agent_name} (Place={best_m.avg_placement:.2f}, "
            f"Top4={best_m.top4_rate*100:.0f}%, Win={best_m.win_rate*100:.0f}%)"
        )
        for m in metrics_list:
            gen_records.append({
                "Generation": m.generation,
                "Agent": m.agent_name,
                "Archetype": m.archetype,
                "Synergy": m.target_synergy,
                "Avg_Placement": round(m.avg_placement, 2),
                "Top4_%": round(m.top4_rate * 100.0, 1),
                "Win_%": round(m.win_rate * 100.0, 1),
                "Avg_Stage": round(m.avg_stage_reached, 1),
                "Avg_2Star": round(m.avg_2star_evos, 2),
                "Avg_3Star": round(m.avg_3star_evos, 2),
            })

    print("\nArquivo MAP-Elites (Melhores Especialistas Descobertos por Nicho [Sinergia x Economia]):")
    archive_df = league.archive.to_dataframe()
    print(archive_df.to_string(index=False))

    # 5. Direct 8-Agent Head-to-Head Lobby Tournament
    print("\n[6/7] Torneio Direto Head-to-Head (Todos os 8 Agentes RL no Mesmo Lobby - 30 Partidas)...")
    h2h_df = league.run_direct_community_lobby_tournament(num_lobbies=30)
    print(h2h_df.to_string(index=False))

    # 6. VIPER Policy Distillation (`scikit-learn` DecisionTreeClassifier)
    print("\n[7/7] Destilando Politicas Neurais PyTorch em Arvores de Decisao (VIPER / `scikit-learn`)...")
    distiller = VIPERPolicyDistiller(max_depth=3, min_samples_leaf=8)
    distilled_reports = []

    for ag in league.agents:
        rep = distiller.distill_agent(
            env=league.env,
            model=ag.model,
            config=ag.config,
            num_episodes=14,
            seed=300,
        )
        distilled_reports.append(rep)
        print("-" * 88)
        print(
            f"Agente: {rep.agent_name} | Arquetipo: {rep.archetype} | "
            f"Fidelidade VIPER (Arvore vs Rede Neural): {rep.fidelity_accuracy*100:.1f}%"
        )
        top_f_str = ", ".join(f"{k}={v*100:.1f}%" for k, v in rep.top_features if v > 0.01)
        print(f"  Variaveis de Estado Mais Importantes: {top_f_str}")
        print("  Regras Extraidas (SE ... ENTAO ...):")
        for rule in rep.extracted_rules_pt[:4]:
            print(f"    * {rule}")

    # Save structured JSON artifact and generate PDF + AI analysis package in resultados/
    root_dir = Path(__file__).resolve().parent
    out_path = root_dir / "pac_rl" / "results_report.json"
    resultados_dir = root_dir / "resultados"
    resultados_dir.mkdir(parents=True, exist_ok=True)

    export_payload = {
        "data_summary": summary,
        "combat_surrogate_metrics": surr_metrics,
        "botv2_meta_top15": bot_meta_df.head(20).to_dict(orient="records"),
        "composition_damage_tank_records": comp_records_df.to_dict(orient="records"),
        "global_top_dps_pokemon": top_dps_global_df.to_dict(orient="records"),
        "global_top_tank_pokemon": top_tank_global_df.to_dict(orient="records"),
        "map_elites_archive": archive_df.to_dict(orient="records"),
        "head_to_head_tournament": h2h_df.to_dict(orient="records"),
        "generation_progression": gen_records,
        "viper_distilled_heuristics": [
            {
                "agent_name": r.agent_name,
                "archetype": r.archetype,
                "fidelity_accuracy": round(r.fidelity_accuracy, 4),
                "top_features": r.top_features,
                "extracted_rules_pt": r.extracted_rules_pt,
                "action_distribution": r.action_distribution,
                "tree_text": r.tree_text,
            }
            for r in distilled_reports
        ],
    }
    out_path.write_text(json.dumps(export_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (resultados_dir / "results_report_cpu.json").write_text(
        json.dumps(export_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    pdf_file = gerar_pdf_resultados(
        out_path,
        resultados_dir / "relatorio_experimento_cpu_pac_rl.pdf",
    )
    ai_json, ai_prompt = gerar_pacote_analise_ia(out_path, resultados_dir)

    elapsed = time.time() - t0
    print("\n" + "=" * 88)
    print(f" Pipeline concluido em {elapsed:.1f}s!")
    print(f"  - JSON Completo : {out_path}")
    print(f"  - PDF Gerado    : {pdf_file}")
    print(f"  - Pacote IA JSON: {ai_json}")
    print(f"  - Prompt para IA: {ai_prompt}")
    print("=" * 88)


if __name__ == "__main__":
    main()
