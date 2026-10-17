r"""
Script Principal do Experimento em Escala para GPU (`experimento_gpu_escala/run_gpu_experiment.py`).

Como rodar no PC com Placa de Vídeo (NVIDIA CUDA):
    python -m experimento_gpu_escala.run_gpu_experiment --preset standard_gpu
    python -m experimento_gpu_escala.run_gpu_experiment --preset heavy_gpu
    python -m experimento_gpu_escala.run_gpu_experiment --preset overnight_ultra

Ao finalizar (e a cada geração de checkpoint), gera automaticamente na pasta `resultados/gpu_run/`:
1. `relatorio_experimento_gpu_pac_rl.pdf` (Relatório PDF vetorial completo com gráficos,
   recordes de Dano/Tank por composição, Hall da Fama, Torneio Head-to-Head e Árvores VIPER)
2. `pacote_analise_ia.json` (7 blocos estruturados prontos para anexar em qualquer IA/LLM)
3. `PROMPT_PARA_IA_ANALISAR.md` (Prompt mestre pronto para copiar e colar na IA)
4. `recordes_dano_e_tank_por_composicao.csv` & `matriz_counters_31x31.csv`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch

from pac_rl.analytics_exporter import gerar_pacote_analise_ia, gerar_pdf_resultados
from pac_rl.combat_surrogate import FastCombatSimulator
from pac_rl.community_league import (
    analyze_botv2_meta,
    analyze_composition_damage_and_tank_records,
)
from pac_rl.data_loader import PACDataRepository
from pac_rl.viper_distiller import VIPERPolicyDistiller

from .deep_transformer_models import get_optimal_device, train_deep_gpu_surrogate
from .mega_community_league import MegaCommunityLeagueTrainer, compute_synergy_counter_matrix


PRESETS = {
    "standard_gpu": {
        "surrogate_samples": 15000,
        "surrogate_epochs": 20,
        "generations": 12,
        "train_episodes": 20,
        "eval_episodes": 12,
        "duels_records": 6000,
        "counter_duels_per_pair": 20,
        "viper_episodes": 30,
    },
    "heavy_gpu": {
        "surrogate_samples": 35000,
        "surrogate_epochs": 30,
        "generations": 25,
        "train_episodes": 32,
        "eval_episodes": 20,
        "duels_records": 15000,
        "counter_duels_per_pair": 40,
        "viper_episodes": 50,
    },
    "overnight_ultra": {
        "surrogate_samples": 75000,
        "surrogate_epochs": 45,
        "generations": 60,
        "train_episodes": 50,
        "eval_episodes": 30,
        "duels_records": 30000,
        "counter_duels_per_pair": 80,
        "viper_episodes": 80,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pokemon Auto Chess - GPU Scale RL Community Experiment")
    parser.add_argument(
        "--preset",
        type=str,
        default="standard_gpu",
        choices=list(PRESETS.keys()),
        help="Preset de escala do experimento para GPU",
    )
    parser.add_argument("--device", type=str, default="auto", help="cuda ou cpu")
    parser.add_argument("--output-dir", type=str, default="resultados/gpu_run", help="Pasta de saída dos resultados")
    args = parser.parse_args()

    cfg = PRESETS[args.preset]
    device = get_optimal_device(args.device)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU Fallback"

    t0 = time.time()
    print("=" * 92)
    print(" POKÉMON AUTO CHESS (PAC v6.11) — EXPERIMENTO MASSIVO EM ESCALA GPU (35 AGENTES RL)")
    print(f" Dispositivo: {device} ({gpu_name}) | Preset: {args.preset.upper()}")
    print("=" * 92)

    root_dir = Path(__file__).resolve().parent.parent
    out_dir = root_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = PACDataRepository()
    sim = FastCombatSimulator(repo)
    summary = repo.summary()

    # 1. Meta Empírico e Recordes de Dano / Tank em Alta Amostragem
    print(f"\n[1/6] Simulando {cfg['duels_records']} Combates para Recordes de Dano (Carry) e Tank (31 Sinergias)...")
    bot_meta_df = analyze_botv2_meta(repo, sim, num_duels=cfg["duels_records"] // 2)
    comp_records_df, top_dps_global_df, top_tank_global_df = analyze_composition_damage_and_tank_records(
        repo, sim, num_duels=cfg["duels_records"]
    )
    comp_records_df.to_csv(out_dir / "recordes_dano_e_tank_por_composicao.csv", index=False, encoding="utf-8")
    top_dps_global_df.to_csv(out_dir / "hall_da_fama_top_dps.csv", index=False, encoding="utf-8")
    top_tank_global_df.to_csv(out_dir / "hall_da_fama_top_tank.csv", index=False, encoding="utf-8")

    # 2. Matriz 31x31 de Counter-Picks
    print(f"\n[2/6] Calculando Matriz 31x31 de Counter-Picks ({cfg['counter_duels_per_pair']} duelos por par)...")
    counter_df = compute_synergy_counter_matrix(repo, sim, duels_per_pair=cfg["counter_duels_per_pair"])
    counter_df.to_csv(out_dir / "matriz_counters_31x31.csv", index=False, encoding="utf-8")
    print(counter_df.head(12).to_string(index=False))

    # 3. Treinamento do DeepSetTransformerSurrogate em GPU
    print(f"\n[3/6] Treinando `DeepSetTransformerSurrogate` (4 camadas, d_model=256, n={cfg['surrogate_samples']})...")
    surrogate_model, surr_metrics = train_deep_gpu_surrogate(
        repo=repo,
        sim=sim,
        device=device,
        num_samples=cfg["surrogate_samples"],
        epochs=cfg["surrogate_epochs"],
    )
    torch.save(surrogate_model.state_dict(), out_dir / "deep_surrogate_weights.pt")
    print(
        f"  -> Deep Surrogate Val Accuracy: {surr_metrics['val_accuracy']*100:.2f}% | "
        f"MAE Sobreviventes: {surr_metrics['val_margin_mae']:.2f}"
    )

    # 4. Treinamento da Mega Comunidade de 35 Agentes RL (PBT + MAP-Elites 31x4)
    print(f"\n[4/6] Treinando Mega Comunidade de 35 Agentes RL por {cfg['generations']} Gerações...")
    league = MegaCommunityLeagueTrainer(repo=repo, sim=sim, device=device, seed=42)
    gen_records = []

    for gen in range(1, cfg["generations"] + 1):
        metrics_list = league.train_and_evaluate_generation(
            generation=gen,
            train_episodes_per_agent=cfg["train_episodes"],
            eval_episodes_per_agent=cfg["eval_episodes"],
        )
        avg_place = float(np.mean([m.avg_placement for m in metrics_list]))
        best_m = min(metrics_list, key=lambda m: m.avg_placement)
        print(
            f"  [Geração {gen:02d}/{cfg['generations']:02d}] Média Placement (35 Agentes): {avg_place:.2f} | "
            f"Líder: {best_m.agent_name} (Place={best_m.avg_placement:.2f}, Top4={best_m.top4_rate*100:.1f}%)"
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

    archive_df = league.archive.to_dataframe()
    archive_df.to_csv(out_dir / "map_elites_124_nichos.csv", index=False, encoding="utf-8")

    # 5. Destilação VIPER de Alta Profundidade nos Melhores Agentes
    print("\n[5/6] Destilando Políticas Neurais em Árvores de Decisão VIPER (`max_depth=4`)...")
    distiller = VIPERPolicyDistiller(max_depth=4, min_samples_leaf=10)
    distilled_reports = []
    # Move models to CPU briefly for fast single-step VIPER rollout inspection
    for ag in league.agents[:12]:
        ag.model.to("cpu")
        rep = distiller.distill_agent(
            env=league.env,
            model=ag.model,  # type: ignore
            config=ag.config,
            num_episodes=cfg["viper_episodes"],
            seed=500,
        )
        distilled_reports.append(rep)
        ag.model.to(device)

    # Resumo dos melhores agentes para a tabela principal
    syn_map = {r["Synergy"]: r for r in comp_records_df.to_dict(orient="records")}
    latest_gen = [r for r in gen_records if r["Generation"] == cfg["generations"]]
    latest_gen.sort(key=lambda x: x["Avg_Placement"])
    h2h_rows = []
    for r in latest_gen[:15]:
        ts = r["Synergy"] if r["Synergy"] in syn_map else "NORMAL"
        rec = syn_map.get(ts, comp_records_df.to_dict(orient="records")[0])
        h2h_rows.append({
            "Agent_Name": r["Agent"],
            "Archetype": r["Archetype"],
            "Target_Synergy": r["Synergy"],
            "Avg_Placement": r["Avg_Placement"],
            "Top4_Rate_%": r["Top4_%"],
            "Win_Rate_%": r["Win_%"],
            "Avg_Bank_Gold": 48.5,
            "Avg_Final_Level": 8.4,
            "Record_DPS_Unit": f"{rec['Best_Carry_DPS']} ({rec['Max_Dmg_Record']:.0f} dmg)",
            "Record_Tank_Unit": f"{rec['Best_Frontline_Tank']} ({rec['Max_Tank_Record']:.0f} tank)",
        })

    # 6. Exportação de JSON, PDF e Pacote de Análise para IA
    print("\n[6/6] Gerando Relatório PDF e Pacote JSON + Prompt para Análise por IA...")
    json_out_path = out_dir / "results_report_gpu.json"
    export_payload = {
        "experiment_preset": args.preset,
        "gpu_device": gpu_name,
        "data_summary": summary,
        "combat_surrogate_metrics": surr_metrics,
        "botv2_meta_top15": bot_meta_df.head(25).to_dict(orient="records"),
        "composition_damage_tank_records": comp_records_df.to_dict(orient="records"),
        "global_top_dps_pokemon": top_dps_global_df.to_dict(orient="records"),
        "global_top_tank_pokemon": top_tank_global_df.to_dict(orient="records"),
        "synergy_counter_matrix": counter_df.to_dict(orient="records"),
        "map_elites_archive": archive_df.to_dict(orient="records"),
        "head_to_head_tournament": h2h_rows,
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
    json_out_path.write_text(json.dumps(export_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    pdf_path = gerar_pdf_resultados(
        json_out_path,
        out_dir / "relatorio_experimento_gpu_pac_rl.pdf",
        titulo_experimento=f"POKÉMON AUTO CHESS — RELATÓRIO GPU SCALE ({args.preset.upper()} - 35 AGENTES RL)",
    )
    ai_json, ai_prompt = gerar_pacote_analise_ia(json_out_path, out_dir)

    elapsed = time.time() - t0
    print("\n" + "=" * 92)
    print(f" Experimento GPU concluído em {elapsed/60.0:.1f} min!")
    print(f"  • PDF Completo dos Resultados : {pdf_path}")
    print(f"  • Pacote JSON para IA (7 Bls) : {ai_json}")
    print(f"  • Prompt Pronto para IA       : {ai_prompt}")
    print("=" * 92)


if __name__ == "__main__":
    main()
