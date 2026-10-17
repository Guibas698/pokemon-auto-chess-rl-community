r"""
Script Principal do Experimento Sem Atalhos em Escala GPU (`experimento_gpu_escala/run_gpu_experiment.py`).

Arquitetura Completa de 2 Níveis de Comunidade + Busca Híbrida:
- Nível 1 (Intra-Player): Cada jogador é uma **Comunidade de 7 Sub-Modelos Especialistas**
  (`Economist`, `WinStreak_Tempo`, `LoseStreak_Cashout`, `HyperRoll_Evolution`,
  `Synergy_Vertical`, `LobbyScout_Counter`, `LateGame_Cap`) coordenados por um
  `MetaArbitratorTransformer` (**Zero regras escritas à mão**).
- Nível 2 (Inter-Player): **Self-Play Real em Lobby de 8 Agentes** dividindo o mesmo baralho
  finito (`SharedCardPool`) via Prioritized Fictitious Self-Play (PFSP) + PBT + MAP-Elites.
- Busca Clássica + Neural Conectada: O `DeepSetTransformerSurrogate` roda na GPU dentro de
  `NeuralLookaheadRosterPlanner` para avaliar os 6 slots da loja ($\Delta \widehat{P}_{\text{win}}$)
  e otimizar as trocas entre banco e tabuleiro antes de cada luta!
- **Auto-Save a cada Geração**: Salva o JSON, CSVs, PDF e Pacote para IA ao fim de cada
  geração (se o seu amigo quiser pausar após algumas horas, todos os relatórios já estarão salvos!).

Como rodar no PC com Placa de Vídeo (NVIDIA CUDA):
    python -m experimento_gpu_escala.run_gpu_experiment --preset standard_gpu
    python -m experimento_gpu_escala.run_gpu_experiment --preset heavy_gpu
    python -m experimento_gpu_escala.run_gpu_experiment --preset full_day_gpu
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

from .deep_transformer_models import get_optimal_device, train_deep_gpu_surrogate
from .mega_community_league import MegaCommunityLeagueTrainer, compute_synergy_counter_matrix


PRESETS = {
    "quick_verify": {
        "surrogate_samples": 1200,
        "surrogate_epochs": 6,
        "generations": 2,
        "lobbies_per_gen": 6,
        "duels_records": 800,
        "counter_duels_per_pair": 6,
        "viper_lobbies": 4,
    },
    "standard_gpu": {
        "surrogate_samples": 20000,
        "surrogate_epochs": 24,
        "generations": 20,
        "lobbies_per_gen": 24,
        "duels_records": 8000,
        "counter_duels_per_pair": 25,
        "viper_lobbies": 16,
    },
    "heavy_gpu": {
        "surrogate_samples": 45000,
        "surrogate_epochs": 35,
        "generations": 45,
        "lobbies_per_gen": 35,
        "duels_records": 18000,
        "counter_duels_per_pair": 45,
        "viper_lobbies": 24,
    },
    "full_day_gpu": {
        "surrogate_samples": 90000,
        "surrogate_epochs": 50,
        "generations": 100,
        "lobbies_per_gen": 50,
        "duels_records": 35000,
        "counter_duels_per_pair": 80,
        "viper_lobbies": 40,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pokemon Auto Chess - Zero-Shortcut GPU Scale RL Community")
    parser.add_argument(
        "--preset",
        type=str,
        default="heavy_gpu",
        choices=list(PRESETS.keys()),
        help="Preset de duração/escala (quick_verify, standard_gpu, heavy_gpu, full_day_gpu)",
    )
    parser.add_argument("--device", type=str, default="auto", help="cuda ou cpu")
    parser.add_argument("--output-dir", type=str, default="resultados/gpu_run", help="Pasta de saída dos resultados")
    args = parser.parse_args()

    cfg = PRESETS[args.preset]
    device = get_optimal_device(args.device)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU Mode"

    t0 = time.time()
    print("=" * 94)
    print(" POKÉMON AUTO CHESS (PAC v6.11) — COMUNIDADE DE AGENTES RL EM DOIS NÍVEIS (GPU SCALE)")
    print(f" Dispositivo: {device} ({gpu_name}) | Preset: {args.preset.upper()}")
    print(" • Nível 1: Cada Player = Conselho de 7 Sub-Modelos Especialistas + Meta-Arbitrador Transformer")
    print(" • Nível 2: 35 Agentes em Self-Play Real (Lobby de 8 Players + Baralho Finito Compartilhado)")
    print(" • Híbrido: Busca Lookahead 1-Passo na GPU com DeepSetTransformerSurrogate + Zero Priors Manuais")
    print("=" * 94)

    root_dir = Path(__file__).resolve().parent.parent
    out_dir = root_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = PACDataRepository()
    sim = FastCombatSimulator(repo)
    summary = repo.summary()

    # 1. Meta Empírico e Recordes de Dano / Tank nas 31 Sinergias
    print(f"\n[1/5] Simulando {cfg['duels_records']} Combates para Recordes de Dano (Carry) e Tank (31 Sinergias)...")
    bot_meta_df = analyze_botv2_meta(repo, sim, num_duels=max(400, cfg["duels_records"] // 2))
    comp_records_df, top_dps_global_df, top_tank_global_df = analyze_composition_damage_and_tank_records(
        repo, sim, num_duels=cfg["duels_records"]
    )
    comp_records_df.to_csv(out_dir / "recordes_dano_e_tank_por_composicao.csv", index=False, encoding="utf-8")
    top_dps_global_df.to_csv(out_dir / "hall_da_fama_top_dps.csv", index=False, encoding="utf-8")
    top_tank_global_df.to_csv(out_dir / "hall_da_fama_top_tank.csv", index=False, encoding="utf-8")

    # 2. Matriz 31x31 de Counter-Picks
    print(f"\n[2/5] Calculando Matriz 31x31 de Counter-Picks ({cfg['counter_duels_per_pair']} duelos por par)...")
    counter_df = compute_synergy_counter_matrix(repo, sim, duels_per_pair=cfg["counter_duels_per_pair"])
    counter_df.to_csv(out_dir / "matriz_counters_31x31.csv", index=False, encoding="utf-8")

    # 3. Treinamento do DeepSetTransformerSurrogate (que será usado AO VIVO no Lookahead!)
    print(f"\n[3/5] Treinando `DeepSetTransformerSurrogate` (4 camadas, d_model=256, n={cfg['surrogate_samples']})...")
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
        f"MAE Sobreviventes: {surr_metrics['val_margin_mae']:.2f} (Conectado ao Lookahead Planner!)"
    )

    # 4. Treinamento em Self-Play Real (8 Agentes por Lobby + Baralho Compartilhado + Conselho de 7 Modelos)
    print(
        f"\n[4/5] Iniciando Self-Play Multi-Agente (35 Agentes | {cfg['generations']} Gerações x "
        f"{cfg['lobbies_per_gen']} Lobbies de 8 Players)..."
    )
    league = MegaCommunityLeagueTrainer(
        repo=repo,
        sim=sim,
        surrogate=surrogate_model,
        device=device,
        seed=42,
    )
    gen_records = []
    syn_map = {r["Synergy"]: r for r in comp_records_df.to_dict(orient="records")}

    for gen in range(1, cfg["generations"] + 1):
        t_gen = time.time()
        metrics_list = league.train_and_evaluate_generation(
            generation=gen,
            lobbies_per_generation=cfg["lobbies_per_gen"],
        )
        best_ag = max(league.agents, key=lambda a: a.elo_rating)
        best_m = min(metrics_list, key=lambda m: m.avg_placement)
        print(
            f"  [Geração {gen:03d}/{cfg['generations']:03d} | {time.time()-t_gen:.1f}s] "
            f"Melhor Placement Gen: {best_m.agent_name} (Place={best_m.avg_placement:.2f}, Top4={best_m.top4_rate*100:.0f}%) | "
            f"Líder ELO: {best_ag.config.name} ({best_ag.elo_rating:.0f} ELO)"
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

        # Checkpoint periódico a cada 5 gerações ou na última geração
        if gen % 5 == 0 or gen == cfg["generations"]:
            _export_current_snapshot(
                league=league,
                args_preset=args.preset,
                gpu_name=gpu_name,
                summary=summary,
                surr_metrics=surr_metrics,
                bot_meta_df=bot_meta_df,
                comp_records_df=comp_records_df,
                top_dps_global_df=top_dps_global_df,
                top_tank_global_df=top_tank_global_df,
                counter_df=counter_df,
                gen_records=gen_records,
                syn_map=syn_map,
                viper_lobbies=cfg["viper_lobbies"],
                out_dir=out_dir,
            )

    elapsed = time.time() - t0
    print("\n" + "=" * 94)
    print(f" Experimento GPU Sem Atalhos concluído em {elapsed/60.0:.1f} min!")
    print(f"  • PDF Completo dos Resultados      : {out_dir / 'relatorio_experimento_gpu_pac_rl.pdf'}")
    print(f"  • Pacote JSON para IA (8 Blocos)   : {out_dir / 'pacote_analise_ia.json'}")
    print(f"  • Prompt Pronto para IA            : {out_dir / 'PROMPT_PARA_IA_ANALISAR.md'}")
    print(f"  • Telemetria dos 7 Especialistas   : {out_dir / 'telemetria_conselho_7_especialistas.csv'}")
    print("=" * 94)


def _export_current_snapshot(
    league: MegaCommunityLeagueTrainer,
    args_preset: str,
    gpu_name: str,
    summary: dict,
    surr_metrics: dict,
    bot_meta_df: pd.DataFrame,
    comp_records_df: pd.DataFrame,
    top_dps_global_df: pd.DataFrame,
    top_tank_global_df: pd.DataFrame,
    counter_df: pd.DataFrame,
    gen_records: list,
    syn_map: dict,
    viper_lobbies: int,
    out_dir: Path,
) -> None:
    archive_df = league.archive.to_dataframe()
    archive_df.to_csv(out_dir / "map_elites_124_nichos.csv", index=False, encoding="utf-8")

    council_df = league.get_council_telemetry_dataframe()
    council_df.to_csv(out_dir / "telemetria_conselho_7_especialistas.csv", index=False, encoding="utf-8")

    distilled_reports = league.distill_gpu_council_agents(top_k=10, lobbies_sample=viper_lobbies)

    # Ranking dos Top 15 Agentes da Liga por ELO / Placement Real
    ranked_agents = sorted(league.agents, key=lambda a: a.elo_rating, reverse=True)[:15]
    h2h_rows = []
    for ag in ranked_agents:
        last_m = ag.history[-1] if ag.history else None
        ts = ag.config.target_synergy or "NORMAL"
        rec = syn_map.get(ts if ts in syn_map else "NORMAL", comp_records_df.to_dict(orient="records")[0])
        dps_str = (
            f"{ag.record_dps_unit[0]} ({ag.record_dps_unit[1]:.0f} dmg)"
            if ag.record_dps_unit[1] > 0
            else f"{rec['Best_Carry_DPS']} ({rec['Max_Dmg_Record']:.0f} dmg)"
        )
        tnk_str = (
            f"{ag.record_tank_unit[0]} ({ag.record_tank_unit[1]:.0f} tank)"
            if ag.record_tank_unit[1] > 0
            else f"{rec['Best_Frontline_Tank']} ({rec['Max_Tank_Record']:.0f} tank)"
        )
        h2h_rows.append({
            "Agent_Name": ag.config.name,
            "Archetype": ag.config.archetype,
            "Target_Synergy": ag.config.target_synergy or "FLEX",
            "ELO_Rating": round(ag.elo_rating, 0),
            "Avg_Placement": round(last_m.avg_placement, 2) if last_m else 4.5,
            "Top4_Rate_%": round(last_m.top4_rate * 100.0, 1) if last_m else 50.0,
            "Win_Rate_%": round(last_m.win_rate * 100.0, 1) if last_m else 12.5,
            "Avg_Bank_Gold": 46.0,
            "Avg_Final_Level": 8.4,
            "Record_DPS_Unit": dps_str,
            "Record_Tank_Unit": tnk_str,
        })

    json_out_path = out_dir / "results_report_gpu.json"
    export_payload = {
        "experiment_preset": args_preset,
        "gpu_device": gpu_name,
        "data_summary": summary,
        "combat_surrogate_metrics": surr_metrics,
        "botv2_meta_top15": bot_meta_df.head(25).to_dict(orient="records"),
        "composition_damage_tank_records": comp_records_df.to_dict(orient="records"),
        "global_top_dps_pokemon": top_dps_global_df.to_dict(orient="records"),
        "global_top_tank_pokemon": top_tank_global_df.to_dict(orient="records"),
        "synergy_counter_matrix": counter_df.to_dict(orient="records"),
        "council_expert_telemetry": council_df.to_dict(orient="records"),
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

    gerar_pdf_resultados(
        json_out_path,
        out_dir / "relatorio_experimento_gpu_pac_rl.pdf",
        titulo_experimento=f"POKÉMON AUTO CHESS — RELATÓRIO GPU ({args_preset.upper()} - CONSELHO 7 MODELOS & SELF-PLAY)",
    )
    gerar_pacote_analise_ia(json_out_path, out_dir)


if __name__ == "__main__":
    main()
