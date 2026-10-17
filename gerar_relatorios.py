"""
Script utilitário para:
1. Calcular / atualizar os Recordes de Dano (Best Carry) e Recordes de Tank (Best Frontline)
   para todas as 31 composições/sinergias do Pokémon Auto Chess.
2. Gerar o PDF executivo multi-página (`resultados/relatorio_experimento_cpu_pac_rl.pdf`
   ou qualquer caminho passado via `--input-json` e `--output-pdf`).
3. Gerar o pacote estruturado de blocos JSON + Prompt para Análise por IA
   (`resultados/pacote_analise_ia.json` e `resultados/PROMPT_PARA_IA_ANALISAR.md`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pac_rl.analytics_exporter import gerar_pacote_analise_ia, gerar_pdf_resultados
from pac_rl.combat_surrogate import FastCombatSimulator
from pac_rl.community_league import analyze_composition_damage_and_tank_records
from pac_rl.data_loader import PACDataRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Gerador de PDF e Pacote de Análise IA para o PAC RL Lab")
    parser.add_argument(
        "--input-json",
        type=str,
        default="pac_rl/results_report.json",
        help="Caminho para o arquivo JSON de resultados do experimento",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="resultados",
        help="Diretório onde serão salvos o PDF, CSVs, JSON de blocos para IA e Prompt Markdown",
    )
    parser.add_argument(
        "--pdf-name",
        type=str,
        default="relatorio_experimento_pac_rl.pdf",
        help="Nome do arquivo PDF de saída",
    )
    parser.add_argument(
        "--num-duels",
        type=int,
        default=1800,
        help="Número de combates táticos para apurar os recordes de Dano e Tank por composição",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    res_path = root / args.input_json
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(res_path.read_text(encoding="utf-8"))

    if "composition_damage_tank_records" not in data or not data["composition_damage_tank_records"]:
        print(f"Calculando recordes de Dano (Carry) e Tank (Frontline) em {args.num_duels} combates...")
        repo = PACDataRepository()
        sim = FastCombatSimulator(repo)
        comp_df, dps_df, tank_df = analyze_composition_damage_and_tank_records(
            repo, sim, num_duels=args.num_duels
        )
        data["composition_damage_tank_records"] = comp_df.to_dict(orient="records")
        data["global_top_dps_pokemon"] = dps_df.to_dict(orient="records")
        data["global_top_tank_pokemon"] = tank_df.to_dict(orient="records")

        syn_map = {r["Synergy"]: r for r in data["composition_damage_tank_records"]}
        for row in data.get("head_to_head_tournament", []):
            ts = row.get("Target_Synergy", "NORMAL")
            rec = syn_map.get(ts if ts != "FLEX" else "NORMAL", data["composition_damage_tank_records"][0])
            row["Record_DPS_Unit"] = f"{rec['Best_Carry_DPS']} ({rec['Max_Dmg_Record']:.0f} dmg)"
            row["Record_Tank_Unit"] = f"{rec['Best_Frontline_Tank']} ({rec['Max_Tank_Record']:.0f} tank)"

        res_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        comp_df.to_csv(out_dir / "recordes_dano_e_tank_por_composicao.csv", index=False, encoding="utf-8")
        dps_df.to_csv(out_dir / "hall_da_fama_top_dps.csv", index=False, encoding="utf-8")
        tank_df.to_csv(out_dir / "hall_da_fama_top_tank.csv", index=False, encoding="utf-8")

        print("\n=== RECORDES DE DANO (CARRY) E TANK (FRONTLINE) POR COMPOSICAO (31 SINERGIAS) ===")
        print(comp_df.to_string(index=False))
        print("\n=== TOP 12 GLOBAL RECORDISTAS DE DANO (DPS) ===")
        print(dps_df.head(12).to_string(index=False))
        print("\n=== TOP 12 GLOBAL RECORDISTAS DE TANK (DANO ABSORVIDO/MITIGADO) ===")
        print(tank_df.head(12).to_string(index=False))

    (out_dir / "results_report_completo.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    pdf_path = gerar_pdf_resultados(res_path, out_dir / args.pdf_name)
    ai_json, ai_prompt = gerar_pacote_analise_ia(res_path, out_dir)

    print("\n" + "=" * 80)
    print(f"PDF gerado com sucesso   : {pdf_path} ({pdf_path.stat().st_size:,} bytes)")
    print(f"Pacote JSON para IA      : {ai_json}")
    print(f"Prompt Markdown para IA  : {ai_prompt}")
    print("=" * 80)


if __name__ == "__main__":
    main()
