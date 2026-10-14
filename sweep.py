"""
Barrido de parámetros (sweep) para la simulación evolutiva de satélites.

Ejecuta la simulación sobre una grilla de configuraciones de
`ConfiguracionFisica` (opcionalmente cruzada con semillas), extrae
métricas de interés de cada corrida y exporta los resultados a un CSV
listo para analizar (pandas, hojas de cálculo, etc.).

Es la base para el análisis evolutivo comparado: por ejemplo, cómo la
probabilidad de mutación saltacional afecta la calidad promedio final
de la constelación, o cómo el costo de evasión modula la presión sobre
`maniobrabilidad_evasiva`.

Uso:
    python sweep.py                              # grilla por defecto
    python sweep.py --salida resultados.csv      # CSV destino
    python sweep.py --dias 365 --semillas 3      # corrida más larga,
                                                 # 3 semillas por celda
"""

import argparse
import csv
import itertools
from datetime import datetime
from typing import Dict, List

import numpy as np

from sat import ConfiguracionFisica, SimulacionSatelites


def construir_grilla(args: argparse.Namespace) -> List[Dict]:
    """
    Construye la lista de overrides de configuración a partir de los
    argumentos: el producto cartesiano de todos los ejes especificados.
    """
    ejes: Dict[str, list] = {}
    if args.prob_saltacional is not None:
        ejes["prob_mutacion_saltacional"] = args.prob_saltacional
    if args.costo_evasion is not None:
        ejes["costo_combustible_evasion_diario_pct"] = args.costo_evasion
    if args.bono_cobertura is not None:
        ejes["peso_bono_cobertura_banda"] = args.bono_cobertura
    if args.dias_no_retorno is not None:
        ejes["dias_sostenidos_punto_no_retorno"] = args.dias_no_retorno
    if not ejes:
        # Grilla por defecto: un eje ilustrativo por mecanismo clave.
        ejes = {
            "prob_mutacion_saltacional": [0.0, 0.01, 0.05],
            "costo_combustible_evasion_diario_pct": [0.0, 0.1],
        }
    nombres = list(ejes.keys())
    return [
        dict(zip(nombres, combinacion))
        for combinacion in itertools.product(*(ejes[n] for n in nombres))
    ]


def ejecutar_celula(overrides: Dict, semilla: int, dias: int,
                    satelites_por_lanzamiento: int) -> Dict:
    """
    Corre una simulación con la configuración y semilla dadas y extrae
    las métricas de interés del estado final y del historial.
    """
    config = ConfiguracionFisica(**overrides)
    sim = SimulacionSatelites(
        tiempo_inicio=datetime(2024, 1, 1),
        semilla=semilla,
        satelites_por_lanzamiento=satelites_por_lanzamiento,
        config=config,
    )
    sim.lanzar_satelites()
    sim.ejecutar_simulacion(dias_totales=dias)

    activos = sim.satelites_activos
    if activos:
        calidad_final = float(np.mean([s.calidad_inherente for s in activos]))
        evasion_final = float(np.mean([s.maniobrabilidad_evasiva for s in activos]))
        agresividad_final = float(np.mean([s.agresividad_reproductiva for s in activos]))
        energia_final = float(np.mean([s.energia for s in activos]))
    else:
        calidad_final = evasion_final = agresividad_final = energia_final = float("nan")

    ultimo = sim.historial[-1] if sim.historial else {}
    causas = sim.causas_muerte_acumuladas
    total_muertes = sum(causas.values())

    metricas = {
        "satelites_activos_final": len(activos),
        "satelites_muertos_total": len(sim.satelites_muertos),
        "reproducidos_total": sim.satelites_reproducidos,
        "colisiones_totales": sim.debris.colisiones_totales,
        "pasivados_total": sim.satelites_pasivados,
        "calidad_promedio_final": calidad_final,
        "evasion_promedio_final": evasion_final,
        "agresividad_promedio_final": agresividad_final,
        "energia_promedio_final": energia_final,
        "densidad_debris_normalizada": float(ultimo.get("densidad_debris", float("nan"))),
        "bandas_punto_no_retorno": "|".join(ultimo.get("bandas_punto_no_retorno", [])),
        "muertes_colision_debris": causas.get("colision_debris", 0),
        "muertes_falla_componentes": causas.get("falla_componentes", 0),
        "muertes_reentrada_orbital": causas.get("reentrada_orbital", 0),
        "muertes_falta_energia": causas.get("falta_energia", 0),
        "muertes_edad_maxima": causas.get("edad_maxima", 0),
        "muertes_totales": total_muertes,
        "costo_total_sostenibilidad": float(
            ultimo.get("costo_total_sostenibilidad", float("nan"))
        ),
    }
    return metricas


def main():
    parser = argparse.ArgumentParser(description="Barrido de parámetros de la simulación.")
    parser.add_argument("--salida", default="sweep_resultados.csv",
                        help="ruta del CSV de salida")
    parser.add_argument("--dias", type=int, default=365,
                        help="días a simular por corrida")
    parser.add_argument("--semillas", type=int, default=1,
                        help="número de semillas por celda de la grilla")
    parser.add_argument("--semilla-base", type=int, default=1000,
                        help="primera semilla; las siguientes son consecutivas")
    parser.add_argument("--satelites-por-lanzamiento", type=int, default=10)
    parser.add_argument("--prob-saltacional", type=float, nargs="+",
                        help="valores de prob_mutacion_saltacional a barrer")
    parser.add_argument("--costo-evasion", type=float, nargs="+",
                        help="valores de costo_combustible_evasion_diario_pct a barrer")
    parser.add_argument("--bono-cobertura", type=float, nargs="+",
                        help="valores de peso_bono_cobertura_banda a barrer")
    parser.add_argument("--dias-no-retorno", type=int, nargs="+",
                        help="valores de dias_sostenidos_punto_no_retorno a barrer")
    args = parser.parse_args()

    grilla = construir_grilla(args)
    total_corridas = len(grilla) * args.semillas
    print(f"Barrido: {len(grilla)} configuraciones x {args.semillas} semilla(s) "
          f"= {total_corridas} corridas de {args.dias} días.\n")

    filas = []
    for idx_celula, overrides in enumerate(grilla, start=1):
        for idx_semilla in range(args.semillas):
            semilla = args.semilla_base + idx_semilla
            etiqueta = ", ".join(f"{k}={v}" for k, v in overrides.items())
            print(f"[{idx_celula}/{len(grilla)}] semilla={semilla} | {etiqueta}")
            metricas = ejecutar_celula(
                overrides, semilla, args.dias, args.satelites_por_lanzamiento
            )
            filas.append({
                "semilla": semilla,
                **{k: v for k, v in overrides.items()},
                **metricas,
            })

    if not filas:
        print("Nada que ejecutar.")
        return

    with open(args.salida, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)

    print(f"\n✓ {len(filas)} corridas exportadas a '{args.salida}'")
    print(f"  Columnas: {', '.join(filas[0].keys())}")


if __name__ == "__main__":
    main()
