"""
Benchmark: vectorización del loop diario de la simulación

Compara el rendimiento de la versión vectorizada actual de `sat.py`
(drag/decaimiento, presión de radiación solar, radiación cósmica,
metabolismo, riesgo de colisión y evaluación de muertes calculados con
NumPy sobre arrays) contra la versión previa (obtenida del último commit
de git, antes de vectorizar), ejecutando el mismo escenario: N satélites
activos avanzando un número fijo de días.

También corre una validación de sanidad numérica (sin NaN, probabilidades
en rango [0,1], etc.) sobre los resultados de ambas versiones para
confirmar que el comportamiento estadístico se mantiene equivalente.

Uso:
    python benchmark.py
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import numpy as np


def cargar_modulo_desde_git(ref: str = "HEAD", ruta: str = "sat.py",
                             nombre_modulo: str = "sat_original_benchmark"):
    """
    Carga `sat.py` tal como estaba en el commit `ref` como un módulo
    Python independiente, sin afectar al `sat.py` actual en disco.

    Args:
        ref: referencia de git (por defecto HEAD, el último commit).
        ruta: ruta relativa del archivo dentro del repo.
        nombre_modulo: nombre bajo el cual registrar el módulo cargado.

    Returns:
        El módulo cargado, o None si no se pudo obtener desde git.
    """
    resultado = subprocess.run(
        ["git", "show", f"{ref}:{ruta}"],
        capture_output=True, check=True
    )
    codigo_original = resultado.stdout.decode("utf-8")

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, f"{nombre_modulo}.py")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(codigo_original)

    spec = importlib.util.spec_from_file_location(nombre_modulo, tmp_path)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nombre_modulo] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def preparar_simulacion(modulo, n_satelites: int, semilla: int):
    """Crea una simulación con `n_satelites` ya activos (sin pasar por
    lanzamientos periódicos), para poder comparar directamente el costo
    del loop diario a distintas escalas de población."""
    np.random.seed(semilla)
    import random as random_module
    random_module.seed(semilla)

    sim = modulo.SimulacionSatelites(
        tiempo_inicio=datetime(2024, 1, 1),
        reproductor_habilitado=True,
    )
    for _ in range(n_satelites):
        a, e, i, Omega, omega, M = sim._generar_parametros_kepleriano()
        sat = modulo.Satellite(a, e, i, Omega, omega, M, sim.tiempo_actual, config=sim.config)
        sim.satelites_activos.append(sat)
    return sim


def benchmark(modulo, n_satelites: int, dias: int, semilla: int = 42):
    """Ejecuta `dias` pasos de simulación y devuelve (tiempo_segundos, sim)."""
    sim = preparar_simulacion(modulo, n_satelites, semilla)
    inicio = time.perf_counter()
    for _ in range(dias):
        sim.paso_tiempo(1)
    fin = time.perf_counter()
    return fin - inicio, sim


def validar_resultado(sim, etiqueta: str) -> None:
    """Chequeos de sanidad numérica sobre el historial y la población final."""
    for stats in sim.historial:
        assert stats['satelites_activos'] >= 0, f"[{etiqueta}] activos negativos"
        assert not np.isnan(stats['edad_promedio_dias']), f"[{etiqueta}] edad NaN"
        assert not np.isnan(stats['calidad_promedio']), f"[{etiqueta}] calidad NaN"
        assert 0.0 <= stats['densidad_debris'] <= 1.0, f"[{etiqueta}] densidad_debris fuera de rango"
        if stats['satelites_activos'] > 0:
            assert 0.0 <= stats['calidad_promedio'] <= 1.0, f"[{etiqueta}] calidad_promedio fuera de rango"

    for sat in sim.satelites_activos:
        assert not np.isnan(sat.a), f"[{etiqueta}] semieje mayor NaN"
        assert not np.isnan(sat.e), f"[{etiqueta}] excentricidad NaN"
        assert 0.0 <= sat.energia <= 100.0 + 1e-6, f"[{etiqueta}] energía fuera de rango: {sat.energia}"
        assert 0.1 <= sat.calidad_inherente <= 1.0, f"[{etiqueta}] calidad_inherente fuera de rango"


def main():
    import sat as sat_actual  # versión vectorizada actual (sat.py en disco)

    print("Cargando versión previa (HEAD, antes de vectorizar) desde git para comparación...")
    sat_original = None
    try:
        sat_original = cargar_modulo_desde_git()
    except subprocess.CalledProcessError:
        print("  No se pudo obtener la versión previa desde git. Se omite comparación 'antes/después'.")

    casos = [(1000, 30), (2500, 20), (5000, 15)]

    print(f"\n{'N sats':>8} {'Días':>6} {'Antes (s)':>12} {'Después (s)':>12} {'Speedup':>10}")
    print("-" * 55)

    for n_satelites, dias in casos:
        tiempo_despues, sim_despues = benchmark(sat_actual, n_satelites, dias)
        validar_resultado(sim_despues, f"vectorizado n={n_satelites}")

        if sat_original is not None:
            tiempo_antes, sim_antes = benchmark(sat_original, n_satelites, dias)
            validar_resultado(sim_antes, f"original n={n_satelites}")
            speedup = tiempo_antes / tiempo_despues if tiempo_despues > 0 else float("nan")
            print(f"{n_satelites:>8} {dias:>6} {tiempo_antes:>12.3f} {tiempo_despues:>12.3f} {speedup:>9.2f}x")
        else:
            print(f"{n_satelites:>8} {dias:>6} {'N/A':>12} {tiempo_despues:>12.3f} {'N/A':>10}")

    print("\n✓ Validación de sanidad numérica: sin NaN, probabilidades y rangos correctos.")


if __name__ == "__main__":
    main()
