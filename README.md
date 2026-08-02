# sats-evolved

Simulación evolutiva de una constelación de satélites en órbita terrestre,
con presión de selección natural sobre la calidad de construcción de los
satélites, y herramientas de visualización de resultados.

## Idea general

Cada satélite nace con una **calidad inherente** aleatoria (`0.1` a `1.0`)
que determina su tasa de falla base. Al reproducirse (heredabilidad), los
satélites "hijo" heredan la calidad de su "padre" con una pequeña mutación.
Con el paso de las generaciones, los satélites de baja calidad mueren antes
y los de alta calidad tienden a dominar la población: una forma simple de
selección natural aplicada a hardware espacial.

Además de la calidad inherente, cada satélite está sujeto a varios
mecanismos de falla que se combinan como tasas aditivas
(`λ_total = λ_inherente + λ_envejecimiento + λ_radiación + λ_debris + λ_energía + λ_combustible`):

- **Envejecimiento**: curva de falla tipo "bañera" (Weibull), más probable
  con la edad del satélite.
- **Degradación orbital (drag atmosférico)**: reduce el semieje mayor día a
  día; si la altitud cae por debajo de 100 km, el satélite reentra y muere.
- **Ciclos solares**: modulan tanto el drag atmosférico como la presión de
  radiación solar en un ciclo sinusoidal de 11 años.
- **Radiación cósmica**: daño acumulativo; el satélite falla al superar un
  umbral de radiación acumulada.
- **Debris espacial**: cada muerte genera nuevo debris, que a su vez
  aumenta el riesgo de colisión (más alto en órbitas LEO congestionadas).
- **Metabolismo (energía/combustible)**: el satélite consume energía a
  diario (recargada parcialmente por luz solar) y puede fallar por falta de
  energía; sin combustible suficiente, aumenta su tasa de falla por no poder
  maniobrar.

## Mecanismos evolutivos a nivel de constelación

Además de la selección individual por calidad inherente, la reproducción
selectiva (`SimulacionSatelites.lanzar_satelites` con `heredables=True`)
incorpora mecanismos evolutivos que operan sobre la constelación como
sistema, no solo satélite por satélite:

- **Selección por aptitud diferencial**: los padres se sortean con
  probabilidad proporcional a su fitness (`Satellite.fitness_reproductivo`,
  hoy la calidad inherente) en vez de uniforme entre candidatos sanos.
- **Fitness de cobertura de constelación**: además de la calidad, se aplica
  un bono a satélites cuyo RAAN (`Omega`) está en una zona poco poblada
  respecto al resto de los candidatos, favoreciendo que la reproducción
  llene huecos de cobertura orbital en vez de concentrarse en posiciones ya
  representadas (selección dependiente de frecuencia).
- **Recombinación sexual (crossover)**: con probabilidad
  `prob_reproduccion_sexual`, un hijo nace combinando los rasgos de 2
  padres (`Satellite.crear_hijo_sexual`) en vez de clonar 1 solo padre con
  mutación, aumentando la diversidad genética de la población.
- **Selección dependiente de densidad (carrying capacity)**: cada intento
  de reproducción puede fallar según qué tan congestionada esté la banda
  orbital del padre (más debris → más probabilidad de que el intento no
  prospere; el hueco se completa con un lanzamiento nuevo sin heredar).
- **Estrategias r/K**: el rasgo heredable `agresividad_reproductiva`
  (0=K-conservador, 1=r-agresivo) determina qué tan exigentes son los
  umbrales de energía/combustible para calificar como candidato a
  reproducirse.
- **Coevolución con el debris**: el rasgo heredable
  `maniobrabilidad_evasiva` reduce la tasa de falla por colisión con
  debris (λ_debris) si el satélite tiene combustible suficiente para
  maniobrar, favoreciendo naturalmente la evasión en bandas congestionadas.

## Estructura del repositorio

```
sat.py           # Motor de simulación: satélites, entornos y evolución
visualizador.py  # Generación de gráficas y dashboards a partir de la simulación
benchmark.py     # Benchmark de rendimiento (antes/después) y validación numérica
tests/           # Suite de tests (pytest)
```

### `sat.py`

Contiene el modelo y el motor de simulación:

- `CicloSolar`, `RadiacionCosmica`, `DebrisEspacial`: subsistemas
  ambientales que afectan a los satélites.
- `Satellite`: satélite individual con parámetros keplerianos, calidad
  inherente, metabolismo (energía/combustible), degradación orbital,
  radiación acumulada y reproducción (`crear_hijo`). Usa `__slots__`
  para reducir memoria y acelerar el acceso a atributos en poblaciones
  grandes (miles de satélites).
- `DistribucionFalla`: modelo legacy de tasa de falla única (se mantiene
  por compatibilidad; el cálculo real usa las tasas aditivas de
  `Satellite.calcular_lambda_total`).
- `SimulacionSatelites`: orquesta lanzamientos periódicos, efectos
  ambientales diarios, evaluación de muertes y reproducción selectiva.
  Guarda un historial de estadísticas diarias en `self.historial`. Expone
  un generador aleatorio reproducible (`self.rng`) y métodos de
  checkpointing (`guardar_checkpoint`/`cargar_checkpoint`).

### `visualizador.py`

Contiene `VisualizadorSatelites`, que toma una `SimulacionSatelites` ya
ejecutada y genera:

- **Gráficas estáticas (Matplotlib)**:
  - `graficar_estadisticas`: panel de 8 gráficas (población, edad, semieje
    mayor, mortalidad, calidad inherente, causas de muerte, etc.).
  - `graficar_calidad_heredabilidad`: distribución de calidad, calidad del
    padre vs. calidad del hijo, y calidad vs. longevidad por causa de
    muerte.
- **Dashboard interactivo (Plotly, HTML)**: `crear_dashboard_interactivo`.
- **Órbitas 3D (Plotly, HTML)**: `crear_orbitas_3d`.
- **Animaciones temporales (Plotly, HTML)**: `crear_animacion_poblacion`,
  `crear_animacion_estadisticas_multiples`, `crear_animacion_scatter_dinamico`.
- **Todo en uno**: `generar_todas_visualizaciones` ejecuta todas las
  anteriores y guarda los archivos en un directorio dado.

## Requisitos

- Python 3.9+
- numpy
- matplotlib
- plotly

Instalación rápida:

```powershell
pip install numpy matplotlib plotly
```

Para desarrollo (correr la suite de tests):

```powershell
pip install pytest
```

## Uso

Ejecutar una simulación básica y ver las estadísticas finales por consola:

```powershell
python sat.py
```

Ejecutar la simulación y generar todas las visualizaciones (PNG y HTML):

```powershell
python visualizador.py
```

O bien, usarlo de forma programática:

```python
from datetime import datetime
from sat import SimulacionSatelites, DistribucionFalla
from visualizador import VisualizadorSatelites

falla = DistribucionFalla(tasa_base=0.0005, parametro_forma=2.0, escala_tiempo=3.0)

sim = SimulacionSatelites(
    distribucion_falla=falla,
    intervalo_lanzamiento_dias=30,
    satelites_por_lanzamiento=5,
    edad_maxima_anos=5,
    tiempo_inicio=datetime(2024, 1, 1),
)

sim.lanzar_satelites()
sim.ejecutar_simulacion(dias_totales=1825)  # 5 años

viz = VisualizadorSatelites(sim)
viz.generar_todas_visualizaciones(directorio="salida")
```

### Reproducibilidad

`SimulacionSatelites` acepta una `semilla` opcional que inicializa un
`np.random.Generator` propio (`sim.rng`), usado en todos los sorteos de
la simulación (calidad inherente, mutaciones, selección de padres,
fallas, colisiones). Con la misma semilla, dos corridas producen
exactamente los mismos resultados:

```python
sim = SimulacionSatelites(tiempo_inicio=datetime(2024, 1, 1), semilla=42)
```

### Checkpointing

Para simulaciones largas, `SimulacionSatelites` puede guardar y restaurar
su estado completo (población, historial, generador aleatorio, etc.) con
`pickle`:

```python
sim.guardar_checkpoint("checkpoint.pkl")
# ... más tarde, o en otro proceso ...
sim = SimulacionSatelites.cargar_checkpoint("checkpoint.pkl")
sim.ejecutar_simulacion(dias_totales=365)  # continúa donde quedó
```

También se puede pedir un checkpoint automático cada N días dentro de
`ejecutar_simulacion`:

```python
sim.ejecutar_simulacion(dias_totales=1825, checkpoint_cada_dias=365,
                       checkpoint_ruta="checkpoint.pkl")
```

### Historial de memoria reducida

Para simulaciones muy largas o con poblaciones muy grandes,
`historial_intervalo_dias` controla cada cuántos días se guarda un
snapshot en `sim.historial` (por defecto 1 = todos los días). El último
snapshot calculado siempre queda disponible en `sim.ultimo_stats`,
incluso si no se guardó en el historial:

```python
sim = SimulacionSatelites(tiempo_inicio=datetime(2024, 1, 1),
                          historial_intervalo_dias=7)  # 1 snapshot por semana
```

Archivos generados por `generar_todas_visualizaciones`:

| Archivo                          | Contenido                                       |
|-----------------------------------|--------------------------------------------------|
| `estadisticas.png`                 | Panel estático de 8 gráficas estadísticas         |
| `calidad_heredabilidad.png`        | Calidad inherente y heredabilidad                 |
| `dashboard.html`                   | Dashboard interactivo (Plotly)                    |
| `orbitas_3d.html`                  | Órbitas 3D animadas alrededor de la Tierra        |
| `animacion_poblacion.html`         | Animación de evolución de población               |
| `animacion_estadisticas.html`      | Animación de múltiples estadísticas en el tiempo  |
| `animacion_scatter.html`           | Animación de edad promedio vs. población activa   |

## Unidades y convenciones

- Distancias orbitales (`a`, altitud, perigeo, apogeo): kilómetros.
- Ángulos (`i`, `Omega`, `omega`, `M`): grados.
- Tiempo de simulación: días (a partir de `tiempo_inicio`).
- Calidad inherente, radiación acumulada y degradación orbital: escala 0-1.
- Energía y combustible: porcentaje, 0-100.

## Rendimiento

El loop diario de `SimulacionSatelites` (drag/decaimiento orbital, presión
de radiación solar, radiación cósmica, metabolismo, riesgo de colisión y
evaluación de muertes) está vectorizado con NumPy a nivel de población
completa (ver las funciones `*_batch` en `sat.py`), en vez de iterar
satélite por satélite en Python puro. Esto es transparente para
`visualizador.py`: los satélites se siguen modelando como objetos
`Satellite` individuales, solo cambia cómo se calculan sus atributos en
cada paso de tiempo.

`benchmark.py` compara la versión vectorizada actual contra la versión
previa (obtenida automáticamente desde `git show HEAD:sat.py`) y valida
que los resultados numéricos sigan en rango (sin NaN, probabilidades
entre 0-1, etc.):

```powershell
python benchmark.py
```

Resultados de referencia (tu hardware puede variar):

| N satélites | Días simulados | Antes (s) | Después (s) | Speedup |
|---|---|---|---|---|
| 1000 | 30 | 0.517 | 0.153 | 3.38x |
| 2500 | 20 | 0.909 | 0.251 | 3.62x |
| 5000 | 15 | 1.431 | 0.375 | 3.82x |

El speedup mejora con la escala de la población, ya que el costo por
satélite de las operaciones matemáticas (raíces, exponenciales, etc.) se
amortiza en operaciones vectoriales de NumPy en vez de llamadas
individuales en Python.

## Tests

La suite de tests (`tests/`) usa `pytest` y cubre: el modelo `Satellite`
(calidad inherente, heredabilidad, `__slots__`), `DebrisEspacial`,
equivalencia numérica entre las funciones vectorizadas `*_batch` y sus
contrapartes escalares, y `SimulacionSatelites` (reproducibilidad con
semilla, muestreo del historial, checkpointing).

```powershell
python -m pytest tests/ -v
```

## Historial de versiones

### v1.3.0
- Nuevos mecanismos evolutivos a nivel de constelación en
  `lanzar_satelites`: selección por aptitud diferencial (fitness
  proporcional a la calidad inherente), fitness de cobertura (bono por
  diversidad de RAAN, calculado en O(n log n) con un barrido circular
  ordenado en vez de una matriz par-a-par O(n²)), recombinación sexual
  (crossover entre 2 padres, `Satellite.crear_hijo_sexual`) y selección
  dependiente de densidad (carrying capacity según congestión de la
  banda orbital del padre).
- Nuevos rasgos heredables en `Satellite`: `agresividad_reproductiva`
  (estrategia r/K, afecta los umbrales de energía/combustible para
  calificar como candidato a reproducirse) y `maniobrabilidad_evasiva`
  (coevolución con el debris, reduce λ_debris si hay combustible
  suficiente para maniobrar).
- Nuevos parámetros en `ConfiguracionFisica` (`factor_capacidad_carga`,
  `factor_reduccion_evasion_max`, umbrales r/K) y en
  `SimulacionSatelites` (`prob_reproduccion_sexual`).
- 11 tests nuevos cubriendo estos mecanismos (`tests/test_sat.py`, 36
  casos en total).

### v1.2.0
- `Satellite` ahora usa `__slots__`: menor uso de memoria y acceso más
  rápido a atributos con poblaciones grandes.
- `SimulacionSatelites` expone un `np.random.Generator` reproducible
  (`self.rng`, configurable con el parámetro `semilla`) usado en todos
  los sorteos de la simulación, incluida la selección de padres al
  reproducirse (vectorizada con `rng.integers` en vez de
  `random.choice` satélite por satélite).
- Nuevo `historial_intervalo_dias` para muestrear `sim.historial` y
  reducir el uso de memoria en corridas largas; `sim.ultimo_stats`
  siempre tiene el snapshot más reciente.
- Nuevo checkpointing (`guardar_checkpoint`/`cargar_checkpoint`, y
  `checkpoint_cada_dias` en `ejecutar_simulacion`) para reanudar
  simulaciones largas tras una interrupción.
- El snapshot diario de estadísticas (`paso_tiempo`) ahora calcula edad,
  energía, radiación, semieje mayor y calidad promedio en una sola
  pasada sobre la población, en vez de 5 pasadas independientes.
- `crear_orbitas_3d` (visualizador.py) vectoriza el cálculo de rotación
  orbital de toda la población a graficar en vez de recalcularlo
  satélite por satélite.
- Nueva suite de tests (`tests/`, pytest) con 25 casos.

### v1.1.0
- Vectorización con NumPy del cálculo de drag/decaimiento orbital, presión
  de radiación solar, radiación cósmica y metabolismo
  (`actualizar_sistemas_ambientales`).
- Vectorización del cálculo de tasas de falla aditivas (λ_total), riesgo
  de colisión y sorteos aleatorios en batch (`evaluar_muertes`), en vez de
  iterar satélite por satélite con `random.random()`.
- Nuevo `benchmark.py` para comparar rendimiento antes/después y validar
  la sanidad numérica de los resultados (1000-5000 satélites): ~3.4x-3.8x
  más rápido, sin cambios de comportamiento ni de interfaz para
  `visualizador.py`.
- Limpieza: se removió la dependencia no utilizada de `scipy`.

### v1.0.0
- Motor de simulación evolutiva de satélites (`sat.py`): calidad
  inherente heredable, drag atmosférico, ciclos solares, radiación
  cósmica, debris espacial por bandas, metabolismo y mitigación de
  debris (pasivación/desorbitado controlado).
- Visualizador (`visualizador.py`): gráficas estáticas, dashboard
  interactivo, órbitas 3D y animaciones temporales.
