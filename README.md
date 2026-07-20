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

## Estructura del repositorio

```
sat.py           # Motor de simulación: satélites, entornos y evolución
visualizador.py  # Generación de gráficas y dashboards a partir de la simulación
```

### `sat.py`

Contiene el modelo y el motor de simulación:

- `CicloSolar`, `RadiacionCosmica`, `DebrisEspacial`: subsistemas
  ambientales que afectan a los satélites.
- `Satellite`: satélite individual con parámetros keplerianos, calidad
  inherente, metabolismo (energía/combustible), degradación orbital,
  radiación acumulada y reproducción (`crear_hijo`).
- `DistribucionFalla`: modelo legacy de tasa de falla única (se mantiene
  por compatibilidad; el cálculo real usa las tasas aditivas de
  `Satellite.calcular_lambda_total`).
- `SimulacionSatelites`: orquesta lanzamientos periódicos, efectos
  ambientales diarios, evaluación de muertes y reproducción selectiva.
  Guarda un historial de estadísticas diarias en `self.historial`.

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
- scipy
- matplotlib
- plotly

Instalación rápida:

```powershell
pip install numpy scipy matplotlib plotly
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
