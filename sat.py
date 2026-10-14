"""
Simulación Evolutiva Avanzada de Satélites en Órbita

Este módulo implementa un modelo de simulación de una constelación de
satélites que envejece, falla y se reproduce con el paso del tiempo,
inspirado en dinámicas de selección natural.

Cada satélite (:class:`Satellite`) nace con una "calidad inherente"
aleatoria que determina su tasa de falla base. La calidad se hereda (con
mutación) al crear satélites "hijo", por lo que, con el paso de las
generaciones, la población tiende a desplazarse hacia mayor calidad
promedio (presión de selección).

El motor de la simulación (:class:`SimulacionSatelites`) avanza el
tiempo día a día aplicando, en orden:

1. Efectos ambientales (drag atmosférico, presión de radiación solar,
   radiación cósmica, metabolismo) que pueden causar la muerte del
   satélite.
2. Evaluación de muertes por edad máxima, falla probabilística (tasas
   de falla aditivas, ver :meth:`Satellite.calcular_lambda_total`) y
   colisión con debris espacial.
3. Lanzamiento de nuevos satélites (nuevos o "hijos" reproducidos) cada
   cierto intervalo de días.

Subsistemas ambientales incluidos:
- Degradación orbital (drag atmosférico)
- Debris espacial (colisiones)
- Ciclos solares + presión de radiación solar
- Radiación cósmica acumulativa
- Maniobrabilidad (cambio orbital)
- Heredabilidad (reproducción selectiva)
- Metabolismo (energía/combustible)

Unidades: las distancias orbitales (a, altitud, perigeo, apogeo) están
en kilómetros, los ángulos (i, Omega, omega, M) en grados, el tiempo en
días y las magnitudes de calidad/energía/combustible/radiación están
normalizadas entre 0 y 1 (o 0-100 para energía/combustible en "%").
"""

import warnings

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pickle

# Constante gravitacional terrestre estándar (GM), en km^3/s^2
GM_TIERRA_KM3_S2 = 398600.4
RADIO_TIERRA_KM = 6371.0
# Factor zonal armónico J2 de la Tierra (aplanamiento del geoide), que
# causa la precesión nodal (deriva del RAAN) y la precesión del perigeo.
J2_TIERRA = 1.08262668e-3
# Velocidad angular media de la Tierra alrededor del Sol (rad/s), usada
# para calcular el ángulo beta de la órbita respecto al terminador Sol-
# Tierra (eclipses).
VEL_ANGULAR_TIERRA_SOL_RAD_S = 1.991e-7

# Generador de números aleatorios usado como fallback cuando un Satellite
# se crea directamente (sin pasar por SimulacionSatelites, que gestiona su
# propio `np.random.Generator` reproducible vía `self.rng`). Evita depender
# del estado aleatorio global de NumPy.
_RNG_POR_DEFECTO = np.random.default_rng()


# =============================================================================
# CONFIGURACIÓN FÍSICA CENTRALIZADA
# =============================================================================

@dataclass
class ConfiguracionFisica:
    """
    Agrupa las constantes físicas y parámetros del modelo que antes
    estaban hardcodeados dentro de los métodos, para facilitar la
    experimentación (ej. barridos de sensibilidad) y las pruebas sin
    tener que editar el código del motor de simulación.

    Los valores por defecto reproducen exactamente el comportamiento
    numérico previo (ninguno de estos cambios altera resultados por sí
    solo, solo centraliza dónde se definen).
    """
    # Orbitales/gravitacionales
    gm_tierra_km3_s2: float = GM_TIERRA_KM3_S2
    radio_tierra_km: float = RADIO_TIERRA_KM

    # Drag atmosférico
    coeficiente_drag_cd: float = 2.2

    # Radiación cósmica
    limite_radiacion: float = 1.0

    # Envejecimiento (curva de bañera, fase de desgaste vía Weibull)
    escala_envejecimiento_anos: float = 3.0
    forma_envejecimiento_k: float = 2.0

    # Metabolismo
    consumo_energico_diario_pct: float = 5.0
    # Recarga solar diaria (puntos porcentuales de energía, ver
    # `actualizar_metabolismo_batch`); antes estaba hardcodeada como un
    # número mágico `+10.0` dentro de la función.
    recarga_solar_diaria_pct: float = 10.0
    # =========================================================================
    # EFECTOS SECULARES Y ECLIPSES (Fase: fidelidad orbital)
    # =========================================================================
    # Propagación de la anomalía media M por paso de tiempo (movimiento
    # kepleriano puro: dM = n·dt). Si es False, M queda estático como en
    # la versión original.
    propagar_anomalia_media: bool = True
    # Precesión nodal J2 (dOmega/dt, Vallado): si es False, el RAAN no
    # deriva y las bandas de RAAN de la constelación quedan fijas.
    precesion_j2_habilitada: bool = True
    # Eclipses por ángulo beta: fracción del día en eclipse estimada como
    # f(beta, altitud). Si es False, la recarga solar es la nominal
    # completa todos los días.
    eclipses_habilitados: bool = True
    # Radio del Sol proyectado sobre la esfera celeste (rad), para el
    # cálculo del ángulo de penumbra en los eclipses.
    radio_angular_sol_rad: float = 0.00465
    # Cobertura mínima de recarga: aún en eclipse total, los paneles
    # entregan una fracción mínima (albedo/reflejos) de la recarga nominal.
    fraccion_recarga_minima: float = 0.1

    # Tasas aditivas de falla (ver Satellite.calcular_lambda_total)
    factor_lambda_radiacion: float = 0.002
    factor_lambda_debris: float = 0.005
    umbral_energia_baja_pct: float = 20.0
    factor_lambda_energia: float = 0.0005
    umbral_combustible_bajo_pct: float = 10.0
    lambda_combustible_bajo: float = 0.002

    # Debris: energía de referencia para fragmentación (colisión/explosión)
    energia_referencia_colision_j: float = 5e10
    fragmentos_base_colision: float = 50.0
    fragmentos_base_explosion: float = 30.0

    # =========================================================================
    # MECANISMOS EVOLUTIVOS A NIVEL DE CONSTELACIÓN
    # =========================================================================

    # Selección dependiente de densidad (carrying capacity ecológico):
    # fracción máxima en que la congestión de una banda orbital reduce la
    # probabilidad de que un intento de reproducción prospere (ver
    # SimulacionSatelites._probabilidad_reproduccion_exitosa).
    factor_capacidad_carga: float = 0.6

    # Coevolución con el debris: reducción máxima de λ_debris para un
    # satélite con `maniobrabilidad_evasiva` = 1.0 y combustible
    # suficiente para maniobrar (ver Satellite.calcular_lambda_total).
    factor_reduccion_evasion_max: float = 0.7
    # Costo diario de combustible (puntos porcentuales) de las
    # maniobras de evasión de debris, proporcional a
    # `maniobrabilidad_evasiva` y solo cuando hay combustible sobre el
    # umbral (trade-off evitación vs. longevidad de maniobras).
    costo_combustible_evasion_diario_pct: float = 0.1
    # =========================================================================
    # ESPECIACIÓN Y COBERTURA (Fase: ecología de la constelación)
    # =========================================================================
    # Aislamiento reproductivo por banda orbital (especiación alopátrica):
    # la recombinación sexual solo puede ocurrir entre padres de la misma
    # banda orbital; si no hay pareja compatible, cae a reproducción
    # asexual del padre sorteado.
    aislamiento_reproductivo_por_banda: bool = True
    # Peso del bono de fitness de cobertura de banda: satélites cuya banda
    # orbital está poco representada entre los candidatos reciben un bono
    # (0 = desactivado; el bono escala hasta 1 + peso para el nicho menos
    # poblado).
    peso_bono_cobertura_banda: float = 0.5
    # Mutación "saltacional" rara: con esta probabilidad por hijo, un rasgo
    # heredable (calidad, agresividad r/K, evasión) salta uniformemente a
    # un valor nuevo del rango completo en vez de mutar gaussianamente.
    # Permite escapar de óptimos locales en el paisaje evolutivo.
    prob_mutacion_saltacional: float = 0.01
    # Días consecutivos con riesgo de cascada > 1 (debris generado >
    # decaído) en una banda para declarar el "punto de no retorno" de
    # Kessler en esa banda.
    dias_sostenidos_punto_no_retorno: int = 30

    # Estrategias r/K (trade-off reproducción vs. longevidad): umbrales de
    # energía/combustible para calificar como candidato a reproducirse,
    # parametrizados según el rasgo heredable `agresividad_reproductiva`
    # (0=K-estrategia conservadora, 1=r-estrategia agresiva). El umbral
    # efectivo es `base - rango * agresividad_reproductiva`, de forma que
    # en agresividad=0.5 se recupera el umbral fijo original (50%/30%).
    umbral_energia_reproduccion_base: float = 70.0
    umbral_energia_reproduccion_rango: float = 40.0
    umbral_combustible_reproduccion_base: float = 45.0
    umbral_combustible_reproduccion_rango: float = 30.0


# =============================================================================
# MODELO ATMOSFÉRICO
# =============================================================================

# Modelo de atmósfera exponencial por capas (referencia estándar en
# astrodinámica, ver Vallado, "Fundamentals of Astrodynamics and
# Applications", tabla de densidad exponencial). Cada tupla es
# (altitud_base_km, densidad_base_kg_m3, altura_de_escala_km). La
# densidad a una altitud h dentro de una capa se calcula como
# rho0 * exp(-(h - h0) / H), lo que reproduce razonablemente bien el
# perfil real de densidad atmosférica entre ~100 y ~1000 km, la zona
# relevante para el drag orbital de satélites LEO.
_TABLA_ATMOSFERA_EXPONENCIAL: List[Tuple[float, float, float]] = [
    (0, 1.225, 7.249),
    (100, 5.297e-7, 5.877),
    (110, 9.661e-8, 7.263),
    (120, 2.438e-8, 9.473),
    (130, 8.484e-9, 12.636),
    (140, 3.845e-9, 16.149),
    (150, 2.070e-9, 22.523),
    (180, 5.464e-10, 29.740),
    (200, 2.789e-10, 37.105),
    (250, 7.248e-11, 45.546),
    (300, 2.418e-11, 53.628),
    (350, 9.518e-12, 53.298),
    (400, 3.725e-12, 58.515),
    (450, 1.585e-12, 60.828),
    (500, 6.967e-13, 63.822),
    (600, 1.454e-13, 71.835),
    (700, 3.614e-14, 88.667),
    (800, 1.170e-14, 124.64),
    (900, 5.245e-15, 181.05),
    (1000, 3.019e-15, 268.00),
]

# Arrays derivados de la tabla anterior, usados para el lookup
# vectorizado de densidad atmosférica (ver
# :func:`densidad_atmosferica_kg_m3_batch`).
_ALTITUDES_BASE = np.array([capa[0] for capa in _TABLA_ATMOSFERA_EXPONENCIAL])
_DENSIDADES_BASE = np.array([capa[1] for capa in _TABLA_ATMOSFERA_EXPONENCIAL])
_ESCALAS_ALTURA = np.array([capa[2] for capa in _TABLA_ATMOSFERA_EXPONENCIAL])


def densidad_atmosferica_kg_m3(altitud_km: float) -> float:
    """
    Densidad atmosférica estimada a una altitud dada, usando un modelo
    exponencial por capas (kg/m^3).

    Debajo de 100 km usa la capa base (troposfera/estratosfera); por
    encima de 1000 km reutiliza la última capa (exosfera, densidad ya
    despreciable para efectos de drag).

    Args:
        altitud_km: altitud sobre la superficie terrestre, en km.

    Returns:
        Densidad atmosférica en kg/m^3.
    """
    altitud_km = max(altitud_km, 0.0)
    capa = _TABLA_ATMOSFERA_EXPONENCIAL[0]
    for candidata in _TABLA_ATMOSFERA_EXPONENCIAL:
        if altitud_km >= candidata[0]:
            capa = candidata
        else:
            break
    altitud_base, densidad_base, altura_escala = capa
    return densidad_base * np.exp(-(altitud_km - altitud_base) / altura_escala)


def densidad_atmosferica_kg_m3_batch(altitudes_km: np.ndarray) -> np.ndarray:
    """
    Versión vectorizada para calcular la densidad atmosférica para un array de altitudes.

    Usa interpolación lineal en el espacio logarítmico de la densidad para evitar
    discontinuidades entre capas. Esto es más preciso que el modelo exponencial
    simple y más rápido que iterar sobre cada altitud individualmente.

    Args:
        altitudes_km: Array de altitudes en km.

    Returns:
        np.ndarray: Array de densidades atmosféricas en kg/m³, mismo shape que la entrada.
    """
    altitudes = np.clip(np.asarray(altitudes_km, dtype=float), 0.0, None)

    # Índices de la capa inferior para cada altitud
    indices = np.searchsorted(_ALTITUDES_BASE, altitudes, side="right") - 1
    indices = np.clip(indices, 0, len(_ALTITUDES_BASE) - 1)

    # Altitudes y densidades de las capas inferior y superior
    h0 = _ALTITUDES_BASE[indices]
    h1 = _ALTITUDES_BASE[np.clip(indices + 1, 0, len(_ALTITUDES_BASE) - 1)]
    rho0 = _DENSIDADES_BASE[indices]
    rho1 = _DENSIDADES_BASE[np.clip(indices + 1, 0, len(_ALTITUDES_BASE) - 1)]
    H = _ESCALAS_ALTURA[indices]

    # Interpolación lineal en log(rho) para evitar discontinuidades
    log_rho = np.log(rho0) + (np.log(rho1) - np.log(rho0)) * (altitudes - h0) / (h1 - h0 + 1e-10)

    # Aplicar decaimiento exponencial dentro de cada capa
    return np.exp(log_rho) * np.exp(-(altitudes - h0) / H)

# =============================================================================
# SISTEMAS AMBIENTALES
# =============================================================================

class CicloSolar:
    """Modela ciclos solares (11 años) que afectan drag y radiación."""
    
    def __init__(self, fase_inicial: float = 0.5):
        """
        Args:
            fase_inicial: fase del ciclo (0 a 1, siendo 0=mínimo, 1=máximo)
        """
        self.fase = fase_inicial
        self.ciclo_dias = 11 * 365.25  # ciclo de 11 años
    
    def actividad_solar(self, dias_transcurridos: float) -> float:
        """
        Retorna actividad solar normalizada (0 a 1).
        
        Args:
            dias_transcurridos: días desde inicio de simulación
            
        Returns:
            actividad solar (0=mínima, 1=máxima)
        """
        posicion_ciclo = (dias_transcurridos / self.ciclo_dias) % 1.0
        # Ciclo sinusoidal
        actividad = 0.5 + 0.5 * np.sin(2 * np.pi * posicion_ciclo)
        return actividad
    
    def presion_radiacion_solar(self, actividad: float) -> float:
        """Presión de radiación solar (afecta órbitas con baja masa)."""
        # Pa (Pascal) normalizado
        presion_base = 4.5e-6  # Pa
        return presion_base * (0.5 + 1.5 * actividad)
    
    def drag_atmosferico(self, actividad: float) -> float:
        """Coeficiente de drag atmosférico normalizado."""
        # Mayor actividad solar → mayor expansión atmosférica → más drag
        return 1.0 + 2.0 * actividad


class RadiacionCosmica:
    """Modela daño acumulativo por radiación cósmica."""
    
    def __init__(self, tasa_base: float = 0.001):
        """
        Args:
            tasa_base: tasa de daño por día (0 a 1)
        """
        self.tasa_base = tasa_base
    
    def damage_diario(self) -> float:
        """Retorna daño acumulativo de un día."""
        return self.tasa_base


class DebrisEspacial:
    """
    Gestiona población de debris que pueden colisionar con satélites.

    A diferencia de un modelo de densidad global única, la población de
    debris se segmenta por banda orbital (inspirado en modelos de
    referencia como NASA ORDEM / ESA MASTER, que discretizan el flujo
    de debris por altitud e inclinación). Cada banda tiene su propia
    densidad, su propia tasa de decaimiento natural (arrastre
    atmosférico: las bandas bajas se "autolimpian" mucho más rápido que
    GEO) y su propio riesgo de colisión asociado.

    Bandas modeladas:
        - "bajo" (<400 km): decae muy rápido por drag atmosférico.
        - "leo_congestionado" (400-2000 km): la zona más poblada de
          debris rastreable (jerga: "LEO congestionado").
        - "meo" (2000-35786 km): decaimiento muy lento (siglos).
        - "geo" (>35786 km): decaimiento casi nulo (milenios).

    Attributes:
        densidad_por_banda: cantidad de objetos de debris estimados por
            banda orbital.
        colisiones_totales: contador acumulado de colisiones satélite-debris.
    """

    # (nombre_banda, altitud_min_km, altitud_max_km, tasa_decaimiento_diaria)
    _BANDAS: List[Tuple[str, float, float, float]] = [
        ("bajo", 0.0, 400.0, 0.010),
        ("leo_congestionado", 400.0, 2000.0, 0.0002),
        ("meo", 2000.0, 35786.0, 0.00002),
        ("geo", 35786.0, float("inf"), 0.000005),
    ]

    # Escala de normalización (0-1) y riesgo base de colisión por banda
    _ESCALA_NORMALIZACION = {
        "bajo": 200.0,
        "leo_congestionado": 1500.0,
        "meo": 500.0,
        "geo": 200.0,
    }
    _RIESGO_BASE = {
        "bajo": 0.0005,
        "leo_congestionado": 0.05,
        "meo": 0.01,
        "geo": 0.0005,
    }

    def __init__(self, densidad_inicial: int = 100, config: Optional[ConfiguracionFisica] = None):
        """
        Args:
            densidad_inicial: cantidad inicial total de debris,
                distribuida entre bandas según su congestión típica
                (mayormente en "leo_congestionado").
            config: configuración física centralizada (ver
                :class:`ConfiguracionFisica`); si es None, se usan los
                valores por defecto.
        """
        self.config = config or ConfiguracionFisica()
        proporciones = {"bajo": 0.05, "leo_congestionado": 0.70,
                       "meo": 0.20, "geo": 0.05}
        self.densidad_por_banda: Dict[str, float] = {
            banda: densidad_inicial * proporciones[banda]
            for banda, _, _, _ in self._BANDAS
        }
        self.generado_ultimo_dia: Dict[str, float] = {
            banda: 0.0 for banda, _, _, _ in self._BANDAS
        }
        self.ultimo_riesgo_cascada: Dict[str, float] = {
            banda: 0.0 for banda, _, _, _ in self._BANDAS
        }
        self.colisiones_totales = 0

    @property
    def densidad(self) -> float:
        """Cantidad total de debris sumando todas las bandas (compatibilidad)."""
        return sum(self.densidad_por_banda.values())

    def _banda_de_altitud(self, altitud_km: float) -> str:
        """Determina a qué banda orbital pertenece una altitud dada."""
        for banda, alt_min, alt_max, _ in self._BANDAS:
            if alt_min <= altitud_km < alt_max:
                return banda
        return self._BANDAS[-1][0]

    def banda_de_altitud(self, altitud_km: float) -> str:
        """
        Versión pública de :meth:`_banda_de_altitud`: banda orbital
        ("bajo", "leo_congestionado", "meo" o "geo") a la que pertenece
        una altitud dada. Usado por `SimulacionSatelites` para criterios
        de selección dependientes de densidad (carrying capacity).
        """
        return self._banda_de_altitud(altitud_km)

    def densidad_normalizada_banda(self, banda: str) -> float:
        """Densidad normalizada (0 a 1) de una banda orbital específica."""
        escala = self._ESCALA_NORMALIZACION[banda]
        return min(self.densidad_por_banda[banda] / escala, 1.0)

    def densidad_normalizada(self) -> float:
        """
        Densidad normalizada (0 a 1) "headline", usando la banda LEO
        congestionada como referencia principal (la más relevante para
        la mayoría de los satélites simulados).
        """
        return self.densidad_normalizada_banda("leo_congestionado")

    def probabilidad_colision(self, altitud_km: float) -> float:
        """
        Calcula probabilidad de colisión diaria según la altitud del
        satélite y la densidad de debris de su banda orbital.

        Args:
            altitud_km: altitud del satélite sobre la Tierra.

        Returns:
            probabilidad de colisión diaria (0 a 0.1).
        """
        banda = self._banda_de_altitud(altitud_km)
        riesgo_base = self._RIESGO_BASE[banda] * self.densidad_normalizada_banda(banda)
        return min(riesgo_base, 0.1)  # máximo 10% de probabilidad

    def probabilidad_colision_batch(self, altitudes_km: np.ndarray) -> np.ndarray:
        """
        Versión vectorizada de :meth:`probabilidad_colision` para calcular
        la probabilidad de colisión diaria de una población completa de
        satélites en una sola pasada (Fase 3: rendimiento y escala).

        En vez de resolver la banda orbital y hacer lookups por
        satélite en un loop de Python, se recorre solo la lista de
        bandas (constante, ~4 elementos) aplicando una máscara booleana
        por banda sobre el array completo de altitudes.

        Args:
            altitudes_km: array de altitudes (km) de todos los satélites.

        Returns:
            array de probabilidades de colisión diaria (0 a 0.1), mismo
            orden/tamaño que `altitudes_km`.
        """
        altitudes = np.asarray(altitudes_km, dtype=float)
        resultado = np.zeros_like(altitudes)
        procesadas = np.zeros(altitudes.shape, dtype=bool)
        for banda, alt_min, alt_max, _ in self._BANDAS:
            mascara = (altitudes >= alt_min) & (altitudes < alt_max)
            riesgo_banda = min(
                self._RIESGO_BASE[banda] * self.densidad_normalizada_banda(banda), 0.1
            )
            resultado[mascara] = riesgo_banda
            procesadas |= mascara
        # Igual que en la version escalar (_banda_de_altitud), las
        # altitudes fuera de toda banda (p. ej. negativas por un semieje
        # degradado) usan la ultima banda como fallback, en vez de
        # quedar con riesgo 0 y divergir de la version escalar.
        banda_fallback = self._BANDAS[-1][0]
        riesgo_fallback = min(
            self._RIESGO_BASE[banda_fallback]
            * self.densidad_normalizada_banda(banda_fallback),
            0.1,
        )
        resultado[~procesadas] = riesgo_fallback
        return resultado

    def generar_debris_por_colision(self, altitud_km: float, masa_kg: float,
                                    velocidad_kms: float):
        """
        Genera fragmentos de debris proporcional a la energía cinética
        de la colisión (masa × velocidad_relativa²), en lugar de una
        cantidad fija. Modela el efecto Kessler: colisiones a alta
        velocidad/masa generan cascadas de fragmentos mucho mayores
        (ej. Cosmos-Iridium 2009: >2000 fragmentos rastreables).

        Args:
            altitud_km: altitud a la que ocurrió la colisión.
            masa_kg: masa del satélite involucrado.
            velocidad_kms: velocidad orbital relativa aproximada (km/s).
        """
        energia_j = 0.5 * masa_kg * (velocidad_kms * 1000.0) ** 2
        # Referencia: 1000 kg a 10 km/s ≈ 5e10 J → ~50 fragmentos base
        energia_referencia_j = self.config.energia_referencia_colision_j
        fragmentos_base = self.config.fragmentos_base_colision
        fragmentos = fragmentos_base * (energia_j / energia_referencia_j)
        fragmentos = float(np.clip(fragmentos, 1.0, 3000.0))

        banda = self._banda_de_altitud(altitud_km)
        self.densidad_por_banda[banda] += fragmentos
        self.generado_ultimo_dia[banda] += fragmentos

    def generar_debris_derelicto(self, altitud_km: float, cantidad: float = 1.0):
        """
        Añade debris por un satélite que queda inactivo sin fragmentarse
        (falla, fin de vida, falta de energía, etc.): el objeto derelicto
        en sí mismo pasa a ser un objeto rastreable más en su banda.

        Args:
            altitud_km: altitud a la que quedó el satélite inactivo.
            cantidad: cantidad de objetos de debris a añadir (por defecto 1,
                representando el satélite derelicto mismo).
        """
        banda = self._banda_de_altitud(altitud_km)
        self.densidad_por_banda[banda] += cantidad
        self.generado_ultimo_dia[banda] += cantidad

    def generar_debris_por_explosion(self, altitud_km: float, masa_kg: float,
                                     fraccion_combustible_residual: float):
        """
        Genera debris por explosión/fragmentación de un satélite que
        finalizó su misión sin ser pasivado y aún conserva combustible o
        presión residual (riesgo real documentado en misiones que no
        siguen buenas prácticas de fin de vida). La cantidad de
        fragmentos escala con la masa del satélite y la fracción de
        combustible residual (proxy de la energía almacenada que puede
        liberarse), siendo más severa que la de un derelicto simple pero
        menor que una colisión de alta energía típica.

        Args:
            altitud_km: altitud a la que ocurrió la explosión.
            masa_kg: masa del satélite involucrado.
            fraccion_combustible_residual: combustible residual (0 a 1).
        """
        fragmentos_base = self.config.fragmentos_base_explosion  # ref: 1000 kg, 100% combustible
        fragmentos = fragmentos_base * (masa_kg / 1000.0) * fraccion_combustible_residual
        fragmentos = float(np.clip(fragmentos, 1.0, 1500.0))

        banda = self._banda_de_altitud(altitud_km)
        self.densidad_por_banda[banda] += fragmentos
        self.generado_ultimo_dia[banda] += fragmentos

    def remover_debris(self, cantidad: float, banda: Optional[str] = None) -> float:
        """
        Remueve debris activamente (ADR - Active Debris Removal),
        simulando misiones de remoción como ClearSpace-1. Si no se
        especifica banda, prioriza automáticamente la banda con mayor
        densidad normalizada (la más congestionada/riesgosa).

        Args:
            cantidad: cantidad de objetos de debris a remover.
            banda: banda específica a targetear, o None para
                autoseleccionar la banda más congestionada.

        Returns:
            Cantidad de debris efectivamente removido (limitada por la
            disponibilidad en la banda seleccionada).
        """
        if banda is None:
            banda = max(self.densidad_por_banda,
                       key=lambda b: self.densidad_normalizada_banda(b))
        removido = min(cantidad, self.densidad_por_banda[banda])
        self.densidad_por_banda[banda] -= removido
        return removido

    def aplicar_decaimiento_diario(self) -> Dict[str, float]:
        """
        Aplica el decaimiento natural (arrastre atmosférico y
        perturbaciones orbitales) a la densidad de debris de cada banda,
        y calcula el "riesgo de cascada" de Kessler: la razón entre
        debris generado y debris removido en el último día para cada
        banda. Un valor > 1 indica que la población de debris de esa
        banda está creciendo netamente (riesgo de cascada); < 1 indica
        que el decaimiento natural supera a la generación.

        Returns:
            Diccionario {banda: riesgo_cascada} para cada banda orbital.
        """
        riesgo: Dict[str, float] = {}
        for banda, _, _, tasa_decaimiento in self._BANDAS:
            densidad_actual = self.densidad_por_banda[banda]
            decaido = densidad_actual * tasa_decaimiento
            self.densidad_por_banda[banda] = max(0.0, densidad_actual - decaido)

            generado = self.generado_ultimo_dia[banda]
            riesgo[banda] = generado / (decaido + 1e-6)
            self.generado_ultimo_dia[banda] = 0.0

        self.ultimo_riesgo_cascada = riesgo
        return riesgo


# =============================================================================
# SATÉLITE MEJORADO
# =============================================================================

class Satellite:
    """
    Satélite avanzado con metabolismo, heredabilidad y degradación.

    Modela un satélite individual con parámetros orbitales keplerianos,
    un "metabolismo" simplificado (energía/combustible), daño acumulado
    por radiación y degradación orbital, y una calidad inherente que se
    hereda (con mutación) a sus satélites "hijo".

    Attributes:
        id: identificador único, autoincremental.
        a, e, i, Omega, omega, M: parámetros keplerianos (a en km,
            ángulos en grados).
        masa_kg: masa del satélite en kg.
        area_seccion: área de sección transversal simplificada (masa/1000).
        calidad_inherente: calidad de construcción (0.1 a 1.0). A mayor
            calidad, menor tasa de falla inherente.
        lambda_inherente: tasa de falla base derivada de la calidad,
            en fallos/día.
        combustible: nivel de combustible para maniobras (0-100%).
        energia: nivel de batería/energía (0-100%).
        radiacion_acumulada: daño acumulado por radiación cósmica (0-1).
        degradacion_orbital: degradación orbital acumulada por drag (0-1).
        tiempo_lanzamiento: fecha/hora de lanzamiento del satélite.
        activo: True mientras el satélite siga operativo.
        causa_muerte: motivo de la falla una vez que `activo` es False.
        pariente_id: id del satélite "padre", si fue reproducido.
        pasivado: True si el satélite fue pasivado (combustible/batería
            purgados) al fin de su misión, práctica estándar de
            mitigación de debris que evita explosiones post-misión.
    """
    
    _contador = 0

    # `__slots__` evita crear un `__dict__` por instancia: con poblaciones
    # de miles de satélites esto reduce notablemente la memoria por objeto
    # y acelera el acceso a atributos (usado intensivamente en las
    # funciones `*_batch` que recolectan atributos de toda la población).
    __slots__ = (
        'id', 'config', 'a', 'e', 'i', 'Omega', 'omega', 'M',
        'masa_kg', 'area_seccion', 'calidad_inherente', 'lambda_inherente',
        'combustible', 'energia', 'consumo_energico_diario',
        'radiacion_acumulada', 'limite_radiacion', 'degradacion_orbital',
        'tiempo_lanzamiento', 'activo', 'causa_muerte', 'pariente_id',
        'pasivado', 'agresividad_reproductiva', 'maniobrabilidad_evasiva',
        'pariente_id_secundario',
    )
    
    def __init__(self, a: float, e: float, i: float, Omega: float, 
                 omega: float, M: float, tiempo_lanzamiento: datetime,
                 combustible: float = 100.0, energia: float = 100.0,
                 masa_kg: float = 1000.0, pariente_id: Optional[int] = None,
                 calidad_inherente: Optional[float] = None,
                 config: Optional[ConfiguracionFisica] = None,
                 rng: Optional[np.random.Generator] = None,
                 agresividad_reproductiva: Optional[float] = None,
                 maniobrabilidad_evasiva: Optional[float] = None):
        """
        Args:
            a, e, i, Omega, omega, M: parámetros keplerianos
            tiempo_lanzamiento: datetime de lanzamiento
            combustible: 0-100 (%)
            energia: 0-100 (%)
            masa_kg: masa del satélite
            pariente_id: ID del satélite "padre" (para heredabilidad)
            calidad_inherente: calidad de construcción (0.1 a 1.0)
            config: configuración física centralizada (ver
                :class:`ConfiguracionFisica`); si es None, se usan los
                valores por defecto.
            rng: generador aleatorio (`np.random.Generator`) usado para
                sortear la calidad inherente y los rasgos heredables
                cuando no se especifican explícitamente. Si es None, se
                usa un generador de fallback a nivel de módulo (no
                reproducible). Al crear satélites desde
                `SimulacionSatelites` siempre se pasa su `self.rng`
                reproducible.
            agresividad_reproductiva: rasgo heredable de estrategia r/K
                (0=K-estrategia conservadora, 1=r-estrategia agresiva)
                que determina qué tan exigentes son los umbrales de
                energía/combustible para calificar como candidato a
                reproducirse (ver
                :meth:`SimulacionSatelites._califica_para_reproduccion`).
                Si es None, se sortea uniformemente entre 0 y 1.
            maniobrabilidad_evasiva: rasgo heredable de evasión de
                debris (0=sin capacidad, 1=máxima); reduce λ_debris si
                el satélite tiene combustible suficiente para maniobrar
                (ver :meth:`calcular_lambda_total`). Si es None, se
                sortea uniformemente entre 0 y 1.
        """
        Satellite._contador += 1
        self.id = Satellite._contador
        self.config = config or ConfiguracionFisica()
        
        # Parámetros Keplerianos
        self.a = a  # semieje mayor (km)
        self.e = e  # excentricidad
        self.i = i  # inclinación (grados)
        self.Omega = Omega  # RAAN (grados)
        self.omega = omega  # argumento del perigeo (grados)
        self.M = M  # anomalía media (grados)
        
        # Propiedades físicas
        self.masa_kg = masa_kg
        self.area_seccion = masa_kg / 1000.0  # área simplificada
        
        # =====================================================================
        # VARIABILIDAD INHERENTE: Cada satélite nace con diferente calidad
        # =====================================================================
        if calidad_inherente is None:
            # Distribución normal: mayoría mediocres, algunos buenos/malos
            rng_efectivo = rng if rng is not None else _RNG_POR_DEFECTO
            self.calidad_inherente = np.clip(rng_efectivo.normal(0.5, 0.15), 0.1, 1.0)
        else:
            self.calidad_inherente = np.clip(calidad_inherente, 0.1, 1.0)
        rng_efectivo = rng if rng is not None else _RNG_POR_DEFECTO
        
        # Tasa de falla inherente (inversamente proporcional a calidad)
        # Satélite excelente (1.0): lambda_inherente ≈ 0.0001/día
        # Satélite pobre (0.1): lambda_inherente ≈ 0.0009/día
        self.lambda_inherente = (1.0 - self.calidad_inherente) * 0.001
        
        # Metabolismo
        self.combustible = combustible  # para maniobras
        self.energia = energia  # batería solar
        self.consumo_energico_diario = self.config.consumo_energico_diario_pct  # % por día en operación
        
        # Radiación cósmica
        self.radiacion_acumulada = 0.0  # 0 a 1
        self.limite_radiacion = self.config.limite_radiacion
        
        # Degradación orbital
        self.degradacion_orbital = 0.0  # 0 a 1
        
        # Estado
        self.tiempo_lanzamiento = tiempo_lanzamiento
        self.activo = True
        self.causa_muerte: Optional[str] = None
        self.pariente_id = pariente_id  # heredabilidad
        self.pariente_id_secundario: Optional[int] = None  # segundo padre (reproducción sexual)
        
        # =====================================================================
        # RASGOS HEREDABLES DE MECANISMOS EVOLUTIVOS A NIVEL DE CONSTELACIÓN
        # =====================================================================
        if agresividad_reproductiva is None:
            # Estrategia r/K: sorteo uniforme (0=K-conservador, 1=r-agresivo)
            self.agresividad_reproductiva = float(rng_efectivo.uniform(0.0, 1.0))
        else:
            self.agresividad_reproductiva = float(np.clip(agresividad_reproductiva, 0.0, 1.0))
        
        if maniobrabilidad_evasiva is None:
            # Capacidad de evasión de debris (coevolución depredador-presa)
            self.maniobrabilidad_evasiva = float(rng_efectivo.uniform(0.0, 1.0))
        else:
            self.maniobrabilidad_evasiva = float(np.clip(maniobrabilidad_evasiva, 0.0, 1.0))
        
        # Mitigación de debris: si el satélite fue pasivado (purga de
        # combustible/batería residual) al fin de su misión.
        self.pasivado = False
    
    def edad_dias(self, tiempo_actual: datetime) -> float:
        """Calcula edad en días."""
        return (tiempo_actual - self.tiempo_lanzamiento).total_seconds() / (24 * 3600)
    
    def altitud_km(self) -> float:
        """Altitud sobre superficie terrestre."""
        return self.a - self.config.radio_tierra_km
    
    def distancia_perigeo(self) -> float:
        """Distancia del perigeo en km."""
        r_p = self.a * (1 - self.e)
        return r_p - self.config.radio_tierra_km
    
    def distancia_apogeo(self) -> float:
        """Distancia del apogeo en km."""
        r_a = self.a * (1 + self.e)
        return r_a - self.config.radio_tierra_km
    
    def periodo_orbital(self) -> float:
        """Período orbital en horas."""
        T_segundos = 2 * np.pi * np.sqrt(self.a**3 / self.config.gm_tierra_km3_s2)
        return T_segundos / 3600
    
    def velocidad_orbital_kms(self) -> float:
        """
        Velocidad orbital instantánea aproximada (ecuación vis-viva),
        asumiendo órbita cuasi-circular (r ≈ a), en km/s.

        Returns:
            Velocidad orbital en km/s.
        """
        return float(np.sqrt(self.config.gm_tierra_km3_s2 / self.a))
    
    # =========================================================================
    # DEGRADACIÓN ORBITAL
    # =========================================================================
    
    def aplicar_drag_atmosferico(self, ciclo_solar: CicloSolar, dias_transcurridos: float):
        """
        Reduce semieje mayor por drag atmosférico usando la ecuación
        física real de decaimiento orbital.

        Se calcula la deceleración por drag en el perigeo/altitud actual
        como α₀ = ½·ρ(R)·v²·cd·A/m (ecuación estándar de arrastre
        atmosférico) y se obtiene la tasa de decaimiento del semieje
        mayor promediada en una órbita como dR/dt = α₀·T/π, donde T es
        el período orbital. La densidad atmosférica ρ(R) se obtiene de
        un modelo exponencial por capas (:func:`densidad_atmosferica_kg_m3`)
        modulado por la actividad solar (mayor actividad → atmósfera más
        expandida → más drag). Esta formulación es la misma usada por
        herramientas de referencia como el propagador de decaimiento
        orbital ORBITM, validada contra telemetría real con ~5% de error.

        Efecto secundario: si la altitud resultante cae por debajo de
        100 km, marca al satélite como inactivo con causa "reentrada_orbital".

        Args:
            ciclo_solar: instancia de CicloSolar usada para obtener la
                actividad solar actual y el coeficiente de drag asociado.
            dias_transcurridos: días desde el inicio de la simulación.
        """
        actividad = ciclo_solar.actividad_solar(dias_transcurridos)
        factor_solar = ciclo_solar.drag_atmosferico(actividad)  # 1.0 a 3.0
        
        cd = self.config.coeficiente_drag_cd  # coeficiente de forma típico (adimensional)
        area_m2 = self.area_seccion  # aproximación simplificada (m^2)
        
        # Densidad atmosférica real a la altitud actual, ajustada por
        # la expansión atmosférica asociada a la actividad solar.
        rho = densidad_atmosferica_kg_m3(self.altitud_km()) * factor_solar
        
        # Velocidad orbital (m/s) y período orbital (s)
        v_ms = self.velocidad_orbital_kms() * 1000.0
        T_s = self.periodo_orbital() * 3600.0
        
        # Deceleración por drag: α₀ = ½·ρ·v²·cd·A/m  (m/s^2)
        alpha_0 = 0.5 * rho * v_ms**2 * cd * area_m2 / self.masa_kg
        
        # Tasa de decaimiento del semieje mayor promediada en una órbita
        # dR/dt = α₀·T/π  (m/s) → integrada un día completo (86400 s)
        dR_dt_m_s = alpha_0 * T_s / np.pi
        perdida_a_km = (dR_dt_m_s * 86400.0) / 1000.0  # km perdidos en 1 día
        
        self.a -= perdida_a_km
        self.degradacion_orbital += perdida_a_km / 100  # track degradación
        
        # Reentrada orbital
        if self.altitud_km() < 100:
            self.activo = False
            self.causa_muerte = "reentrada_orbital"
    
    # =========================================================================
    # PRESIÓN DE RADIACIÓN SOLAR
    # =========================================================================
    
    def aplicar_presion_radiacion_solar(self, ciclo_solar: CicloSolar, 
                                        dias_transcurridos: float):
        """
        Afecta principalmente satélites ligeros.
        Puede cambiar órbita ligeramente.

        Incrementa levemente la excentricidad en función de la relación
        área/masa del satélite y de la presión de radiación solar actual
        (limitada a un máximo de 0.3).

        Args:
            ciclo_solar: instancia de CicloSolar usada para obtener la
                presión de radiación solar según la actividad actual.
            dias_transcurridos: días desde el inicio de la simulación.
        """
        actividad = ciclo_solar.actividad_solar(dias_transcurridos)
        presion = ciclo_solar.presion_radiacion_solar(actividad)
        
        # Satélites más ligeros se ven más afectados
        area_masa_ratio = self.area_seccion / self.masa_kg
        
        # Efecto en excentricidad (aumenta muy levemente)
        delta_e = area_masa_ratio * presion * 1e-6
        self.e += delta_e
        self.e = min(self.e, 0.3)  # limitar excentricidad
    
    # =========================================================================
    # RADIACIÓN CÓSMICA
    # =========================================================================
    
    def aplicar_radiacion_cosmica(self, radiacion_cosmica: RadiacionCosmica):
        """
        Acumula daño de radiación cósmica.

        Si el daño acumulado alcanza `limite_radiacion`, marca al
        satélite como inactivo con causa "radiacion_cosmica".

        Args:
            radiacion_cosmica: instancia de RadiacionCosmica que provee
                el daño diario a acumular.
        """
        damage = radiacion_cosmica.damage_diario()
        self.radiacion_acumulada += damage
        
        if self.radiacion_acumulada >= self.limite_radiacion:
            self.activo = False
            self.causa_muerte = "radiacion_cosmica"
    
    # =========================================================================
    # METABOLISMO
    # =========================================================================
    
    def consumir_energia(self) -> bool:
        """
        Consume energía diariamente.
        Retorna False si se queda sin energía.
        """
        self.energia -= self.consumo_energico_diario
        if self.energia <= 0:
            self.activo = False
            self.causa_muerte = "falta_energia"
            return False
        return True
    
    def recargar_energia(self, cantidad: float):
        """
        Recarga energía (luz solar), sin superar el 100%.

        Args:
            cantidad: puntos porcentuales de energía a recargar.
        """
        self.energia = min(self.energia + cantidad, 100.0)
    
    def maniobra_orbital(self, delta_a: float, delta_e: float) -> bool:
        """
        Realiza maniobra orbital consumiendo combustible.
        
        Args:
            delta_a, delta_e: cambios en semieje y excentricidad
            
        Returns:
            True si la maniobra fue exitosa
        """
        costo_combustible = abs(delta_a) * 0.5 + abs(delta_e) * 10.0
        
        if self.combustible >= costo_combustible:
            self.combustible -= costo_combustible
            self.a += delta_a
            self.e += delta_e
            return True
        return False
    
    def recargar_combustible(self, cantidad: float):
        """
        Recarga tanque de combustible, sin superar el 100%.

        Args:
            cantidad: puntos porcentuales de combustible a recargar.
        """
        self.combustible = min(self.combustible + cantidad, 100.0)
    
    # =========================================================================
    # MITIGACIÓN DE DEBRIS: PASIVACIÓN Y DESORBITADO CONTROLADO
    # =========================================================================
    
    def pasivar(self):
        """
        Pasiva el satélite al fin de su vida útil: purga el combustible
        y la energía residual para prevenir explosiones/fragmentaciones
        post-misión (práctica estándar de mitigación de debris, ej.
        directrices IADC/ISO 24113).
        """
        self.combustible = 0.0
        self.energia = 0.0
        self.pasivado = True
    
    def intentar_desorbitado_controlado(self, altitud_objetivo_km: float = 300.0) -> bool:
        """
        Intenta una maniobra de desorbitación controlada al fin de la
        misión, reduciendo el semieje mayor hacia una altitud objetivo
        donde el decaimiento natural por drag es rápido, para cumplir
        la "regla de los 25 años" (el satélite debe reentrar dentro de
        25 años desde el fin de su misión). Reutiliza :meth:`maniobra_orbital`.

        Args:
            altitud_objetivo_km: altitud objetivo tras la maniobra (por
                defecto 300 km).

        Returns:
            True si la maniobra se pudo ejecutar (combustible suficiente
            o el satélite ya estaba por debajo de la altitud objetivo).
        """
        delta_a = altitud_objetivo_km - self.altitud_km()
        if delta_a >= 0:
            return True  # ya está por debajo de la altitud objetivo
        return self.maniobra_orbital(delta_a, 0.0)
    
    # =========================================================================
    # HEREDABILIDAD
    # =========================================================================
    
    # =========================================================================
    # CÁLCULO DE TASAS ADITIVAS DE FALLA
    # =========================================================================
    
    def calcular_lambda_total(self, edad_dias: float, 
                             riesgo_debris: float) -> float:
        """
        Calcula tasa total de falla como SUMA de mecanismos independientes.
        
        Cada mecanismo contribuye con su propia tasa:
        λ_total = λ_inherente + λ_envejecimiento + λ_radiación + 
                  λ_debris + λ_energía + λ_combustible
        
        Args:
            edad_dias: edad en días
            riesgo_debris: probabilidad base de colisión (0-1)
            
        Returns:
            tasa de falla total (en unidades de fallos por día)
        """
        
        # 1. λ_INHERENTE: Constante para cada satélite (su "genética")
        lambda_inherente = self.lambda_inherente
        
        # 2. λ_ENVEJECIMIENTO: Curva de bañera Weibull
        # Baja al inicio (rodaje), sube con edad (wear-out)
        escala_dias = self.config.escala_envejecimiento_anos * 365.25  # años en días
        k_forma = self.config.forma_envejecimiento_k  # parámetro de forma
        
        if edad_dias == 0:
            lambda_envejecimiento = 0
        else:
            lambda_envejecimiento = (k_forma / escala_dias) * \
                                   (edad_dias / escala_dias) ** (k_forma - 1)
        
        # 3. λ_RADIACIÓN: Acumula daño con el tiempo
        # Mayor exposición = más daño acumulado
        lambda_radiacion = self.radiacion_acumulada * self.config.factor_lambda_radiacion
        
        # 4. λ_DEBRIS: Riesgo de colisión (ambiental), reducido por la
        # capacidad de evasión heredable del satélite (coevolución con
        # el debris: solo puede evadir si tiene combustible suficiente
        # para maniobrar; si no, el rasgo no tiene efecto).
        lambda_debris_base = riesgo_debris * self.config.factor_lambda_debris
        if self.combustible >= self.config.umbral_combustible_bajo_pct:
            reduccion_evasion = self.config.factor_reduccion_evasion_max * self.maniobrabilidad_evasiva
            lambda_debris = lambda_debris_base * (1.0 - reduccion_evasion)
        else:
            lambda_debris = lambda_debris_base
        
        # 5. λ_ENERGÍA: Si batería baja, sistemas fallan
        # Feedback metabolismo: energía baja = más fallos
        umbral_energia = self.config.umbral_energia_baja_pct
        if self.energia < umbral_energia:
            lambda_energia = (umbral_energia - self.energia) * self.config.factor_lambda_energia
        else:
            lambda_energia = 0
        
        # 6. λ_COMBUSTIBLE: Sin combustible no puede maniobrar
        # No puede evitar debris o ajustar órbita
        if self.combustible < self.config.umbral_combustible_bajo_pct:
            lambda_combustible = self.config.lambda_combustible_bajo
        else:
            lambda_combustible = 0
        
        # SUMA TOTAL DE TASAS
        lambda_total = (lambda_inherente + lambda_envejecimiento + 
                       lambda_radiacion + lambda_debris + 
                       lambda_energia + lambda_combustible)
        
        return lambda_total
    
    def probabilidad_falla_diaria_aditiva(self, edad_dias: float,
                                         riesgo_debris: float) -> float:
        """
        Convierte tasa total a probabilidad de falla en un día.
        
        P(falla en 1 día) = 1 - exp(-λ_total)
        
        Args:
            edad_dias, riesgo_debris: como calcular_lambda_total
            
        Returns:
            probabilidad de falla (0-1)
        """
        lambda_total = self.calcular_lambda_total(edad_dias, riesgo_debris)
        # Convertir tasa a probabilidad de falla diaria
        prob = 1 - np.exp(-lambda_total)
        return min(prob, 1.0)
    
    def fitness_reproductivo(self) -> float:
        """
        Aptitud reproductiva usada para la selección de padres
        proporcional a fitness (selección por aptitud diferencial, ver
        :meth:`SimulacionSatelites._pesos_seleccion_padres`), en vez de
        una selección uniforme entre candidatos. Hoy se basa únicamente
        en la calidad inherente; se expone como método separado para
        poder combinarla con más criterios en el futuro sin tocar el
        resto del código.

        Returns:
            Valor de aptitud (> 0), mayor es mejor.
        """
        return self.calidad_inherente
    
    def crear_hijo(self, tiempo_lanzamiento: datetime, 
                   tasa_mutacion: float = 0.05,
                   rng: Optional[np.random.Generator] = None) -> 'Satellite':
        """
        Crea un satélite hijo heredando parámetros con mutación.
        
        HEREDABILIDAD: El hijo hereda la calidad inherente del padre
        (con mutación pequeña)
        
        Args:
            tiempo_lanzamiento: datetime del lanzamiento del hijo
            tasa_mutacion: tasa de mutación (0-1)
            rng: generador aleatorio (`np.random.Generator`) usado para
                las mutaciones. Si es None, se usa un generador de
                fallback a nivel de módulo (no reproducible).
            
        Returns:
            nuevo Satellite
        """
        rng_efectivo = rng if rng is not None else _RNG_POR_DEFECTO

        # Heredar parámetros orbitales con mutación
        mutacion = lambda x: x + rng_efectivo.normal(0, tasa_mutacion * x)
        
        a_hijo = mutacion(self.a)
        e_hijo = np.clip(mutacion(self.e), 0.0, 0.3)
        i_hijo = np.clip(mutacion(self.i), 0, 90)
        Omega_hijo = mutacion(self.Omega) % 360
        omega_hijo = mutacion(self.omega) % 360
        M_hijo = rng_efectivo.uniform(0, 360)  # nuevo en órbita
        
        # =====================================================================
        # HEREDABILIDAD DE CALIDAD: Hijo hereda robustez del padre
        # =====================================================================
        # Pequeña variación genética en calidad
        calidad_hijo = self.calidad_inherente + rng_efectivo.normal(0, 0.03)
        calidad_hijo = np.clip(calidad_hijo, 0.1, 1.0)
        # Resultado: familias de satélites "buenos" o "malos"
        
        # Herencia de los rasgos r/K (agresividad_reproductiva) y de
        # evasión de debris (maniobrabilidad_evasiva), con mutación
        # gaussiana pequeña, igual que la calidad inherente.
        agresividad_hijo = np.clip(
            self.agresividad_reproductiva + rng_efectivo.normal(0, 0.05), 0.0, 1.0
        )
        evasion_hijo = np.clip(
            self.maniobrabilidad_evasiva + rng_efectivo.normal(0, 0.05), 0.0, 1.0
        )

        # Mutación saltacional rara: con prob
        # `prob_mutacion_saltacional`, un rasgo salta uniformemente a un
        # valor nuevo del rango completo (escapa de óptimos locales).
        p_salto = self.config.prob_mutacion_saltacional
        if p_salto > 0 and rng_efectivo.random() < p_salto:
            rasgo_salto = rng_efectivo.integers(0, 3)
            if rasgo_salto == 0:
                calidad_hijo = float(rng_efectivo.uniform(0.1, 1.0))
            elif rasgo_salto == 1:
                agresividad_hijo = float(rng_efectivo.uniform(0.0, 1.0))
            else:
                evasion_hijo = float(rng_efectivo.uniform(0.0, 1.0))
        
        # El hijo hereda propiedades físicas
        combustible_hijo = self.combustible * 0.9
        energia_hijo = self.energia * 0.9
        masa_hijo = self.masa_kg * rng_efectivo.uniform(0.9, 1.1)
        
        return Satellite(
            a_hijo, e_hijo, i_hijo, Omega_hijo, omega_hijo, M_hijo,
            tiempo_lanzamiento,
            combustible=combustible_hijo,
            energia=energia_hijo,
            masa_kg=masa_hijo,
            pariente_id=self.id,
            calidad_inherente=calidad_hijo,  # Heredar calidad
            config=self.config,
            rng=rng_efectivo,
            agresividad_reproductiva=agresividad_hijo,
            maniobrabilidad_evasiva=evasion_hijo
        )
    
    def crear_hijo_sexual(self, otro_padre: 'Satellite', tiempo_lanzamiento: datetime,
                         tasa_mutacion: float = 0.05,
                         rng: Optional[np.random.Generator] = None) -> 'Satellite':
        """
        Crea un satélite hijo por recombinación sexual (crossover) entre
        dos padres, en vez de clonar un único padre con mutación (ver
        :meth:`crear_hijo`). Cada rasgo heredable del hijo es una mezcla
        aleatoria (crossover aritmético, análogo a BLX-alpha: una
        combinación convexa con peso `alpha` sorteado por rasgo) de
        ambos padres, seguida de la misma mutación gaussiana que en la
        reproducción asexual. Aumenta la diversidad genética de la
        población frente a la clonación de un único linaje dominante.

        Args:
            otro_padre: segundo progenitor.
            tiempo_lanzamiento: datetime del lanzamiento del hijo.
            tasa_mutacion: tasa de mutación (0-1).
            rng: generador aleatorio; si es None, usa el de fallback del
                módulo.

        Returns:
            Nuevo Satellite, con `pariente_id` = id de este satélite (el
            primer padre) y `pariente_id_secundario` = id de `otro_padre`.
        """
        rng_efectivo = rng if rng is not None else _RNG_POR_DEFECTO

        def combinar(x: float, y: float) -> float:
            alpha = rng_efectivo.uniform(0.0, 1.0)
            base = alpha * x + (1 - alpha) * y
            return base + rng_efectivo.normal(0, tasa_mutacion * abs(base))

        a_hijo = combinar(self.a, otro_padre.a)
        e_hijo = np.clip(combinar(self.e, otro_padre.e), 0.0, 0.3)
        i_hijo = np.clip(combinar(self.i, otro_padre.i), 0, 90)
        Omega_hijo = combinar(self.Omega, otro_padre.Omega) % 360
        omega_hijo = combinar(self.omega, otro_padre.omega) % 360
        M_hijo = rng_efectivo.uniform(0, 360)

        calidad_hijo = np.clip(
            combinar(self.calidad_inherente, otro_padre.calidad_inherente), 0.1, 1.0
        )
        agresividad_hijo = np.clip(
            combinar(self.agresividad_reproductiva, otro_padre.agresividad_reproductiva), 0.0, 1.0
        )
        evasion_hijo = np.clip(
            combinar(self.maniobrabilidad_evasiva, otro_padre.maniobrabilidad_evasiva), 0.0, 1.0
        )

        # Mutación saltacional rara (igual que en la vía asexual):
        # con prob `prob_mutacion_saltacional`, un rasgo salta
        # uniformemente a un valor nuevo del rango completo.
        p_salto = self.config.prob_mutacion_saltacional
        if p_salto > 0 and rng_efectivo.random() < p_salto:
            rasgo_salto = rng_efectivo.integers(0, 3)
            if rasgo_salto == 0:
                calidad_hijo = float(rng_efectivo.uniform(0.1, 1.0))
            elif rasgo_salto == 1:
                agresividad_hijo = float(rng_efectivo.uniform(0.0, 1.0))
            else:
                evasion_hijo = float(rng_efectivo.uniform(0.0, 1.0))

        combustible_hijo = (self.combustible + otro_padre.combustible) / 2 * 0.9
        energia_hijo = (self.energia + otro_padre.energia) / 2 * 0.9
        masa_hijo = (self.masa_kg + otro_padre.masa_kg) / 2 * rng_efectivo.uniform(0.9, 1.1)

        hijo = Satellite(
            a_hijo, e_hijo, i_hijo, Omega_hijo, omega_hijo, M_hijo,
            tiempo_lanzamiento,
            combustible=combustible_hijo,
            energia=energia_hijo,
            masa_kg=masa_hijo,
            pariente_id=self.id,
            calidad_inherente=calidad_hijo,
            config=self.config,
            rng=rng_efectivo,
            agresividad_reproductiva=agresividad_hijo,
            maniobrabilidad_evasiva=evasion_hijo
        )
        hijo.pariente_id_secundario = otro_padre.id
        return hijo
    
    def info(self) -> Dict:
        """Información completa del satélite."""
        return {
            'id': self.id,
            'pariente_id': self.pariente_id,
            'pariente_id_secundario': self.pariente_id_secundario,
            'calidad_inherente': self.calidad_inherente,
            'lambda_inherente': self.lambda_inherente,
            'agresividad_reproductiva': self.agresividad_reproductiva,
            'maniobrabilidad_evasiva': self.maniobrabilidad_evasiva,
            'semieje_mayor_km': self.a,
            'excentricidad': self.e,
            'inclinacion_grados': self.i,
            'altitud_km': self.altitud_km(),
            'perigeo_km': self.distancia_perigeo(),
            'apogeo_km': self.distancia_apogeo(),
            'periodo_horas': self.periodo_orbital(),
            'masa_kg': self.masa_kg,
            'combustible': self.combustible,
            'energia': self.energia,
            'radiacion_acumulada': self.radiacion_acumulada,
            'degradacion_orbital': self.degradacion_orbital,
            'activo': self.activo,
            'causa_muerte': self.causa_muerte
        }


# =============================================================================
# OPERACIONES VECTORIZADAS SOBRE POBLACIONES DE SATÉLITES
# =============================================================================
#
# Las funciones de esta sección replican, para una lista completa de
# satélites, la matemática de los métodos equivalentes de `Satellite`
# (aplicar_drag_atmosferico, aplicar_presion_radiacion_solar,
# aplicar_radiacion_cosmica, consumir_energia/recargar_energia y
# calcular_lambda_total), pero operando sobre arrays de NumPy en vez de
# iterar en Python satélite por satélite. Esto evita miles de llamadas a
# sqrt/exp/pow individuales cuando la población crece (1000-5000+
# satélites), que es el cuello de botella real del loop diario.
#
# Cada función recibe la lista de satélites afectados, muta sus atributos
# in-place (igual que los métodos originales) y devuelve únicamente la
# información adicional que el llamador necesita (máscaras de
# muerte/reentrada, o el array de lambda total). La forma de los objetos
# `Satellite` no cambia, por lo que `visualizador.py` sigue funcionando
# sin modificaciones.

def aplicar_drag_atmosferico_batch(satelites: List['Satellite'], ciclo_solar: 'CicloSolar',
                                    dias_transcurridos: float,
                                    config: ConfiguracionFisica) -> np.ndarray:
    """
    Versión vectorizada de :meth:`Satellite.aplicar_drag_atmosferico` para
    una lista completa de satélites activos.

    Muta `a` y `degradacion_orbital` de cada satélite in-place, igual que
    el método original, pero calculando la densidad atmosférica, la
    velocidad orbital y el período orbital de toda la población en una
    sola pasada vectorizada.

    Args:
        satelites: lista de satélites activos a actualizar.
        ciclo_solar: instancia de CicloSolar (actividad solar global).
        dias_transcurridos: días desde el inicio de la simulación.
        config: configuración física compartida.

    Returns:
        Array booleano (mismo orden que `satelites`) indicando qué
        satélites reentraron (altitud resultante < 100 km).
    """
    n = len(satelites)
    if n == 0:
        return np.zeros(0, dtype=bool)

    a_arr = np.array([s.a for s in satelites], dtype=float)
    masa_arr = np.array([s.masa_kg for s in satelites], dtype=float)
    area_arr = np.array([s.area_seccion for s in satelites], dtype=float)

    actividad = ciclo_solar.actividad_solar(dias_transcurridos)
    factor_solar = ciclo_solar.drag_atmosferico(actividad)

    altitud_arr = a_arr - config.radio_tierra_km
    rho = densidad_atmosferica_kg_m3_batch(altitud_arr) * factor_solar

    v_ms = np.sqrt(config.gm_tierra_km3_s2 / a_arr) * 1000.0
    T_s = 2 * np.pi * np.sqrt(a_arr ** 3 / config.gm_tierra_km3_s2)

    alpha_0 = 0.5 * rho * v_ms ** 2 * config.coeficiente_drag_cd * area_arr / masa_arr
    dR_dt_m_s = alpha_0 * T_s / np.pi
    perdida_a_km = (dR_dt_m_s * 86400.0) / 1000.0

    nueva_a = a_arr - perdida_a_km

    for idx, sat in enumerate(satelites):
        sat.a = float(nueva_a[idx])
        sat.degradacion_orbital += float(perdida_a_km[idx]) / 100

    return (nueva_a - config.radio_tierra_km) < 100


def aplicar_presion_radiacion_solar_batch(satelites: List['Satellite'], ciclo_solar: 'CicloSolar',
                                          dias_transcurridos: float) -> None:
    """
    Versión vectorizada de :meth:`Satellite.aplicar_presion_radiacion_solar`.

    Muta `e` de cada satélite in-place.

    Args:
        satelites: lista de satélites a actualizar.
        ciclo_solar: instancia de CicloSolar.
        dias_transcurridos: días desde el inicio de la simulación.
    """
    n = len(satelites)
    if n == 0:
        return

    actividad = ciclo_solar.actividad_solar(dias_transcurridos)
    presion = ciclo_solar.presion_radiacion_solar(actividad)

    area_masa_ratio = np.array([s.area_seccion / s.masa_kg for s in satelites], dtype=float)
    delta_e = area_masa_ratio * presion * 1e-6

    for idx, sat in enumerate(satelites):
        sat.e = min(sat.e + float(delta_e[idx]), 0.3)


def aplicar_radiacion_cosmica_batch(satelites: List['Satellite'],
                                    radiacion_cosmica: 'RadiacionCosmica') -> np.ndarray:
    """
    Versión vectorizada de :meth:`Satellite.aplicar_radiacion_cosmica`.

    Muta `radiacion_acumulada` de cada satélite in-place.

    Args:
        satelites: lista de satélites a actualizar.
        radiacion_cosmica: instancia de RadiacionCosmica.

    Returns:
        Array booleano indicando qué satélites superaron su límite de
        radiación (deberían morir con causa "radiacion_cosmica").
    """
    n = len(satelites)
    if n == 0:
        return np.zeros(0, dtype=bool)

    damage = radiacion_cosmica.damage_diario()
    radiacion_arr = np.array([s.radiacion_acumulada for s in satelites], dtype=float) + damage
    limite_arr = np.array([s.limite_radiacion for s in satelites], dtype=float)

    for idx, sat in enumerate(satelites):
        sat.radiacion_acumulada = float(radiacion_arr[idx])

    return radiacion_arr >= limite_arr


def propagar_elementos_orbitales_batch(satelites: List['Satellite'],
                                       dt_dias: float,
                                       config: ConfiguracionFisica) -> None:
    """
    Propaga la anomalía media y aplica la precesión secular J2 de toda
    la población en una sola pasada vectorizada (Fase: fidelidad orbital).
    1. Anomalía media: dM = n·dt con n = sqrt(GM/a³) (movimiento
       kepleriano puro), en grados (M %= 360).
    2. Precesión nodal J2 (Vallado, "Fundamentals of Astrodynamics",
       ecuación estándar de precesión nodal):
       dOmega/dt = -3/2 · J2 · n · (Re/p)² · cos(i)
       con p = a(1-e²). Para órbitas retrógradas (i>90°) el RAAN
       retrocede; para i=90° (polares) no hay deriva.
    Muta `M` y `Omega` de cada satélite in-place. `omega` (argumento del
    perigeo) también precesa por J2, pero no afecta ninguno de los
    mecanismos de la simulación (los rasgos dependen de Omega, a, e, i),
    así que se omite por rendimiento.
    Args:
        satelites: lista de satélites a propagar.
        dt_dias: paso de tiempo en días.
        config: configuración física compartida.
    """
    n = len(satelites)
    if n == 0:
        return
    a_arr = np.array([s.a for s in satelites], dtype=float)
    e_arr = np.array([s.e for s in satelites], dtype=float)
    i_arr = np.radians(np.array([s.i for s in satelites], dtype=float))

    # Movimiento medio (rad/s) y avance de la anomalía media en grados
    n_rad_s = np.sqrt(config.gm_tierra_km3_s2 / a_arr ** 3)
    if config.propagar_anomalia_media:
        dM_grados = np.degrees(n_rad_s * dt_dias * 86400.0) % 360.0
        for idx, sat in enumerate(satelites):
            sat.M = (sat.M + float(dM_grados[idx])) % 360.0

    if not config.precesion_j2_habilitada:
        return
    # Precesión nodal: dOmega/dt = -3/2 J2 n (Re/p)^2 cos(i)
    p_arr = a_arr * (1.0 - e_arr ** 2)
    factor = -1.5 * J2_TIERRA * n_rad_s * (config.radio_tierra_km / p_arr) ** 2
    dOmega_rad = factor * np.cos(i_arr) * dt_dias * 86400.0
    dOmega_grados = np.degrees(dOmega_rad)
    for idx, sat in enumerate(satelites):
        sat.Omega = (sat.Omega + float(dOmega_grados[idx])) % 360.0


def fraccion_dia_en_eclipse(beta_abs: float, altitud_km: float,
                            config: ConfiguracionFisica) -> float:
    """
    Fracción del día que un satélite pasa en (pen)umbra de la Tierra,
    dado el ángulo beta (ángulo entre el plano orbital y la dirección
    Sol-Tierra) y la altitud de la órbita.
    Aproximación estándar para órbita circular (p. ej. Wertz, "Spacecraft
    Attitude Determination and Control"): el satélite entra en eclipse
    cuando el ángulo entre su posición y la dirección al Sol es menor
    que el ángulo de eclipses rho_e = arccos(Re / r) + radio_angular_sol.
    La fracción del día en eclipse es f = (pi - 2·beta) / (2·pi) cuando
    beta < rho_e, y 0 cuando beta >= rho_e (órbita siempre iluminada).
    Para beta=0 (plano orbital contiene al Sol: eclipse central máximo)
    la fracción depende del arco eclipsado.
    Args:
        beta_abs: valor absoluto del ángulo beta en radianes.
        altitud_km: altitud de la órbita (km).
        config: configuración física (radio terrestre, radio angular del Sol).
    Returns:
        Fracción del día en eclipse (0 a ~0.4 para LEO).
    """
    r = altitud_km + config.radio_tierra_km
    rho_eclipse = np.arccos(config.radio_tierra_km / r) + config.radio_angular_sol_rad
    if beta_abs >= rho_eclipse:
        return 0.0
    # Longitud de arco de la órbita que cae en sombra, sobre el total 2pi.
    # Para órbita circular el arco eclipsado es 2·(rho_e - beta), limitado
    # a [0, 2·rho_e].
    arco_eclipsado = 2.0 * max(rho_eclipse - beta_abs, 0.0)
    return float(min(arco_eclipsado / (2.0 * np.pi), 0.5))


def calcular_beta_rad(dias_transcurridos: float,
                      inclinacion_grados: float) -> float:
    """
    Aproximación del ángulo beta de una órbita: ángulo entre el plano
    orbital y el vector Tierra→Sol.
    beta ≈ asin( sin(i) · sin(δ_solar) ) + corrección por estación,
    donde δ_solar es la declinación solar. Aproximamos la declinación
    solar con la órbita anual: δ = 23.44° · sin(2π · (d - 80) / 365.25)
    (solsticio ~d=172, equinoccio ~d=80).
    Args:
        dias_transcurridos: días desde el inicio de la simulación (se
            asume que el inicio corresponde aprox. al equinoccio de
            primavera para la fase de la declinación; para una simulación
            evolutiva de años solo importa el ciclo anual, no la fase).
        inclinacion_grados: inclinación orbital en grados.
    Returns:
        beta en radianes (valor absoluto; la recarga depende de |beta|).
    """
    declinacion_grados = 23.44 * np.sin(2 * np.pi * (dias_transcurridos - 80) / 365.25)
    beta = np.arcsin(np.clip(
        np.sin(np.radians(inclinacion_grados)) * np.sin(np.radians(declinacion_grados)),
        -1.0, 1.0,
    ))
    return float(abs(beta))


def actualizar_metabolismo_batch(satelites: List['Satellite'],
                                config: Optional[ConfiguracionFisica] = None,
                                dias_transcurridos: float = 0.0) -> np.ndarray:
    """
    Versión vectorizada de la recarga/consumo diario de energía
    (:meth:`Satellite.recargar_energia` + :meth:`Satellite.consumir_energia`).

    Muta `energia` de cada satélite in-place. Si `config.eclipses_habilitados`,
    la recarga solar de cada satélite se escala por (1 - fracción del día
    en eclipse), estimada desde su inclinación y el día del año
    (ver :func:`fraccion_dia_en_eclipse` y :func:`calcular_beta_rad`);
    en eclipse total la recarga no baja de `fraccion_recarga_minima`.

    Args:
        satelites: lista de satélites a actualizar.
        config: configuración física compartida que provee
            `recarga_solar_diaria_pct`; si es None, se usan los valores
            por defecto (compatibilidad con llamadas previas).
        dias_transcurridos: días desde el inicio de la simulación, usados
            para la declinación solar (eclipses). Solo relevante si
            `config.eclipses_habilitados`.

    Returns:
        Array booleano indicando qué satélites se quedaron sin energía
        (deberían morir con causa "falta_energia").
    """
    n = len(satelites)
    if n == 0:
        return np.zeros(0, dtype=bool)

    config_efectiva = config or ConfiguracionFisica()
    energia_arr = np.array([s.energia for s in satelites], dtype=float)
    consumo_arr = np.array([s.consumo_energico_diario for s in satelites], dtype=float)

    # Recarga solar efectiva por satélite: la nominal escalada por la
    # fracción del día fuera de eclipse (Fase: eclipses por beta-angle).
    recarga_arr = np.full(n, config_efectiva.recarga_solar_diaria_pct)
    if config_efectiva.eclipses_habilitados:
        fracciones_eclipse = np.array([
            fraccion_dia_en_eclipse(
                calcular_beta_rad(dias_transcurridos, sat.i),
                sat.altitud_km(),
                config_efectiva,
            )
            for sat in satelites
        ])
        recarga_arr = recarga_arr * (1.0 - fracciones_eclipse)
        # En eclipse total la recarga no baja del mínimo (albedo):
        piso = (config_efectiva.recarga_solar_diaria_pct
                * config_efectiva.fraccion_recarga_minima)
        recarga_arr = np.maximum(recarga_arr, piso)

    energia_arr = np.minimum(energia_arr + recarga_arr, 100.0) - consumo_arr

    for idx, sat in enumerate(satelites):
        sat.energia = float(energia_arr[idx])

    return energia_arr <= 0


def aplicar_costo_evasion_batch(satelites: List['Satellite'],
                               config: ConfiguracionFisica) -> None:
    """
    Cobra el costo diario de combustible de las maniobras de evasión de
    debris, proporcional a `maniobrabilidad_evasiva` y solo para
    satélites con combustible sobre el umbral (Fase: coevolución con
    costo). Muta `combustible` de cada satélite in-place.
    Introducir el costo en una fase separada de `calcular_lambda_total`
    (que es de solo-lectura, igual que su versión batch) preserva la
    equivalencia batch/escalar verificada por los tests.
    Args:
        satelites: lista de satélites a actualizar.
        config: configuración física compartida.
    """
    if not satelites:
        return
    umbral = config.umbral_combustible_bajo_pct
    for sat in satelites:
        if sat.combustible >= umbral:
            sat.combustible = max(
                0.0,
                sat.combustible
                - config.costo_combustible_evasion_diario_pct
                * sat.maniobrabilidad_evasiva,
            )


def calcular_lambda_total_batch(satelites: List['Satellite'], edades_dias: np.ndarray,
                                riesgos_debris: np.ndarray,
                                config: ConfiguracionFisica) -> np.ndarray:
    """
    Versión vectorizada de :meth:`Satellite.calcular_lambda_total` para
    una lista completa de satélites.

    No muta ningún estado; solo calcula la tasa total de falla aditiva
    de cada satélite en una sola pasada de NumPy.

    Args:
        satelites: lista de satélites (mismo orden que `edades_dias` y
            `riesgos_debris`).
        edades_dias: array de edades (días) de cada satélite.
        riesgos_debris: array de probabilidades base de colisión (0-1)
            de cada satélite (ver
            :meth:`DebrisEspacial.probabilidad_colision_batch`).
        config: configuración física compartida.

    Returns:
        Array de tasas de falla totales (fallos/día), mismo orden que
        `satelites`.
    """
    n = len(satelites)
    if n == 0:
        return np.zeros(0)

    lambda_inherente = np.array([s.lambda_inherente for s in satelites], dtype=float)
    radiacion_acumulada = np.array([s.radiacion_acumulada for s in satelites], dtype=float)
    energia = np.array([s.energia for s in satelites], dtype=float)
    combustible = np.array([s.combustible for s in satelites], dtype=float)
    maniobrabilidad_evasiva = np.array([s.maniobrabilidad_evasiva for s in satelites], dtype=float)

    escala_dias = config.escala_envejecimiento_anos * 365.25
    k_forma = config.forma_envejecimiento_k

    edades_seguras = np.where(edades_dias == 0, 1.0, edades_dias)  # evita 0**(k-1) con k<1
    lambda_envejecimiento = np.where(
        edades_dias == 0,
        0.0,
        (k_forma / escala_dias) * (edades_seguras / escala_dias) ** (k_forma - 1)
    )

    lambda_radiacion = radiacion_acumulada * config.factor_lambda_radiacion

    # λ_DEBRIS: reducido por la capacidad de evasión heredable
    # (coevolución con el debris), solo si hay combustible suficiente
    # para maniobrar (ver Satellite.calcular_lambda_total, versión
    # escalar equivalente).
    umbral_combustible_bajo = config.umbral_combustible_bajo_pct
    lambda_debris_base = riesgos_debris * config.factor_lambda_debris
    reduccion_evasion = np.where(
        combustible >= umbral_combustible_bajo,
        config.factor_reduccion_evasion_max * maniobrabilidad_evasiva,
        0.0
    )
    lambda_debris = lambda_debris_base * (1.0 - reduccion_evasion)

    umbral_energia = config.umbral_energia_baja_pct
    lambda_energia = np.where(
        energia < umbral_energia,
        (umbral_energia - energia) * config.factor_lambda_energia,
        0.0
    )

    umbral_combustible = config.umbral_combustible_bajo_pct
    lambda_combustible = np.where(
        combustible < umbral_combustible,
        config.lambda_combustible_bajo,
        0.0
    )

    return (lambda_inherente + lambda_envejecimiento + lambda_radiacion +
            lambda_debris + lambda_energia + lambda_combustible)


# =============================================================================
# DISTRIBUCIÓN DE FALLA MEJORADA
# =============================================================================

class DistribucionFalla:
    """
    Probabilidad de falla basada en condiciones.

    Modelo legacy de tasa de falla única basada en una curva tipo
    Weibull (envejecimiento) modificada por radiación y degradación
    orbital. Se mantiene por compatibilidad y para configurar
    `SimulacionSatelites`, pero el cálculo real de fallas usa el modelo
    de tasas aditivas de :meth:`Satellite.calcular_lambda_total`.
    """
    
    def __init__(self, tasa_base: float = 0.001, parametro_forma: float = 1.5,
                 escala_tiempo: float = 3.0):
        """
        Args:
            tasa_base: tasa de falla base para satélites recién lanzados.
            parametro_forma: parámetro de forma de la curva Weibull
                (k > 1 implica tasa de falla creciente con la edad).
            escala_tiempo: escala de tiempo en años usada para normalizar
                la curva de envejecimiento.
        """
        self.tasa_base = tasa_base
        self.parametro_forma = parametro_forma
        self.escala_tiempo = escala_tiempo
    
    def probabilidad_falla_diaria(self, edad_dias: float, 
                                 radiacion: float = 0.0,
                                 degradacion: float = 0.0) -> float:
        """
        Método legacy para compatibilidad.
        NOTA: Este método está deprecado, usar calcular_lambda_total() en
        Satellite (emite DeprecationWarning al invocarse).
        
        Args:
            edad_dias: edad en días
            radiacion: radiación acumulada (0-1)
            degradacion: degradación orbital (0-1)
            
        Returns:
            probabilidad de falla (0-1)
        """
        warnings.warn(
            "DistribucionFalla.probabilidad_falla_diaria está deprecado; "
            "usar Satellite.calcular_lambda_total (modelo de tasas aditivas).",
            DeprecationWarning,
            stacklevel=2,
        )
        escala_dias = self.escala_tiempo * 365.25
        
        if edad_dias == 0:
            tasa = self.tasa_base
        else:
            tasa = (self.parametro_forma / escala_dias) * \
                   (edad_dias / escala_dias) ** (self.parametro_forma - 1)
        
        # Modificar por radiación y degradación
        tasa *= (1.0 + radiacion * 2.0 + degradacion * 1.0)
        
        prob = 1 - np.exp(-tasa)
        return min(prob, 1.0)


# =============================================================================
# SIMULACIÓN MEJORADA
# =============================================================================

class SimulacionSatelites:
    """
    Simulación evolutiva avanzada de satélites.

    Orquesta el ciclo de vida completo de una población de satélites:
    lanzamientos periódicos, efectos ambientales diarios, evaluación de
    muertes y reproducción selectiva (heredabilidad). El historial de
    estadísticas por día queda disponible en `self.historial` para su
    posterior visualización (ver `visualizador.py`).

    Attributes:
        distribucion_falla: modelo legacy de tasa de falla (ver
            :class:`DistribucionFalla`).
        intervalo_lanzamiento: `timedelta` entre lanzamientos programados.
        satelites_por_lanzamiento: cantidad de satélites por lanzamiento.
        edad_maxima_dias: edad máxima operativa, en días.
        tiempo_actual: fecha/hora actual de la simulación.
        dias_transcurridos: contador de días simulados desde el inicio.
        ciclo_solar, radiacion_cosmica, debris: subsistemas ambientales.
        satelites_activos: lista de satélites actualmente operativos.
        satelites_muertos: lista de satélites que dejaron de operar.
        historial: lista de diccionarios con estadísticas por día.
        reproductor_habilitado: si se permite la reproducción selectiva.
        tasa_mutacion: tasa de mutación usada al crear satélites hijo.
        satelites_reproducidos: contador total de satélites reproducidos.
        causas_muerte_acumuladas: conteo acumulado de muertes por causa.
        desorbitado_controlado_habilitado: si al llegar a la edad máxima
            se intenta una maniobra de desorbitado controlado + pasivación
            (mitigación de debris, "regla de los 25 años").
        altitud_objetivo_desorbitado_km: altitud objetivo de la maniobra
            de desorbitado controlado.
        prob_explosion_no_pasivado: probabilidad de explosión/fragmentación
            para un satélite que termina su misión sin pasivarse y con
            combustible residual por encima de `combustible_umbral_riesgo`.
        adr_habilitado: si se simula remoción activa de debris (ADR).
        adr_tasa_remocion_anual: cantidad de objetos de debris removidos
            por año cuando `adr_habilitado` es True.
        costo_lanzamiento_unitario, costo_remocion_unitario: costos
            (en unidades arbitrarias) usados para la métrica simple de
            sostenibilidad económica.
        satelites_pasivados: contador de satélites pasivados exitosamente.
        debris_removido_total: total de debris removido por ADR.
        costo_total_lanzamientos, costo_total_remocion: costos acumulados.
    """
    
    def __init__(self, distribucion_falla: Optional[DistribucionFalla] = None,
                 intervalo_lanzamiento_dias: int = 30,
                 satelites_por_lanzamiento: int = 5,
                 edad_maxima_anos: int = 5,
                 tiempo_inicio: Optional[datetime] = None,
                 reproductor_habilitado: bool = True,
                 tasa_mutacion: float = 0.05,
                 desorbitado_controlado_habilitado: bool = True,
                 altitud_objetivo_desorbitado_km: float = 300.0,
                 prob_explosion_no_pasivado: float = 0.1,
                 combustible_umbral_riesgo: float = 10.0,
                 adr_habilitado: bool = False,
                 adr_tasa_remocion_anual: float = 0.0,
                 costo_lanzamiento_unitario: float = 100.0,
                 costo_remocion_unitario: float = 20.0,
                 config: Optional[ConfiguracionFisica] = None,
                 semilla: Optional[int] = None,
                 historial_intervalo_dias: int = 1,
                 prob_reproduccion_sexual: float = 0.3):
        """
        Args:
            distribucion_falla: modelo de falla
            intervalo_lanzamiento_dias: días entre lanzamientos
            satelites_por_lanzamiento: satélites por lanzamiento
            edad_maxima_anos: edad máxima
            tiempo_inicio: inicio simulación
            reproductor_habilitado: si se permite reproducción selectiva
            tasa_mutacion: tasa de mutación en heredabilidad
            desorbitado_controlado_habilitado: si se intenta desorbitado
                controlado + pasivación al fin de vida (regla de 25 años)
            altitud_objetivo_desorbitado_km: altitud objetivo de la
                maniobra de desorbitado controlado
            prob_explosion_no_pasivado: probabilidad diaria de explosión
                para satélites no pasivados con combustible residual
            combustible_umbral_riesgo: umbral de combustible (%) por
                encima del cual un satélite no pasivado puede explotar
            adr_habilitado: si se simula remoción activa de debris (ADR)
            adr_tasa_remocion_anual: objetos de debris removidos por año
            costo_lanzamiento_unitario: costo (unidades arbitrarias) por
                satélite lanzado, para la métrica de sostenibilidad
            costo_remocion_unitario: costo (unidades arbitrarias) por
                objeto de debris removido vía ADR
            config: configuración física centralizada (ver
                :class:`ConfiguracionFisica`) compartida por todos los
                satélites y el sistema de debris; si es None, se usan
                los valores por defecto.
            semilla: semilla para el generador aleatorio reproducible
                (`self.rng`, un `np.random.Generator`) usado en todos los
                sorteos de la simulación (calidad inherente, mutaciones,
                selección de padres, fallas, colisiones). Si es None, la
                simulación no es reproducible entre corridas.
            historial_intervalo_dias: cada cuántos días se guarda un
                snapshot en `self.historial` (por defecto 1 = todos los
                días, igual que antes). Aumentarlo reduce el uso de
                memoria en simulaciones muy largas o con poblaciones muy
                grandes; el último snapshot calculado siempre queda
                disponible en `self.ultimo_stats`.
            prob_reproduccion_sexual: probabilidad (0-1) de que un hijo
                nazca por recombinación sexual (crossover entre 2 padres,
                ver :meth:`Satellite.crear_hijo_sexual`) en vez de la
                reproducción asexual clásica (clonar 1 padre + mutación).
        """
        self.config = config or ConfiguracionFisica()
        self.rng = np.random.default_rng(semilla)
        self.distribucion_falla = distribucion_falla or DistribucionFalla()
        self.intervalo_lanzamiento = timedelta(days=intervalo_lanzamiento_dias)
        self.satelites_por_lanzamiento = satelites_por_lanzamiento
        self.edad_maxima_dias = edad_maxima_anos * 365.25
        
        self.tiempo_actual = tiempo_inicio or datetime.now()
        self.tiempo_proximo_lanzamiento = self.tiempo_actual
        self.dias_transcurridos = 0.0
        
        # Sistemas ambientales
        self.ciclo_solar = CicloSolar()
        self.radiacion_cosmica = RadiacionCosmica()
        self.debris = DebrisEspacial(config=self.config)
        
        # Población
        self.satelites_activos: List[Satellite] = []
        self.satelites_muertos: List[Satellite] = []
        self.historial: List[Dict] = []
        self.historial_intervalo_dias = max(1, historial_intervalo_dias)
        self.ultimo_stats: Optional[Dict] = None
        
        # Reproducción
        self.reproductor_habilitado = reproductor_habilitado
        self.tasa_mutacion = tasa_mutacion
        self.satelites_reproducidos = 0
        self.prob_reproduccion_sexual = prob_reproduccion_sexual
        
        # Tracking de causas de muerte acumuladas (para visualización)
        self.causas_muerte_acumuladas: Dict[str, int] = {}

        # Detección del punto de no retorno de Kessler por banda orbital:
        # días consecutivos con riesgo de cascada > 1, y bandas que ya
        # cruzaron el umbral sostenido (una vez cruzado, es permanente
        # para la corrida: la cascada se autoabastece).
        self._dias_cascada_sostenida: Dict[str, int] = {}
        self.bandas_punto_no_retorno: set = set()
        
        # Mitigación de debris: pasivación + desorbitado controlado
        self.desorbitado_controlado_habilitado = desorbitado_controlado_habilitado
        self.altitud_objetivo_desorbitado_km = altitud_objetivo_desorbitado_km
        self.prob_explosion_no_pasivado = prob_explosion_no_pasivado
        self.combustible_umbral_riesgo = combustible_umbral_riesgo
        self.satelites_pasivados = 0
        
        # Remoción activa de debris (ADR)
        self.adr_habilitado = adr_habilitado
        self.adr_tasa_remocion_anual = adr_tasa_remocion_anual
        self.debris_removido_total = 0.0
        
        # Dimensión económica simple (trade-offs de sostenibilidad)
        self.costo_lanzamiento_unitario = costo_lanzamiento_unitario
        self.costo_remocion_unitario = costo_remocion_unitario
        self.costo_total_lanzamientos = 0.0
        self.costo_total_remocion = 0.0
    
    def _registrar_causa_muerte(self, causa: Optional[str]):
        """
        Registra una muerte en el conteo acumulado por causa.

        Args:
            causa: nombre de la causa de muerte (o None, registrado
                como "desconocida").
        """
        clave = causa or "desconocida"
        self.causas_muerte_acumuladas[clave] = self.causas_muerte_acumuladas.get(clave, 0) + 1
    
    def _generar_parametros_kepleriano(self) -> Tuple[float, float, float, float, float, float]:
        """
        Genera parámetros keplerianos aleatorios para un nuevo satélite.

        Returns:
            Tupla (a, e, i, Omega, omega, M) con semieje mayor en km
            (rango LEO 6600-7200) y ángulos en grados.
        """
        a = self.rng.uniform(6600, 7200)
        e = self.rng.uniform(0.0, 0.02)
        i = self.rng.uniform(0, 90)
        Omega = self.rng.uniform(0, 360)
        omega = self.rng.uniform(0, 360)
        M = self.rng.uniform(0, 360)
        return a, e, i, Omega, omega, M
    
    def _califica_para_reproduccion(self, sat: Satellite) -> bool:
        """
        Determina si un satélite califica como candidato a reproducirse.

        Estrategias r/K (trade-off reproducción vs. longevidad): el
        umbral de energía/combustible requerido no es fijo, se ajusta
        según el rasgo heredable `agresividad_reproductiva` de cada
        satélite. Satélites más "r-seleccionados" (agresividad alta)
        califican con menos energía/combustible (reproducción temprana y
        arriesgada, a costa de quedar más expuestos a fallar después por
        falta de energía/combustible); satélites más "K-seleccionados"
        (agresividad baja) requieren estar casi al máximo (reproducción
        conservadora, prioriza la propia supervivencia). En
        agresividad=0.5 se recupera el umbral fijo original (50%/30%).

        Args:
            sat: satélite a evaluar.

        Returns:
            True si el satélite califica como candidato a reproducirse.
        """
        umbral_energia = (self.config.umbral_energia_reproduccion_base -
                          self.config.umbral_energia_reproduccion_rango * sat.agresividad_reproductiva)
        umbral_combustible = (self.config.umbral_combustible_reproduccion_base -
                              self.config.umbral_combustible_reproduccion_rango * sat.agresividad_reproductiva)
        return sat.energia > umbral_energia and sat.combustible > umbral_combustible
    
    def _bono_diversidad_orbital(self, candidatos: List[Satellite]) -> np.ndarray:
        """
        Bono de fitness por diversidad orbital (fitness de cobertura de
        constelación): satélites cuyo RAAN (`Omega`) está en una zona
        poco poblada dentro de los candidatos reciben un bono mayor a
        1.0 en su peso de selección como padres, favoreciendo que la
        reproducción llene huecos de cobertura en vez de concentrarse en
        posiciones ya representadas (selección disruptiva / dependiente
        de frecuencia, análoga a partición de nicho ecológico).

        Implementación: para puntos sobre un círculo, el vecino más
        cercano de cada punto siempre es uno de sus dos vecinos en el
        orden circular ordenado (propiedad estándar de vecino más
        cercano en 1D/circular). Esto permite calcular la distancia
        mínima de cada satélite en O(n log n) (ordenando) en vez de la
        matriz de distancias par-a-par O(n²), relevante cuando la
        cantidad de candidatos a reproducirse escala a miles.

        Args:
            candidatos: lista de satélites candidatos a reproducirse.

        Returns:
            Array de bonos positivos (mismo orden que `candidatos`); un
            valor mayor a 1.0 indica una posición de RAAN menos
            congestionada que el promedio de los candidatos.
        """
        n = len(candidatos)
        if n <= 1:
            return np.ones(n)

        omegas = np.array([c.Omega for c in candidatos], dtype=float) % 360.0
        orden = np.argsort(omegas)
        omegas_ordenadas = omegas[orden]

        # Distancia circular de cada punto (ya ordenado) a su siguiente
        # vecino; la distancia a su vecino anterior es la distancia "al
        # siguiente" del punto previo (desplazada una posición).
        distancia_siguiente = (np.roll(omegas_ordenadas, -1) - omegas_ordenadas) % 360.0
        distancia_anterior = np.roll(distancia_siguiente, 1)
        distancia_min_ordenada = np.minimum(distancia_siguiente, distancia_anterior)

        distancia_min = np.empty(n)
        distancia_min[orden] = distancia_min_ordenada

        promedio = distancia_min.mean()
        if promedio <= 1e-9:
            return np.ones(n)
        return distancia_min / promedio
    
    def _pesos_seleccion_padres(self, candidatos: List[Satellite]) -> np.ndarray:
        """
        Calcula los pesos (sin normalizar) usados para elegir padres al
        reproducirse: selección por aptitud diferencial (proporcional a
        :meth:`Satellite.fitness_reproductivo`, hoy la calidad inherente)
        combinada con el bono de diversidad orbital de
        :meth:`_bono_diversidad_orbital`, para repartir mejor la
        cobertura de la constelación en vez de concentrar la
        reproducción en unos pocos linajes/posiciones.

        Args:
            candidatos: lista de satélites candidatos a reproducirse.

        Returns:
            Array de pesos positivos (mismo orden que `candidatos`); se
            debe normalizar (dividir por su suma) para usarlos como
            probabilidades.
        """
        pesos_fitness = np.array([c.fitness_reproductivo() for c in candidatos])
        bono_diversidad = self._bono_diversidad_orbital(candidatos)
        return pesos_fitness * bono_diversidad * self._bono_cobertura_banda(candidatos)

    def _bono_cobertura_banda(self, candidatos: List[Satellite]) -> np.ndarray:
        """
        Bono de fitness de cobertura de banda orbital (selección de
        nicho a nivel de constelación, complementario al bono de RAAN):
        satélites cuya banda orbital ("bajo", "leo_congestionado", "meo",
        "geo") está menos representada entre los candidatos reciben un
        bono mayor, favoreciendo que la reproducción pueble nichos de
        altitud poco cubiertos en vez de concentrarse en la banda dominante.
        El bono del nicho menos poblado escala hasta
        `1 + peso_bono_cobertura_banda` (y hasta 1.0 para el más poblado);
        con `peso_bono_cobertura_banda` = 0 queda desactivado.
        Args:
            candidatos: lista de satélites candidatos a reproducirse.
        Returns:
            Array de bonos positivos (mismo orden que `candidatos`).
        """
        peso = self.config.peso_bono_cobertura_banda
        n = len(candidatos)
        if peso <= 0.0 or n <= 1:
            return np.ones(n)
        bandas = [self.debris.banda_de_altitud(c.altitud_km()) for c in candidatos]
        conteo: Dict[str, int] = {}
        for banda in bandas:
            conteo[banda] = conteo.get(banda, 0) + 1
        max_conteo = max(conteo.values())
        bono_por_banda = {
            banda: 1.0 + peso * (1.0 - conteo[banda] / max_conteo)
            for banda in conteo
        }
        return np.array([bono_por_banda[b] for b in bandas])
    
    def _probabilidad_reproduccion_exitosa(self, padre: Satellite) -> float:
        """
        Probabilidad de que un intento de reproducción prospere, según
        la congestión de la banda orbital donde vive el padre (selección
        dependiente de densidad / capacidad de carga ecológica): a mayor
        densidad de debris en la banda, menor probabilidad de que la
        reproducción tenga éxito. Un intento fallido no genera hijo; el
        hueco se completa en :meth:`lanzar_satelites` con un lanzamiento
        nuevo (sin heredabilidad) en vez de uno heredado.

        Args:
            padre: satélite candidato a reproducirse.

        Returns:
            Probabilidad de éxito (0-1).
        """
        banda = self.debris.banda_de_altitud(padre.altitud_km())
        densidad_banda = self.debris.densidad_normalizada_banda(banda)
        return max(0.0, 1.0 - self.config.factor_capacidad_carga * densidad_banda)
    
    def lanzar_satelites(self, heredables: bool = False) -> List[Satellite]:
        """
        Lanza nuevos satélites.
        
        Args:
            heredables: si los nuevos pueden reproducirse
            
        Returns:
            lista de satélites lanzados

        Mecanismos evolutivos a nivel de constelación aplicados durante
        la reproducción selectiva (cuando `heredables=True`):

        1. Selección por aptitud diferencial + fitness de cobertura: los
           padres se sortean con probabilidad proporcional a
           :meth:`_pesos_seleccion_padres` (calidad inherente × bono de
           diversidad orbital), en vez de uniforme.
        2. Selección dependiente de densidad (carrying capacity): cada
           intento de reproducción puede fallar según
           :meth:`_probabilidad_reproduccion_exitosa` (más congestión en
           la banda del padre → más probabilidad de fallar).
        3. Recombinación sexual (crossover): con probabilidad
           `self.prob_reproduccion_sexual` el hijo nace de 2 padres
           (:meth:`Satellite.crear_hijo_sexual`) en vez de 1 solo padre.
        4. Estrategias r/K: el filtro de candidatos usa umbrales
           variables por satélite según `agresividad_reproductiva` (ver
           :meth:`_califica_para_reproduccion`).
        """
        nuevos = []
        
        if heredables and self.reproductor_habilitado and self.satelites_activos:
            # Reproducción selectiva: satélites sanos (según su umbral
            # r/K individual) pueden reproducirse
            padres_candidatos = [s for s in self.satelites_activos 
                               if self._califica_para_reproduccion(s)]
            
            if padres_candidatos:
                cantidad = min(self.satelites_por_lanzamiento, len(padres_candidatos))

                pesos = self._pesos_seleccion_padres(padres_candidatos)
                probabilidades = pesos / pesos.sum()

                for _ in range(cantidad):
                    # 1. Selección por aptitud diferencial + cobertura
                    idx_padre = self.rng.choice(len(padres_candidatos), p=probabilidades)
                    padre = padres_candidatos[idx_padre]

                    # 2. Selección dependiente de densidad (carrying capacity)
                    if self.rng.random() > self._probabilidad_reproduccion_exitosa(padre):
                        continue  # intento fallido por saturación orbital

                    # 3. Recombinación sexual (crossover) vs. asexual,
                    #    con aislamiento reproductivo por banda orbital
                    #    (especiación alopátrica): el segundo padre debe
                    #    pertenecer a la misma banda que el primero; si no
                    #    hay pareja compatible, cae a asexual.
                    if (len(padres_candidatos) > 1 and
                            self.rng.random() < self.prob_reproduccion_sexual):
                        if self.config.aislamiento_reproductivo_por_banda:
                            banda_padre = self.debris.banda_de_altitud(padre.altitud_km())
                            compatibles = [
                                idx for idx, cand in enumerate(padres_candidatos)
                                if idx != idx_padre
                                and self.debris.banda_de_altitud(cand.altitud_km()) == banda_padre
                            ]
                        else:
                            compatibles = [
                                idx for idx in range(len(padres_candidatos))
                                if idx != idx_padre
                            ]
                        if compatibles:
                            prob_pareja = probabilidades[compatibles] / probabilidades[compatibles].sum()
                            idx_padre2 = int(self.rng.choice(len(compatibles), p=prob_pareja))
                            padre2 = padres_candidatos[compatibles[idx_padre2]]
                            hijo = padre.crear_hijo_sexual(
                                padre2, self.tiempo_actual, self.tasa_mutacion, rng=self.rng
                            )
                        else:
                            hijo = padre.crear_hijo(self.tiempo_actual, self.tasa_mutacion, rng=self.rng)
                    else:
                        hijo = padre.crear_hijo(self.tiempo_actual, self.tasa_mutacion, rng=self.rng)

                    self.satelites_activos.append(hijo)
                    nuevos.append(hijo)
                    self.satelites_reproducidos += 1
                    
                    # Padre gasta combustible en "reproducción"
                    padre.combustible -= 10
        
        # Completar con lanzamientos nuevos
        faltantes = self.satelites_por_lanzamiento - len(nuevos)
        for _ in range(faltantes):
            a, e, i, Omega, omega, M = self._generar_parametros_kepleriano()
            sat = Satellite(a, e, i, Omega, omega, M, self.tiempo_actual, config=self.config, rng=self.rng)
            self.satelites_activos.append(sat)
            nuevos.append(sat)
        
        # Costo económico de los lanzamientos (dimensión de sostenibilidad)
        self.costo_total_lanzamientos += self.costo_lanzamiento_unitario * len(nuevos)
        
        self.tiempo_proximo_lanzamiento += self.intervalo_lanzamiento
        return nuevos
    
    def evaluar_muertes(self):
        """
        Evalúa muertes por múltiples causas.

        Recorre `satelites_activos` y marca como muertos (moviéndolos a
        `satelites_muertos` y registrando la causa) a quienes:
        1. Alcanzaron la edad máxima ("edad_maxima", "fin_mision_controlado"
           o "explosion_fragmentacion", según si se intenta mitigación de
           fin de vida, ver más abajo).
        2. Fallan probabilísticamente según la tasa aditiva total
           ("falla_componentes", ver
           :meth:`Satellite.probabilidad_falla_diaria_aditiva`).
        3. Colisionan con debris espacial ("colision_debris").

        Al llegar a la edad máxima, si `desorbitado_controlado_habilitado`
        está activo, se intenta una maniobra de desorbitado controlado
        (:meth:`Satellite.intentar_desorbitado_controlado`) y una
        pasivación (:meth:`Satellite.pasivar`, con probabilidad de éxito
        igual a la calidad inherente del satélite). Un satélite que
        cumple ambas se marca como "fin_mision_controlado"; uno que no
        se pasiva y conserva combustible residual por encima de
        `combustible_umbral_riesgo` puede sufrir "explosion_fragmentacion"
        (con `prob_explosion_no_pasivado` de probabilidad), generando
        varias veces más debris que un derelicto simple.

        Cada muerte genera debris en la banda orbital correspondiente:
        las colisiones y explosiones generan fragmentos proporcionales
        a la energía involucrada (ver
        :meth:`DebrisEspacial.generar_debris_por_colision` y
        :meth:`DebrisEspacial.generar_debris_por_explosion`), mientras
        que el resto de causas dejan al satélite mismo como un objeto
        derelicto rastreable (ver
        :meth:`DebrisEspacial.generar_debris_derelicto`). Al finalizar,
        actualiza `satelites_activos` con los supervivientes.

        Implementación: las edades, riesgos de colisión y tasas de
        falla aditivas de toda la población se calculan en una sola
        pasada vectorizada (ver
        :func:`calcular_lambda_total_batch` y
        :meth:`DebrisEspacial.probabilidad_colision_batch`), y los
        sorteos aleatorios de falla/colisión se generan en batch con
        `self.rng.random`, en vez de iterar satélite por satélite. El
        grupo (típicamente pequeño) que llega a la edad máxima conserva
        su lógica secuencial, ya que involucra decisiones de mitigación
        de debris satélite por satélite.
        """
        activos = self.satelites_activos
        if not activos:
            return

        edades = np.array([sat.edad_dias(self.tiempo_actual) for sat in activos])
        mascara_edad_maxima = edades >= self.edad_maxima_dias
        indices_edad_maxima = np.nonzero(mascara_edad_maxima)[0]
        indices_resto = np.nonzero(~mascara_edad_maxima)[0]

        # 1. Edad máxima (fin de vida): mitigación opcional de debris
        for idx in indices_edad_maxima:
            sat = activos[idx]
            sat.activo = False

            maniobra_ok = True
            if self.desorbitado_controlado_habilitado:
                maniobra_ok = sat.intentar_desorbitado_controlado(
                    self.altitud_objetivo_desorbitado_km
                )
                if self.rng.random() < sat.calidad_inherente:
                    sat.pasivar()
                    self.satelites_pasivados += 1

            if self.desorbitado_controlado_habilitado and maniobra_ok and sat.pasivado:
                sat.causa_muerte = "fin_mision_controlado"
            elif (not sat.pasivado and sat.combustible > self.combustible_umbral_riesgo
                  and self.rng.random() < self.prob_explosion_no_pasivado):
                sat.causa_muerte = "explosion_fragmentacion"
            else:
                sat.causa_muerte = "edad_maxima"

            self.satelites_muertos.append(sat)
            self._registrar_causa_muerte(sat.causa_muerte)

            if sat.causa_muerte == "explosion_fragmentacion":
                self.debris.generar_debris_por_explosion(
                    sat.altitud_km(), sat.masa_kg, sat.combustible / 100.0
                )
            else:
                self.debris.generar_debris_derelicto(sat.altitud_km())

        if len(indices_resto) == 0:
            self.satelites_activos = []
            return

        satelites_resto = [activos[idx] for idx in indices_resto]
        edades_resto = edades[indices_resto]
        altitudes_resto = np.array([sat.altitud_km() for sat in satelites_resto])
        riesgos_debris = self.debris.probabilidad_colision_batch(altitudes_resto)

        # 2. Falla probabilística (tasas aditivas, vectorizado)
        lambda_total = calcular_lambda_total_batch(
            satelites_resto, edades_resto, riesgos_debris, self.config
        )
        prob_falla = np.clip(1 - np.exp(-lambda_total), 0.0, 1.0)
        mascara_falla = self.rng.random(len(satelites_resto)) < prob_falla

        # 3. Colisión con debris (vectorizado)
        mascara_colision = self.rng.random(len(satelites_resto)) < riesgos_debris

        satelites_supervivientes = []
        for idx_local, sat in enumerate(satelites_resto):
            if mascara_falla[idx_local]:
                sat.activo = False
                sat.causa_muerte = "falla_componentes"
                self.satelites_muertos.append(sat)
                self._registrar_causa_muerte(sat.causa_muerte)
                self.debris.generar_debris_derelicto(sat.altitud_km())
                continue

            if mascara_colision[idx_local]:
                sat.activo = False
                sat.causa_muerte = "colision_debris"
                self.satelites_muertos.append(sat)
                self._registrar_causa_muerte(sat.causa_muerte)
                self.debris.colisiones_totales += 1
                self.debris.generar_debris_por_colision(
                    sat.altitud_km(), sat.masa_kg, sat.velocidad_orbital_kms()
                )
                continue

            satelites_supervivientes.append(sat)

        self.satelites_activos = satelites_supervivientes

        # Costo diario de combustible de las maniobras de evasión
        # (coevolución con costo: evadir debris gasta longevidad de
        # maniobras). Fase separada del cálculo de lambda, que es de
        # solo-lectura.
        aplicar_costo_evasion_batch(self.satelites_activos, self.config)
    
    def actualizar_sistemas_ambientales(self):
        """
        Aplica efectos ambientales a satélites activos.

        Para cada satélite activo aplica, en orden: drag atmosférico
        (posible reentrada), presión de radiación solar, radiación
        cósmica (posible falla por radiación) y consumo/recarga de
        energía (posible falla por falta de energía). Los satélites que
        dejan de estar activos se mueven a `satelites_muertos`, su causa
        de muerte se registra en el conteo acumulado y el satélite
        derelicto pasa a formar parte del debris de su banda orbital.

        Implementación: cada etapa (drag, presión de radiación solar,
        radiación cósmica, metabolismo) se calcula para toda la
        población activa en una sola pasada vectorizada (ver las
        funciones `*_batch` a nivel de módulo), en vez de iterar
        satélite por satélite en Python. Entre etapas se filtran los
        satélites que murieron, preservando el mismo orden/semántica
        que la versión secuencial original (un satélite que muere en
        una etapa no continúa a las siguientes).
        """
        activos = self.satelites_activos
        if not activos:
            return

        # 1. Drag atmosférico y reentrada (vectorizado)
        mascara_reentrada = aplicar_drag_atmosferico_batch(
            activos, self.ciclo_solar, self.dias_transcurridos, self.config
        )
        for sat, reentro in zip(activos, mascara_reentrada):
            if reentro:
                sat.activo = False
                sat.causa_muerte = "reentrada_orbital"
                self.satelites_muertos.append(sat)
                self._registrar_causa_muerte(sat.causa_muerte)
                # A diferencia del resto de causas, un satélite que
                # reentra se desintegra en la atmósfera: no deja debris
                # derelicto en su banda orbital (el drag elimina el
                # objeto, no lo convierte en riesgo de colisión).

        supervivientes_drag = [sat for sat in activos if sat.activo]

        # 2. Presión de radiación solar (vectorizado, no mata satélites)
        aplicar_presion_radiacion_solar_batch(
            supervivientes_drag, self.ciclo_solar, self.dias_transcurridos
        )

        # 3. Radiación cósmica (vectorizado)
        mascara_radiacion = aplicar_radiacion_cosmica_batch(
            supervivientes_drag, self.radiacion_cosmica
        )
        for sat, murio in zip(supervivientes_drag, mascara_radiacion):
            if murio:
                sat.activo = False
                sat.causa_muerte = "radiacion_cosmica"
                self.satelites_muertos.append(sat)
                self._registrar_causa_muerte(sat.causa_muerte)
                self.debris.generar_debris_derelicto(sat.altitud_km())

        supervivientes_radiacion = [sat for sat in supervivientes_drag if sat.activo]

        # 4. Metabolismo: recarga y consumo de energía (vectorizado)
        mascara_sin_energia = actualizar_metabolismo_batch(
            supervivientes_radiacion, self.config, self.dias_transcurridos
        )
        for sat, sin_energia in zip(supervivientes_radiacion, mascara_sin_energia):
            if sin_energia:
                sat.activo = False
                sat.causa_muerte = "falta_energia"
                self.satelites_muertos.append(sat)
                self._registrar_causa_muerte(sat.causa_muerte)
                self.debris.generar_debris_derelicto(sat.altitud_km())

        # Limpiar muertos del activos
        self.satelites_activos = [sat for sat in supervivientes_radiacion if sat.activo]

        # 5. Propagacion orbital: anomalia media (dM = n·dt) y precesion
        # nodal J2 del RAAN (Fase: fidelidad orbital). Muta M/Omega
        # in-place para toda la poblacion en una sola pasada.
        propagar_elementos_orbitales_batch(
            self.satelites_activos, 1.0, self.config
        )
    
    def paso_tiempo(self, dias: int = 1) -> Dict:
        """
        Avanza simulación varios días.

        Por cada día: aplica efectos ambientales, evalúa muertes
        naturales/colisiones y, si corresponde, lanza nuevos satélites
        (alternando entre lanzamientos nuevos y reproducción selectiva).
        Al final registra un snapshot de estadísticas en `self.historial`.

        Args:
            dias: cantidad de días a avanzar en esta llamada.

        Returns:
            Diccionario con las estadísticas del último día simulado
            (también agregado a `self.historial`).
        """
        for _ in range(dias):
            self.tiempo_actual += timedelta(days=1)
            self.dias_transcurridos += 1
            
            # Aplicar efectos ambientales
            self.actualizar_sistemas_ambientales()
            
            # Evaluar muertes naturales/colisiones
            self.evaluar_muertes()
            
            # Decaimiento natural del debris + riesgo de cascada del día
            riesgo_cascada = self.debris.aplicar_decaimiento_diario()

            # Detección del "punto de no retorno" de Kessler por banda:
            # la banda entra en cascada sostenida cuando el debris
            # generado supera al decaído (riesgo > 1) durante
            # `dias_sostenidos_punto_no_retorno` días consecutivos.
            for banda, riesgo in riesgo_cascada.items():
                if riesgo > 1.0:
                    self._dias_cascada_sostenida[banda] = (
                        self._dias_cascada_sostenida.get(banda, 0) + 1
                    )
                else:
                    self._dias_cascada_sostenida[banda] = 0
                if (banda not in self.bandas_punto_no_retorno
                        and self._dias_cascada_sostenida[banda]
                        >= self.config.dias_sostenidos_punto_no_retorno):
                    self.bandas_punto_no_retorno.add(banda)
            
            # Remoción activa de debris (ADR), si está habilitada
            if self.adr_habilitado and self.adr_tasa_remocion_anual > 0:
                cantidad_diaria = self.adr_tasa_remocion_anual / 365.25
                removido = self.debris.remover_debris(cantidad_diaria)
                self.debris_removido_total += removido
                self.costo_total_remocion += removido * self.costo_remocion_unitario
            
            # Lanzar nuevos satélites si es tiempo
            nuevos = []
            if self.tiempo_actual >= self.tiempo_proximo_lanzamiento:
                # Alternar entre lanzamientos nuevos y reproducción
                heredables = (len(self.satelites_muertos) % 100) < 50
                nuevos = self.lanzar_satelites(heredables=heredables)
        
        (edad_promedio, energia_promedio, radiacion_promedio,
         semieje_promedio, calidad_promedio) = self._estadisticas_poblacion_activa()

        stats = {
            'tiempo': self.tiempo_actual,
            'dias': int(self.dias_transcurridos),
            'satelites_activos': len(self.satelites_activos),
            'satelites_muertos': len(self.satelites_muertos),
            'nuevos_lanzados': len(nuevos),
            'reproducidos': self.satelites_reproducidos,
            'actividad_solar': float(self.ciclo_solar.actividad_solar(self.dias_transcurridos)),
            'densidad_debris': float(self.debris.densidad_normalizada()),
            'densidad_debris_por_banda': {
                banda: float(self.debris.densidad_normalizada_banda(banda))
                for banda in self.debris.densidad_por_banda
            },
            'riesgo_cascada': riesgo_cascada,
            'bandas_punto_no_retorno': sorted(self.bandas_punto_no_retorno),
            'colisiones_totales': self.debris.colisiones_totales,
            'edad_promedio_dias': edad_promedio,
            'energia_promedio': energia_promedio,
            'radiacion_promedio': radiacion_promedio,
            'semieje_mayor_promedio': semieje_promedio,
            'calidad_promedio': calidad_promedio,
            'causas_muerte_acumuladas': dict(self.causas_muerte_acumuladas),
            'satelites_pasivados': self.satelites_pasivados,
            'debris_removido_total': float(self.debris_removido_total),
            'costo_total_lanzamientos': float(self.costo_total_lanzamientos),
            'costo_total_remocion': float(self.costo_total_remocion),
            'costo_total_sostenibilidad': float(self.costo_total_lanzamientos + self.costo_total_remocion),
        }
        
        self.ultimo_stats = stats
        # Muestreo del historial: por defecto (historial_intervalo_dias=1)
        # se guarda un snapshot por día, igual que antes. Con un intervalo
        # mayor se reduce el uso de memoria en simulaciones largas o con
        # poblaciones muy grandes; `self.ultimo_stats` siempre tiene el
        # snapshot más reciente, se haya guardado en el historial o no.
        if int(self.dias_transcurridos) % self.historial_intervalo_dias == 0:
            self.historial.append(stats)
        return stats
    
    def _estadisticas_poblacion_activa(self) -> Tuple[float, float, float, float, float]:
        """
        Calcula edad, energía, radiación acumulada, semieje mayor y
        calidad inherente promedio de los satélites activos en una sola
        pasada sobre la población, en vez de 5 pasadas independientes
        (una por cada `_*_promedio`). Se usa en el snapshot diario de
        :meth:`paso_tiempo`, que es el punto donde más importa evitar
        recorridos repetidos de la población completa.

        Returns:
            Tupla (edad_promedio_dias, energia_promedio, radiacion_promedio,
            semieje_mayor_promedio, calidad_promedio). Todos 0.0 si no hay
            satélites activos.
        """
        activos = self.satelites_activos
        if not activos:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        tiempo_actual = self.tiempo_actual
        n = len(activos)
        edades = np.empty(n)
        energias = np.empty(n)
        radiaciones = np.empty(n)
        semiejes = np.empty(n)
        calidades = np.empty(n)

        for idx, sat in enumerate(activos):
            edades[idx] = sat.edad_dias(tiempo_actual)
            energias[idx] = sat.energia
            radiaciones[idx] = sat.radiacion_acumulada
            semiejes[idx] = sat.a
            calidades[idx] = sat.calidad_inherente

        return (
            float(np.mean(edades)),
            float(np.mean(energias)),
            float(np.mean(radiaciones)),
            float(np.mean(semiejes)),
            float(np.mean(calidades)),
        )
    
    def _edad_promedio(self) -> float:
        """Edad promedio (en días) de los satélites activos."""
        if not self.satelites_activos:
            return 0
        edades = [sat.edad_dias(self.tiempo_actual) for sat in self.satelites_activos]
        return float(np.mean(edades))
    
    def _energia_promedio(self) -> float:
        """Energía promedio (%) de los satélites activos."""
        if not self.satelites_activos:
            return 0
        return float(np.mean([sat.energia for sat in self.satelites_activos]))
    
    def _radiacion_promedio(self) -> float:
        """Radiación cósmica acumulada promedio (0-1) de los satélites activos."""
        if not self.satelites_activos:
            return 0
        return float(np.mean([sat.radiacion_acumulada for sat in self.satelites_activos]))
    
    def _semieje_promedio(self) -> float:
        """Semieje mayor promedio (km) de los satélites activos."""
        if not self.satelites_activos:
            return 0
        return float(np.mean([sat.a for sat in self.satelites_activos]))
    
    def _calidad_promedio(self) -> float:
        """Calidad inherente promedio (0-1) de los satélites activos."""
        if not self.satelites_activos:
            return 0
        return float(np.mean([sat.calidad_inherente for sat in self.satelites_activos]))
    
    def ejecutar_simulacion(self, dias_totales: int = 1825,
                           heredables: Optional[bool] = None,
                           checkpoint_cada_dias: Optional[int] = None,
                           checkpoint_ruta: str = "checkpoint_simulacion.pkl") -> List[Dict]:
        """
        Ejecuta simulación completa.

        Avanza la simulación día a día hasta `dias_totales`, imprimiendo
        un resumen de progreso cada 365 días.

        Args:
            dias_totales: cantidad total de días a simular (por defecto
                1825 días = 5 años).
            heredables: si es True/False, sobrescribe `reproductor_habilitado`
                durante la ejecución (equivalente a pasar el parámetro al
                constructor). Si es None (por defecto), se usa el valor del
                constructor. Se acepta por compatibilidad con la firma
                documentada en el README.
            checkpoint_cada_dias: si se especifica, guarda un checkpoint
                (ver :meth:`guardar_checkpoint`) cada esta cantidad de
                días, para poder reanudar la simulación tras una
                interrupción sin perder el progreso. Si es None (por
                defecto), no se guardan checkpoints automáticos.
            checkpoint_ruta: ruta del archivo donde guardar los
                checkpoints automáticos.

        Returns:
            Historial completo de estadísticas por día (`self.historial`).
        """
        print(f"Iniciando simulación avanzada por {dias_totales} días...")
        print(f"  - Ciclos solares: Habilitados")
        print(f"  - Debris espacial: Habilitado")
        print(f"  - Radiación cósmica: Habilitada")
        if heredables is not None:
            self.reproductor_habilitado = bool(heredables)
        print(f"  - Heredabilidad: {'Habilitada' if self.reproductor_habilitado else 'Deshabilitada'}")
        print(f"Tiempo inicial: {self.tiempo_actual}\n")
        
        for i in range(dias_totales):
            self.paso_tiempo(1)
            if (i + 1) % 365 == 0:
                print(f"Año {(i+1)//365}: {len(self.satelites_activos)} activos, "
                      f"{len(self.satelites_muertos)} muertos, "
                      f"Debris: {self.debris.densidad:.0f}")
            if checkpoint_cada_dias and (i + 1) % checkpoint_cada_dias == 0:
                self.guardar_checkpoint(checkpoint_ruta)
                print(f"  ✓ Checkpoint guardado en '{checkpoint_ruta}' (día {i + 1})")
        
        print(f"\nSimulación completada. Tiempo final: {self.tiempo_actual}")
        return self.historial
    
    # Version del esquema de checkpoint: incrementar al cambiar la
    # estructura serializada (atributos de SimulacionSatelites/Satellite/
    # subsistemas) para que `cargar_checkpoint` rechace archivos
    # incompatibles con un error claro en vez de fallar de forma
    # silenciosa o impredecible al deserializar.
    _VERSION_ESQUEMA_CHECKPOINT = 1
    
    def guardar_checkpoint(self, ruta: str) -> None:
        """
        Serializa el estado completo de la simulación (satélites activos y
        muertos, historial, generador aleatorio reproducible, contadores,
        subsistemas ambientales, etc.) a un archivo con `pickle`, para
        poder reanudar una simulación larga tras una interrupción sin
        perder el progreso.

        Args:
            ruta: ruta del archivo donde guardar el checkpoint.
        """
        self._version_esquema_checkpoint = self._VERSION_ESQUEMA_CHECKPOINT
        with open(ruta, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def cargar_checkpoint(ruta: str) -> "SimulacionSatelites":
        """
        Restaura una instancia de `SimulacionSatelites` previamente
        guardada con :meth:`guardar_checkpoint`, lista para continuar
        avanzando con :meth:`paso_tiempo` o :meth:`ejecutar_simulacion`.

        Args:
            ruta: ruta del archivo de checkpoint.

        Returns:
            Instancia restaurada de `SimulacionSatelites`.

        Raises:
            ValueError: si el checkpoint fue guardado con una versión
                de esquema incompatible con el código actual.
        """
        with open(ruta, "rb") as f:
            simulacion = pickle.load(f)
        version_archivo = getattr(
            simulacion, "_version_esquema_checkpoint", None
        )
        version_actual = SimulacionSatelites._VERSION_ESQUEMA_CHECKPOINT
        if version_archivo != version_actual:
            raise ValueError(
                f"Checkpoint con esquema incompatible: el archivo fue "
                f"guardado con versión {version_archivo!r} y el código "
                f"actual espera la versión {version_actual}. Regenera el "
                f"checkpoint con la versión actual del simulador."
            )
        # `Satellite._contador` es un atributo de clase y no se serializa
        # con el checkpoint: al restaurarlo en un proceso nuevo arrancaria
        # en 0 y los proximos satelites lanzados recibirian IDs ya usados
        # (colision de IDs que romperia `pariente_id` y los linajes).
        ids_existentes = [s.id for s in simulacion.satelites_activos]
        ids_existentes += [s.id for s in simulacion.satelites_muertos]
        if ids_existentes:
            Satellite._contador = max(Satellite._contador, max(ids_existentes))
        return simulacion
    
    def obtener_estadisticas_finales(self) -> Dict:
        """
        Estadísticas finales detalladas.

        Returns:
            Diccionario con totales de satélites lanzados/activos/muertos,
            reproducidos, causas de muerte agregadas, estado del debris
            y promedios finales (edad, energía, radiación, semieje mayor).
        """
        total_lanzados = len(self.satelites_activos) + len(self.satelites_muertos)
        
        causas_muerte = {}
        for s in self.satelites_muertos:
            causa = s.causa_muerte or "desconocida"
            causas_muerte[causa] = causas_muerte.get(causa, 0) + 1
        
        return {
            'tiempo_simulacion': str(self.tiempo_actual),
            'dias_totales': int(self.dias_transcurridos),
            'satelites_activos_finales': len(self.satelites_activos),
            'satelites_muertos_totales': len(self.satelites_muertos),
            'satelites_lanzados_totales': total_lanzados,
            'satelites_reproducidos': self.satelites_reproducidos,
            'causas_muerte': causas_muerte,
            'debris_final': int(self.debris.densidad),
            'debris_por_banda_final': {
                banda: round(cantidad, 1) for banda, cantidad in self.debris.densidad_por_banda.items()
            },
            'colisiones_totales': self.debris.colisiones_totales,
            'edad_promedio_activos_dias': self._edad_promedio(),
            'energia_promedio': self._energia_promedio(),
            'radiacion_promedio': self._radiacion_promedio(),
            'semieje_mayor_promedio_km': self._semieje_promedio(),
            'satelites_pasivados': self.satelites_pasivados,
            'debris_removido_total_adr': round(self.debris_removido_total, 1),
            'costo_total_lanzamientos': round(self.costo_total_lanzamientos, 2),
            'costo_total_remocion': round(self.costo_total_remocion, 2),
            'costo_total_sostenibilidad': round(self.costo_total_lanzamientos + self.costo_total_remocion, 2),
        }
    
    def listar_satelites_activos(self, limite: int = 10) -> List[Dict]:
        """
        Lista satélites activos con info completa.

        Args:
            limite: cantidad máxima de satélites a incluir.

        Returns:
            Lista de diccionarios (ver :meth:`Satellite.info`) con los
            primeros `limite` satélites activos.
        """
        return [sat.info() for sat in self.satelites_activos[:limite]]


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    falla = DistribucionFalla(
        tasa_base=0.0003,
        parametro_forma=2.0,
        escala_tiempo=3.0
    )
    
    sim = SimulacionSatelites(
        distribucion_falla=falla,
        intervalo_lanzamiento_dias=30,
        satelites_por_lanzamiento=5,
        edad_maxima_anos=5,
        tiempo_inicio=datetime(2024, 1, 1),
        reproductor_habilitado=True,
        tasa_mutacion=0.05
    )
    
    sim.lanzar_satelites()
    historial = sim.ejecutar_simulacion(dias_totales=1825)
    
    print("\n" + "="*70)
    print("ESTADÍSTICAS FINALES - SIMULACIÓN AVANZADA")
    print("="*70)
    stats = sim.obtener_estadisticas_finales()
    for clave, valor in stats.items():
        print(f"{clave:.<45} {valor}")
    
    print("\n" + "="*70)
    print("SATÉLITES ACTIVOS (muestra)")
    print("="*70)
    activos = sim.listar_satelites_activos(limite=5)
    for sat in activos:
        print(f"\nSAT-{sat['id']} (padre: SAT-{sat['pariente_id']})")
        print(f"  Calidad: {sat['calidad_inherente']:.2f} | λ_inherente: {sat['lambda_inherente']:.6f}/día")
        print(f"  Altitud: {sat['altitud_km']:.0f} km | Energía: {sat['energia']:.1f}%")
        print(f"  Combustible: {sat['combustible']:.1f}% | Radiación: {sat['radiacion_acumulada']:.3f}")
        print(f"  Masa: {sat['masa_kg']:.1f} kg | Degradación: {sat['degradacion_orbital']:.4f}")
