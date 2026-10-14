"""
Suite de tests para `sat.py`.

Cubre: modelo de `Satellite` (calidad inherente, heredabilidad,
`__slots__`), `DebrisEspacial` (bandas orbitales, batch vs. escalar),
las funciones vectorizadas `*_batch` (equivalencia numérica contra sus
contrapartes escalares) y `SimulacionSatelites` (reproducibilidad con
semilla, muestreo de `historial`, checkpointing).

Ejecutar con:
    python -m pytest tests/ -v
"""

import pickle
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pytest

from sat import (
    ConfiguracionFisica,
    Satellite,
    DebrisEspacial,
    CicloSolar,
    RadiacionCosmica,
    SimulacionSatelites,
    DistribucionFalla,
    aplicar_drag_atmosferico_batch,
    aplicar_presion_radiacion_solar_batch,
    aplicar_radiacion_cosmica_batch,
    actualizar_metabolismo_batch,
    calcular_lambda_total_batch,
    propagar_elementos_orbitales_batch,
    fraccion_dia_en_eclipse,
    calcular_beta_rad,
    aplicar_costo_evasion_batch,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def config():
    return ConfiguracionFisica()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _crear_satelite(config, rng, **overrides):
    """Crea un Satellite de prueba en órbita LEO típica."""
    kwargs: Dict[str, Any] = dict(
        a=7000.0, e=0.001, i=45.0, Omega=10.0, omega=20.0, M=30.0,
        tiempo_lanzamiento=datetime(2024, 1, 1),
        masa_kg=1000.0, config=config, rng=rng,
    )
    kwargs.update(overrides)
    return Satellite(**kwargs)


# =============================================================================
# Satellite: calidad inherente, __slots__, heredabilidad
# =============================================================================

class TestSatellite:
    def test_calidad_inherente_explicita_se_clippea(self, config, rng):
        sat = _crear_satelite(config, rng, calidad_inherente=5.0)
        assert sat.calidad_inherente == 1.0

        sat_bajo = _crear_satelite(config, rng, calidad_inherente=-3.0)
        assert sat_bajo.calidad_inherente == 0.1

    def test_calidad_inherente_default_en_rango(self, config, rng):
        for _ in range(50):
            sat = _crear_satelite(config, rng, calidad_inherente=None)
            assert 0.1 <= sat.calidad_inherente <= 1.0

    def test_lambda_inherente_formula(self, config, rng):
        sat = _crear_satelite(config, rng, calidad_inherente=1.0)
        assert sat.lambda_inherente == pytest.approx(0.0)

        sat_malo = _crear_satelite(config, rng, calidad_inherente=0.1)
        assert sat_malo.lambda_inherente == pytest.approx((1.0 - 0.1) * 0.001)

    def test_slots_impide_atributos_nuevos(self, config, rng):
        sat = _crear_satelite(config, rng)
        with pytest.raises(AttributeError):
            sat.atributo_inexistente = 123  # type: ignore[attr-defined]

    def test_altitud_y_periodo_orbital_positivos(self, config, rng):
        sat = _crear_satelite(config, rng)
        assert sat.altitud_km() > 0
        assert sat.periodo_orbital() > 0
        assert sat.velocidad_orbital_kms() > 0

    def test_crear_hijo_hereda_calidad_con_mutacion_acotada(self, config, rng):
        padre = _crear_satelite(config, rng, calidad_inherente=0.7)
        hijo = padre.crear_hijo(datetime(2024, 2, 1), tasa_mutacion=0.05, rng=rng)

        assert hijo.pariente_id == padre.id
        assert 0.1 <= hijo.calidad_inherente <= 1.0
        # La mutación de calidad tiene sigma=0.03: en la práctica no debería
        # desviarse groseramente del valor del padre.
        assert abs(hijo.calidad_inherente - padre.calidad_inherente) < 0.3

    def test_crear_hijo_parametros_orbitales_en_rango(self, config, rng):
        padre = _crear_satelite(config, rng)
        hijo = padre.crear_hijo(datetime(2024, 2, 1), rng=rng)
        assert 0.0 <= hijo.e <= 0.3
        assert 0.0 <= hijo.i <= 90.0
        assert 0.0 <= hijo.Omega < 360.0
        assert 0.0 <= hijo.omega < 360.0

    def test_pasivar_purga_combustible_y_energia(self, config, rng):
        sat = _crear_satelite(config, rng, combustible=80.0, energia=90.0)
        sat.pasivar()
        assert sat.combustible == 0.0
        assert sat.energia == 0.0
        assert sat.pasivado is True


# =============================================================================
# DebrisEspacial
# =============================================================================

class TestDebrisEspacial:
    def test_probabilidad_colision_en_rango(self, config):
        debris = DebrisEspacial(densidad_inicial=500, config=config)
        for altitud in (200, 800, 10000, 40000):
            prob = debris.probabilidad_colision(altitud)
            assert 0.0 <= prob <= 0.1

    def test_probabilidad_colision_batch_coincide_con_escalar(self, config):
        debris = DebrisEspacial(densidad_inicial=800, config=config)
        altitudes = np.array([150.0, 500.0, 1200.0, 5000.0, 40000.0, 300.0])

        esperado = np.array([debris.probabilidad_colision(a) for a in altitudes])
        obtenido = debris.probabilidad_colision_batch(altitudes)

        np.testing.assert_allclose(obtenido, esperado)

    def test_probabilidad_colision_batch_coincide_con_escalar_fuera_de_bandas(self, config):
        # Las altitudes fuera de toda banda (negativas, por encima del
        # tope GEO) deben usar el mismo fallback que la version escalar
        # (_banda_de_altitud), no quedar con riesgo 0.
        debris = DebrisEspacial(densidad_inicial=800, config=config)
        altitudes = np.array([-50.0, -0.1, 0.0, 399.999, 2000.0, 35786.0, 1e9])

        esperado = np.array([debris.probabilidad_colision(a) for a in altitudes])
        obtenido = debris.probabilidad_colision_batch(altitudes)

        np.testing.assert_allclose(obtenido, esperado)

    def test_generar_debris_por_colision_incrementa_banda(self, config):
        debris = DebrisEspacial(densidad_inicial=100, config=config)
        antes = debris.densidad_por_banda["leo_congestionado"]
        debris.generar_debris_por_colision(altitud_km=800, masa_kg=1000, velocidad_kms=10)
        assert debris.densidad_por_banda["leo_congestionado"] > antes


# =============================================================================
# Funciones vectorizadas *_batch vs. equivalentes escalares
# =============================================================================

class TestFuncionesBatch:
    def _crear_poblacion(self, config, rng, n=15):
        satelites = []
        for idx in range(n):
            sat = Satellite(
                a=6700.0 + idx * 20, e=0.001 * idx, i=30.0 + idx,
                Omega=idx * 10 % 360, omega=idx * 15 % 360, M=idx * 5 % 360,
                tiempo_lanzamiento=datetime(2024, 1, 1) if idx % 2 == 0 else datetime(2023, 6, 1),
                masa_kg=500.0 + idx * 50, config=config, rng=rng,
            )
            sat.energia = 100.0 - idx * 3
            sat.combustible = 100.0 - idx * 4
            sat.radiacion_acumulada = 0.01 * idx
            satelites.append(sat)
        return satelites

    def _clonar_poblacion(self, poblacion, config, rng):
        """Clona una población de satélites preservando su estado mutable
        (energía, combustible, radiación acumulada), para poder comparar
        una ruta de cálculo `_batch` contra la equivalente escalar sin que
        ambas partan de estados distintos."""
        clones = []
        for s in poblacion:
            clon = Satellite(s.a, s.e, s.i, s.Omega, s.omega, s.M, s.tiempo_lanzamiento,
                             masa_kg=s.masa_kg, config=config, rng=rng)
            clon.energia = s.energia
            clon.combustible = s.combustible
            clon.radiacion_acumulada = s.radiacion_acumulada
            clones.append(clon)
        return clones

    def test_drag_batch_coincide_con_escalar(self, config, rng):
        poblacion_batch = self._crear_poblacion(config, rng)
        poblacion_escalar = self._clonar_poblacion(poblacion_batch, config, rng)

        ciclo_solar = CicloSolar()
        dias = 42.0

        mascara_reentrada = aplicar_drag_atmosferico_batch(poblacion_batch, ciclo_solar, dias, config)
        for sat in poblacion_escalar:
            sat.aplicar_drag_atmosferico(ciclo_solar, dias)

        a_batch = np.array([s.a for s in poblacion_batch])
        a_escalar = np.array([s.a for s in poblacion_escalar])
        # Nota: `densidad_atmosferica_kg_m3_batch` usa interpolación
        # log-lineal entre capas (más precisa), mientras que la versión
        # escalar `densidad_atmosferica_kg_m3` aplica el decaimiento
        # exponencial simple desde la base de la capa (ver sus
        # docstrings en sat.py). Por diseño no son bit-a-bit idénticas;
        # se tolera la pequeña diferencia relativa esperada entre ambos
        # modelos de densidad atmosférica.
        np.testing.assert_allclose(a_batch, a_escalar, rtol=1e-4)

        reentrada_escalar = np.array([not s.activo for s in poblacion_escalar])
        np.testing.assert_array_equal(mascara_reentrada, reentrada_escalar)

    def test_presion_radiacion_solar_batch_coincide_con_escalar(self, config, rng):
        poblacion_batch = self._crear_poblacion(config, rng)
        poblacion_escalar = self._clonar_poblacion(poblacion_batch, config, rng)
        ciclo_solar = CicloSolar()

        aplicar_presion_radiacion_solar_batch(poblacion_batch, ciclo_solar, 10.0)
        for sat in poblacion_escalar:
            sat.aplicar_presion_radiacion_solar(ciclo_solar, 10.0)

        e_batch = np.array([s.e for s in poblacion_batch])
        e_escalar = np.array([s.e for s in poblacion_escalar])
        np.testing.assert_allclose(e_batch, e_escalar, rtol=1e-10)

    def test_radiacion_cosmica_batch_coincide_con_escalar(self, config, rng):
        poblacion_batch = self._crear_poblacion(config, rng)
        poblacion_escalar = self._clonar_poblacion(poblacion_batch, config, rng)
        radiacion_cosmica = RadiacionCosmica()

        mascara_muerte = aplicar_radiacion_cosmica_batch(poblacion_batch, radiacion_cosmica)
        for sat in poblacion_escalar:
            sat.aplicar_radiacion_cosmica(radiacion_cosmica)

        rad_batch = np.array([s.radiacion_acumulada for s in poblacion_batch])
        rad_escalar = np.array([s.radiacion_acumulada for s in poblacion_escalar])
        np.testing.assert_allclose(rad_batch, rad_escalar, rtol=1e-10)

        muerte_escalar = np.array([not s.activo for s in poblacion_escalar])
        np.testing.assert_array_equal(mascara_muerte, muerte_escalar)

    def test_metabolismo_batch_coincide_con_escalar(self, config, rng):
        poblacion_batch = self._crear_poblacion(config, rng)
        poblacion_escalar = self._clonar_poblacion(poblacion_batch, config, rng)
        for s in poblacion_batch:
            s.energia = 100.0
        for s in poblacion_escalar:
            s.energia = 100.0

        mascara_sin_energia = actualizar_metabolismo_batch(poblacion_batch)
        for sat in poblacion_escalar:
            sat.recargar_energia(10.0)
            sat.consumir_energia()

        energia_batch = np.array([s.energia for s in poblacion_batch])
        energia_escalar = np.array([s.energia for s in poblacion_escalar])
        np.testing.assert_allclose(energia_batch, energia_escalar, rtol=1e-10)

        muerte_escalar = np.array([not s.activo for s in poblacion_escalar])
        np.testing.assert_array_equal(mascara_sin_energia, muerte_escalar)

    def test_lambda_total_batch_coincide_con_escalar(self, config, rng):
        poblacion = self._crear_poblacion(config, rng)
        ciclo_solar = CicloSolar()
        edades = np.array([sat.edad_dias(datetime(2024, 6, 1)) for sat in poblacion])
        riesgos_debris = np.array([0.01 * (idx + 1) for idx in range(len(poblacion))])

        lambda_batch = calcular_lambda_total_batch(poblacion, edades, riesgos_debris, config)
        lambda_escalar = np.array([
            sat.calcular_lambda_total(edad, riesgo)
            for sat, edad, riesgo in zip(poblacion, edades, riesgos_debris)
        ])

        np.testing.assert_allclose(lambda_batch, lambda_escalar, rtol=1e-10)

    def test_metabolismo_batch_respeta_recarga_solar_configurable(self, config, rng):
        # La recarga solar (+10 por defecto) debe venir de la
        # configuracion, no estar hardcodeada en la funcion batch.
        # eclipses desactivados para probar solo la recarga configurable
        config_recarga = ConfiguracionFisica(
            recarga_solar_diaria_pct=20.0, eclipses_habilitados=False
        )
        poblacion = self._crear_poblacion(config_recarga, rng)
        for sat in poblacion:
            sat.energia = 50.0
        actualizar_metabolismo_batch(poblacion, config_recarga)
        # 50 + 20 de recarga - 5 de consumo
        np.testing.assert_allclose(
            np.array([s.energia for s in poblacion]), 65.0, rtol=1e-10
        )


    def test_propagacion_anomalia_media(self, config, rng):
        # a=6771 km (400 km alt): n = sqrt(GM/a^3); tras 1 dia M avanza
        # n*86400 en grados (mod 360).
        sat = _crear_satelite(config, rng, a=6771.0)
        M_inicial = sat.M
        propagar_elementos_orbitales_batch([sat], 1.0, config)
        n_rad_s = np.sqrt(config.gm_tierra_km3_s2 / 6771.0 ** 3)
        esperado = (M_inicial + np.degrees(n_rad_s * 86400.0)) % 360.0
        assert sat.M == pytest.approx(esperado, abs=1e-6)

    def test_propagacion_respetada_como_opt_in(self, config, rng):
        cfg_off = ConfiguracionFisica(propagar_anomalia_media=False,
                                      precesion_j2_habilitada=False)
        sat = _crear_satelite(config, rng, a=6771.0, Omega=42.0)
        M_inicial, Omega_inicial = sat.M, sat.Omega
        propagar_elementos_orbitales_batch([sat], 10.0, cfg_off)
        assert sat.M == M_inicial
        assert sat.Omega == Omega_inicial

    def test_precesion_j2_iss(self, config, rng):
        # ISS-like (i=51.6, a=6771): deriva del RAAN ~ -5 grados/dia.
        sat = _crear_satelite(config, rng, a=6771.0, i=51.6, e=0.001, Omega=0.0)
        propagar_elementos_orbitales_batch([sat], 1.0, config)
        assert -5.5 < (sat.Omega - 360.0 if sat.Omega > 180 else sat.Omega) < -4.5

    def test_precesion_j2_nula_en_orbita_polar(self, config, rng):
        # cos(90)=0: una orbita polar no deriva su RAAN por J2.
        sat = _crear_satelite(config, rng, a=6771.0, i=90.0, e=0.001, Omega=10.0)
        propagar_elementos_orbitales_batch([sat], 365.0, config)
        assert sat.Omega == pytest.approx(10.0, abs=1e-9)

    def test_fraccion_eclipse_beta_cero_y_sin_eclipse(self, config):
        # beta=0 y alt=400 km: fraccion = 2*rho_e/(2*pi) ~ 0.111
        f = fraccion_dia_en_eclipse(0.0, 400.0, config)
        rho = np.arccos(config.radio_tierra_km / (400.0 + config.radio_tierra_km)) \
            + config.radio_angular_sol_rad
        assert f == pytest.approx(2 * rho / (2 * np.pi), rel=1e-9)
        assert 0.0 < f < 0.5
        # beta mayor al angulo de eclipse: orbita siempre iluminada
        assert fraccion_dia_en_eclipse(1.4, 400.0, config) == 0.0

    def test_beta_maximo_en_solsticio(self, config):
        # Solsticio (~dia 172), i=51.6: beta ~ 23.44*sin(51.6) = 18.3 deg.
        beta = calcular_beta_rad(172, 51.6)
        assert np.degrees(beta) == pytest.approx(18.3, abs=0.2)

    def test_eclipses_reducen_recarga_solar(self, config, rng):
        # Con eclipses activados, un satelite con beta pequeño recarga
        # menos que uno en orbita siempre iluminada; nunca menos del piso.
        cfg_ecl = ConfiguracionFisica(eclipses_habilitados=True)
        sat_beta_bajo = _crear_satelite(config, rng, i=51.6)
        sat_beta_bajo.energia = 50.0
        actualizar_metabolismo_batch([sat_beta_bajo], cfg_ecl,
                                     dias_transcurridos=100)
        # recarga efectiva < nominal (10): energia < 50+10-5 = 55
        assert sat_beta_bajo.energia < 55.0
        assert sat_beta_bajo.energia > 50.0  # el piso evita quedarse a 0 de recarga
        # Sin eclipses (control): recarga nominal completa
        cfg_no_ecl = ConfiguracionFisica(eclipses_habilitados=False)
        sat_control = _crear_satelite(config, rng, i=51.6)
        sat_control.energia = 50.0
        actualizar_metabolismo_batch([sat_control], cfg_no_ecl,
                                     dias_transcurridos=100)
        assert sat_control.energia == pytest.approx(55.0)


# =============================================================================
# SimulacionSatelites: reproducibilidad, historial, checkpointing
# =============================================================================

class TestSimulacionSatelites:
    def _correr(self, semilla, dias=40, **kwargs):
        sim = SimulacionSatelites(
            tiempo_inicio=datetime(2024, 1, 1),
            intervalo_lanzamiento_dias=15,
            satelites_por_lanzamiento=4,
            edad_maxima_anos=2,
            semilla=semilla,
            **kwargs,
        )
        sim.lanzar_satelites()
        sim.ejecutar_simulacion(dias_totales=dias)
        return sim

    def test_simulacion_sin_nan_ni_valores_fuera_de_rango(self):
        sim = self._correr(semilla=1)
        for stats in sim.historial:
            assert stats["satelites_activos"] >= 0
            assert not np.isnan(stats["calidad_promedio"])
            assert not np.isnan(stats["edad_promedio_dias"])
            assert 0.0 <= stats["densidad_debris"] <= 1.0
        for sat in sim.satelites_activos:
            assert not np.isnan(sat.a)
            assert 0.1 <= sat.calidad_inherente <= 1.0

    def test_misma_semilla_produce_resultados_identicos(self):
        sim1 = self._correr(semilla=123)
        sim2 = self._correr(semilla=123)

        assert len(sim1.satelites_activos) == len(sim2.satelites_activos)
        assert len(sim1.satelites_muertos) == len(sim2.satelites_muertos)
        assert sim1.historial[-1]["calidad_promedio"] == pytest.approx(
            sim2.historial[-1]["calidad_promedio"]
        )
        assert [s.a for s in sim1.satelites_activos] == pytest.approx(
            [s.a for s in sim2.satelites_activos]
        )

    def test_semillas_distintas_producen_resultados_distintos(self):
        sim1 = self._correr(semilla=1)
        sim2 = self._correr(semilla=2)
        # Con alta probabilidad los semiejes mayores no serán idénticos.
        assert [s.a for s in sim1.satelites_activos] != [s.a for s in sim2.satelites_activos]

    def test_historial_intervalo_dias_reduce_entradas(self):
        dias = 40
        sim_diario = self._correr(semilla=9, dias=dias, historial_intervalo_dias=1)
        sim_muestreado = self._correr(semilla=9, dias=dias, historial_intervalo_dias=10)

        assert len(sim_diario.historial) == dias
        assert len(sim_muestreado.historial) == dias // 10
        # `ultimo_stats` siempre refleja el último día simulado, aunque no
        # se haya guardado en el historial.
        assert sim_muestreado.ultimo_stats is not None
        assert sim_muestreado.ultimo_stats["dias"] == dias

    def test_checkpoint_guarda_y_restaura_estado(self, tmp_path):
        sim = self._correr(semilla=55, dias=30)
        ruta = tmp_path / "checkpoint.pkl"
        sim.guardar_checkpoint(str(ruta))

        assert ruta.exists()
        restaurado = SimulacionSatelites.cargar_checkpoint(str(ruta))

        assert restaurado.dias_transcurridos == sim.dias_transcurridos
        assert len(restaurado.satelites_activos) == len(sim.satelites_activos)
        assert len(restaurado.historial) == len(sim.historial)

        # Debe poder seguir avanzando sin errores tras restaurarse.
        restaurado.ejecutar_simulacion(dias_totales=5)
        assert restaurado.dias_transcurridos == sim.dias_transcurridos + 5

    def test_checkpoint_restaura_contador_de_ids(self, tmp_path):
        # `Satellite._contador` es un atributo de clase y no viaja en el
        # pickle de la simulacion: cargar un checkpoint en un proceso
        # nuevo no debe reutilizar IDs ya asignados (romperia linajes).
        sim = self._correr(semilla=57, dias=20)
        ruta = tmp_path / "checkpoint_ids.pkl"
        sim.guardar_checkpoint(str(ruta))

        contador_previo = Satellite._contador
        Satellite._contador = 0  # simula un proceso nuevo
        try:
            restaurado = SimulacionSatelites.cargar_checkpoint(str(ruta))
            ids_existentes = {s.id for s in
                               restaurado.satelites_activos + restaurado.satelites_muertos}
            nuevos = restaurado.lanzar_satelites()
            for sat in nuevos:
                assert sat.id not in ids_existentes
            assert Satellite._contador >= max(ids_existentes)
        finally:
            Satellite._contador = contador_previo

    def test_ejecutar_simulacion_parametro_heredables(self):
        # El README documenta `ejecutar_simulacion(..., heredables=...)`;
        # debe sobrescribir reproductor_habilitado durante la ejecucion.
        sim = SimulacionSatelites(
            tiempo_inicio=datetime(2024, 1, 1), semilla=61,
            reproductor_habilitado=True,
        )
        sim.lanzar_satelites()
        sim.ejecutar_simulacion(dias_totales=10, heredables=False)
        assert sim.reproductor_habilitado is False

        sim2 = SimulacionSatelites(
            tiempo_inicio=datetime(2024, 1, 1), semilla=62,
            reproductor_habilitado=False,
        )
        sim2.lanzar_satelites()
        sim2.ejecutar_simulacion(dias_totales=10, heredables=True)
        assert sim2.reproductor_habilitado is True

    def test_reentrada_orbital_no_genera_debris(self):
        # Un satelite que reentra se desintegra en la atmosfera: no debe
        # dejar un derelicto en su banda orbital como las otras causas.
        sim = SimulacionSatelites(
            tiempo_inicio=datetime(2024, 1, 1), semilla=63,
            intervalo_lanzamiento_dias=1000,  # sin lanzamientos nuevos
        )
        sim.lanzar_satelites()
        debris_antes = sim.debris.densidad
        for sat in sim.satelites_activos:
            sat.a = sim.config.radio_tierra_km + 50.0  # fuerza reentrada
        sim.paso_tiempo(1)
        reentrados = [s for s in sim.satelites_muertos
                      if s.causa_muerte == "reentrada_orbital"]
        assert len(reentrados) > 0
        assert sim.debris.densidad <= debris_antes


    def test_checkpoint_rechaza_esquema_incompatible(self, tmp_path):
        # Un checkpoint guardado con otra version de esquema debe
        # rechazarse con un error claro, no fallar de forma
        # impredecible al deserializar.
        sim = self._correr(semilla=71, dias=10)
        ruta = tmp_path / "checkpoint_bad.pkl"
        sim.guardar_checkpoint(str(ruta))
        with open(ruta, "rb") as f:
            objeto = pickle.load(f)
        objeto._version_esquema_checkpoint = (
            SimulacionSatelites._VERSION_ESQUEMA_CHECKPOINT + 1
        )
        with open(ruta, "wb") as f:
            pickle.dump(objeto, f)
        with pytest.raises(ValueError, match="esquema incompatible"):
            SimulacionSatelites.cargar_checkpoint(str(ruta))

    def test_distribucion_falla_legacy_emite_deprecation_warning(self):
        # DistribucionFalla.probabilidad_falla_diaria es legacy: el
        # motor usa Satellite.calcular_lambda_total (tasas aditivas).
        with pytest.warns(DeprecationWarning, match="deprecado"):
            DistribucionFalla().probabilidad_falla_diaria(100.0)

    def test_checkpoint_es_picklable_directamente(self):
        sim = self._correr(semilla=3, dias=10)
        datos = pickle.dumps(sim)
        restaurado = pickle.loads(datos)
        assert len(restaurado.satelites_activos) == len(sim.satelites_activos)

    def test_ejecutar_simulacion_checkpoint_automatico(self, tmp_path):
        ruta = tmp_path / "auto_checkpoint.pkl"
        sim = SimulacionSatelites(
            tiempo_inicio=datetime(2024, 1, 1), semilla=8,
            satelites_por_lanzamiento=3,
        )
        sim.lanzar_satelites()
        sim.ejecutar_simulacion(
            dias_totales=20, checkpoint_cada_dias=10, checkpoint_ruta=str(ruta)
        )
        assert ruta.exists()
        restaurado = SimulacionSatelites.cargar_checkpoint(str(ruta))
        # El último checkpoint automático se guardó en el día 20 (múltiplo
        # de 10 más cercano al final).
        assert restaurado.dias_transcurridos == 20.0

    def test_lanzar_satelites_reproduccion_selectiva_genera_hijos(self):
        sim = SimulacionSatelites(
            tiempo_inicio=datetime(2024, 1, 1), semilla=21,
            satelites_por_lanzamiento=6, reproductor_habilitado=True,
        )
        sim.lanzar_satelites()  # primeros satélites (no heredables)
        nuevos = sim.lanzar_satelites(heredables=True)

        assert len(nuevos) == 6
        assert sim.satelites_reproducidos > 0
        hijos = [s for s in nuevos if s.pariente_id is not None]
        assert len(hijos) == sim.satelites_reproducidos

    def test_obtener_estadisticas_finales_consistente(self):
        sim = self._correr(semilla=4, dias=30)
        stats = sim.obtener_estadisticas_finales()
        assert stats["satelites_activos_finales"] == len(sim.satelites_activos)
        assert stats["satelites_muertos_totales"] == len(sim.satelites_muertos)
        assert stats["dias_totales"] == 30


# =============================================================================
# Mecanismos evolutivos a nivel de constelación (selección por aptitud,
# recombinación sexual, capacidad de carga, r/K, evasión de debris)
# =============================================================================

class TestMecanismosEvolutivosConstelacion:
    def test_costo_evasion_consume_combustible(self, config, rng):
        # Evadir debris cuesta combustible proporcional a la capacidad
        # de evasion; bajo el umbral de combustible no hay maniobra ni costo.
        sat = _crear_satelite(config, rng, maniobrabilidad_evasiva=1.0,
                              combustible=50.0)
        aplicar_costo_evasion_batch([sat], config)
        assert sat.combustible == pytest.approx(
            50.0 - config.costo_combustible_evasion_diario_pct
        )
        sat_bajo = _crear_satelite(config, rng, maniobrabilidad_evasiva=1.0,
                                   combustible=config.umbral_combustible_bajo_pct - 1.0)
        aplicar_costo_evasion_batch([sat_bajo], config)
        assert sat_bajo.combustible == pytest.approx(
            config.umbral_combustible_bajo_pct - 1.0
        )

    def test_bono_cobertura_banda_favorece_nicho_minoritario(self):
        # Satelites de la banda minoritaria entre los candidatos deben
        # recibir un bono mayor que los de la banda dominante.
        sim = SimulacionSatelites(
            tiempo_inicio=datetime(2024, 1, 1), semilla=91,
            satelites_por_lanzamiento=5,
        )
        sim.lanzar_satelites()
        # Forzar candidatos de dos bandas: 2 LEO + 1 MEO
        cands = sim.satelites_activos[:3]
        cands[0].a = sim.config.radio_tierra_km + 800.0
        cands[1].a = sim.config.radio_tierra_km + 800.0
        cands[2].a = sim.config.radio_tierra_km + 5000.0
        bonos = sim._bono_cobertura_banda(cands)
        assert bonos[2] > bonos[0] == bonos[1]
        # Con peso 0 el bono queda desactivado
        cfg_off = ConfiguracionFisica(peso_bono_cobertura_banda=0.0)
        sim.config = cfg_off
        assert np.all(sim._bono_cobertura_banda(cands) == 1.0)

    def test_especiacion_aislamiento_reproductivo_por_banda(self):
        # Con prob_reproduccion_sexual=1 y aislamiento activo, todo hijo
        # con 2 padres debe tener padres de la misma banda orbital.
        sim = SimulacionSatelites(
            tiempo_inicio=datetime(2024, 1, 1), semilla=92,
            satelites_por_lanzamiento=20, prob_reproduccion_sexual=1.0,
        )
        sim.lanzar_satelites()
        sim.ejecutar_simulacion(dias_totales=90)
        todos = {s.id: s for s in sim.satelites_activos + sim.satelites_muertos}
        con_dos_padres = [s for s in todos.values()
                          if s.pariente_id_secundario is not None]
        assert con_dos_padres, "deberia haber hijos sexuales en 90 dias"
        for hijo in con_dos_padres:
            p1 = todos.get(hijo.pariente_id)
            p2 = todos.get(hijo.pariente_id_secundario)
            if p1 is None or p2 is None:
                continue  # padre ya no rastreable (checkpoint parcial)
            banda1 = sim.debris.banda_de_altitud(p1.altitud_km())
            banda2 = sim.debris.banda_de_altitud(p2.altitud_km())
            assert banda1 == banda2

    def test_punto_no_retorno_kessler(self):
        # Con cascada sostenida (riesgo > 1 durante N dias consecutivos),
        # la banda se declara en punto de no retorno y queda registrada.
        cfg = ConfiguracionFisica(dias_sostenidos_punto_no_retorno=3)
        sim = SimulacionSatelites(
            tiempo_inicio=datetime(2024, 1, 1), semilla=93,
            satelites_por_lanzamiento=60, intervalo_lanzamiento_dias=5,
            config=cfg,
        )
        sim.lanzar_satelites()
        # Forzar una cascada: debris enorme en LEO genera colisiones
        # diarias sostenidas.
        sim.debris.densidad_por_banda["leo_congestionado"] = 5000.0
        sim.ejecutar_simulacion(dias_totales=40)
        assert "leo_congestionado" in sim.bandas_punto_no_retorno
        assert "leo_congestionado" in sim.historial[-1]["bandas_punto_no_retorno"]

    def test_mutacion_saltacional_rara(self, config, rng):
        # Con prob 1.0, todo hijo asexual sufre un salto en un rasgo
        # (uniforme en el rango completo); con prob 0.0, nunca.
        cfg_salto = ConfiguracionFisica(prob_mutacion_saltacional=1.0)
        padre = _crear_satelite(
            config, rng, calidad_inherente=0.9,
            agresividad_reproductiva=0.9, maniobrabilidad_evasiva=0.9,
        )
        padre.config = cfg_salto
        rasgos_saltales = 0
        for _ in range(60):
            hijo = padre.crear_hijo(datetime(2024, 1, 2), rng=rng)
            if (hijo.calidad_inherente < 0.75
                    or hijo.agresividad_reproductiva < 0.75
                    or hijo.maniobrabilidad_evasiva < 0.75):
                rasgos_saltales += 1
        assert rasgos_saltales > 30  # con prob 1.0, la mayoria salta lejos
        cfg_cero = ConfiguracionFisica(prob_mutacion_saltacional=0.0)
        padre.config = cfg_cero
        for _ in range(60):
            hijo = padre.crear_hijo(datetime(2024, 1, 2), rng=rng)
            # sin saltos: los tres rasgos quedan cerca del parental
            assert hijo.calidad_inherente > 0.75
            assert hijo.agresividad_reproductiva > 0.75
            assert hijo.maniobrabilidad_evasiva > 0.75

    def test_mutacion_saltacional_en_via_sexual(self, config, rng):
        # Igual que la via asexual: el salto tambien aplica al crossover.
        cfg_salto = ConfiguracionFisica(prob_mutacion_saltacional=1.0)
        p1 = _crear_satelite(config, rng, calidad_inherente=0.9,
                             agresividad_reproductiva=0.9,
                             maniobrabilidad_evasiva=0.9)
        p2 = _crear_satelite(config, rng, calidad_inherente=0.9,
                             agresividad_reproductiva=0.9,
                             maniobrabilidad_evasiva=0.9)
        p1.config = p2.config = cfg_salto
        saltos = 0
        for _ in range(60):
            hijo = p1.crear_hijo_sexual(p2, datetime(2024, 1, 2), rng=rng)
            if (hijo.calidad_inherente < 0.75
                    or hijo.agresividad_reproductiva < 0.75
                    or hijo.maniobrabilidad_evasiva < 0.75):
                saltos += 1
        assert saltos > 30

    def test_fitness_reproductivo_es_calidad_inherente(self, config, rng):
        sat = _crear_satelite(config, rng, calidad_inherente=0.83)
        assert sat.fitness_reproductivo() == pytest.approx(0.83)

    def test_crear_hijo_sexual_combina_ambos_padres(self, config, rng):
        padre1 = _crear_satelite(
            config, rng, calidad_inherente=0.9, a=6700.0,
            agresividad_reproductiva=0.2, maniobrabilidad_evasiva=0.8,
        )
        padre2 = _crear_satelite(
            config, rng, calidad_inherente=0.3, a=7100.0,
            agresividad_reproductiva=0.8, maniobrabilidad_evasiva=0.2,
        )
        hijo = padre1.crear_hijo_sexual(padre2, datetime(2024, 2, 1), tasa_mutacion=0.05, rng=rng)

        assert hijo.pariente_id == padre1.id
        assert hijo.pariente_id_secundario == padre2.id
        # Los rasgos heredados deben quedar en rango válido (el crossover
        # es una combinación convexa de ambos padres + mutación pequeña,
        # por lo que deberían quedar cerca del rango de los padres).
        assert 0.1 <= hijo.calidad_inherente <= 1.0
        assert 0.0 <= hijo.agresividad_reproductiva <= 1.0
        assert 0.0 <= hijo.maniobrabilidad_evasiva <= 1.0
        # `a` no se clippea (igual que en la reproducción asexual), pero
        # debería seguir siendo un valor razonable cercano al de los
        # padres (crossover + mutación pequeña), no un valor disparatado.
        assert 5000.0 <= hijo.a <= 9000.0

    def test_crear_hijo_asexual_hereda_rasgos_rk_y_evasion(self, config, rng):
        padre = _crear_satelite(
            config, rng, agresividad_reproductiva=0.4, maniobrabilidad_evasiva=0.6
        )
        hijo = padre.crear_hijo(datetime(2024, 2, 1), rng=rng)
        assert hijo.pariente_id_secundario is None
        assert 0.0 <= hijo.agresividad_reproductiva <= 1.0
        assert 0.0 <= hijo.maniobrabilidad_evasiva <= 1.0
        # Con sigma=0.05 no debería desviarse groseramente del padre.
        assert abs(hijo.agresividad_reproductiva - 0.4) < 0.3
        assert abs(hijo.maniobrabilidad_evasiva - 0.6) < 0.3

    def test_lambda_debris_reducido_por_evasion_con_combustible(self, config, rng):
        ciclo_solar = CicloSolar()
        sat_sin_evasion = _crear_satelite(
            config, rng, calidad_inherente=0.5,
            maniobrabilidad_evasiva=0.0, combustible=100.0
        )
        sat_con_evasion = _crear_satelite(
            config, rng, calidad_inherente=0.5,
            maniobrabilidad_evasiva=1.0, combustible=100.0
        )
        lam_sin = sat_sin_evasion.calcular_lambda_total(100, 0.05)
        lam_con = sat_con_evasion.calcular_lambda_total(100, 0.05)
        assert lam_con < lam_sin

    def test_evasion_no_reduce_lambda_sin_combustible(self, config, rng):
        ciclo_solar = CicloSolar()
        combustible_bajo = config.umbral_combustible_bajo_pct - 1.0
        sat_sin_evasion = _crear_satelite(
            config, rng, calidad_inherente=0.5,
            maniobrabilidad_evasiva=0.0, combustible=combustible_bajo
        )
        sat_con_evasion = _crear_satelite(
            config, rng, calidad_inherente=0.5,
            maniobrabilidad_evasiva=1.0, combustible=combustible_bajo
        )
        lam_sin = sat_sin_evasion.calcular_lambda_total(100, 0.05)
        lam_con = sat_con_evasion.calcular_lambda_total(100, 0.05)
        # Sin combustible suficiente para maniobrar, el rasgo de evasión
        # no tiene efecto (mismo lambda con o sin el rasgo).
        assert lam_con == pytest.approx(lam_sin)

    def test_lambda_total_batch_incluye_reduccion_por_evasion(self, config, rng):
        satelites = [
            _crear_satelite(config, rng, calidad_inherente=0.5,
                           maniobrabilidad_evasiva=0.0, combustible=100.0),
            _crear_satelite(config, rng, calidad_inherente=0.5,
                           maniobrabilidad_evasiva=1.0, combustible=100.0),
        ]
        edades = np.array([100.0, 100.0])
        riesgos = np.array([0.05, 0.05])
        lam = calcular_lambda_total_batch(satelites, edades, riesgos, config)
        assert lam[1] < lam[0]

    def test_calificacion_reproduccion_varia_con_agresividad(self):
        sim = SimulacionSatelites(tiempo_inicio=datetime(2024, 1, 1), semilla=1)
        sat_r = Satellite(7000.0, 0.001, 45.0, 0.0, 0.0, 0.0, datetime(2024, 1, 1),
                          config=sim.config, rng=sim.rng, agresividad_reproductiva=1.0,
                          energia=35.0, combustible=20.0)
        sat_k = Satellite(7000.0, 0.001, 45.0, 0.0, 0.0, 0.0, datetime(2024, 1, 1),
                          config=sim.config, rng=sim.rng, agresividad_reproductiva=0.0,
                          energia=35.0, combustible=20.0)
        # Con la misma energía/combustible bajos, el r-estratega (agresivo)
        # califica y el K-estratega (conservador) no.
        assert sim._califica_para_reproduccion(sat_r) is True
        assert sim._califica_para_reproduccion(sat_k) is False

    def test_pesos_seleccion_padres_favorece_mayor_calidad(self):
        sim = SimulacionSatelites(tiempo_inicio=datetime(2024, 1, 1), semilla=2)
        candidatos = [
            Satellite(7000.0, 0.001, 45.0, 10.0, 0.0, 0.0, datetime(2024, 1, 1),
                     config=sim.config, rng=sim.rng, calidad_inherente=1.0),
            Satellite(7000.0, 0.001, 45.0, 10.0, 0.0, 0.0, datetime(2024, 1, 1),
                     config=sim.config, rng=sim.rng, calidad_inherente=0.1),
        ]
        pesos = sim._pesos_seleccion_padres(candidatos)
        # Mismo RAAN (bono de diversidad neutro) → la proporción de pesos
        # debe reflejar exactamente la proporción de calidad inherente.
        assert pesos[0] / pesos[1] == pytest.approx(1.0 / 0.1)

    def test_bono_diversidad_orbital_favorece_posiciones_aisladas(self):
        sim = SimulacionSatelites(tiempo_inicio=datetime(2024, 1, 1), semilla=3)
        candidatos = [
            Satellite(7000.0, 0.001, 45.0, 0.0, 0.0, 0.0, datetime(2024, 1, 1),
                     config=sim.config, rng=sim.rng),
            Satellite(7000.0, 0.001, 45.0, 10.0, 0.0, 0.0, datetime(2024, 1, 1),
                     config=sim.config, rng=sim.rng),
            Satellite(7000.0, 0.001, 45.0, 200.0, 0.0, 0.0, datetime(2024, 1, 1),
                     config=sim.config, rng=sim.rng),
        ]
        bono = sim._bono_diversidad_orbital(candidatos)
        # El satélite aislado (RAAN=200, lejos del par 0/10) debe recibir
        # un bono de diversidad mayor que los dos que están muy juntos.
        assert bono[2] > bono[0]
        assert bono[2] > bono[1]

    def test_probabilidad_reproduccion_exitosa_disminuye_con_densidad(self):
        sim_vacio = SimulacionSatelites(tiempo_inicio=datetime(2024, 1, 1), semilla=4)
        sim_vacio.debris.densidad_por_banda["leo_congestionado"] = 0.0

        sim_congestionado = SimulacionSatelites(tiempo_inicio=datetime(2024, 1, 1), semilla=4)
        sim_congestionado.debris.densidad_por_banda["leo_congestionado"] = 1500.0  # satura la banda

        padre = Satellite(6771.0, 0.001, 45.0, 0.0, 0.0, 0.0, datetime(2024, 1, 1),
                         config=sim_vacio.config, rng=sim_vacio.rng)  # altitud ~400km

        prob_vacio = sim_vacio._probabilidad_reproduccion_exitosa(padre)
        prob_congestionado = sim_congestionado._probabilidad_reproduccion_exitosa(padre)

        assert prob_vacio == pytest.approx(1.0)
        assert prob_congestionado < prob_vacio
        assert prob_congestionado == pytest.approx(1.0 - sim_congestionado.config.factor_capacidad_carga)

    def test_simulacion_con_mecanismos_evolutivos_sin_nan(self):
        sim = SimulacionSatelites(
            tiempo_inicio=datetime(2024, 1, 1), semilla=11,
            satelites_por_lanzamiento=8, prob_reproduccion_sexual=0.5,
        )
        sim.lanzar_satelites()
        sim.ejecutar_simulacion(dias_totales=60)

        for sat in sim.satelites_activos + sim.satelites_muertos:
            assert 0.0 <= sat.agresividad_reproductiva <= 1.0
            assert 0.0 <= sat.maniobrabilidad_evasiva <= 1.0
            assert not np.isnan(sat.agresividad_reproductiva)
            assert not np.isnan(sat.maniobrabilidad_evasiva)

        # Con prob_reproduccion_sexual alto y suficientes generaciones,
        # debería haber al menos algún hijo con 2 padres registrados.
        todos = sim.satelites_activos + sim.satelites_muertos
        con_dos_padres = [s for s in todos if s.pariente_id_secundario is not None]
        assert len(con_dos_padres) > 0
