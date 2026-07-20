"""
Visualizador Evolutivo de Satélites

Genera visualizaciones a partir del historial producido por
:class:`sat.SimulacionSatelites` (ver `sat.py`). Requiere que la
simulación ya haya sido ejecutada (`sim.ejecutar_simulacion(...)`) antes
de crear el visualizador, ya que se basa en `sim.historial` y en las
listas `sim.satelites_activos` / `sim.satelites_muertos`.

Tipos de visualización disponibles:
1. Gráficas estadísticas estáticas con Matplotlib (PNG).
2. Dashboard interactivo con Plotly (HTML).
3. Animación 3D de órbitas alrededor de la Tierra (HTML).
4. Animaciones temporales de población y estadísticas (HTML).

Uso típico:
    >>> sim = SimulacionSatelites(...)
    >>> sim.lanzar_satelites()
    >>> sim.ejecutar_simulacion(dias_totales=1825)
    >>> viz = VisualizadorSatelites(sim)
    >>> viz.generar_todas_visualizaciones(directorio='salida')
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from typing import List, Dict, Optional
import json

from sat import SimulacionSatelites, Satellite, DistribucionFalla


class VisualizadorSatelites:
    """
    Visualiza la evolución de la simulación de satélites.

    Attributes:
        sim: instancia de `SimulacionSatelites` ya ejecutada, usada como
            fuente de datos para todas las visualizaciones.
        historial: atajo a `sim.historial` (lista de estadísticas diarias).
    """
    
    def __init__(self, simulacion: SimulacionSatelites):
        """
        Inicializa el visualizador.
        
        Args:
            simulacion: instancia de SimulacionSatelites con datos ejecutados
        """
        self.sim = simulacion
        self.historial = simulacion.historial
    
    # =========================================================================
    # MATPLOTLIB: Gráficas estadísticas
    # =========================================================================
    
    def graficar_estadisticas(self, guardar_como: Optional[str] = None, mostrar: bool = True) -> Figure:
        """
        Genera panel de 8 gráficas estadísticas con Matplotlib.
        
        Args:
            guardar_como: ruta donde guardar la imagen (ej: 'stats.png')
            mostrar: si mostrar el gráfico en pantalla
        """
        fig = plt.figure(figsize=(16, 16))
        gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Extraer datos
        tiempos = [h['tiempo'] for h in self.historial]
        dias = [(t - tiempos[0]).days for t in tiempos]
        activos = [h['satelites_activos'] for h in self.historial]
        muertos = [h['satelites_muertos'] for h in self.historial]
        edades = [h['edad_promedio_dias'] for h in self.historial]
        semiejes = [h['semieje_mayor_promedio'] for h in self.historial]
        calidades = [h.get('calidad_promedio', 0) for h in self.historial]
        
        # 1. Población: Satélites activos vs muertos
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(dias, activos, label='Activos', linewidth=2.5, color='#2ecc71')
        ax1.plot(dias, muertos, label='Muertos', linewidth=2.5, color='#e74c3c')
        ax1.fill_between(dias, activos, alpha=0.3, color='#2ecc71')
        ax1.fill_between(dias, muertos, alpha=0.3, color='#e74c3c')
        ax1.set_xlabel('Días')
        ax1.set_ylabel('Cantidad de satélites')
        ax1.set_title('Evolución de la Población', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Población total lanzada
        ax2 = fig.add_subplot(gs[0, 1])
        total_lanzados = [a + m for a, m in zip(activos, muertos)]
        ax2.bar(dias[::30], total_lanzados[::30], width=20, color='#3498db', alpha=0.7)
        ax2.set_xlabel('Días')
        ax2.set_ylabel('Satélites lanzados')
        ax2.set_title('Satélites Lanzados Acumulados (cada 30 días)', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')
        
        # 3. Edad promedio
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(dias, edades, linewidth=2.5, color='#9b59b6')
        ax3.fill_between(dias, edades, alpha=0.3, color='#9b59b6')
        ax3.axhline(y=5*365.25, color='red', linestyle='--', label='Edad máxima (5 años)', linewidth=2)
        ax3.set_xlabel('Días')
        ax3.set_ylabel('Edad promedio (días)')
        ax3.set_title('Edad Promedio de Satélites Activos', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Semieje mayor promedio
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(dias, semiejes, linewidth=2.5, color='#e67e22')
        ax4.fill_between(dias, semiejes, alpha=0.3, color='#e67e22')
        ax4.set_xlabel('Días')
        ax4.set_ylabel('Semieje mayor (km)')
        ax4.set_title('Semieje Mayor Promedio', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
        
        # 5. Tasa de mortalidad (muertos por día)
        ax5 = fig.add_subplot(gs[2, 0])
        tasas_mortalidad = [0] + [muertos[i] - muertos[i-1] for i in range(1, len(muertos))]
        ax5.bar(dias, tasas_mortalidad, width=1, color='#c0392b', alpha=0.7)
        ax5.set_xlabel('Días')
        ax5.set_ylabel('Satélites muertos/día')
        ax5.set_title('Tasa de Mortalidad Diaria', fontsize=12, fontweight='bold')
        ax5.grid(alpha=0.3, axis='y')
        
        # 6. Proporción activos/muertos en el tiempo
        ax6 = fig.add_subplot(gs[2, 1])
        proporciones = [a / (a + m) if (a + m) > 0 else 0 
                       for a, m in zip(activos, muertos)]
        ax6.plot(dias, proporciones, linewidth=2.5, color='#1abc9c')
        ax6.fill_between(dias, proporciones, alpha=0.3, color='#1abc9c')
        ax6.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
        ax6.set_xlabel('Días')
        ax6.set_ylabel('Proporción activos')
        ax6.set_ylim(bottom=0, top=1)
        ax6.set_title('Proporción de Satélites Activos', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(alpha=0.3)
        
        # 7. Evolución de calidad inherente promedio (presión de selección)
        ax7 = fig.add_subplot(gs[3, 0])
        ax7.plot(dias, calidades, linewidth=2.5, color='#f39c12')
        ax7.fill_between(dias, calidades, alpha=0.3, color='#f39c12')
        ax7.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Calidad inicial esperada (0.5)')
        ax7.set_xlabel('Días')
        ax7.set_ylabel('Calidad inherente promedio')
        ax7.set_ylim(bottom=0, top=1)
        ax7.set_title('Evolución de Calidad Inherente (Selección Natural)', fontsize=12, fontweight='bold')
        ax7.legend()
        ax7.grid(alpha=0.3)
        
        # 8. Causas de muerte (distribución final)
        ax8 = fig.add_subplot(gs[3, 1])
        causas_finales = self.historial[-1].get('causas_muerte_acumuladas', {}) if self.historial else {}
        if causas_finales:
            etiquetas = list(causas_finales.keys())
            valores = list(causas_finales.values())
            colores_causas = plt.get_cmap('tab10')(np.linspace(0, 1, len(etiquetas)))
            ax8.barh(etiquetas, valores, color=colores_causas)
            for idx, v in enumerate(valores):
                ax8.text(v, idx, f' {v}', va='center', fontsize=9)
        ax8.set_xlabel('Cantidad de satélites')
        ax8.set_title('Causas de Muerte (Acumulado Final)', fontsize=12, fontweight='bold')
        ax8.grid(alpha=0.3, axis='x')
        
        plt.suptitle('Simulación Evolutiva de Satélites - Análisis Estadístico',
                     fontsize=14, fontweight='bold', y=0.995)
        
        if guardar_como:
            plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
            print(f"✓ Gráficas guardadas en: {guardar_como}")
        
        if mostrar:
            plt.show()
        
        return fig
    
    # =========================================================================
    # MATPLOTLIB: Calidad inherente y heredabilidad
    # =========================================================================
    
    def graficar_calidad_heredabilidad(self, guardar_como: Optional[str] = None,
                                       mostrar: bool = True) -> Figure:
        """
        Genera panel de 3 gráficas sobre la calidad inherente de los satélites
        y su heredabilidad a través de generaciones.
        
        Args:
            guardar_como: ruta donde guardar la imagen
            mostrar: si mostrar el gráfico en pantalla
        """
        todos = self.sim.satelites_activos + self.sim.satelites_muertos
        
        fig = plt.figure(figsize=(16, 5))
        gs = GridSpec(1, 3, figure=fig, wspace=0.35)
        
        # 1. Histograma de distribución de calidad inherente
        ax1 = fig.add_subplot(gs[0, 0])
        calidades = [sat.calidad_inherente for sat in todos]
        ax1.hist(calidades, bins=25, color='#f39c12', alpha=0.8, edgecolor='black')
        ax1.axvline(x=float(np.mean(calidades)) if calidades else 0, color='red',
                   linestyle='--', linewidth=2, label='Promedio')
        ax1.set_xlabel('Calidad inherente')
        ax1.set_ylabel('Cantidad de satélites')
        ax1.set_title(f'Distribución de Calidad\n(n={len(todos)})', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Heredabilidad: calidad del padre vs calidad del hijo
        ax2 = fig.add_subplot(gs[0, 1])
        satelites_por_id = {sat.id: sat for sat in todos}
        calidad_padres, calidad_hijos = [], []
        for sat in todos:
            if sat.pariente_id is not None and sat.pariente_id in satelites_por_id:
                calidad_padres.append(satelites_por_id[sat.pariente_id].calidad_inherente)
                calidad_hijos.append(sat.calidad_inherente)
        
        if calidad_padres:
            ax2.scatter(calidad_padres, calidad_hijos, alpha=0.5, color='#3498db', s=25)
            ax2.plot([0.0, 1.0], [0.0, 1.0], color='gray', linestyle='--', linewidth=1.5, label='Sin mutación (y=x)')
            ax2.set_xlim(0.0, 1.0)
            ax2.set_ylim(0.0, 1.0)
        ax2.set_xlabel('Calidad del padre')
        ax2.set_ylabel('Calidad del hijo')
        ax2.set_title(f'Heredabilidad de Calidad\n(n={len(calidad_padres)} pares)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Calidad vs edad alcanzada al morir, coloreada por causa
        ax3 = fig.add_subplot(gs[0, 2])
        muertos = self.sim.satelites_muertos
        causas_unicas = sorted(set(s.causa_muerte or "desconocida" for s in muertos))
        colores_mapa = plt.get_cmap('tab10')(np.linspace(0, 1, max(len(causas_unicas), 1)))
        for causa, color in zip(causas_unicas, colores_mapa):
            subset = [s for s in muertos if (s.causa_muerte or "desconocida") == causa]
            edades_muerte = [s.edad_dias(self.sim.tiempo_actual) for s in subset]
            calidades_subset = [s.calidad_inherente for s in subset]
            ax3.scatter(calidades_subset, edades_muerte, alpha=0.5, s=20, 
                       color=color, label=causa)
        ax3.set_xlabel('Calidad inherente')
        ax3.set_ylabel('Edad al morir (días)')
        ax3.set_title('Calidad vs Longevidad por Causa de Muerte', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)
        
        plt.suptitle('Análisis de Calidad Inherente y Heredabilidad',
                     fontsize=14, fontweight='bold', y=1.05)
        
        if guardar_como:
            plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
            print(f"✓ Gráficas de calidad/heredabilidad guardadas en: {guardar_como}")
        
        if mostrar:
            plt.show()
        
        return fig
    
    # =========================================================================
    # PLOTLY: Dashboard interactivo
    # =========================================================================
    
    def crear_dashboard_interactivo(self, guardar_como: str = 'dashboard.html'):
        """
        Crea un dashboard interactivo HTML con Plotly.

        Genera un panel de 8 gráficas interactivas (población, lanzamientos
        acumulados, edad promedio, semieje mayor, tasa de mortalidad,
        proporción de activos, calidad inherente promedio y causas de
        muerte) análogas a las de :meth:`graficar_estadisticas`, pero
        navegables e interactivas en el navegador.
        
        Args:
            guardar_como: ruta del archivo HTML a generar.
        """
        # Extraer datos
        tiempos = [h['tiempo'] for h in self.historial]
        dias = [(t - tiempos[0]).days for t in tiempos]
        activos = [h['satelites_activos'] for h in self.historial]
        muertos = [h['satelites_muertos'] for h in self.historial]
        edades = [h['edad_promedio_dias'] for h in self.historial]
        semiejes = [h['semieje_mayor_promedio'] for h in self.historial]
        calidades = [h.get('calidad_promedio', 0) for h in self.historial]
        causas_finales = self.historial[-1].get('causas_muerte_acumuladas', {}) if self.historial else {}
        
        # Crear subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=('Población: Activos vs Muertos',
                          'Satélites Lanzados Acumulados',
                          'Edad Promedio (días)',
                          'Semieje Mayor Promedio (km)',
                          'Tasa de Mortalidad Diaria',
                          'Proporción Activos/Total',
                          'Calidad Inherente Promedio (Selección Natural)',
                          'Causas de Muerte (Distribución Final)'),
            specs=[[{}, {}],
                   [{}, {}],
                   [{}, {}],
                   [{}, {'type': 'pie'}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Población
        fig.add_trace(
            go.Scatter(x=dias, y=activos, name='Activos', mode='lines',
                      line=dict(color='#2ecc71', width=3),
                      fill='tozeroy', fillcolor='rgba(46, 204, 113, 0.2)',
                      hovertemplate='Día %{x}<br>Activos: %{y}<extra></extra>'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=dias, y=muertos, name='Muertos', mode='lines',
                      line=dict(color='#e74c3c', width=3),
                      fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.2)',
                      hovertemplate='Día %{x}<br>Muertos: %{y}<extra></extra>'),
            row=1, col=1
        )
        
        # 2. Lanzados acumulados
        total_lanzados = [a + m for a, m in zip(activos, muertos)]
        fig.add_trace(
            go.Bar(x=dias[::30], y=total_lanzados[::30], name='Lanzados',
                  marker=dict(color='#3498db'),
                  hovertemplate='Día %{x}<br>Total: %{y}<extra></extra>'),
            row=1, col=2
        )
        
        # 3. Edad promedio
        fig.add_trace(
            go.Scatter(x=dias, y=edades, name='Edad promedio', mode='lines',
                      line=dict(color='#9b59b6', width=3),
                      fill='tozeroy', fillcolor='rgba(155, 89, 182, 0.2)',
                      hovertemplate='Día %{x}<br>Edad: %{y:.1f} días<extra></extra>'),
            row=2, col=1
        )
        # Línea horizontal para edad máxima
        fig.add_trace(
            go.Scatter(x=dias, y=[5*365.25]*len(dias), mode='lines', name='Edad máxima (5 años)',
                      line=dict(color='red', width=2, dash='dash'),
                      hovertemplate='Edad máxima<extra></extra>'),
            row=2, col=1
        )
        
        # 4. Semieje mayor promedio
        fig.add_trace(
            go.Scatter(x=dias, y=semiejes, name='Semieje mayor', mode='lines',
                      line=dict(color='#e67e22', width=3),
                      fill='tozeroy', fillcolor='rgba(230, 126, 34, 0.2)',
                      hovertemplate='Día %{x}<br>Semieje: %{y:.1f} km<extra></extra>'),
            row=2, col=2
        )
        
        # 5. Tasa de mortalidad
        tasas_mortalidad = [0] + [muertos[i] - muertos[i-1] for i in range(1, len(muertos))]
        fig.add_trace(
            go.Bar(x=dias, y=tasas_mortalidad, name='Muertes/día',
                  marker=dict(color='#c0392b'),
                  hovertemplate='Día %{x}<br>Muertes: %{y}<extra></extra>'),
            row=3, col=1
        )
        
        # 6. Proporción activos
        proporciones = [a / (a + m) if (a + m) > 0 else 0 
                       for a, m in zip(activos, muertos)]
        fig.add_trace(
            go.Scatter(x=dias, y=proporciones, name='Proporción activos',
                      mode='lines',
                      line=dict(color='#1abc9c', width=3),
                      fill='tozeroy', fillcolor='rgba(26, 188, 156, 0.2)',
                      hovertemplate='Día %{x}<br>Proporción: %{y:.2%}<extra></extra>'),
            row=3, col=2
        )
        
        # 7. Calidad inherente promedio (selección natural)
        fig.add_trace(
            go.Scatter(x=dias, y=calidades, name='Calidad promedio', mode='lines',
                      line=dict(color='#f39c12', width=3),
                      fill='tozeroy', fillcolor='rgba(243, 156, 18, 0.2)',
                      hovertemplate='Día %{x}<br>Calidad: %{y:.3f}<extra></extra>'),
            row=4, col=1
        )
        
        # 8. Causas de muerte (pie chart)
        if causas_finales:
            fig.add_trace(
                go.Pie(labels=list(causas_finales.keys()), values=list(causas_finales.values()),
                      hovertemplate='%{label}<br>%{value} (%{percent})<extra></extra>'),
                row=4, col=2
            )
        
        # Actualizar ejes
        fig.update_xaxes(title_text='Días', row=1, col=1)
        fig.update_xaxes(title_text='Días', row=1, col=2)
        fig.update_xaxes(title_text='Días', row=2, col=1)
        fig.update_xaxes(title_text='Días', row=2, col=2)
        fig.update_xaxes(title_text='Días', row=3, col=1)
        fig.update_xaxes(title_text='Días', row=3, col=2)
        fig.update_xaxes(title_text='Días', row=4, col=1)
        
        fig.update_yaxes(title_text='Cantidad', row=1, col=1)
        fig.update_yaxes(title_text='Satélites', row=1, col=2)
        fig.update_yaxes(title_text='Días', row=2, col=1)
        fig.update_yaxes(title_text='km', row=2, col=2)
        fig.update_yaxes(title_text='Muertes/día', row=3, col=1)
        fig.update_yaxes(title_text='Proporción', row=3, col=2)
        fig.update_yaxes(title_text='Calidad (0-1)', row=4, col=1)
        
        fig.update_layout(
            title_text='Dashboard Interactivo - Simulación de Satélites',
            height=1600,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.write_html(guardar_como)
        print(f"✓ Dashboard interactivo guardado en: {guardar_como}")
    
    # =========================================================================
    # PLOTLY 3D: Órbitas animadas
    # =========================================================================
    
    def crear_orbitas_3d(self, guardar_como: str = 'orbitas_3d.html',
                         incluir_tierra: bool = True, 
                         mostrar_satelites: int = 20):
        """
        Crea animación 3D de órbitas de satélites alrededor de la Tierra.

        Toma los primeros `mostrar_satelites` satélites activos de la
        simulación y dibuja su traza orbital en 3D junto con la Tierra
        (esfera de radio 6371 km) como referencia visual.
        
        Args:
            guardar_como: ruta del archivo HTML.
            incluir_tierra: si mostrar la Tierra en el centro.
            mostrar_satelites: número máximo de satélites a visualizar.
        """
        # Seleccionar satélites activos
        satelites_a_mostrar = self.sim.satelites_activos[:mostrar_satelites]
        
        # Radio de la Tierra (km)
        R_tierra = 6371
        
        # Crear figura
        fig = go.Figure()
        
        # Agregar Tierra (esfera azul)
        if incluir_tierra:
            # Crear esfera de la Tierra
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            x_tierra = R_tierra * np.outer(np.cos(u), np.sin(v))
            y_tierra = R_tierra * np.outer(np.sin(u), np.sin(v))
            z_tierra = R_tierra * np.outer(np.ones(np.size(u)), np.cos(v))
            
            fig.add_trace(go.Surface(
                x=x_tierra, y=y_tierra, z=z_tierra,
                colorscale='Blues', showscale=False,
                opacity=0.8, name='Tierra'
            ))
        
        # Colores para los satélites
        colores = px.colors.qualitative.Plotly
        
        # Agregar órbitas y satélites
        for idx, sat in enumerate(satelites_a_mostrar):
            color = colores[idx % len(colores)]
            
            # Generar puntos de la órbita (elipse Kepleriana simplificada)
            nu = np.linspace(0, 2*np.pi, 200)  # anomalía verdadera
            
            # Coordenadas en plano orbital
            r = sat.a * (1 - sat.e**2) / (1 + sat.e * np.cos(nu))
            x_orb = r * np.cos(nu)
            y_orb = r * np.sin(nu)
            z_orb = np.zeros_like(x_orb)
            
            # Rotaciones de Euler (simplificadas)
            omega_rad = np.radians(sat.omega)
            Omega_rad = np.radians(sat.Omega)
            i_rad = np.radians(sat.i)
            
            # Matriz de rotación
            x_rot = (x_orb * (np.cos(omega_rad)*np.cos(Omega_rad) - 
                             np.sin(omega_rad)*np.cos(i_rad)*np.sin(Omega_rad)) -
                    y_orb * (np.sin(omega_rad)*np.cos(Omega_rad) + 
                             np.cos(omega_rad)*np.cos(i_rad)*np.sin(Omega_rad)))
            
            y_rot = (x_orb * (np.cos(omega_rad)*np.sin(Omega_rad) + 
                             np.sin(omega_rad)*np.cos(i_rad)*np.cos(Omega_rad)) +
                    y_orb * (-np.sin(omega_rad)*np.sin(Omega_rad) + 
                             np.cos(omega_rad)*np.cos(i_rad)*np.cos(Omega_rad)))
            
            z_rot = (x_orb * np.sin(omega_rad)*np.sin(i_rad) + 
                    y_orb * np.cos(omega_rad)*np.sin(i_rad))
            
            # Agregar órbita (línea)
            fig.add_trace(go.Scatter3d(
                x=x_rot, y=y_rot, z=z_rot,
                mode='lines',
                line=dict(color=color, width=2),
                name=f'Órbita SAT-{sat.id}',
                hovertemplate='SAT-{}<br>x: %{{x:.0f}} km<br>y: %{{y:.0f}} km<br>z: %{{z:.0f}} km<extra></extra>'.format(sat.id),
                visible=True
            ))
            
            # Agregar satélite (punto actual)
            # Usar anomalía media como posición actual
            M_rad = np.radians(sat.M)
            r_sat = sat.a * (1 - sat.e**2) / (1 + sat.e * np.cos(M_rad))
            x_sat_orb = r_sat * np.cos(M_rad)
            y_sat_orb = r_sat * np.sin(M_rad)
            z_sat_orb = 0
            
            x_sat = (x_sat_orb * (np.cos(omega_rad)*np.cos(Omega_rad) - 
                                 np.sin(omega_rad)*np.cos(i_rad)*np.sin(Omega_rad)) -
                    y_sat_orb * (np.sin(omega_rad)*np.cos(Omega_rad) + 
                                 np.cos(omega_rad)*np.cos(i_rad)*np.sin(Omega_rad)))
            
            y_sat = (x_sat_orb * (np.cos(omega_rad)*np.sin(Omega_rad) + 
                                 np.sin(omega_rad)*np.cos(i_rad)*np.cos(Omega_rad)) +
                    y_sat_orb * (-np.sin(omega_rad)*np.sin(Omega_rad) + 
                                 np.cos(omega_rad)*np.cos(i_rad)*np.cos(Omega_rad)))
            
            z_sat = (x_sat_orb * np.sin(omega_rad)*np.sin(i_rad) + 
                    y_sat_orb * np.cos(omega_rad)*np.sin(i_rad))
            
            info_sat = sat.info()
            fig.add_trace(go.Scatter3d(
                x=[x_sat], y=[y_sat], z=[z_sat],
                mode='markers+text',
                marker=dict(size=8, color=color),
                text=[f'SAT-{sat.id}'],
                textposition='top center',
                name=f'SAT-{sat.id}',
                hovertemplate=(
                    f'SAT-{sat.id}<br>'
                    f'Calidad inherente: {info_sat["calidad_inherente"]:.2f}<br>'
                    f'a: {info_sat["semieje_mayor_km"]:.1f} km<br>'
                    f'e: {info_sat["excentricidad"]:.4f}<br>'
                    f'i: {info_sat["inclinacion_grados"]:.1f}°<br>'
                    f'Perigeo: {info_sat["perigeo_km"]:.1f} km<br>'
                    f'Apogeo: {info_sat["apogeo_km"]:.1f} km<extra></extra>'
                ),
                visible=True
            ))
        
        # Actualizar layout
        fig.update_layout(
            title=f'Órbitas 3D de Satélites (mostrando {min(mostrar_satelites, len(satelites_a_mostrar))} de {len(self.sim.satelites_activos)} activos)',
            scene=dict(
                xaxis_title='X (km)',
                yaxis_title='Y (km)',
                zaxis_title='Z (km)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=900,
            showlegend=True,
            hovermode='closest'
        )
        
        fig.write_html(guardar_como)
        print(f"✓ Visualización 3D guardada en: {guardar_como}")
    
    # =========================================================================
    # ANIMACIONES TEMPORALES
    # =========================================================================
    
    def crear_animacion_poblacion(self, guardar_como: str = 'animacion_poblacion.html',
                                 velocidad: int = 50):
        """
        Crea una animación temporal de la evolución de población.
        
        Args:
            guardar_como: ruta del archivo HTML
            velocidad: velocidad de animación en ms entre frames
        """
        # Extraer datos
        tiempos = [h['tiempo'] for h in self.historial]
        dias = [(t - tiempos[0]).days for t in tiempos]
        activos = [h['satelites_activos'] for h in self.historial]
        muertos = [h['satelites_muertos'] for h in self.historial]
        total = [a + m for a, m in zip(activos, muertos)]
        
        # Crear figura base
        fig = go.Figure()
        
        # Agregar traza inicial
        fig.add_trace(
            go.Bar(x=['Activos'], y=[activos[0]], name='Activos',
                   marker=dict(color='#2ecc71'), showlegend=True)
        )
        fig.add_trace(
            go.Bar(x=['Muertos'], y=[muertos[0]], name='Muertos',
                   marker=dict(color='#e74c3c'), showlegend=True)
        )
        fig.add_trace(
            go.Bar(x=['Total'], y=[total[0]], name='Total',
                   marker=dict(color='#3498db'), showlegend=True)
        )
        
        # Crear frames para la animación
        frames = []
        for i, (a, m, t) in enumerate(zip(activos, muertos, total)):
            frame = go.Frame(
                data=[
                    go.Bar(x=['Activos'], y=[a], marker=dict(color='#2ecc71')),
                    go.Bar(x=['Muertos'], y=[m], marker=dict(color='#e74c3c')),
                    go.Bar(x=['Total'], y=[t], marker=dict(color='#3498db'))
                ],
                name=str(dias[i]),
                layout=go.Layout(
                    title_text=f'Población de Satélites - Día {dias[i]}'
                )
            )
            frames.append(frame)
        
        fig.frames = frames
        
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': '▶ Play', 'method': 'animate',
                     'args': [None, {'frame': {'duration': velocidad, 'redraw': True},
                                     'fromcurrent': True}]},
                    {'label': '⏸ Pause', 'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                      'mode': 'immediate'}]}
                ]
            }],
            sliders=[{
                'active': 0,
                'steps': [{'args': [[f.name], {'frame': {'duration': velocidad, 'redraw': True},
                                               'mode': 'immediate'}],
                          'label': f'Día {dias[j]}',
                          'method': 'animate'}
                         for j, f in enumerate(frames)],
                'x': 0.1, 'len': 0.9, 'y': 0, 'yanchor': 'top'
            }],
            title='Animación: Evolución de Población de Satélites',
            xaxis_title='Categoría',
            yaxis_title='Cantidad de satélites',
            height=600,
            barmode='group',
            template='plotly_white'
        )
        
        fig.write_html(guardar_como)
        print(f"✓ Animación de población guardada en: {guardar_como}")
    
    def crear_animacion_estadisticas_multiples(self, guardar_como: str = 'animacion_estadisticas.html',
                                               velocidad: int = 30):
        """
        Crea una animación de múltiples estadísticas a través del tiempo.
        
        Args:
            guardar_como: ruta del archivo HTML
            velocidad: velocidad de animación en ms
        """
        # Extraer datos
        tiempos = [h['tiempo'] for h in self.historial]
        dias = [(t - tiempos[0]).days for t in tiempos]
        activos = [h['satelites_activos'] for h in self.historial]
        muertos = [h['satelites_muertos'] for h in self.historial]
        edades = [h['edad_promedio_dias'] for h in self.historial]
        semiejes = [h['semieje_mayor_promedio'] for h in self.historial]
        calidades = [h.get('calidad_promedio', 0) for h in self.historial]
        
        # Crear figura con subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Población', 'Edad Promedio (días)',
                          'Semieje Mayor Promedio (km)', 'Proporción Activos',
                          'Calidad Inherente Promedio', ''),
            specs=[[{}, {}], [{}, {}], [{}, None]]
        )
        
        # Población
        fig.add_trace(
            go.Scatter(x=[dias[0]], y=[activos[0]], name='Activos',
                      mode='lines+markers', line=dict(color='#2ecc71', width=3),
                      marker=dict(size=8)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[dias[0]], y=[muertos[0]], name='Muertos',
                      mode='lines+markers', line=dict(color='#e74c3c', width=3),
                      marker=dict(size=8)),
            row=1, col=1
        )
        
        # Edad
        fig.add_trace(
            go.Scatter(x=[dias[0]], y=[edades[0]], name='Edad promedio',
                      mode='lines+markers', line=dict(color='#9b59b6', width=3),
                      marker=dict(size=8)),
            row=1, col=2
        )
        
        # Semieje
        fig.add_trace(
            go.Scatter(x=[dias[0]], y=[semiejes[0]], name='Semieje',
                      mode='lines+markers', line=dict(color='#e67e22', width=3),
                      marker=dict(size=8)),
            row=2, col=1
        )
        
        # Proporción
        prop = [a / (a + m) if (a + m) > 0 else 0 for a, m in zip(activos, muertos)]
        fig.add_trace(
            go.Scatter(x=[dias[0]], y=[prop[0]], name='Proporción',
                      mode='lines+markers', line=dict(color='#1abc9c', width=3),
                      marker=dict(size=8)),
            row=2, col=2
        )
        
        # Calidad inherente promedio
        fig.add_trace(
            go.Scatter(x=[dias[0]], y=[calidades[0]], name='Calidad promedio',
                      mode='lines+markers', line=dict(color='#f39c12', width=3),
                      marker=dict(size=8)),
            row=3, col=1
        )
        
        # Crear frames
        frames = []
        for i in range(len(dias)):
            frame = go.Frame(
                data=[
                    go.Scatter(x=dias[:i+1], y=activos[:i+1]),
                    go.Scatter(x=dias[:i+1], y=muertos[:i+1]),
                    go.Scatter(x=dias[:i+1], y=edades[:i+1]),
                    go.Scatter(x=dias[:i+1], y=semiejes[:i+1]),
                    go.Scatter(x=dias[:i+1], y=prop[:i+1]),
                    go.Scatter(x=dias[:i+1], y=calidades[:i+1])
                ],
                name=str(dias[i])
            )
            frames.append(frame)
        
        fig.frames = frames
        
        # Actualizar ejes
        fig.update_xaxes(title_text='Días', row=1, col=1)
        fig.update_xaxes(title_text='Días', row=1, col=2)
        fig.update_xaxes(title_text='Días', row=2, col=1)
        fig.update_xaxes(title_text='Días', row=2, col=2)
        fig.update_xaxes(title_text='Días', row=3, col=1)
        fig.update_yaxes(title_text='Cantidad', row=1, col=1)
        fig.update_yaxes(title_text='Días', row=1, col=2)
        fig.update_yaxes(title_text='km', row=2, col=1)
        fig.update_yaxes(title_text='Proporción', row=2, col=2)
        fig.update_yaxes(title_text='Calidad (0-1)', row=3, col=1)
        
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': '▶ Play', 'method': 'animate',
                     'args': [None, {'frame': {'duration': velocidad, 'redraw': True},
                                     'fromcurrent': True}]},
                    {'label': '⏸ Pause', 'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                      'mode': 'immediate'}]}
                ]
            }],
            sliders=[{
                'active': 0,
                'steps': [{'args': [[f.name], {'frame': {'duration': velocidad, 'redraw': True},
                                               'mode': 'immediate'}],
                          'label': f'Día {dias[j]}',
                          'method': 'animate'}
                         for j, f in enumerate(frames)],
                'x': 0.1, 'len': 0.9, 'y': -0.1, 'yanchor': 'top'
            }],
            title='Animación: Evolución de Estadísticas',
            height=1100,
            template='plotly_white',
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.write_html(guardar_como)
        print(f"✓ Animación de estadísticas guardada en: {guardar_como}")
    
    def crear_animacion_scatter_dinamico(self, guardar_como: str = 'animacion_scatter.html',
                                        velocidad: int = 50):
        """
        Crea una animación de dispersión (scatter) mostrando relación entre 
        edad promedio y población.
        
        Args:
            guardar_como: ruta del archivo HTML
            velocidad: velocidad de animación en ms
        """
        # Extraer datos
        tiempos = [h['tiempo'] for h in self.historial]
        dias = [(t - tiempos[0]).days for t in tiempos]
        activos = [h['satelites_activos'] for h in self.historial]
        muertos = [h['satelites_muertos'] for h in self.historial]
        edades = [h['edad_promedio_dias'] for h in self.historial]
        total = [a + m for a, m in zip(activos, muertos)]
        prop = [a / (a + m) if (a + m) > 0 else 0 for a, m in zip(activos, muertos)]
        
        # Crear figura base
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=[edades[0]], y=[activos[0]], mode='markers+text',
                marker=dict(size=20, color='#2ecc71'),
                text=[f'Día 0<br>E: {edades[0]:.0f}d<br>A: {activos[0]}'],
                textposition='top center',
                name='Satélites Activos',
                hovertemplate='Edad: %{x:.1f} días<br>Activos: %{y}<extra></extra>'
            )
        )
        
        # Crear frames
        frames = []
        for i in range(len(dias)):
            frame = go.Frame(
                data=[go.Scatter(
                    x=edades[:i+1], y=activos[:i+1], mode='markers+lines',
                    marker=dict(size=12, color=dias[:i+1], 
                              colorscale='Viridis', showscale=True,
                              colorbar=dict(title='Días')),
                    line=dict(color='#2ecc71', width=2),
                    text=[f'Día {dias[j]}<br>Edad: {edades[j]:.0f}d<br>Activos: {activos[j]}' 
                         for j in range(i+1)],
                    hovertemplate='%{text}<extra></extra>'
                )],
                name=str(dias[i])
            )
            frames.append(frame)
        
        fig.frames = frames
        
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': '▶ Play', 'method': 'animate',
                     'args': [None, {'frame': {'duration': velocidad, 'redraw': True},
                                     'fromcurrent': True}]},
                    {'label': '⏸ Pause', 'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                      'mode': 'immediate'}]}
                ]
            }],
            sliders=[{
                'active': 0,
                'steps': [{'args': [[f.name], {'frame': {'duration': velocidad, 'redraw': True},
                                               'mode': 'immediate'}],
                          'label': f'Día {dias[j]}',
                          'method': 'animate'}
                         for j, f in enumerate(frames)],
                'x': 0.1, 'len': 0.9, 'y': -0.1, 'yanchor': 'top'
            }],
            title='Animación: Relación Edad vs Población Activa',
            xaxis_title='Edad Promedio (días)',
            yaxis_title='Satélites Activos',
            height=700,
            template='plotly_white'
        )
        
        fig.write_html(guardar_como)
        print(f"✓ Animación scatter guardada en: {guardar_como}")
    
    # =========================================================================
    # Método general: Generar todo
    # =========================================================================
    
    def generar_todas_visualizaciones(self, directorio: str = '.'):
        """
        Genera todas las visualizaciones.

        Ejecuta en secuencia: gráficas estáticas de estadísticas, gráficas
        de calidad/heredabilidad, dashboard interactivo, órbitas 3D y las
        tres animaciones temporales (población, estadísticas múltiples y
        scatter edad vs población). Crea `directorio` si no existe.
        
        Args:
            directorio: directorio donde guardar los archivos.
        """
        print("\n" + "="*60)
        print("GENERANDO VISUALIZACIONES")
        print("="*60)
        
        import os
        
        # Crear directorio si no existe
        if directorio != '.' and not os.path.exists(directorio):
            os.makedirs(directorio)
        
        # Gráficas Matplotlib
        print("\n1. Generando gráficas estadísticas (Matplotlib)...")
        ruta_stats = os.path.join(directorio, 'estadisticas.png')
        self.graficar_estadisticas(guardar_como=ruta_stats, mostrar=False)
        
        # Calidad inherente y heredabilidad
        print("2. Generando gráficas de calidad y heredabilidad (Matplotlib)...")
        ruta_calidad = os.path.join(directorio, 'calidad_heredabilidad.png')
        self.graficar_calidad_heredabilidad(guardar_como=ruta_calidad, mostrar=False)
        
        # Dashboard Plotly
        print("3. Generando dashboard interactivo (Plotly)...")
        ruta_dashboard = os.path.join(directorio, 'dashboard.html')
        self.crear_dashboard_interactivo(guardar_como=ruta_dashboard)
        
        # Órbitas 3D
        print("4. Generando visualización 3D de órbitas...")
        ruta_orbitas = os.path.join(directorio, 'orbitas_3d.html')
        self.crear_orbitas_3d(guardar_como=ruta_orbitas, mostrar_satelites=15)
        
        # Animaciones temporales
        print("5. Generando animación de población...")
        ruta_anim_pop = os.path.join(directorio, 'animacion_poblacion.html')
        self.crear_animacion_poblacion(guardar_como=ruta_anim_pop, velocidad=50)
        
        print("6. Generando animación de estadísticas múltiples...")
        ruta_anim_stats = os.path.join(directorio, 'animacion_estadisticas.html')
        self.crear_animacion_estadisticas_multiples(guardar_como=ruta_anim_stats, velocidad=30)
        
        print("7. Generando animación scatter (Edad vs Población)...")
        ruta_anim_scatter = os.path.join(directorio, 'animacion_scatter.html')
        self.crear_animacion_scatter_dinamico(guardar_como=ruta_anim_scatter, velocidad=50)
        
        print("\n" + "="*60)
        print("✓ Todas las visualizaciones generadas exitosamente")
        print("="*60 + "\n")
        
        print("Archivos generados:")
        print("  - estadisticas.png (gráficas estáticas)")
        print("  - calidad_heredabilidad.png (calidad inherente y heredabilidad)")
        print("  - dashboard.html (dashboard interactivo)")
        print("  - orbitas_3d.html (órbitas 3D)")
        print("  - animacion_poblacion.html (animación barras)")
        print("  - animacion_estadisticas.html (animación líneas múltiples)")
        print("  - animacion_scatter.html (animación scatter dinámico)")
        print()


# =============================================================================
# Ejemplo de uso
# =============================================================================

if __name__ == "__main__":
    # Configurar la distribución de falla
    falla = DistribucionFalla(
        tasa_base=0.0005,
        parametro_forma=2.0,
        escala_tiempo=3.0
    )
    
    # Crear simulación
    sim = SimulacionSatelites(
        distribucion_falla=falla,
        intervalo_lanzamiento_dias=30,
        satelites_por_lanzamiento=5,
        edad_maxima_anos=5,
        tiempo_inicio=datetime(2024, 1, 1)
    )
    
    # Ejecutar simulación
    print("Ejecutando simulación...")
    sim.lanzar_satelites()
    sim.ejecutar_simulacion(dias_totales=1825)
    
    # Crear visualizador
    viz = VisualizadorSatelites(sim)
    
    # Generar todas las visualizaciones
    viz.generar_todas_visualizaciones(directorio='.')
    
    # Opcionalmente, generar solo una específica
    print("\nGenerando visualización Matplotlib (mostrada en pantalla)...")
    viz.graficar_estadisticas(mostrar=True)
