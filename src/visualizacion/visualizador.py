"""
Módulo de visualización de datos para el proyecto
de Calidad del Aire en el GAM de Costa Rica.
Genera gráficos de líneas, barras, heatmaps y mapas interactivos.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from basedatos.gestor_base_datos import GestorBaseDatos


class Visualizador:
    """
    Clase encargada de generar todas las visualizaciones del proyecto.
    Incluye gráficos de flujo vehicular, contaminantes, clima y mapas.
    """

    def __init__(self):
        """Inicializa el visualizador conectando a la base de datos."""
        self.gestor_bd = GestorBaseDatos()
        self.df_flujo = None
        self.df_aire = None
        self.df_clima = None

    def cargar_datos(self):
        """Carga los datos desde SQL Server."""
        self.df_flujo = self.gestor_bd.consultar("SELECT * FROM flujo_vehicular")
        self.df_aire = self.gestor_bd.consultar("SELECT * FROM calidad_aire")
        self.df_clima = self.gestor_bd.consultar("SELECT * FROM clima")
        print("[Visualizador] Datos cargados correctamente.")

    # ─────────────────────────────────────────
    # FLUJO VEHICULAR
    # ─────────────────────────────────────────

    def grafico_flujo_por_anio(self):
        """
        Genera un gráfico de barras del total de vehículos por año.
        """
        df = self.df_flujo.groupby('anio')['total'].sum().reset_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='anio', y='total', palette='Blues_d')
        plt.title('Flujo Vehicular Total por Año — Ruta 27', fontsize=14)
        plt.xlabel('Año')
        plt.ylabel('Total de Vehículos')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('data/processed/flujo_por_anio.png')
        plt.show()
        print("[Visualizador] Gráfico flujo por año generado.")

    def grafico_flujo_por_mes(self):
        """
        Genera un gráfico de líneas del flujo vehicular promedio por mes.
        """

        def grafico_flujo_por_mes(self):
            """
            Genera un gráfico de líneas del flujo vehicular promedio por mes.
            """
            orden_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                           'Julio', 'Agosto', 'Setiembre', 'Octubre', 'Noviembre', 'Diciembre']
            df = self.df_flujo.groupby('mes')['total'].mean().reset_index()
            df['mes'] = pd.Categorical(df['mes'], categories=orden_meses, ordered=True)
            df = df.sort_values('mes')
            df = df.dropna(subset=['mes'])
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=df, x='mes', y='total', marker='o', color='steelblue')
            plt.title('Flujo Vehicular Promedio por Mes — Ruta 27', fontsize=14)
            plt.xlabel('Mes')
            plt.ylabel('Promedio de Vehículos')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('data/processed/flujo_por_mes.png')
            plt.show()
            print("[Visualizador] Gráfico flujo por mes generado.")

    # ─────────────────────────────────────────
    # CALIDAD DEL AIRE
    # ─────────────────────────────────────────

    def grafico_contaminantes_por_anio(self):
        """
        Genera un gráfico de líneas de PM2.5 y NO2 por año.
        """
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=self.df_aire, x='anio', y='pm2_5',
                     marker='o', label='PM2.5', color='red')
        sns.lineplot(data=self.df_aire, x='anio', y='nitrogen_dioxide',
                     marker='s', label='NO2', color='orange')
        plt.title('Evolución de Contaminantes por Año — GAM Costa Rica', fontsize=14)
        plt.xlabel('Año')
        plt.ylabel('Concentración')
        plt.legend()
        plt.tight_layout()
        plt.savefig('data/processed/contaminantes_por_anio.png')
        plt.show()
        print("[Visualizador] Gráfico contaminantes por año generado.")

    def heatmap_correlacion(self):
        """
        Genera un heatmap de correlación entre flujo vehicular y contaminantes.
        """
        df_merge = pd.merge(
            self.df_flujo.groupby(['anio', 'mes'])['total'].sum().reset_index(),
            self.df_aire[['anio', 'mes', 'pm2_5', 'nitrogen_dioxide', 'ozone']],
            on=['anio', 'mes'],
            how='inner'
        )
        correlacion = df_merge[['total', 'pm2_5', 'nitrogen_dioxide', 'ozone']].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Heatmap de Correlación — Flujo Vehicular vs Contaminantes', fontsize=14)
        plt.tight_layout()
        plt.savefig('data/processed/heatmap_correlacion.png')
        plt.show()
        print("[Visualizador] Heatmap de correlación generado.")

    # ─────────────────────────────────────────
    # CLIMA
    # ─────────────────────────────────────────

    def grafico_temperatura_por_anio(self):
        """
        Genera un gráfico de líneas de temperatura promedio por año.
        """
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=self.df_clima, x='anio', y='temperature_2m',
                     marker='o', color='tomato')
        plt.title('Temperatura Promedio por Año — GAM Costa Rica', fontsize=14)
        plt.xlabel('Año')
        plt.ylabel('Temperatura (°C)')
        plt.tight_layout()
        plt.savefig('data/processed/temperatura_por_anio.png')
        plt.show()
        print("[Visualizador] Gráfico temperatura generado.")

    # ─────────────────────────────────────────
    # MAPA INTERACTIVO
    # ─────────────────────────────────────────

    def mapa_ruta_27(self):
        """
        Genera un mapa interactivo de la Ruta 27 con Folium
        mostrando los puntos de conteo vehicular.
        """
        mapa = folium.Map(location=[9.9281, -84.0907], zoom_start=10)

        puntos = [
            {"nombre": "San José - Inicio Ruta 27", "lat": 9.9281, "lon": -84.0907},
            {"nombre": "Punto Conteo 10540", "lat": 9.8968, "lon": -84.2819},
            {"nombre": "Caldera - Final Ruta 27", "lat": 9.9019, "lon": -84.7356},
        ]

        for punto in puntos:
            folium.Marker(
                location=[punto["lat"], punto["lon"]],
                popup=punto["nombre"],
                tooltip=punto["nombre"],
                icon=folium.Icon(color='blue', icon='car', prefix='fa')
            ).add_to(mapa)

        mapa.save('data/processed/mapa_ruta_27.html')
        print("[Visualizador] Mapa Ruta 27 guardado en data/processed/mapa_ruta_27.html")
        return mapa

    # ─────────────────────────────────────────
    # EJECUTAR TODO
    # ─────────────────────────────────────────

    def ejecutar_visualizaciones(self):
        """
        Ejecuta todas las visualizaciones del proyecto.
        """
        print("\n GENERANDO VISUALIZACIONES\n")
        self.cargar_datos()
        self.grafico_flujo_por_anio()
        self.grafico_flujo_por_mes()
        self.grafico_contaminantes_por_anio()
        self.heatmap_correlacion()
        self.grafico_temperatura_por_anio()
        self.mapa_ruta_27()
        print("\n VISUALIZACIONES COMPLETADAS\n")