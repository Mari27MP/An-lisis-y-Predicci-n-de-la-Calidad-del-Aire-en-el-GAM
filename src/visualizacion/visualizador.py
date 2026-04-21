"""
Módulo de visualización de datos para el proyecto
de Calidad del Aire en el GAM de Costa Rica.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from basedatos.gestor_base_datos import GestorBaseDatos


class Visualizador:

    def __init__(self):
        self.gestor_bd = GestorBaseDatos()
        self.df_flujo = None
        self.df_aire = None
        self.df_clima = None
        self.base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def cargar_datos(self):
        self.df_flujo = self.gestor_bd.consultar("SELECT * FROM flujo_vehicular")
        self.df_aire = self.gestor_bd.consultar("SELECT * FROM calidad_aire")
        self.df_clima = self.gestor_bd.consultar("SELECT * FROM clima")
        print("[Visualizador] Datos cargados correctamente.")

    def grafico_flujo_por_anio(self):
        df = self.df_flujo.groupby('anio')['total'].sum().reset_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='anio', y='total', hue='anio', palette='Blues_d', legend=False)
        plt.title('Flujo Vehicular Total por Año — Ruta 27', fontsize=14)
        plt.xlabel('Año')
        plt.ylabel('Total de Vehículos')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.base, 'data/processed/flujo_por_anio.png'))
        plt.show()
        print("[Visualizador] Gráfico flujo por año generado.")

    def grafico_flujo_por_mes(self):
        orden_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                       'Julio', 'Agosto', 'Setiembre', 'Octubre', 'Noviembre', 'Diciembre']
        df = self.df_flujo.groupby('mes')['total'].mean().reset_index()
        df['mes'] = pd.Categorical(df['mes'], categories=orden_meses, ordered=True)
        df = df.sort_values('mes').dropna(subset=['mes'])
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='mes', y='total', marker='o', color='steelblue')
        plt.title('Flujo Vehicular Promedio por Mes — Ruta 27', fontsize=14)
        plt.xlabel('Mes')
        plt.ylabel('Promedio de Vehículos')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.base, 'data/processed/flujo_por_mes.png'))
        plt.show()
        print("[Visualizador] Gráfico flujo por mes generado.")

    def grafico_pm25_por_mes(self):
        orden_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                       'Julio', 'Agosto', 'Setiembre', 'Octubre', 'Noviembre', 'Diciembre']
        df = self.df_aire.copy()
        df['mes'] = pd.Categorical(df['mes'], categories=orden_meses, ordered=True)
        df = df.sort_values(['anio', 'mes'])
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='mes', y='pm2_5', hue='anio', marker='o')
        plt.title('PM2.5 Promedio por Mes — GAM Costa Rica', fontsize=14)
        plt.xlabel('Mes')
        plt.ylabel('PM2.5 (μg/m³)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.base, 'data/processed/pm25_por_mes.png'))
        plt.show()
        print("[Visualizador] Gráfico PM2.5 por mes generado.")

    def grafico_contaminantes_por_anio(self):
        df = self.df_aire.groupby('anio')[['pm2_5', 'nitrogen_dioxide', 'ozone']].mean().reset_index()
        df_melted = df.melt(id_vars='anio', var_name='Contaminante', value_name='Promedio')
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_melted, x='anio', y='Promedio', hue='Contaminante')
        plt.title('Promedio de Contaminantes por Año — GAM Costa Rica', fontsize=14)
        plt.xlabel('Año')
        plt.ylabel('Promedio')
        plt.tight_layout()
        plt.savefig(os.path.join(self.base, 'data/processed/contaminantes_por_anio.png'))
        plt.show()
        print("[Visualizador] Gráfico contaminantes por año generado.")

    def heatmap_correlacion(self):
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
        plt.savefig(os.path.join(self.base, 'data/processed/heatmap_correlacion.png'))
        plt.show()
        print("[Visualizador] Heatmap de correlación generado.")

    def grafico_temperatura_por_anio(self):
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=self.df_clima, x='anio', y='temperature_2m',
                     marker='o', color='tomato')
        plt.title('Temperatura Promedio por Año — GAM Costa Rica', fontsize=14)
        plt.xlabel('Año')
        plt.ylabel('Temperatura (°C)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.base, 'data/processed/temperatura_por_anio.png'))
        plt.show()
        print("[Visualizador] Gráfico temperatura generado.")

    def mapa_ruta_27(self):
        mapa = folium.Map(location=[9.9281, -84.0907], zoom_start=10)

        puntos = [
            {"nombre": "Sabana San José", "km": 500, "lat": 9.9281, "lon": -84.0907},
            {"nombre": "Escazú", "km": 2900, "lat": 9.9141, "lon": -84.1366},
            {"nombre": "Santa Ana", "km": 7100, "lat": 9.9033, "lon": -84.1833},
            {"nombre": "Santa Ana - Ciudad Colón", "km": 10540, "lat": 9.8968, "lon": -84.2819},
            {"nombre": "Ciudad Colón", "km": 22450, "lat": 9.8897, "lon": -84.3197},
            {"nombre": "Castro Madriz Sector A", "km": 30620, "lat": 9.8750, "lon": -84.4100},
            {"nombre": "Castro Madriz Sector B", "km": 31550, "lat": 9.8720, "lon": -84.4250},
            {"nombre": "Orotina", "km": 41960, "lat": 9.9019, "lon": -84.5247},
            {"nombre": "Turrucares", "km": 54950, "lat": 9.9200, "lon": -84.4800},
            {"nombre": "Balsa de Atenas", "km": 62180, "lat": 9.9800, "lon": -84.3800},
            {"nombre": "Pozón de Orotina", "km": 71335, "lat": 9.8900, "lon": -84.6500},
            {"nombre": "Caldera", "km": 75400, "lat": 9.9019, "lon": -84.7356},
        ]

        flujo_promedio = self.df_flujo.groupby('punto_conteo')['total'].mean().to_dict()

        coordenadas = [[p["lat"], p["lon"]] for p in puntos]
        folium.PolyLine(
            coordenadas,
            color='#00D4AA',
            weight=4,
            opacity=0.8,
            tooltip="Ruta 27 — San José a Caldera"
        ).add_to(mapa)

        for i, punto in enumerate(puntos):
            promedio = flujo_promedio.get(punto["km"], 0)
            popup_html = f"""
            <div style='font-family: Arial; font-size: 13px; min-width: 200px;'>
                <b style='color:#0066CC;'>📍 {punto['nombre']}</b><br>
                <b>Km:</b> {punto['km']}<br>
                <b>Flujo promedio:</b> {promedio:,.0f} veh/mes
            </div>
            """
            folium.Marker(
                location=[punto["lat"], punto["lon"]],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Punto {i+1}: {punto['nombre']}",
                icon=folium.Icon(color='blue', icon='car', prefix='fa')
            ).add_to(mapa)

        mapa.save(os.path.join(self.base, 'data/processed/mapa_ruta_27.html'))
        print("[Visualizador] Mapa Ruta 27 guardado.")
        return mapa

    def ejecutar_visualizaciones(self):
        print("\n GENERANDO VISUALIZACIONES\n")
        self.cargar_datos()
        self.grafico_flujo_por_anio()
        self.grafico_flujo_por_mes()
        self.grafico_pm25_por_mes()
        self.grafico_contaminantes_por_anio()
        self.heatmap_correlacion()
        self.grafico_temperatura_por_anio()
        self.mapa_ruta_27()
        print("\n VISUALIZACIONES COMPLETADAS\n")