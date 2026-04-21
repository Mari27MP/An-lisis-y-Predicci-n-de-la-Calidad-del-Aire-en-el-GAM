"""
Dashboard interactivo para el análisis de Calidad del Aire en el GAM.
Proyecto: Big Data BD-143 — Colegio Universitario de Cartago
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from basedatos.gestor_base_datos import GestorBaseDatos
from modelos.modelo_ml import ModeloML

st.set_page_config(
    page_title="Calidad del Aire GAM",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Análisis y Predicción de la Calidad del Aire en el GAM")
st.markdown("**Colegio Universitario de Cartago — Big Data BD-143**")
st.markdown("---")

@st.cache_data
def cargar_datos():
    gestor_bd = GestorBaseDatos()
    df_flujo = gestor_bd.consultar("SELECT * FROM flujo_vehicular")
    df_aire = gestor_bd.consultar("SELECT * FROM calidad_aire")
    df_clima = gestor_bd.consultar("SELECT * FROM clima")
    return df_flujo, df_aire, df_clima

df_flujo, df_aire, df_clima = cargar_datos()

st.sidebar.title("📊 Navegación")
seccion = st.sidebar.radio("Seleccione una sección:", [
    "📈 Flujo Vehicular",
    "💨 Calidad del Aire",
    "🌡️ Clima",
    "🤖 Predicción ICA"
])

if seccion == "📈 Flujo Vehicular":
    st.header("📈 Flujo Vehicular Ruta 27")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Registros", f"{len(df_flujo):,}")
    col2.metric("Años Disponibles", f"{df_flujo['anio'].min()} - {df_flujo['anio'].max()}")
    col3.metric("Puntos de Conteo", f"{df_flujo['punto_conteo'].nunique()}")

    st.subheader("Flujo Vehicular Total por Año")
    df_anio = df_flujo.groupby('anio')['total'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df_anio, x='anio', y='total', hue='anio', palette='Blues_d', legend=False, ax=ax)
    ax.set_xlabel('Año')
    ax.set_ylabel('Total de Vehículos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Flujo por Punto de Conteo")
    df_punto = df_flujo.groupby('ubicacion')['total'].sum().reset_index().sort_values('total', ascending=False)
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df_punto, x='total', y='ubicacion', palette='Blues_d', ax=ax2)
    ax2.set_xlabel('Total de Vehículos')
    ax2.set_ylabel('Ubicación')
    plt.tight_layout()
    st.pyplot(fig2)

elif seccion == "💨 Calidad del Aire":
    st.header("💨 Calidad del Aire — GAM Costa Rica")

    col1, col2, col3 = st.columns(3)
    col1.metric("PM2.5 Promedio", f"{df_aire['pm2_5'].mean():.2f} μg/m³")
    col2.metric("NO2 Promedio", f"{df_aire['nitrogen_dioxide'].mean():.2f}")
    col3.metric("Ozono Promedio", f"{df_aire['ozone'].mean():.2f}")

    st.subheader("PM2.5 por Mes")
    orden_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                   'Julio', 'Agosto', 'Setiembre', 'Octubre', 'Noviembre', 'Diciembre']
    df_mes = df_aire.copy()
    df_mes['mes'] = pd.Categorical(df_mes['mes'], categories=orden_meses, ordered=True)
    df_mes = df_mes.sort_values(['anio', 'mes'])
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=df_mes, x='mes', y='pm2_5', hue='anio', marker='o', ax=ax)
    ax.set_xlabel('Mes')
    ax.set_ylabel('PM2.5 (μg/m³)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Contaminantes por Año")
    df_cont = df_aire.groupby('anio')[['pm2_5', 'nitrogen_dioxide', 'ozone']].mean().reset_index()
    df_melted = df_cont.melt(id_vars='anio', var_name='Contaminante', value_name='Promedio')
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df_melted, x='anio', y='Promedio', hue='Contaminante', ax=ax2)
    plt.tight_layout()
    st.pyplot(fig2)

elif seccion == "🌡️ Clima":
    st.header("🌡️ Variables Climáticas — GAM Costa Rica")

    col1, col2, col3 = st.columns(3)
    col1.metric("Temperatura Promedio", f"{df_clima['temperature_2m'].mean():.1f}°C")
    col2.metric("Humedad Promedio", f"{df_clima['relative_humidity_2m'].mean():.1f}%")
    col3.metric("Viento Promedio", f"{df_clima['windspeed_10m'].mean():.1f} km/h")

    st.subheader("Temperatura por Año")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=df_clima, x='anio', y='temperature_2m', marker='o', color='tomato', ax=ax)
    ax.set_xlabel('Año')
    ax.set_ylabel('Temperatura (°C)')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Correlación: Flujo Vehicular vs Contaminantes")
    df_merge = pd.merge(
        df_flujo.groupby(['anio', 'mes'])['total'].sum().reset_index(),
        df_aire[['anio', 'mes', 'pm2_5', 'nitrogen_dioxide', 'ozone']],
        on=['anio', 'mes'], how='inner'
    )
    correlacion = df_merge[['total', 'pm2_5', 'nitrogen_dioxide', 'ozone']].corr()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
    plt.tight_layout()
    st.pyplot(fig2)

elif seccion == "🤖 Predicción ICA":
    st.header("🤖 Predicción de Categoría ICA")
    st.markdown("Ingrese los valores para predecir la calidad del aire:")

    col1, col2 = st.columns(2)
    with col1:
        total_vehiculos = st.number_input("Total de vehículos", min_value=0, value=950000)
        pm2_5 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, value=8.5)
        nitrogen_dioxide = st.number_input("NO2", min_value=0.0, value=15.2)
        ozone = st.number_input("Ozono", min_value=0.0, value=32.1)
    with col2:
        temperature_2m = st.number_input("Temperatura (°C)", min_value=0.0, value=20.5)
        relative_humidity_2m = st.number_input("Humedad (%)", min_value=0.0, value=78.0)
        windspeed_10m = st.number_input("Viento (km/h)", min_value=0.0, value=7.5)

    if st.button("🔍 Predecir"):
        modelo = ModeloML()
        modelo.cargar_y_preparar_datos()
        modelo.dividir_datos()
        categoria = modelo.predecir_nuevo(
            total_vehiculos=total_vehiculos,
            pm2_5=pm2_5,
            nitrogen_dioxide=nitrogen_dioxide,
            ozone=ozone,
            temperature_2m=temperature_2m,
            relative_humidity_2m=relative_humidity_2m,
            windspeed_10m=windspeed_10m
        )
        colores = {"Buena": "🟢", "Moderada": "🟡", "Mala": "🟠", "Muy Mala": "🔴"}
        icono = colores.get(categoria, "⚪")
        st.success(f"### {icono} Categoría ICA predicha: **{categoria}**")