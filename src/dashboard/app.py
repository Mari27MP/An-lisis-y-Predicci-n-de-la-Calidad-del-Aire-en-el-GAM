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
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .portada {
        background: linear-gradient(135deg, #0E1117 0%, #1A1D24 100%);
        border: 2px solid #00D4AA;
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        margin-bottom: 30px;
    }
    .portada h1 { color: #00D4AA; font-size: 2.5em; }
    .portada h3 { color: #FFFFFF; }
    .portada p { color: #AAAAAA; font-size: 1.1em; }
    .badge {
        background-color: #00D4AA;
        color: #0E1117;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def cargar_datos():
    gestor_bd = GestorBaseDatos()
    df_flujo = gestor_bd.consultar("SELECT * FROM flujo_vehicular")
    df_aire = gestor_bd.consultar("SELECT * FROM calidad_aire")
    df_clima = gestor_bd.consultar("SELECT * FROM clima")
    return df_flujo, df_aire, df_clima

df_flujo, df_aire, df_clima = cargar_datos()

st.sidebar.markdown("## 📊 Panel de Control")
seccion = st.sidebar.radio("Seleccione una sección:", [
    "🏠 Portada",
    "📈 Flujo Vehicular",
    "💨 Calidad del Aire",
    "🌡️ Clima",
    "🔬 EDA",
    "🤖 Predicción ICA"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filtros")
anios_disponibles = sorted(df_flujo['anio'].unique())
anio_sel = st.sidebar.selectbox("Año de análisis:", ["Todos"] + list(anios_disponibles))

if seccion == "🏠 Portada":
    st.markdown("""
    <div class="portada">
        <h1>🌿 Análisis y Predicción de la Calidad del Aire en el GAM</h1>
        <br>
        <h3>Colegio Universitario de Cartago</h3>
        <p>Curso: Big Data (BD-143) — III Cuatrimestre 2025</p>
        <p>Profesor: Osvaldo González Chaves</p>
        <br>
        <span class="badge">Mariana Méndez</span>
        <span class="badge">Claret Rodríguez</span>
        <br><br>
        <p style="color: #00D4AA; font-size: 1.3em;"><b>2025</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Descripción del Proyecto")
    st.markdown("""
    Este proyecto analiza la relación entre el **flujo vehicular** en la Ruta 27 (San José - Caldera)
    y la **calidad del aire** en el Gran Área Metropolitana (GAM) de Costa Rica.

    **Pregunta central:** ¿Cuando hay más vehículos en circulación, hay peor calidad del aire en el GAM?

    Se integran tres fuentes de datos:
    - 🚗 **Flujo vehicular** de la Ruta 27 (ARESEP) — 2014 a 2024
    - 💨 **Calidad del aire** (PM2.5, PM10, NO2, CO, Ozono) — Open-Meteo API
    - 🌡️ **Clima histórico** (temperatura, humedad, viento) — Open-Meteo Archive API
    """)

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 Período Análisis", "2022 — 2024")
    col2.metric("🚗 Registros Flujo", f"{len(df_flujo):,}")
    col3.metric("💨 Registros Aire", f"{len(df_aire):,}")
    col4.metric("🌡️ Registros Clima", f"{len(df_clima):,}")

elif seccion == "📈 Flujo Vehicular":
    st.markdown("## 📈 Flujo Vehicular Ruta 27")

    df_f = df_flujo if anio_sel == "Todos" else df_flujo[df_flujo['anio'] == anio_sel]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Vehículos", f"{df_f['total'].sum():,.0f}")
    col2.metric("Promedio Mensual", f"{df_f['total'].mean():,.0f}")
    col3.metric("Puntos de Conteo", f"{df_f['punto_conteo'].nunique()}")

    st.markdown("### Flujo Vehicular Total por Año")
    df_anio = df_flujo.groupby('anio')['total'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#1A1D24')
    ax.set_facecolor('#1A1D24')
    sns.barplot(data=df_anio, x='anio', y='total', hue='anio', palette='Blues_d', legend=False, ax=ax)
    ax.set_xlabel('Año', color='white')
    ax.set_ylabel('Total de Vehículos', color='white')
    ax.tick_params(colors='white')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Flujo por Punto de Conteo")
    df_punto = df_f.groupby('ubicacion')['total'].sum().reset_index().sort_values('total', ascending=True)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    fig2.patch.set_facecolor('#1A1D24')
    ax2.set_facecolor('#1A1D24')
    sns.barplot(data=df_punto, x='total', y='ubicacion', palette='Blues_d', ax=ax2)
    ax2.set_xlabel('Total de Vehículos', color='white')
    ax2.set_ylabel('Ubicación', color='white')
    ax2.tick_params(colors='white')
    plt.tight_layout()
    st.pyplot(fig2)

elif seccion == "💨 Calidad del Aire":
    st.markdown("## 💨 Calidad del Aire — GAM Costa Rica")

    col1, col2, col3 = st.columns(3)
    col1.metric("PM2.5 Promedio", f"{df_aire['pm2_5'].mean():.2f} μg/m³")
    col2.metric("NO2 Promedio", f"{df_aire['nitrogen_dioxide'].mean():.2f}")
    col3.metric("Ozono Promedio", f"{df_aire['ozone'].mean():.2f}")

    st.markdown("### PM2.5 por Mes")
    orden_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                   'Julio', 'Agosto', 'Setiembre', 'Octubre', 'Noviembre', 'Diciembre']
    df_mes = df_aire.copy()
    df_mes['mes'] = pd.Categorical(df_mes['mes'], categories=orden_meses, ordered=True)
    df_mes = df_mes.sort_values(['anio', 'mes'])
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#1A1D24')
    ax.set_facecolor('#1A1D24')
    sns.lineplot(data=df_mes, x='mes', y='pm2_5', hue='anio', marker='o', ax=ax)
    ax.set_xlabel('Mes', color='white')
    ax.set_ylabel('PM2.5 (μg/m³)', color='white')
    ax.tick_params(colors='white')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Contaminantes por Año")
    df_cont = df_aire.groupby('anio')[['pm2_5', 'nitrogen_dioxide', 'ozone']].mean().reset_index()
    df_melted = df_cont.melt(id_vars='anio', var_name='Contaminante', value_name='Promedio')
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    fig2.patch.set_facecolor('#1A1D24')
    ax2.set_facecolor('#1A1D24')
    sns.barplot(data=df_melted, x='anio', y='Promedio', hue='Contaminante', ax=ax2)
    ax2.tick_params(colors='white')
    plt.tight_layout()
    st.pyplot(fig2)

elif seccion == "🌡️ Clima":
    st.markdown("## 🌡️ Variables Climáticas — GAM Costa Rica")

    col1, col2, col3 = st.columns(3)
    col1.metric("Temperatura Promedio", f"{df_clima['temperature_2m'].mean():.1f}°C")
    col2.metric("Humedad Promedio", f"{df_clima['relative_humidity_2m'].mean():.1f}%")
    col3.metric("Viento Promedio", f"{df_clima['windspeed_10m'].mean():.1f} km/h")

    st.markdown("### Temperatura por Año")
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#1A1D24')
    ax.set_facecolor('#1A1D24')
    sns.lineplot(data=df_clima, x='anio', y='temperature_2m', marker='o', color='tomato', ax=ax)
    ax.set_xlabel('Año', color='white')
    ax.set_ylabel('Temperatura (°C)', color='white')
    ax.tick_params(colors='white')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Correlación: Flujo Vehicular vs Contaminantes")
    df_merge = pd.merge(
        df_flujo.groupby(['anio', 'mes'])['total'].sum().reset_index(),
        df_aire[['anio', 'mes', 'pm2_5', 'nitrogen_dioxide', 'ozone']],
        on=['anio', 'mes'], how='inner'
    )
    correlacion = df_merge[['total', 'pm2_5', 'nitrogen_dioxide', 'ozone']].corr()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    fig2.patch.set_facecolor('#1A1D24')
    ax2.set_facecolor('#1A1D24')
    sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
    ax2.tick_params(colors='white')
    plt.tight_layout()
    st.pyplot(fig2)

elif seccion == "🔬 EDA":
    st.markdown("## 🔬 Análisis Exploratorio de Datos")

    tab1, tab2, tab3 = st.tabs(["Flujo Vehicular", "Calidad del Aire", "Clima"])

    with tab1:
        st.markdown("### Estadísticas Descriptivas — Flujo Vehicular")
        st.dataframe(df_flujo[['anio', 'liviano', 'dos_tres_ejes', 'autobus', 'total']].describe().round(2))
        st.markdown("### Flujo por Año")
        st.dataframe(df_flujo.groupby('anio')['total'].sum().reset_index())

    with tab2:
        st.markdown("### Estadísticas Descriptivas — Calidad del Aire")
        st.dataframe(df_aire[['pm2_5', 'pm10', 'nitrogen_dioxide', 'carbon_monoxide', 'ozone']].describe().round(2))
        st.markdown("### Promedio por Año")
        st.dataframe(df_aire.groupby('anio')[['pm2_5', 'nitrogen_dioxide', 'ozone']].mean().round(2))

    with tab3:
        st.markdown("### Estadísticas Descriptivas — Clima")
        st.dataframe(df_clima[['temperature_2m', 'relative_humidity_2m', 'windspeed_10m']].describe().round(2))
        st.markdown("### Promedio por Año")
        st.dataframe(df_clima.groupby('anio')[['temperature_2m', 'relative_humidity_2m', 'windspeed_10m']].mean().round(2))

elif seccion == "🤖 Predicción ICA":
    st.markdown("## 🤖 Predicción de Calidad del Aire — 2025")
    st.markdown("Seleccione un mes de 2025 para predecir la categoría ICA.")

    orden_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                   'Julio', 'Agosto', 'Setiembre', 'Octubre', 'Noviembre', 'Diciembre']
    meses_num = {m: i+1 for i, m in enumerate(orden_meses)}

    mes_sel = st.selectbox("📅 Seleccione el mes de 2025:", orden_meses)
    mes_num = meses_num[mes_sel]

    clima_mes = df_clima[df_clima['mes_num'] == mes_num][['temperature_2m', 'relative_humidity_2m', 'windspeed_10m']].mean()
    aire_mes = df_aire[df_aire['mes_num'] == mes_num][['pm2_5', 'nitrogen_dioxide', 'ozone']].mean()
    flujo_mes = df_flujo[df_flujo['mes'] == mes_sel]['total'].mean()

    st.markdown(f"### Valores históricos promedio para {mes_sel}:")
    col1, col2, col3 = st.columns(3)
    col1.metric("🌡️ Temperatura", f"{clima_mes['temperature_2m']:.1f}°C" if not pd.isna(clima_mes['temperature_2m']) else "N/A")
    col2.metric("💧 Humedad", f"{clima_mes['relative_humidity_2m']:.1f}%" if not pd.isna(clima_mes['relative_humidity_2m']) else "N/A")
    col3.metric("💨 Viento", f"{clima_mes['windspeed_10m']:.1f} km/h" if not pd.isna(clima_mes['windspeed_10m']) else "N/A")

    st.markdown("### Ajuste el flujo vehicular esperado para 2025:")
    total_vehiculos = st.slider("🚗 Total de vehículos", 100000, 5000000,
                                 int(flujo_mes) if not pd.isna(flujo_mes) else 950000, 50000)

    st.markdown("---")

    if st.button(f"🔍 Predecir Calidad del Aire para {mes_sel} 2025", use_container_width=True):
        with st.spinner("Ejecutando modelo..."):
            modelo = ModeloML()
            modelo.cargar_y_preparar_datos()
            modelo.dividir_datos()
            categoria = modelo.predecir_nuevo(
                total_vehiculos=total_vehiculos,
                pm2_5=float(aire_mes['pm2_5']) if not pd.isna(aire_mes['pm2_5']) else 8.5,
                nitrogen_dioxide=float(aire_mes['nitrogen_dioxide']) if not pd.isna(aire_mes['nitrogen_dioxide']) else 15.2,
                ozone=float(aire_mes['ozone']) if not pd.isna(aire_mes['ozone']) else 32.1,
                temperature_2m=float(clima_mes['temperature_2m']) if not pd.isna(clima_mes['temperature_2m']) else 20.5,
                relative_humidity_2m=float(clima_mes['relative_humidity_2m']) if not pd.isna(clima_mes['relative_humidity_2m']) else 78.0,
                windspeed_10m=float(clima_mes['windspeed_10m']) if not pd.isna(clima_mes['windspeed_10m']) else 7.5
            )

        colores = {"Buena": "🟢", "Moderada": "🟡", "Mala": "🟠", "Muy Mala": "🔴"}
        icono = colores.get(categoria, "⚪")
        st.success(f"### {icono} Predicción para {mes_sel} 2025: **{categoria}**")

        st.markdown("### Escala ICA de referencia:")
        col1, col2, col3, col4 = st.columns(4)
        col1.success("🟢 Buena\nPM2.5 ≤ 12")
        col2.warning("🟡 Moderada\nPM2.5 ≤ 35")
        col3.error("🟠 Mala\nPM2.5 ≤ 55")
        col4.error("🔴 Muy Mala\nPM2.5 > 55")