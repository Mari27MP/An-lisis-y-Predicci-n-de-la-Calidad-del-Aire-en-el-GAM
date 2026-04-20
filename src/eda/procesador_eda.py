"""
Módulo para el análisis exploratorio de datos (EDA).
Realiza análisis estadístico, limpieza y exploración inicial de los datos.
"""

import pandas as pd
import numpy as np
from basedatos.gestor_base_datos import GestorBaseDatos


class ProcesadorEDA:

    def __init__(self):
        """Inicializa el procesador conectando a la base de datos."""
        self.gestor_bd = GestorBaseDatos()
        self.df_flujo = None
        self.df_aire = None
        self.df_clima = None

    def cargar_datos(self):
        """Carga los datos desde SQL Server."""
        self.df_flujo = self.gestor_bd.consultar("SELECT * FROM flujo_vehicular")
        self.df_aire = self.gestor_bd.consultar("SELECT * FROM calidad_aire")
        self.df_clima = self.gestor_bd.consultar("SELECT * FROM clima")
        print("[ProcesadorEDA] Datos cargados correctamente.")

    # ─────────────────────────────────────────
    # ESTADÍSTICAS DESCRIPTIVAS
    # ─────────────────────────────────────────

    def resumen_general(self, df: pd.DataFrame, nombre: str) -> None:
        """
        Imprime un resumen general del DataFrame indicado.

        Args:
            df (pd.DataFrame): DataFrame a analizar.
            nombre (str): Nombre del conjunto de datos.
        """
        print(f"\n{'='*50}")
        print(f" RESUMEN: {nombre}")
        print(f"{'='*50}")
        print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
        print(f"\nColumnas: {df.columns.tolist()}")
        print(f"\nTipos de datos:\n{df.dtypes}")
        print(f"\nEstadísticas descriptivas:\n{df.describe()}")

    def valores_nulos(self, df: pd.DataFrame, nombre: str) -> pd.DataFrame:
        """
        Detecta y reporta valores nulos en el DataFrame.

        Args:
            df (pd.DataFrame): DataFrame a analizar.
            nombre (str): Nombre del conjunto de datos.

        Returns:
            pd.DataFrame: Reporte de valores nulos por columna.
        """
        nulos = df.isnull().sum()
        porcentaje = (nulos / len(df)) * 100
        reporte = pd.DataFrame({
            'Valores_Nulos': nulos,
            'Porcentaje': porcentaje
        })
        print(f"\n{'='*50}")
        print(f" VALORES NULOS: {nombre}")
        print(f"{'='*50}")
        print(reporte)
        return reporte

    # ─────────────────────────────────────────
    # ANÁLISIS TEMPORAL
    # ─────────────────────────────────────────

    def flujo_por_anio(self) -> pd.DataFrame:
        """
        Calcula el total de vehículos por año en la Ruta 27.

        Returns:
            pd.DataFrame: Total de flujo vehicular agrupado por año.
        """
        resultado = self.df_flujo.groupby('anio')['total'].sum().reset_index()
        resultado.columns = ['Año', 'Total_Vehiculos']
        print(f"\n{'='*50}")
        print(" FLUJO VEHICULAR POR AÑO")
        print(f"{'='*50}")
        print(resultado)
        return resultado

    def promedio_aire_por_anio(self) -> pd.DataFrame:
        """
        Calcula el promedio anual de PM2.5 y NO2.

        Returns:
            pd.DataFrame: Promedios anuales de contaminantes.
        """
        resultado = self.df_aire.groupby('anio')[['pm2_5', 'nitrogen_dioxide']].mean().reset_index()
        print(f"\n{'='*50}")
        print(" PROMEDIO ANUAL DE CONTAMINANTES")
        print(f"{'='*50}")
        print(resultado)
        return resultado

    def promedio_clima_por_anio(self) -> pd.DataFrame:
        """
        Calcula el promedio anual de temperatura, humedad y viento.

        Returns:
            pd.DataFrame: Promedios anuales de variables climáticas.
        """
        resultado = self.df_clima.groupby('anio')[['temperature_2m', 'relative_humidity_2m', 'windspeed_10m']].mean().reset_index()
        print(f"\n{'='*50}")
        print(" PROMEDIO ANUAL DE CLIMA")
        print(f"{'='*50}")
        print(resultado)
        return resultado

    # ─────────────────────────────────────────
    # CORRELACIONES
    # ─────────────────────────────────────────

    def correlacion_flujo_aire(self) -> pd.DataFrame:
        """
        Calcula la correlación entre flujo vehicular y contaminantes
        uniendo los DataFrames por año y mes.

        Returns:
            pd.DataFrame: Matriz de correlación entre variables.
        """
        df_merge = pd.merge(
            self.df_flujo.groupby(['anio', 'mes'])['total'].sum().reset_index(),
            self.df_aire[['anio', 'mes', 'pm2_5', 'nitrogen_dioxide', 'ozone']],
            on=['anio', 'mes'],
            how='inner'
        )
        correlacion = df_merge[['total', 'pm2_5', 'nitrogen_dioxide', 'ozone']].corr()
        print(f"\n{'=' * 50}")
        print(" CORRELACIÓN: FLUJO VEHICULAR vs CONTAMINANTES")
        print(f"{'=' * 50}")
        print(correlacion)
        return correlacion

    # ─────────────────────────────────────────
    # EDA COMPLETO
    # ─────────────────────────────────────────

    def ejecutar_eda_completo(self) -> None:
        """
        Ejecuta el análisis exploratorio completo sobre los tres DataFrames.
        """
        print("\n🔍 INICIANDO ANÁLISIS EXPLORATORIO DE DATOS (EDA)\n")

        self.cargar_datos()

        self.resumen_general(self.df_flujo, "Flujo Vehicular Ruta 27")
        self.resumen_general(self.df_aire, "Calidad del Aire")
        self.resumen_general(self.df_clima, "Clima Histórico")

        self.valores_nulos(self.df_flujo, "Flujo Vehicular")
        self.valores_nulos(self.df_aire, "Calidad del Aire")
        self.valores_nulos(self.df_clima, "Clima")

        self.flujo_por_anio()
        self.promedio_aire_por_anio()
        self.promedio_clima_por_anio()
        self.correlacion_flujo_aire()

        print("\n✅ EDA COMPLETADO\n")