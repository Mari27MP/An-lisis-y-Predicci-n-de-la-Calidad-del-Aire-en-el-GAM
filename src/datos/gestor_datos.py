"""
Módulo para carga, limpieza y transformación del dataset de flujo vehicular Ruta 27.
Fuente: ARESEP - Flujos vehiculares Ruta 27 (San José - Caldera)
"""

import pandas as pd
import os
from itertools import product


class GestorDatos:
    """
    Encargada de cargar, transformar y exportar el dataset de flujo vehicular.
    Aplica limpieza, imputación de datos faltantes y estandarización de columnas.
    """

    # Nombres de los lugares correspondientes a cada punto de conteo
    # Los números representan el kilómetro de la Ruta 27 donde está el sensor
    UBICACIONES = {
        500:   'Sabana San Jose',
        2900:  'Escazu',
        7100:  'Santa Ana',
        10540: 'Santa Ana Ciudad Colon',
        22450: 'Ciudad Colon',
        30620: 'Castro Madriz Sector A',
        31550: 'Castro Madriz Sector B',
        41960: 'Orotina',
        54950: 'Turrucares',
        62180: 'Balsa de Atenas',
        71335: 'Pozon de Orotina',
        75400: 'Caldera'
    }

    MESES = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
             'Julio', 'Agosto', 'Setiembre', 'Octubre', 'Noviembre', 'Diciembre']

    def __init__(self, ruta_archivo: str):
        self.ruta_archivo = ruta_archivo
        self.df = None
        self.df_limpio = None

    def cargar_datos(self) -> pd.DataFrame:
        """Carga el archivo Excel del dataset de Ruta 27."""
        if not os.path.exists(self.ruta_archivo):
            raise FileNotFoundError(f"Archivo no encontrado: {self.ruta_archivo}")
        self.df = pd.read_excel(self.ruta_archivo)
        print(f"[GestorDatos] Archivo cargado: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
        return self.df

    def limpiar_datos(self, incluir_2020: bool = False) -> pd.DataFrame:
        """
        Limpia y transforma el dataset de Ruta 27.
        Por defecto excluye el 2020 por ser un año atipico debido al COVID.
        Si incluir_2020=True se incluye para comparar con el modelo sin ese año.
        """
        if self.df is None:
            self.cargar_datos()

        df = self.df.copy()

        # 1. Nos quedamos solo con los datos del 2014 al 2024
        #    porque antes del 2014 no todos los puntos de conteo estaban activos
        df = df[df['Año'].between(2014, 2024)]

        # 2. El 2020 se puede excluir porque el COVID alteró
        #    completamente los patrones de tráfico ese año
        if not incluir_2020:
            df = df[df['Año'] != 2020]
            print("[GestorDatos] Año 2020 eliminado")

        # 3. Los meses vienen con espacios al final en el archivo original
        #    por ejemplo 'Enero     ' en vez de 'Enero'
        df['Mes'] = df['Mes'].str.strip()

        # 4. Algunos puntos de conteo no reportaron datos en ciertos meses
        #    Creamos todas las combinaciones posibles para identificar esas filas faltantes
        años = [a for a in range(2014, 2025) if a != 2020 or incluir_2020]
        puntos = df['Punto Conteo'].unique().tolist()
        combinaciones = pd.DataFrame(
            list(product(años, self.MESES, puntos)),
            columns=['Año', 'Mes', 'Punto Conteo']
        )
        df = combinaciones.merge(df, on=['Año', 'Mes', 'Punto Conteo'], how='left')

        # 5. Las filas faltantes se rellenan con el promedio historico
        #    del mismo mes y punto de conteo en los demas años
        cols_numericas = ['Liviano', 'Dos Tres Ejes', 'Cuatro Ejes', 'Cinco Más Ejes', 'Autobus', 'Total']
        for col in cols_numericas:
            df[col] = df.groupby(['Punto Conteo', 'Mes'])[col].transform(
                lambda x: x.fillna(x.mean())
            )

        # 6. Agregamos el nombre del lugar correspondiente a cada punto de conteo
        #    ya que en el archivo original solo viene el numero de kilometro
        df['ubicacion'] = df['Punto Conteo'].map(self.UBICACIONES)

        # 7. Renombramos las columnas para evitar problemas con tildes y espacios
        #    especialmente al insertar los datos en SQL Server
        df.rename(columns={
            'Año':            'anio',
            'Mes':            'mes',
            'Punto Conteo':   'punto_conteo',
            'Liviano':        'liviano',
            'Dos Tres Ejes':  'dos_tres_ejes',
            'Cuatro Ejes':    'cuatro_ejes',
            'Cinco Más Ejes': 'cinco_mas_ejes',
            'Autobus':        'autobus',
            'Total':          'total'
        }, inplace=True)

        # 8. Las columnas numericas venian como float, las convertimos a int
        #    porque son conteos de vehiculos y no tienen decimales
        cols_int = ['anio', 'punto_conteo', 'liviano', 'dos_tres_ejes',
                    'cuatro_ejes', 'cinco_mas_ejes', 'autobus', 'total']
        df[cols_int] = df[cols_int].astype(int)

        self.df_limpio = df
        print(f"[GestorDatos] Limpieza completada: {len(df)} filas")
        return df

    def exportar_procesado(self, ruta_salida: str) -> None:
        """Exporta el DataFrame limpio a data/processed/ en formato CSV."""
        if self.df_limpio is None:
            raise ValueError("No hay datos limpios. Ejecute limpiar_datos() primero.")
        self.df_limpio.to_csv(ruta_salida, index=False)
        print(f"[GestorDatos] Archivo exportado: {ruta_salida}")

    def limpiar_aire(self, ruta_archivo: str) -> pd.DataFrame:
        """
        Limpia el CSV crudo de calidad del aire descargado de la API.
        Convierte time a datetime, extrae anio y mes, y agrega a nivel mensual.
        """
        df = pd.read_csv(ruta_archivo)

        # 1. Convertir time a datetime
        df['time'] = pd.to_datetime(df['time'])

        # 2. Extraer anio y mes numero
        df['anio'] = df['time'].dt.year
        df['mes_num'] = df['time'].dt.month

        # 3. Agregar a nivel mensual con promedio
        df_mensual = df.groupby(['anio', 'mes_num'])[
            ['pm2_5', 'pm10', 'nitrogen_dioxide', 'carbon_monoxide', 'ozone']
        ].mean().reset_index()

        # 4. Agregar columna mes en texto usando el diccionario de meses
        meses = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
                 5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
                 9: 'Setiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
        df_mensual['mes'] = df_mensual['mes_num'].map(meses)

        # 5. Redondear a 2 decimales
        cols = ['pm2_5', 'pm10', 'nitrogen_dioxide', 'carbon_monoxide', 'ozone']
        df_mensual[cols] = df_mensual[cols].round(2)

        self.df_limpio = df_mensual
        print(f"[GestorDatos] Aire limpio: {len(df_mensual)} filas")
        return df_mensual

    def limpiar_clima(self, ruta_archivo: str) -> pd.DataFrame:
        """
        Limpia el CSV crudo de clima descargado de la API.
        Convierte time a datetime, extrae anio y mes, y agrega a nivel mensual.
        """
        df = pd.read_csv(ruta_archivo)

        # 1. Convertir time a datetime
        df['time'] = pd.to_datetime(df['time'])

        # 2. Extraer anio y mes numero
        df['anio'] = df['time'].dt.year
        df['mes_num'] = df['time'].dt.month

        # 3. Agregar a nivel mensual con promedio
        df_mensual = df.groupby(['anio', 'mes_num'])[
            ['temperature_2m', 'relative_humidity_2m', 'windspeed_10m']
        ].mean().reset_index()

        # 4. Agregar columna mes en texto
        meses = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
                 5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
                 9: 'Setiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
        df_mensual['mes'] = df_mensual['mes_num'].map(meses)

        # 5. Redondear a 2 decimales
        cols = ['temperature_2m', 'relative_humidity_2m', 'windspeed_10m']
        df_mensual[cols] = df_mensual[cols].round(2)

        self.df_limpio = df_mensual
        print(f"[GestorDatos] Clima limpio: {len(df_mensual)} filas")
        return df_mensual