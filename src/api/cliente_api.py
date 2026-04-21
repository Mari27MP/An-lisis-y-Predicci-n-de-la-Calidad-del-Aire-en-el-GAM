import requests
import pandas as pd
import os


class ClienteAPI:

    def __init__(self):
        self.url_aire = "https://air-quality-api.open-meteo.com/v1/air-quality"
        self.url_clima = "https://archive-api.open-meteo.com/v1/archive"
        self.fecha_inicio = "2022-08-04"
        self.fecha_fin = "2024-12-31"
        self.carpeta_salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/raw")
        os.makedirs(self.carpeta_salida, exist_ok=True)

    def obtener_aire(self):
        params = {
            "latitude": 9.9281,
            "longitude": -84.0907,
            "hourly": "pm2_5,pm10,nitrogen_dioxide,carbon_monoxide,ozone",
            "start_date": self.fecha_inicio,
            "end_date": self.fecha_fin
        }

        response = requests.get(self.url_aire, params=params)

        if response.status_code == 200:
            print("Datos de calidad del aire obtenidos exitosamente.")
            df = pd.DataFrame(response.json()["hourly"])
            return df
        else:
            print(f"Error al obtener datos de aire: {response.status_code}")
            return None

    def obtener_clima(self):
        params = {
            "latitude": 9.9281,
            "longitude": -84.0907,
            "hourly": "temperature_2m,relative_humidity_2m,windspeed_10m",
            "start_date": self.fecha_inicio,
            "end_date": self.fecha_fin,
            "timezone": "America/Costa_Rica"
        }

        response = requests.get(self.url_clima, params=params)

        if response.status_code == 200:
            print("Datos de clima obtenidos exitosamente.")
            df = pd.DataFrame(response.json()["hourly"])
            return df
        else:
            print(f"Error al obtener datos de clima: {response.status_code}")
            return None

    def exportar_csv(self, df, nombre_archivo):
        if df is not None:
            ruta = os.path.join(self.carpeta_salida, nombre_archivo)
            df.to_csv(ruta, index=False)
            print(f"Archivo guardado en: {ruta}")
        else:
            print("No hay datos para exportar.")