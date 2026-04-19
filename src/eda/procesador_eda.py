"""
Módulo para el análisis exploratorio de datos (EDA).
Realiza análisis estadístico, limpieza y exploración inicial de los datos.
"""

import pandas as pd
from basedatos.gestor_base_datos import GestorBaseDatos


class ProcesadorEDA:

    def __init__(self):
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