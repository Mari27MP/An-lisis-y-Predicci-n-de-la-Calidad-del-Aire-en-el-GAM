"""
Módulo de utilidades generales reutilizables para el proyecto.
Proporciona funciones auxiliares de validación, formateo y logging.
"""

import os
import logging
from datetime import datetime


class Utilidades:

    def __init__(self):
        self.logger = self._configurar_logger()

    def _configurar_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        return logging.getLogger(__name__)

    def log_info(self, mensaje: str):
        self.logger.info(mensaje)

    def log_error(self, mensaje: str):
        self.logger.error(mensaje)

    def validar_archivo(self, ruta: str) -> bool:
        existe = os.path.exists(ruta)
        if not existe:
            self.log_error(f"Archivo no encontrado: {ruta}")
        else:
            self.log_info(f"Archivo validado correctamente: {ruta}")
        return existe

    def obtener_timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def mes_a_numero(self, mes: str) -> int:
        meses = {
            "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
            "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
            "Setiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
        }
        return meses.get(mes.strip(), 0)

    def numero_a_mes(self, numero: int) -> str:
        meses = {
            1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
            5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
            9: "Setiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
        }
        return meses.get(numero, "")

    def categorizar_ica(self, pm25: float) -> str:
        if pm25 <= 12:
            return "Buena"
        elif pm25 <= 35.4:
            return "Moderada"
        elif pm25 <= 55.4:
            return "Mala"
        else:
            return "Muy Mala"

    def validar_dataframe(self, df, columnas_requeridas: list) -> bool:
        faltantes = [c for c in columnas_requeridas if c not in df.columns]
        if faltantes:
            self.log_error(f"Columnas faltantes: {faltantes}")
            return False
        self.log_info("DataFrame validado correctamente")
        return True