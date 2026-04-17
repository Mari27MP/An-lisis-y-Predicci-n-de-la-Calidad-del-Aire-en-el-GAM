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