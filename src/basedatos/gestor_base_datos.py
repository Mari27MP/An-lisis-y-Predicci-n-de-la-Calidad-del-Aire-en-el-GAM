""""
Módulo para conexión e integración con la base de datos SQL Server.
"""

import pandas as pd
from sqlalchemy import create_engine, text


class GestorBaseDatos:

    def __init__(self):
        self.servidor = "CLARETRJ\\MMSP"
        self.base_datos = "CalidadAireGAM"
        self.usuario = "sa"
        self.contrasena = "Cla.06bin"
        self.engine = self._conectar()

    def _conectar(self):
        try:
            cadena = (
                f"mssql+pyodbc://{self.usuario}:{self.contrasena}"
                f"@{self.servidor}/{self.base_datos}"
                f"?driver=ODBC+Driver+17+for+SQL+Server"
                f"&TrustServerCertificate=yes"
            )
            engine = create_engine(cadena)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("[GestorBaseDatos] Conexion exitosa a SQL Server.")
            return engine
        except Exception as e:
            print(f"[GestorBaseDatos] Error al conectar: {e}")
            return None

    def limpiar_tabla(self, tabla: str) -> None:
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"DELETE FROM {tabla}"))
                conn.commit()
            print(f"[GestorBaseDatos] Tabla '{tabla}' limpiada.")
        except Exception as e:
            print(f"[GestorBaseDatos] Error al limpiar '{tabla}': {e}")

    def insertar_datos(self, df: pd.DataFrame, tabla: str) -> None:
        try:
            df.to_sql(tabla, con=self.engine, if_exists='append', index=False)
            print(f"[GestorBaseDatos] {len(df)} filas insertadas en '{tabla}'.")
        except Exception as e:
            print(f"[GestorBaseDatos] Error al insertar en '{tabla}': {e}")

    def consultar(self, query: str) -> pd.DataFrame:
        try:
            df = pd.read_sql(text(query), con=self.engine)
            print(f"[GestorBaseDatos] Consulta ejecutada: {len(df)} filas.")
            return df
        except Exception as e:
            print(f"[GestorBaseDatos] Error en consulta: {e}")
            return None