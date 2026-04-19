import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from basedatos.gestor_base_datos import GestorBaseDatos

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

gestor_bd = GestorBaseDatos()

df_ruta = pd.read_csv(os.path.join(BASE, 'data/processed/ruta27_limpio.csv'))
gestor_bd.insertar_datos(df_ruta, 'flujo_vehicular')

df_aire = pd.read_csv(os.path.join(BASE, 'data/processed/aire_limpio.csv'))
gestor_bd.insertar_datos(df_aire, 'calidad_aire')

df_clima = pd.read_csv(os.path.join(BASE, 'data/processed/clima_limpio.csv'))
gestor_bd.insertar_datos(df_clima, 'clima')