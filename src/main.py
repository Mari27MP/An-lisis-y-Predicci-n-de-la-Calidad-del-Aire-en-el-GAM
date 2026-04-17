import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datos.gestor_datos import GestorDatos


def main():
    print("=== Proyecto: Analisis y Prediccion de Calidad del Aire GAM ===\n")

    # ── Paso 1: Limpieza de datos de flujo vehicular Ruta 27 ──────────────
    print("--- Cargando y limpiando datos de flujo vehicular ---")

    # Version sin 2020 (version principal)
    gestor = GestorDatos('../data/raw/Datos_Abiertos_ARESEP_Flujos_vehiculares_ruta_27_.xlsx')
    df_limpio = gestor.limpiar_datos(incluir_2020=False)
    gestor.exportar_procesado('../data/processed/ruta27_limpio.csv')

    # Version con 2020 (para comparar con el modelo)
    gestor_2020 = GestorDatos('../data/raw/Datos_Abiertos_ARESEP_Flujos_vehiculares_ruta_27_.xlsx')
    df_con_2020 = gestor_2020.limpiar_datos(incluir_2020=True)
    gestor_2020.exportar_procesado('../data/processed/ruta27_con_2020.csv')

    print("\n=== Paso 1 completado ===")


if __name__ == "__main__":
    main()