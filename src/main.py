import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datos.gestor_datos import GestorDatos


def main():
    print("=== Proyecto: Analisis y Prediccion de Calidad del Aire GAM ===\n")

    # ── Paso 1: Limpieza de datos de flujo vehicular Ruta 27 ──────────────
    print("--- Cargando y limpiando datos de flujo vehicular ---")

    gestor = GestorDatos('../data/raw/Datos_Abiertos_ARESEP_Flujos_vehiculares_ruta_27_.xlsx')
    df_limpio = gestor.limpiar_datos(incluir_2020=False)
    gestor.exportar_procesado('../data/processed/ruta27_limpio.csv')

    gestor_2020 = GestorDatos('../data/raw/Datos_Abiertos_ARESEP_Flujos_vehiculares_ruta_27_.xlsx')
    df_con_2020 = gestor_2020.limpiar_datos(incluir_2020=True)
    gestor_2020.exportar_procesado('../data/processed/ruta27_con_2020.csv')

    print("\n=== Paso 1 completado ===")

    # ── Paso 3: Limpieza de CSVs de APIs ──────────────────────────────────
    print("\n--- Limpiando datos de calidad del aire y clima ---")

    gestor_aire = GestorDatos('../data/raw/aire_crudo.csv')
    df_aire_limpio = gestor_aire.limpiar_aire('../data/raw/aire_crudo.csv')
    gestor_aire.exportar_procesado('../data/processed/aire_limpio.csv')

    gestor_clima = GestorDatos('../data/raw/clima_crudo.csv')
    df_clima_limpio = gestor_clima.limpiar_clima('../data/raw/clima_crudo.csv')
    gestor_clima.exportar_procesado('../data/processed/clima_limpio.csv')

    print("\n=== Paso 3 completado ===")


if __name__ == "__main__":
    main()