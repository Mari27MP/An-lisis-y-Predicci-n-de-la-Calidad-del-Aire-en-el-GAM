# Análisis y Predicción de la Calidad del Aire en el GAM

**Colegio Universitario de Cartago**  
**Curso**: Big Data (BD-143) — III Cuatrimestre 2025  
**Profesor**: Osvaldo González Chaves  
**Estudiantes**: Mariana Méndez | Claret Rodríguez  

---

## Descripción
Proyecto de ciencia de datos que analiza y predice la calidad del aire en el Gran Área Metropolitana de Costa Rica, correlacionando flujo vehicular con datos meteorológicos y contaminantes atmosféricos. Busca responder: ¿Cuándo hay más carros en la calle, hay peor calidad del aire en el GAM? Se predice la categoría de calidad del aire según el Índice Costarricense de Calidad del Aire (ICCA): Buena, Moderada, Mala o Muy Mala.

## Objetivos
- Analizar la relación entre flujo vehicular y niveles de contaminación en el GAM
- Identificar patrones temporales de contaminación por mes y año
- Predecir la categoría de calidad del aire usando Machine Learning supervisado
- Visualizar los resultados en un dashboard interactivo con mapas del GAM

## Fuentes de datos
- **Flujo vehicular CONAVI**: CSV de ARESEP con datos de peajes Zurquí, Naranjo, Tres Ríos y Alajuela (2003-2025)
- **Flujo vehicular Ruta 27**: XLSX de ARESEP con datos de la ruta San José-Caldera (2009-2025)
- **Calidad del aire**: API Open-Meteo Air Quality — PM2.5, PM10, NO2, CO, Ozono (desde 2013)
- **Clima histórico**: API Open-Meteo Archive — temperatura, humedad, velocidad del viento (desde 2003)
- **Base de datos**: SQL Server con tablas de flujo vehicular, calidad del aire, clima y predicciones

## Tecnologías utilizadas
- **Lenguaje**: Python 3
- **Manipulación de datos**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Random Forest, KNN, Árbol de Decisión, Regresión Logística)
- **Visualización**: Matplotlib, Seaborn, Folium (mapas interactivos)
- **Base de datos**: SQL Server con pyodbc
- **Dashboard**: Streamlit
- **Control de versiones**: Git, GitHub, SourceTree
- **Entorno**: Anaconda, Jupyter Notebook, PyCharm

## Arquitectura del proyecto (POO)
src/
├── datos/          # Clase GestorDatos: carga, limpia y exporta CSV/Excel
├── basedatos/      # Clase GestorBaseDatos: conexión y consultas SQL Server
├── api/            # Clase ClienteAPI: peticiones a APIs y transformación a DataFrames
├── eda/            # Clase ProcesadorEDA: análisis estadístico y exploración de datos
├── visualizacion/  # Clase Visualizador: gráficos, mapas y heatmaps
├── modelos/        # Clase ModeloML: entrenamiento y evaluación de modelos supervisados
├── helpers/        # Clase Utilidades: funciones auxiliares reutilizables
└── main.py         # Punto de entrada del proyecto
data/
├── raw/            # Archivos CSV y Excel originales
└── processed/      # Archivos procesados y limpios
notebooks/
└── exploracion_inicial.ipynb  # Jupyter notebook con EDA inicial

## Modelos de Machine Learning
- **Tipo**: Clasificación supervisada
- **Variable objetivo**: Categoría ICA (Buena / Moderada / Mala / Muy Mala)
- **Variables de entrada**: Flujo vehicular, temperatura, humedad, velocidad del viento, mes, año
- **Algoritmos**: Random Forest, KNN, Árbol de Decisión, Regresión Logística
- **Métrica de evaluación**: Accuracy (mínimo 90%)

## Rúbrica del proyecto
| Criterio | Porcentaje |
|---|---|
| Uso de datos (CSV, APIs, limpieza) | 25% |
| Conexión e integración con base de datos | 25% |
| Análisis exploratorio de datos (EDA) | 15% |
| Visualización de datos | 10% |
| Modelo de Machine Learning supervisado | 15% |
| Dashboard interactivo con Streamlit | 10% |
| Mapas como visualización adicional | +2% |
