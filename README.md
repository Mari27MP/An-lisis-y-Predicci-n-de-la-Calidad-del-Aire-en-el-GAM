# Análisis y Predicción de la Calidad del Aire en el GAM

*Colegio Universitario de Cartago*  
*Curso*: Big Data (BD-143) — III Cuatrimestre 2025  
*Profesor*: Osvaldo González Chaves  
*Estudiantes*: Mariana Méndez | Claret Rodríguez  

---

## Descripción
Proyecto de ciencia de datos que analiza y predice la calidad del aire en el Gran Área Metropolitana de Costa Rica, correlacionando flujo vehicular con datos meteorológicos y contaminantes atmosféricos. Busca responder: ¿Cuándo hay más carros en la calle, hay peor calidad del aire en el GAM? Se predice la categoría de calidad del aire según el Índice Costarricense de Calidad del Aire (ICCA): Buena, Moderada, Mala o Muy Mala.

## Objetivos
- Analizar la relación entre flujo vehicular y niveles de contaminación en el GAM
- Identificar patrones temporales de contaminación por mes y año
- Predecir la categoría de calidad del aire usando Machine Learning supervisado
- Visualizar los resultados en un dashboard interactivo con mapas del GAM

## Fuentes de datos
- *Flujo vehicular CONAVI*: CSV de ARESEP con datos de peajes Zurquí, Naranjo, Tres Ríos y Alajuela (2003-2025)
- *Flujo vehicular Ruta 27*: XLSX de ARESEP con datos de la ruta San José-Caldera (2009-2025)
- *Calidad del aire*: API Open-Meteo Air Quality — PM2.5, PM10, NO2, CO, Ozono (desde 2013)
- *Clima histórico*: API Open-Meteo Archive — temperatura, humedad, velocidad del viento (desde 2003)
- *Base de datos*: SQL Server con tablas de flujo vehicular, calidad del aire, clima y predicciones

## Tecnologías utilizadas
- *Lenguaje*: Python 3
- *Manipulación de datos*: Pandas, NumPy
- *Machine Learning*: Scikit-learn (Random Forest, KNN, Árbol de Decisión, Regresión Logística)
- *Visualización*: Matplotlib, Seaborn, Folium (mapas interactivos)
- *Base de datos*: SQL Server con pyodbc
- *Dashboard*: Streamlit
- *Control de versiones*: Git, GitHub, SourceTree
- *Entorno*: Anaconda, Jupyter Notebook, PyCharm


## Arquitectura del proyecto (POO)
bash
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
└── exploracion_inicial.ipynb

## Modelos de Machine Learning

- *Tipo*: Clasificación supervisada  
- *Variable objetivo*: Categoría ICA (Buena / Moderada / Mala / Muy Mala)  

### Variables de entrada

Se utilizan variables relacionadas con la movilidad, la calidad del aire y las condiciones climáticas:

- Flujo vehicular total  
- Dióxido de nitrógeno (NO2)  
- Ozono  
- Temperatura  
- Humedad relativa  
- Velocidad del viento  

> Nota: La variable PM2.5 no se utiliza como variable de entrada, ya que a partir de ella se construye la variable objetivo (categoría ICA).

---

### Algoritmos utilizados

Se entrenaron y compararon distintos modelos de Machine Learning:

- Random Forest  
- K-Nearest Neighbors (KNN)  
- Árbol de Decisión  
- Regresión Logística  

---

### Evaluación del modelo

El desempeño de los modelos se evaluó utilizando:

- *Accuracy* como métrica principal  

Todos los modelos alcanzaron un accuracy del 100%, sin embargo, este resultado debe interpretarse con cautela.

---

### Limitaciones del modelo

El dataset presenta un desbalance significativo en la variable objetivo, ya que la mayoría de los registros pertenecen a la categoría "Buena" y muy pocos a otras categorías.

Esto provoca que los modelos obtengan altos valores de accuracy, pero estos resultados no reflejan necesariamente una alta capacidad predictiva en escenarios reales.

Además, debido a la baja cantidad de registros en algunas clases, no fue posible aplicar correctamente técnicas como:

- Validación cruzada (Cross-Validation)  
- Optimización de hiperparámetros (GridSearchCV)  

---

### Mejora aplicada al modelo

Inicialmente se utilizó la variable PM2.5 como variable de entrada. Sin embargo, se identificó que esto generaba fuga de información, ya que la variable objetivo (categoría ICA) se construye a partir de PM2.5.

Por esta razón, se corrigió el modelo eliminando PM2.5 de las variables de entrada, utilizando únicamente variables independientes como el flujo vehicular, variables climáticas y otros contaminantes.

---

### Conclusión

El modelo implementado sigue el pipeline completo de Machine Learning, incluyendo carga de datos, entrenamiento, evaluación y predicción.

Aunque los resultados obtenidos son altos, estos deben interpretarse considerando las limitaciones del dataset. Como mejora futura, se recomienda aumentar la cantidad de datos y balancear las clases para obtener un modelo más robusto.


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