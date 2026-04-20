"""
Módulo de entrenamiento y evaluación de modelos de Machine Learning
para la predicción de calidad del aire en el GAM de Costa Rica.
Implementa clasificación supervisada con múltiples algoritmos.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from basedatos.gestor_base_datos import GestorBaseDatos


class ModeloML:
    """
    Clase encargada del entrenamiento y evaluación de modelos
    de Machine Learning supervisado para clasificación de
    calidad del aire según el Índice ICA.
    """

    def __init__(self):
        """Inicializa el modelo conectando a la base de datos."""
        self.gestor_bd = GestorBaseDatos()
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.resultados = {}

    # ─────────────────────────────────────────
    # PREPARACIÓN DE DATOS
    # ─────────────────────────────────────────

    def cargar_y_preparar_datos(self):
        """
        Carga y une los datos de flujo vehicular, calidad del aire
        y clima desde SQL Server para preparar el dataset de entrenamiento.
        """
        df_flujo = self.gestor_bd.consultar(
            "SELECT anio, mes, SUM(total) as total_vehiculos FROM flujo_vehicular GROUP BY anio, mes"
        )
        df_aire = self.gestor_bd.consultar(
            "SELECT anio, mes, pm2_5, nitrogen_dioxide, ozone FROM calidad_aire"
        )
        df_clima = self.gestor_bd.consultar(
            "SELECT anio, mes, temperature_2m, relative_humidity_2m, windspeed_10m FROM clima"
        )

        df = pd.merge(df_flujo, df_aire, on=['anio', 'mes'], how='inner')
        df = pd.merge(df, df_clima, on=['anio', 'mes'], how='inner')
        df['categoria_ica'] = df['pm2_5'].apply(self._categorizar_ica)

        self.df = df
        print(f"[ModeloML] Dataset preparado: {df.shape[0]} filas, {df.shape[1]} columnas")
        print(f"[ModeloML] Distribución ICA:\n{df['categoria_ica'].value_counts()}")
        return df

    def _categorizar_ica(self, pm2_5: float) -> str:
        """
        Categoriza la calidad del aire según el valor de PM2.5.

        Args:
            pm2_5 (float): Concentración de PM2.5 en μg/m³.

        Returns:
            str: Categoría ICA (Buena/Moderada/Mala/Muy Mala).
        """
        if pm2_5 <= 12:
            return 'Buena'
        elif pm2_5 <= 35:
            return 'Moderada'
        elif pm2_5 <= 55:
            return 'Mala'
        else:
            return 'Muy Mala'

    def dividir_datos(self, test_size: float = 0.2):
        """
        Divide el dataset en conjuntos de entrenamiento y prueba.

        Args:
            test_size (float): Proporción de datos para prueba (default 0.2).
        """
        features = ['total_vehiculos', 'pm2_5', 'nitrogen_dioxide',
                    'ozone', 'temperature_2m', 'relative_humidity_2m', 'windspeed_10m']

        X = self.df[features]
        y = self.le.fit_transform(self.df['categoria_ica'])
        X_scaled = self.scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        print(f"[ModeloML] Entrenamiento: {len(self.X_train)} muestras | Prueba: {len(self.X_test)} muestras")

    # ─────────────────────────────────────────
    # MODELOS
    # ─────────────────────────────────────────

    def entrenar_random_forest(self):
        """
        Entrena y evalúa un modelo Random Forest.

        Returns:
            float: Accuracy del modelo.
        """
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(self.X_train, self.y_train)
        y_pred = modelo.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.resultados['Random Forest'] = accuracy
        print(f"\n{'='*50}")
        print(f" RANDOM FOREST — Accuracy: {accuracy:.2%}")
        print(f"{'='*50}")
        print(classification_report(self.y_test, y_pred,
              target_names=self.le.classes_, zero_division=0))
        return accuracy

    def entrenar_arbol_decision(self):
        """
        Entrena y evalúa un modelo de Árbol de Decisión.

        Returns:
            float: Accuracy del modelo.
        """
        modelo = DecisionTreeClassifier(random_state=42)
        modelo.fit(self.X_train, self.y_train)
        y_pred = modelo.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.resultados['Árbol de Decisión'] = accuracy
        print(f"\n{'='*50}")
        print(f" ÁRBOL DE DECISIÓN — Accuracy: {accuracy:.2%}")
        print(f"{'='*50}")
        print(classification_report(self.y_test, y_pred,
              target_names=self.le.classes_, zero_division=0))
        return accuracy

    def entrenar_knn(self):
        """
        Entrena y evalúa un modelo KNN.

        Returns:
            float: Accuracy del modelo.
        """
        modelo = KNeighborsClassifier(n_neighbors=3)
        modelo.fit(self.X_train, self.y_train)
        y_pred = modelo.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.resultados['KNN'] = accuracy
        print(f"\n{'='*50}")
        print(f" KNN — Accuracy: {accuracy:.2%}")
        print(f"{'='*50}")
        print(classification_report(self.y_test, y_pred,
              target_names=self.le.classes_, zero_division=0))
        return accuracy

    # ─────────────────────────────────────────
    # CROSS-VALIDACIÓN
    # ─────────────────────────────────────────

    def cross_validacion(self) -> None:
        """
        Evalúa los modelos usando Cross-Validación con 5 folds.
        Proporciona una evaluación más robusta del rendimiento real.
        """
        X = self.scaler.fit_transform(
            self.df[['total_vehiculos', 'pm2_5', 'nitrogen_dioxide',
                     'ozone', 'temperature_2m', 'relative_humidity_2m', 'windspeed_10m']]
        )
        y = self.le.fit_transform(self.df['categoria_ica'])

        modelos = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Árbol de Decisión': DecisionTreeClassifier(random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=3)
        }

        print(f"\n{'='*50}")
        print(" CROSS-VALIDACIÓN (5 Folds)")
        print(f"{'='*50}")

        for nombre, modelo in modelos.items():
            scores = cross_val_score(modelo, X, y, cv=5, scoring='accuracy')
            print(f"\n{nombre}:")
            print(f"  Accuracy por fold: {[f'{s:.2%}' for s in scores]}")
            print(f"  Promedio: {scores.mean():.2%} | Desviación: {scores.std():.2%}")

    # ─────────────────────────────────────────
    # GRID SEARCH
    # ─────────────────────────────────────────

    def optimizar_random_forest(self) -> dict:
        """
        Optimiza los hiperparámetros del Random Forest usando GridSearchCV.

        Returns:
            dict: Mejores parámetros encontrados.
        """
        X = self.scaler.fit_transform(
            self.df[['total_vehiculos', 'pm2_5', 'nitrogen_dioxide',
                     'ozone', 'temperature_2m', 'relative_humidity_2m', 'windspeed_10m']]
        )
        y = self.le.fit_transform(self.df['categoria_ica'])

        parametros = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }

        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            parametros,
            cv=5,
            scoring='accuracy'
        )
        grid.fit(X, y)

        print(f"\n{'='*50}")
        print(" OPTIMIZACIÓN GRID SEARCH — Random Forest")
        print(f"{'='*50}")
        print(f"Mejores parámetros: {grid.best_params_}")
        print(f"Mejor accuracy: {grid.best_score_:.2%}")

        return grid.best_params_

    # ─────────────────────────────────────────
    # PREDICCIÓN DE NUEVOS DATOS
    # ─────────────────────────────────────────

    def predecir_nuevo(self, total_vehiculos: int, pm2_5: float,
                       nitrogen_dioxide: float, ozone: float,
                       temperature_2m: float, relative_humidity_2m: float,
                       windspeed_10m: float) -> str:
        """
        Predice la categoría ICA para un nuevo registro de datos.

        Args:
            total_vehiculos (int): Total de vehículos en circulación.
            pm2_5 (float): Concentración de PM2.5 en μg/m³.
            nitrogen_dioxide (float): Concentración de NO2.
            ozone (float): Concentración de Ozono.
            temperature_2m (float): Temperatura en °C.
            relative_humidity_2m (float): Humedad relativa en %.
            windspeed_10m (float): Velocidad del viento en km/h.

        Returns:
            str: Categoría ICA predicha.
        """
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        X = self.scaler.fit_transform(
            self.df[['total_vehiculos', 'pm2_5', 'nitrogen_dioxide',
                     'ozone', 'temperature_2m', 'relative_humidity_2m', 'windspeed_10m']]
        )
        y = self.le.fit_transform(self.df['categoria_ica'])
        modelo.fit(X, y)

        nuevo = self.scaler.transform([[
            total_vehiculos, pm2_5, nitrogen_dioxide,
            ozone, temperature_2m, relative_humidity_2m, windspeed_10m
        ]])
        prediccion = modelo.predict(nuevo)
        categoria = self.le.inverse_transform(prediccion)[0]

        print(f"\n{'='*50}")
        print(" PREDICCIÓN PARA NUEVO REGISTRO")
        print(f"{'='*50}")
        print(f"  Total vehículos: {total_vehiculos:,}")
        print(f"  PM2.5: {pm2_5} μg/m³")
        print(f"  NO2: {nitrogen_dioxide}")
        print(f"  Ozono: {ozone}")
        print(f"  Temperatura: {temperature_2m}°C")
        print(f"  Humedad: {relative_humidity_2m}%")
        print(f"  Viento: {windspeed_10m} km/h")
        print(f"\n   Categoría ICA predicha: {categoria}")
        return categoria

    # ─────────────────────────────────────────
    # COMPARACIÓN
    # ─────────────────────────────────────────

    def comparar_modelos(self):
        """
        Muestra una tabla comparativa de accuracy de todos los modelos.
        """
        print(f"\n{'='*50}")
        print(" COMPARACIÓN DE MODELOS")
        print(f"{'='*50}")
        df_resultados = pd.DataFrame(
            list(self.resultados.items()),
            columns=['Modelo', 'Accuracy']
        )
        df_resultados = df_resultados.sort_values('Accuracy', ascending=False)
        df_resultados['Accuracy'] = df_resultados['Accuracy'].apply(lambda x: f"{x:.2%}")
        print(df_resultados.to_string(index=False))
        return df_resultados

    # ─────────────────────────────────────────
    # EJECUTAR TODO
    # ─────────────────────────────────────────

    def ejecutar_modelos(self):
        """
        Ejecuta el pipeline completo de Machine Learning.
        """
        print("\n INICIANDO ENTRENAMIENTO DE MODELOS ML\n")
        self.cargar_y_preparar_datos()
        self.dividir_datos()
        self.entrenar_random_forest()
        self.entrenar_arbol_decision()
        self.entrenar_knn()
        # Regresión Logística excluida por dataset desbalanceado
        # self.entrenar_regresion_logistica()
        self.comparar_modelos()
        self.cross_validacion()
        self.optimizar_random_forest()
        self.predecir_nuevo(
            total_vehiculos=950000,
            pm2_5=8.5,
            nitrogen_dioxide=15.2,
            ozone=32.1,
            temperature_2m=20.5,
            relative_humidity_2m=78.0,
            windspeed_10m=7.5
        )
        print("\n ENTRENAMIENTO COMPLETADO\n")