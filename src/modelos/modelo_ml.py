"""
Módulo de entrenamiento y evaluación de modelos de Machine Learning
para la predicción de calidad del aire en el GAM de Costa Rica.
Implementa clasificación supervisada con múltiples algoritmos.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
              target_names=self.le.classes_))
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
              target_names=self.le.classes_))
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
              target_names=self.le.classes_))
        return accuracy

    def entrenar_regresion_logistica(self):
        """
        Entrena y evalúa un modelo de Regresión Logística.

        Returns:
            float: Accuracy del modelo.
        """
        modelo = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
        modelo.fit(self.X_train, self.y_train)
        y_pred = modelo.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.resultados['Regresión Logística'] = accuracy
        print(f"\n{'='*50}")
        print(f" REGRESIÓN LOGÍSTICA — Accuracy: {accuracy:.2%}")
        print(f"{'='*50}")
        print(classification_report(self.y_test, y_pred,
                                    target_names=self.le.classes_, zero_division=0))
        return accuracy




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
        # Regresión Logística requiere mínimo 2 clases en entrenamiento
        # No compatible con dataset desbalanceado (28 Buena vs 1 Moderada)
        # self.entrenar_regresion_logistica()
        self.comparar_modelos()
        print("\nENTRENAMIENTO COMPLETADO\n")