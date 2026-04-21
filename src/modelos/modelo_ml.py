"""
Módulo de entrenamiento y evaluación de modelos de Machine Learning
para la predicción de calidad del aire en el GAM de Costa Rica.
Implementa clasificación supervisada con múltiples algoritmos.
"""

import numpy as np
import pandas as pd
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
            "SELECT anio, mes, SUM(total) AS total_vehiculos "
            "FROM flujo_vehicular "
            "GROUP BY anio, mes"
        )

        df_aire = self.gestor_bd.consultar(
            "SELECT anio, mes, pm2_5, nitrogen_dioxide, ozone "
            "FROM calidad_aire"
        )

        df_clima = self.gestor_bd.consultar(
            "SELECT anio, mes, temperature_2m, relative_humidity_2m, windspeed_10m "
            "FROM clima"
        )

        df = pd.merge(df_flujo, df_aire, on=["anio", "mes"], how="inner")
        df = pd.merge(df, df_clima, on=["anio", "mes"], how="inner")

        # Crear variable objetivo a partir de PM2.5
        df["categoria_ica"] = df["pm2_5"].apply(self._categorizar_ica)

        # Eliminar nulos si existieran
        df = df.dropna().copy()

        self.df = df

        print(f"[ModeloML] Dataset preparado: {df.shape[0]} filas, {df.shape[1]} columnas")
        print(f"[ModeloML] Distribución ICA:\n{df['categoria_ica'].value_counts()}")

        return df

    def _categorizar_ica(self, pm2_5: float) -> str:
        """
        Categoriza la calidad del aire según el valor de PM2.5.
        """
        if pm2_5 <= 12:
            return "Buena"
        elif pm2_5 <= 35:
            return "Moderada"
        elif pm2_5 <= 55:
            return "Mala"
        else:
            return "Muy Mala"

    def _obtener_features(self):
        """
        Devuelve las variables de entrada del modelo.
        pm2_5 NO se usa porque de ella se crea categoria_ica.
        """
        return [
            "total_vehiculos",
            "nitrogen_dioxide",
            "ozone",
            "temperature_2m",
            "relative_humidity_2m",
            "windspeed_10m",
        ]

    def dividir_datos(self, test_size: float = 0.2):
        """
        Divide el dataset en entrenamiento y prueba.

        Si hay clases con solo 1 registro, evita usar stratify
        y fuerza que esos registros raros queden en entrenamiento
        para que el modelo pueda aprender ambas clases.
        """
        features = self._obtener_features()

        X = self.df[features].copy()
        y = self.le.fit_transform(self.df["categoria_ica"])

        X_scaled = self.scaler.fit_transform(X)

        clases, conteos = np.unique(y, return_counts=True)
        min_clase = conteos.min()

        if min_clase >= 2:
            # Caso normal: se puede usar stratify
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y
            )
            print("[ModeloML] División con stratify aplicada correctamente.")
        else:
            # Caso especial: alguna clase tiene solo 1 registro
            print("[ModeloML] Advertencia: hay clases con muy pocos registros.")
            print("[ModeloML] Se usará una división especial sin stratify.")

            indices = np.arange(len(y))

            # Índices de clases raras (conteo == 1)
            indices_raros = []
            for clase, conteo in zip(clases, conteos):
                if conteo == 1:
                    idx_clase = indices[y == clase][0]
                    indices_raros.append(idx_clase)

            indices_raros = np.array(indices_raros, dtype=int)
            indices_restantes = np.array(
                [i for i in indices if i not in indices_raros],
                dtype=int
            )

            # Si no hay suficientes restantes para dividir, todo va a entrenamiento
            if len(indices_restantes) < 2:
                self.X_train = X_scaled
                self.y_train = y
                self.X_test = X_scaled
                self.y_test = y
                print("[ModeloML] Muy pocos datos para separar train/test. Se usarán todos los datos.")
            else:
                X_rest = X_scaled[indices_restantes]
                y_rest = y[indices_restantes]

                X_train_rest, X_test_rest, y_train_rest, y_test_rest = train_test_split(
                    X_rest,
                    y_rest,
                    test_size=test_size,
                    random_state=42
                )

                # Forzar clases raras al entrenamiento
                X_train_raros = X_scaled[indices_raros]
                y_train_raros = y[indices_raros]

                self.X_train = np.vstack([X_train_rest, X_train_raros])
                self.y_train = np.concatenate([y_train_rest, y_train_raros])
                self.X_test = X_test_rest
                self.y_test = y_test_rest

        print(f"[ModeloML] Entrenamiento: {len(self.X_train)} muestras | Prueba: {len(self.X_test)} muestras")
        print(f"[ModeloML] Clases en entrenamiento: {np.unique(self.y_train)}")
        print(f"[ModeloML] Clases en prueba: {np.unique(self.y_test)}")

    # ─────────────────────────────────────────
    # MODELOS
    # ─────────────────────────────────────────

    def _puede_entrenar_modelo(self):
        """
        Verifica si hay al menos 2 clases en entrenamiento.
        """
        clases_train = np.unique(self.y_train)
        if len(clases_train) < 2:
            print("[ModeloML] No se puede entrenar el modelo: entrenamiento con una sola clase.")
            return False
        return True

    def entrenar_random_forest(self):
        if not self._puede_entrenar_modelo():
            return None

        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(self.X_train, self.y_train)
        y_pred = modelo.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        self.resultados["Random Forest"] = accuracy

        print(f"\n{'=' * 50}")
        print(f" RANDOM FOREST — Accuracy: {accuracy:.2%}")
        print(f"{'=' * 50}")
        print(classification_report(
            self.y_test,
            y_pred,
            labels=np.unique(self.y_test),
            target_names=self.le.inverse_transform(np.unique(self.y_test)),
            zero_division=0
        ))

        return accuracy

    def entrenar_arbol_decision(self):
        if not self._puede_entrenar_modelo():
            return None

        modelo = DecisionTreeClassifier(random_state=42)
        modelo.fit(self.X_train, self.y_train)
        y_pred = modelo.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        self.resultados["Árbol de Decisión"] = accuracy

        print(f"\n{'=' * 50}")
        print(f" ÁRBOL DE DECISIÓN — Accuracy: {accuracy:.2%}")
        print(f"{'=' * 50}")
        print(classification_report(
            self.y_test,
            y_pred,
            labels=np.unique(self.y_test),
            target_names=self.le.inverse_transform(np.unique(self.y_test)),
            zero_division=0
        ))

        return accuracy

    def entrenar_knn(self):
        if not self._puede_entrenar_modelo():
            return None

        # Ajustar k para datasets pequeños
        k = min(3, len(self.X_train))
        if k < 1:
            print("[ModeloML] No hay suficientes datos para entrenar KNN.")
            return None

        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(self.X_train, self.y_train)
        y_pred = modelo.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        self.resultados["KNN"] = accuracy

        print(f"\n{'=' * 50}")
        print(f" KNN — Accuracy: {accuracy:.2%}")
        print(f"{'=' * 50}")
        print(classification_report(
            self.y_test,
            y_pred,
            labels=np.unique(self.y_test),
            target_names=self.le.inverse_transform(np.unique(self.y_test)),
            zero_division=0
        ))

        return accuracy

    def entrenar_regresion_logistica(self):
        if not self._puede_entrenar_modelo():
            return None

        modelo = LogisticRegression(max_iter=1000, random_state=42)
        modelo.fit(self.X_train, self.y_train)
        y_pred = modelo.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        self.resultados["Regresión Logística"] = accuracy

        print(f"\n{'=' * 50}")
        print(f" REGRESIÓN LOGÍSTICA — Accuracy: {accuracy:.2%}")
        print(f"{'=' * 50}")
        print(classification_report(
            self.y_test,
            y_pred,
            labels=np.unique(self.y_test),
            target_names=self.le.inverse_transform(np.unique(self.y_test)),
            zero_division=0
        ))

        return accuracy

    # ─────────────────────────────────────────
    # CROSS-VALIDACIÓN
    # ─────────────────────────────────────────

    def cross_validacion(self):
        """
        Evalúa los modelos usando Cross-Validación.
        Si alguna clase tiene menos de 2 registros, se omite.
        """
        features = self._obtener_features()

        X = self.scaler.fit_transform(self.df[features])
        y = self.le.fit_transform(self.df["categoria_ica"])

        clases, conteos = np.unique(y, return_counts=True)
        min_clase = conteos.min()

        if min_clase < 2:
            print(f"\n{'=' * 50}")
            print(" CROSS-VALIDACIÓN")
            print(f"{'=' * 50}")
            print("[ModeloML] No se puede aplicar cross-validation.")
            print("[ModeloML] Motivo: hay clases con menos de 2 registros.")
            return

        cv_folds = min(5, min_clase)

        modelos = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=min(3, len(X))),
            "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42),
        }

        print(f"\n{'=' * 50}")
        print(f" CROSS-VALIDACIÓN ({cv_folds} folds)")
        print(f"{'=' * 50}")

        for nombre, modelo in modelos.items():
            scores = cross_val_score(modelo, X, y, cv=cv_folds, scoring="accuracy")
            print(f"\n{nombre}:")
            print(f"  Accuracy por fold: {[f'{s:.2%}' for s in scores]}")
            print(f"  Promedio: {scores.mean():.2%} | Desviación: {scores.std():.2%}")

    # ─────────────────────────────────────────
    # GRID SEARCH
    # ─────────────────────────────────────────

    def optimizar_random_forest(self):
        """
        Optimiza hiperparámetros del Random Forest.
        Si alguna clase tiene menos de 2 registros, se omite.
        """
        features = self._obtener_features()

        X = self.scaler.fit_transform(self.df[features])
        y = self.le.fit_transform(self.df["categoria_ica"])

        clases, conteos = np.unique(y, return_counts=True)
        min_clase = conteos.min()

        if min_clase < 2:
            print(f"\n{'=' * 50}")
            print(" OPTIMIZACIÓN GRID SEARCH — Random Forest")
            print(f"{'=' * 50}")
            print("[ModeloML] No se puede aplicar GridSearchCV.")
            print("[ModeloML] Motivo: hay clases con menos de 2 registros.")
            return None

        cv_folds = min(5, min_clase)

        parametros = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
        }

        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            parametros,
            cv=cv_folds,
            scoring="accuracy"
        )

        grid.fit(X, y)

        print(f"\n{'=' * 50}")
        print(" OPTIMIZACIÓN GRID SEARCH — Random Forest")
        print(f"{'=' * 50}")
        print(f"Mejores parámetros: {grid.best_params_}")
        print(f"Mejor accuracy: {grid.best_score_:.2%}")

        return grid.best_params_

    # ─────────────────────────────────────────
    # PREDICCIÓN DE NUEVOS DATOS
    # ─────────────────────────────────────────

    def predecir_nuevo(
        self,
        total_vehiculos: int,
        nitrogen_dioxide: float,
        ozone: float,
        temperature_2m: float,
        relative_humidity_2m: float,
        windspeed_10m: float
    ) -> str:
        """
        Predice la categoría ICA para un nuevo registro.
        """
        features = self._obtener_features()

        X = self.scaler.fit_transform(self.df[features])
        y = self.le.fit_transform(self.df["categoria_ica"])

        if len(np.unique(y)) < 2:
            print("[ModeloML] No se puede predecir: solo hay una clase en el dataset.")
            return "No disponible"

        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X, y)

        nuevo = self.scaler.transform([[
            total_vehiculos,
            nitrogen_dioxide,
            ozone,
            temperature_2m,
            relative_humidity_2m,
            windspeed_10m
        ]])

        prediccion = modelo.predict(nuevo)
        categoria = self.le.inverse_transform(prediccion)[0]

        print(f"\n{'=' * 50}")
        print(" PREDICCIÓN PARA NUEVO REGISTRO")
        print(f"{'=' * 50}")
        print(f"  Total vehículos: {total_vehiculos:,}")
        print(f"  NO2: {nitrogen_dioxide}")
        print(f"  Ozono: {ozone}")
        print(f"  Temperatura: {temperature_2m} °C")
        print(f"  Humedad: {relative_humidity_2m}%")
        print(f"  Viento: {windspeed_10m} km/h")
        print(f"\n   Categoría ICA predicha: {categoria}")

        return categoria

    # ─────────────────────────────────────────
    # COMPARACIÓN
    # ─────────────────────────────────────────

    def comparar_modelos(self):
        """
        Muestra una tabla comparativa de accuracy de los modelos entrenados.
        """
        print(f"\n{'=' * 50}")
        print(" COMPARACIÓN DE MODELOS")
        print(f"{'=' * 50}")

        if not self.resultados:
            print("[ModeloML] No hay resultados para comparar.")
            return None

        df_resultados = pd.DataFrame(
            list(self.resultados.items()),
            columns=["Modelo", "Accuracy"]
        )

        df_resultados = df_resultados.sort_values("Accuracy", ascending=False)
        df_resultados["Accuracy"] = df_resultados["Accuracy"].apply(lambda x: f"{x:.2%}")

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
        self.entrenar_regresion_logistica()

        self.comparar_modelos()
        self.cross_validacion()
        self.optimizar_random_forest()

        self.predecir_nuevo(
            total_vehiculos=950000,
            nitrogen_dioxide=15.2,
            ozone=32.1,
            temperature_2m=20.5,
            relative_humidity_2m=78.0,
            windspeed_10m=7.5
        )

        print("\n ENTRENAMIENTO COMPLETADO\n")


if __name__ == "__main__":
    modelo = ModeloML()
    modelo.ejecutar_modelos()