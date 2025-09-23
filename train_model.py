import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# ----------------------
# Funciones
# ----------------------

def cargar_datos_csv(ruta_csv="data.csv", test_size=0.2, random_state=42):
    """
    Carga el dataset Breast Cancer desde CSV, preprocesa y divide en train/test.
    
    Args:
        ruta_csv (str): Ruta al archivo CSV.
        test_size (float): Proporción para el set de prueba.
        random_state (int): Semilla para reproducibilidad.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    df = pd.read_csv("data.csv")

    # ----------------------
    # Preprocesamiento
    # ----------------------
    # Eliminar duplicados
    df = df.drop_duplicates()

    # Eliminar columnas innecesarias
    cols_a_eliminar = []
    if "id" in df.columns:
        cols_a_eliminar.append("id")
    if "Unnamed: 32" in df.columns:
        cols_a_eliminar.append("Unnamed: 32")

    if cols_a_eliminar:
        df = df.drop(columns=cols_a_eliminar)

    # Separar target (diagnosis) y features
    target_col = "diagnosis"
    if df[target_col].dtype == object:
        # Convertir M/B a 1/0 si es categórico
        df[target_col] = df[target_col].map({"M": 1, "B": 0})

    # Eliminar filas con NaN
    df = df.dropna()

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def entrenar_modelo(X_train, y_train, n_iter=5, cv=3, random_state=42):
    """
    Entrena un RandomForestClassifier con búsqueda aleatoria de hiperparámetros.
    """
    rf = RandomForestClassifier(random_state=random_state)
    param_dist = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, None],
        "min_samples_split": [2, 5, 10]
    }

    busqueda = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="accuracy",
        random_state=random_state,
        n_jobs=-1
    )

    busqueda.fit(X_train, y_train)
    return busqueda.best_estimator_, busqueda.best_params_


def evaluar_modelo(modelo, X_test, y_test):
    """
    Evalúa el modelo en el conjunto de prueba.
    """
    return modelo.score(X_test, y_test)


def guardar_modelo(modelo, nombre_archivo="modelo.pkl"):
    """
    Guarda el modelo entrenado en un archivo usando joblib.
    """
    joblib.dump(modelo, nombre_archivo)
    print(f"\nModelo guardado en {nombre_archivo}")


# ----------------------
# Ejecución principal
# ----------------------
if __name__ == "__main__":
    # Configurar MLflow
    mlflow.set_experiment("breast-cancer-experiment")

    # Cargar datos CSV
    X_train, X_test, y_train, y_test = cargar_datos_csv("data.csv")

    with mlflow.start_run():
        # Entrenamiento
        modelo, mejores_parametros = entrenar_modelo(X_train, y_train)

        # Registrar hiperparámetros
        mlflow.log_params(mejores_parametros)

        # Evaluación
        accuracy = evaluar_modelo(modelo, X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        print("Mejores hiperparámetros:", mejores_parametros)
        print("Accuracy en test:", accuracy)

        # Guardar modelo en MLflow
        mlflow.sklearn.log_model(modelo, name="modelo")

        # Guardar también localmente
        guardar_modelo(modelo)