import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def train_model(df: pd.DataFrame, target_column: str, algorithm: str):
    """
    Entrena un modelo de machine learning.

    Args:
        df (pd.DataFrame): El DataFrame preprocesado.
        target_column (str): El nombre de la columna objetivo (ej. 'koi_disposition').
        algorithm (str): El algoritmo a usar ('random_forest' o 'xgboost').

    Returns:
        El objeto del modelo entrenado y su precisión (accuracy).
    """
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no se encuentra en el DataFrame.")

    # 1. Separar características (X) y etiqueta (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Seleccionar el modelo basado en el parámetro
    print(f"INFO: Seleccionando el algoritmo: {algorithm}")
    if algorithm == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif algorithm == 'xgboost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        raise ValueError(f"Algoritmo '{algorithm}' no soportado.")

    # 3. Entrenar el modelo
    print("INFO: Iniciando entrenamiento del modelo...")
    model.fit(X_train, y_train)
    print("INFO: Entrenamiento completado.")

    # 4. Evaluar y devolver resultados
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"INFO: Precisión del modelo en set de prueba: {accuracy:.4f}")

    return model, accuracy