import pandas as pd

def preprocess_kepler_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simula el preprocesamiento para un dataset de Kepler.
    """
    print("INFO: Ejecutando el preprocesador para datos de KEPLER.")
    # Aquí iría la lógica real: limpiar, normalizar, etc.
    print(f"INFO: DataFrame recibido con {df.shape[0]} filas y {df.shape[1]} columnas.")
    # Por ahora, solo lo devolvemos como está.
    return df

def preprocess_tess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simula el preprocesamiento para un dataset de TESS.
    """
    print("INFO: Ejecutando el preprocesador para datos de TESS.")
    print(f"INFO: DataFrame recibido con {df.shape[0]} filas y {df.shape[1]} columnas.")
    return df

# Diccionario para mapear la fuente de datos a la función correcta.
# Esta es una práctica recomendada para evitar usar if/else anidados.
PROCESSORS = {
    "kepler": preprocess_kepler_data,
    "tess": preprocess_tess_data,
}