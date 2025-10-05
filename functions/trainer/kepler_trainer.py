import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
import pickle
from datetime import datetime

# --- Importaciones de Google Cloud ---
from google.cloud import storage, firestore

# =============================================================================
# CLASE DE CONFIGURACIÓN (Tu código original, sin cambios)
# =============================================================================
class ModelConfig:
    test_size = 0.2
    random_state = 42
    imputation_strategy = 'median'
    rf_n_estimators = 200
    rf_max_depth = 20
    rf_class_weight = 'balanced'
    gb_n_estimators = 200
    gb_max_depth = 5
    gb_learning_rate = 0.1
    xgb_n_estimators = 200
    xgb_max_depth = 6
    xgb_learning_rate = 0.1
    cv_splits = 5
    top_features_to_show = 20
    # Los modelos a entrenar se decidirán por el parámetro 'algorithm'

# =============================================================================
# FUNCIONES DEL PIPELINE (Tu código original, adaptado)
# =============================================================================

# La función load_data no es necesaria aquí, ya que el DataFrame se pasará directamente.

def select_features(df):
    """Selecciona las features más importantes"""
    print("PASO 1: SELECCIÓN DE FEATURES")
    feature_groups = {
        'flags': ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_score'],
        'planeta': ['koi_period', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_sma', 'koi_eccen', 'koi_incl'],
        'transito': ['koi_duration', 'koi_depth', 'koi_ror', 'koi_impact', 'koi_model_snr'],
        'estrella': ['koi_steff', 'koi_slogg', 'koi_srad', 'koi_smass', 'koi_smet'],
        'calidad': ['koi_count', 'koi_num_transits']
    }
    selected_features = [f for group in feature_groups.values() for f in group]
    available_features = [f for f in selected_features if f in df.columns]
    X = df[available_features].copy()
    y = df['koi_disposition'].copy()
    print(f"✓ Usando {X.shape[1]} features disponibles.")
    return X, y

def engineer_features(X):
    """Crea features derivadas"""
    print("PASO 2: FEATURE ENGINEERING")
    X_eng = X.copy()
    if 'koi_prad' in X.columns and 'koi_srad' in X.columns:
        X_eng['planet_star_ratio'] = X_eng['koi_prad'] / (X_eng['koi_srad'] * 109.1)
    if 'koi_prad' in X.columns and 'koi_period' in X.columns:
        X_eng['density_proxy'] = X_eng['koi_prad'] / (X_eng['koi_period'] ** (1/3))
    flag_cols = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
    available_flags = [c for c in flag_cols if c in X.columns]
    if available_flags:
        X_eng['total_fp_flags'] = X_eng[available_flags].sum(axis=1)
    print(f"✓ Features de ingeniería creadas.")
    return X_eng

def preprocess_data(X, y, config):
    """Preprocesa los datos"""
    print("PASO 3: PREPROCESAMIENTO")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    imputer = SimpleImputer(strategy=config.imputation_strategy)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns, index=X.index)
    print("✓ Preprocesamiento completo.")
    return X_scaled, y_encoded, le, imputer, scaler

def train_and_evaluate(X_train, y_train, X_test, y_test, algorithm, config):
    """Entrena un modelo específico y devuelve el modelo y sus métricas."""
    print(f"PASO 4: ENTRENANDO MODELO: {algorithm}")
    
    models = {
        'random_forest': RandomForestClassifier(n_estimators=config.rf_n_estimators, max_depth=config.rf_max_depth, class_weight=config.rf_class_weight, random_state=config.random_state, n_jobs=-1),
        'xgboost': xgb.XGBClassifier(n_estimators=config.xgb_n_estimators, max_depth=config.xgb_max_depth, learning_rate=config.xgb_learning_rate, random_state=config.random_state, eval_metric='mlogloss'),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=config.gb_n_estimators, max_depth=config.gb_max_depth, learning_rate=config.gb_learning_rate, random_state=config.random_state)
    }
    
    model = models.get(algorithm)
    if model is None:
        raise ValueError(f"Algoritmo '{algorithm}' no está definido en el pipeline de entrenamiento.")

    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"✓ Entrenamiento completo. F1-Score: {f1:.4f}")
    return model, f1, report

# =============================================================================
# FUNCIÓN DE PIPELINE ADAPTADA
# =============================================================================

def run_training_pipeline(df, algorithm, config):
    """
    Ejecuta el pipeline completo desde un DataFrame hasta el modelo entrenado.
    """
    X, y = select_features(df)
    X = engineer_features(X)
    X_processed, y_encoded, le, imputer, scaler = preprocess_data(X, y, config)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y_encoded
    )
    
    model, f1, report = train_and_evaluate(X_train, y_train, X_test, y_test, algorithm, config)
    
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X_processed.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(config.top_features_to_show).to_dict('records')

    # Empaquetar todos los artefactos para guardarlos
    artifacts = {
        'model': model,
        'scaler': scaler,
        'imputer': imputer,
        'label_encoder': le,
        'feature_names': X_processed.columns.tolist(),
    }
    
    # Empaquetar los metadatos para guardarlos
    metadata = {
        'f1_score': f1,
        'classification_report': report,
        'feature_importance': feature_importance,
        'model_config': config.__dict__
    }
    
    return artifacts, metadata

# =============================================================================
# NUEVA FUNCIÓN PRINCIPAL PARA LA NUBE
# =============================================================================

def train_and_save_model(job_id, df, algorithm, bucket_name):
    """
    Función orquestadora que ejecuta el pipeline y guarda los resultados en GCP.
    """
    print("🚀 INICIANDO PIPELINE DE ENTRENAMIENTO EN LA NUBE 🚀")
    
    # Inicializar clientes de GCP
    storage_client = storage.Client()
    firestore_client = firestore.Client()
    
    # Configurar y ejecutar el pipeline
    config = ModelConfig()
    artifacts, metadata = run_training_pipeline(df, algorithm, config)
    
    # --- Guardar artefactos en Cloud Storage ---
    print(f"💾 Guardando artefactos del modelo en Cloud Storage...")
    model_filename = f"models/{job_id}_pipeline_artifacts.pkl"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(model_filename)
    
    # Serializar (convertir a bytes) los artefactos y subirlos
    with blob.open("wb") as f:
        pickle.dump(artifacts, f)
        
    gcs_uri = f"gs://{bucket_name}/{blob.name}"
    print(f"✓ Artefactos guardados en: {gcs_uri}")

    # --- Guardar metadatos en Firestore ---
    print(f"📝 Guardando metadatos en Firestore...")
    doc_ref = firestore_client.collection("exo_scout_models").document(job_id)
    
    final_metadata_update = {
        "status": "completed",
        "completed_at": datetime.now(),
        "results": {
            "gcs_artifacts_path": gcs_uri,
            **metadata  # Desempaqueta el diccionario de metadatos aquí
        }
    }
    
    doc_ref.update(final_metadata_update)
    print(f"✓ Metadatos actualizados para el job: {job_id}")
    
    return final_metadata_update