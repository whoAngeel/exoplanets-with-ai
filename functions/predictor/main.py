import functions_framework
from flask import Request, jsonify
from google.cloud import firestore, storage
import pandas as pd
import pickle
import io

# --- INICIALIZACIÓN PEREZOSA (sin cambios) ---
firestore_client = None
storage_client = None

# --- MANEJO DE CORS (sin cambios) ---
CORS_HEADERS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
}

def _get_clients():
    global firestore_client, storage_client
    if firestore_client is None:
        firestore_client = firestore.Client()
    if storage_client is None:
        storage_client = storage.Client()
    return firestore_client, storage_client

def _load_artifacts_from_gcs(gcs_uri):
    # (Esta función no necesita cambios)
    print(f"Cargando artefactos desde: {gcs_uri}")
    _, storage_client = _get_clients()
    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
    blob = storage_client.bucket(bucket_name).blob(blob_name)
    artifacts = pickle.loads(blob.download_as_bytes())
    print("✓ Artefactos cargados exitosamente.")
    return artifacts

def _apply_pipeline(new_data_df, artifacts, data_source):
    """
    Aplica el pipeline de preprocesamiento y feature engineering a los nuevos datos.
    Ahora es retrocompatible con modelos antiguos.
    """
    print("Aplicando pipeline de preprocesamiento...")
    
    scaler = artifacts['scaler']
    imputer = artifacts['imputer']
    
    # --- CORRECCIÓN DE RETROCOMPATIBILIDAD ---
    try:
        # Intenta obtener la lista de features directamente (método nuevo y preferido)
        feature_names = artifacts['feature_names']
        print("INFO: 'feature_names' encontrado en los artefactos del modelo.")
    except KeyError:
        # Fallback para modelos antiguos: obtener las features desde el scaler
        print("WARN: 'feature_names' no se encontró. Infiriendo desde el objeto 'scaler'.")
        # Esta función devuelve los nombres de las columnas con los que se entrenó el scaler
        feature_names = scaler.get_feature_names_out()
    # -----------------------------------------

    X_new = new_data_df.copy()

    # --- Feature Engineering Real ---
    if data_source == 'kepler':
        if 'koi_prad' in X_new.columns and 'koi_srad' in X_new.columns:
            X_new['planet_star_ratio'] = X_new['koi_prad'] / (X_new['koi_srad'] * 109.1)
        if 'koi_prad' in X_new.columns and 'koi_period' in X_new.columns:
            X_new['density_proxy'] = X_new['koi_prad'] / (X_new['koi_period'] ** (1/3))
        flag_cols = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
        available_flags = [c for c in flag_cols if c in X_new.columns]
        if available_flags:
            X_new['total_fp_flags'] = X_new[available_flags].sum(axis=1)
            
    elif data_source == 'k2':
        if 'pl_rade' in X_new.columns and 'st_rad' in X_new.columns:
            X_new['planet_star_ratio'] = X_new['pl_rade'] / (X_new['st_rad'] * 109.1)
        if 'pl_rade' in X_new.columns and 'pl_orbper' in X_new.columns:
            X_new['density_proxy'] = X_new['pl_rade'] / (X_new['pl_orbper'] ** (1/3))

    # Asegurarse de que el DataFrame final tenga exactamente las mismas columnas que el modelo espera
    X_reindexed = X_new.reindex(columns=feature_names, fill_value=0)

    # Aplicar imputer y scaler
    X_imputed = pd.DataFrame(imputer.transform(X_reindexed), columns=feature_names)
    X_scaled = pd.DataFrame(scaler.transform(X_imputed), columns=feature_names)
    
    print("✓ Pipeline aplicado.")
    return X_scaled

@functions_framework.http
def predictor_function(request: Request):
    """
    Función de Inferencia. Recibe un archivo CSV y un job_id, y devuelve las probabilidades.
    """
    if request.method == 'OPTIONS':
        return ('', 204, CORS_HEADERS)

    try:
        firestore_client, storage_client = _get_clients()
        
        # --- CAMBIO 1: Manejar entrada como 'multipart/form-data' ---
        if 'file' not in request.files or 'job_id' not in request.form:
            return (jsonify({"error": "Petición inválida. Se requiere un archivo 'file' y un campo 'job_id'."}), 400, CORS_HEADERS)
        
        job_id = request.form['job_id']
        file = request.files['file']
        
        # Leemos el CSV subido directamente en un DataFrame
        new_data_df = pd.read_csv(io.BytesIO(file.read()), comment='#', engine='python', delimiter=',')

        # 1. Buscar metadatos del modelo en Firestore
        doc_ref = firestore_client.collection("exo_scout_models").document(job_id)
        doc = doc_ref.get()
        if not doc.exists:
            return (jsonify({"error": f"El modelo con job_id '{job_id}' no fue encontrado."}), 404, CORS_HEADERS)
        
        metadata = doc.to_dict()
        gcs_uri = metadata.get("results", {}).get("gcs_artifacts_path")
        data_source = metadata.get("params", {}).get("data_source") # Obtenemos el data_source original
        
        if not gcs_uri or not data_source:
             return (jsonify({"error": "Metadatos incompletos para el modelo. Falta la ruta o la fuente de datos."}), 500, CORS_HEADERS)

        # 2. Cargar los artefactos del modelo
        artifacts = _load_artifacts_from_gcs(gcs_uri)
        model = artifacts['model']
        label_encoder = artifacts['label_encoder']

        # 3. Preparar los nuevos datos aplicando el pipeline correcto
        X_prepared = _apply_pipeline(new_data_df, artifacts, data_source)

        # --- CAMBIO 2: Usar predict_proba para obtener probabilidades ---
        probabilities = model.predict_proba(X_prepared)
        class_names = label_encoder.classes_

        # 4. Formatear la respuesta para que sea fácil de usar en el frontend
        results = []
        for i, prediction_probs in enumerate(probabilities):
            # Crea un diccionario legible: {'CANDIDATE': 0.8, 'CONFIRMED': 0.1, ...}
            prob_dict = {class_names[j]: round(prob, 4) for j, prob in enumerate(prediction_probs)}
            results.append(prob_dict)
        
        return (jsonify({"job_id": job_id, "predictions": results}), 200, CORS_HEADERS)

    except Exception as e:
        print(f"Error en la predicción: {e}")
        return (jsonify({"error": "Ocurrió un error interno al procesar la predicción."}), 500, CORS_HEADERS)