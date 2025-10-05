# main.py

import functions_framework
from flask import Request, jsonify
import pandas as pd
import io
from google.cloud import storage, firestore
from datetime import datetime

# Importar el pipeline específico que necesitamos
from pipelines.kepler_pipeline import KeplerTrainingPipeline
from pipelines.k2_pipeline import K2TrainingPipeline 

from common import gcp_utils

@functions_framework.http
def trainer_function(request: Request):
    """
    Cloud Function Orquestadora.
    Elige y ejecuta el pipeline de entrenamiento apropiado.
    """
    # ... (El código de parseo de la solicitud es el mismo) ...
    request_json = request.get_json(silent=True) or {}
    job_id = request_json.get("job_id")
    gcs_input_uri = request.json.get("gcs_input_uri")
    data_source = request.json.get("data_source")
    algorithm = request.json.get("algorithm")
    model_name = request_json.get("model_name", f"model_{job_id[:8]}") 

    
    if not all([job_id, gcs_input_uri, data_source, algorithm]):
        return ("Error: Faltan parámetros.", 400)

    MODEL_BUCKET_NAME = "exoplanets-nasa-models"
    doc_ref = firestore.Client().collection("exo_scout_models").document(job_id)
    
    try:
        # Registrar inicio del job
        initial_metadata = {
            "job_id": job_id,
            "model_name": model_name,
            "status": "training",
            "created_at": datetime.now(),
            "params": {
                "data_source": data_source,
                "algorithm": algorithm,
                "gcs_input_uri": gcs_input_uri
            }
        }
        doc_ref.set(initial_metadata)
        print(f"INFO: Job {job_id} ({model_name}) registrado en Firestore con estado 'training'.")

    except Exception as e:
        print(f"ERROR CRÍTICO: No se pudo registrar el job {job_id}. Error: {e}")
        return ("Error interno al iniciar el job.", 500)
    
    try:
        # Descargar datos
        storage_client = storage.Client()
        bucket_name, file_name = gcs_input_uri.replace("gs://", "").split("/", 1)
        blob = storage_client.bucket(bucket_name).blob(file_name)
        df = pd.read_csv(io.BytesIO(blob.download_as_bytes()), comment='#')
        
        # --- ORQUESTACIÓN ---
        # Elige el pipeline correcto basado en la fuente de datos
        if data_source == 'kepler':
            pipeline = KeplerTrainingPipeline(df=df, algorithm=algorithm)
        elif data_source == 'k2': # <-- ¡NUEVO!
            pipeline = K2TrainingPipeline(df=df, algorithm=algorithm)
        else:
            raise NotImplementedError(f"El pipeline para '{data_source}' no está implementado.")
        
        # Ejecutar el pipeline
        artifacts, metadata = pipeline.run()
        
        # Guardar resultados
        gcs_uri = gcp_utils.save_artifacts_to_gcs(MODEL_BUCKET_NAME, job_id, artifacts)
        final_results = gcp_utils.update_firestore_metadata(job_id, gcs_uri, metadata)

        return jsonify(final_results), 200

    except Exception as e:
        print(f"ERROR CRÍTICO en el job {job_id}: {e}")
        error_payload = {"status": "error", "error_message": str(e), "failed_at": datetime.now()}
        doc_ref.update(error_payload)
        return (f"Ocurrió un error en el job {job_id}. Revisa Firestore.", 500)