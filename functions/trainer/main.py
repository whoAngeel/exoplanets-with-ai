import functions_framework
from flask import Request, jsonify
import pandas as pd
import pickle
import os
import io

from datetime import datetime

from google.cloud import storage, firestore

from processors import PROCESSORS
from trainer import train_model

# --- Inicialización de Clientes de GCP (sin cambios) ---
try:
    storage_client = storage.Client()
    firestore_client = firestore.Client()
    MODEL_BUCKET_NAME = os.environ.get("MODEL_BUCKET_NAME")
except Exception as e:
    print(f"Error al inicializar clientes de GCP: {e}")
    # ... (código de manejo de error igual)

@functions_framework.http
def trainer_function(request: Request):
    """
    Cloud Function #2: El Entrenador Dedicado.
    Implementa un registro de estado en Firestore.
    """
    # --- ¡NUEVO! Paso 1: Obtener Job ID y validar ---
    request_json = request.get_json(silent=True) or {}
    job_id = request_json.get("job_id")
    if not job_id:
        return ("Error: El parámetro 'job_id' es requerido.", 400)

    # Referencia al documento de Firestore que usaremos durante todo el proceso
    doc_ref = firestore_client.collection("exo_scout_models").document(job_id)
    
    try:
        # --- ¡NUEVO! Paso 2: Registrar inicio en Firestore ---
        initial_metadata = {
            "job_id": job_id,
            "status": "training",
            "created_at": datetime.now(),
            "params": { # Guardamos los parámetros para referencia
                "data_source": request_json.get("data_source", "kepler"),
                "algorithm": request_json.get("algorithm", "random_forest")
            }
        }
        doc_ref.set(initial_metadata) # .set() crea el documento

        gcs_input_uri = request_json.get("gcs_input_uri")
        if not gcs_input_uri:
            raise ValueError("El parámetro 'gcs_input_uri' es requerido.")

        # Parsea el URI para obtener el nombre del bucket y del archivo
        bucket_name = gcs_input_uri.split("/")[2]
        file_name = "/".join(gcs_input_uri.split("/")[3:])
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        
        # Descarga como bytes y lee con Pandas
        csv_data = blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(csv_data))
        
        print(f"INFO: Leído el archivo {gcs_input_uri} exitosamente. Shape: {df.shape}")
        
        # --- El resto de la lógica continúa desde aquí ---
        data_source = initial_metadata["params"]["data_source"]
        algorithm = initial_metadata["params"]["algorithm"]
        
        processed_df = PROCESSORS[data_source](df)
        
        trained_model, model_accuracy = train_model(
            df=processed_df,
            target_column='koi_disposition',
            algorithm=algorithm
        )

        # --- Paso 4: Guardar artefacto en Cloud Storage ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{job_id}_{algorithm}_{timestamp}.pkl"
        bucket = storage_client.bucket(MODEL_BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_string(pickle.dumps(trained_model))
        gcs_uri = f"gs://{MODEL_BUCKET_NAME}/{blob.name}"

        # --- ¡NUEVO! Paso 5: Actualizar Firestore con el resultado exitoso ---
        success_metadata = {
            "status": "completed",
            "completed_at": datetime.now(),
            "results": {
                "gcs_path": gcs_uri,
                "accuracy": model_accuracy
            }
        }
        doc_ref.update(success_metadata) # .update() añade o modifica campos
        
        return jsonify(success_metadata), 200

    except Exception as e:
        # --- ¡NUEVO! Paso 6: Manejo de errores ---
        print(f"ERROR en job {job_id}: {e}")
        error_payload = {
            "status": "error",
            "error_message": str(e),
            "failed_at": datetime.now()
        }
        # Actualizamos el documento en Firestore para registrar el fallo
        doc_ref.update(error_payload)
        
        # Respondemos con un error HTTP 500
        return (f"Ocurrió un error en el job {job_id}. Revisa Firestore para más detalles.", 500)