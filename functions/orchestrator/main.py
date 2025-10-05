import functions_framework
from flask import Request, jsonify
import os
import uuid
import json
import google.generativeai as genai
from google.cloud import storage, tasks_v2

# --- Inicialización de Clientes de GCP ---
try:
    storage_client = storage.Client()
    tasks_client = tasks_v2.CloudTasksClient()
    
    # Configura Gemini (requiere una variable de entorno con tu API Key o configuración ADC)
    # Para el Space Apps Challenge, puedes obtener una API key gratuita.
    # genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    # O, si usas Vertex AI, la autenticación es automática en GCP.
    genai.configure(project=os.environ.get("GCP_PROJECT"))
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')

except Exception as e:
    print(f"Error al inicializar clientes: {e}")

# --- Configuración (leer desde variables de entorno) ---
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
TASKS_QUEUE = os.environ.get("TASKS_QUEUE", "exo-scout-queue")
TRAINER_FUNCTION_URL = os.environ.get("TRAINER_FUNCTION_URL") # URL de la CF2 desplegada
UPLOAD_BUCKET_NAME = "exoplanets-nasa-models" # ¡Tu bucket!

def get_data_source_from_headers(headers: str) -> str:
    """Usa Gemini para identificar la fuente de datos a partir de las columnas del CSV."""
    prompt = f"""
    Eres un asistente de astronomía. Dada la siguiente lista de nombres de columnas de un archivo CSV de datos de exoplanetas, identifica si los datos provienen del telescopio 'kepler' o 'tess'. Responde únicamente con la palabra 'kepler' o 'tess'.

    Columnas: "{headers}"
    """
    try:
        response = gemini_model.generate_content(prompt)
        # Limpia la respuesta para asegurarnos que solo contenga la palabra clave
        data_source = response.text.strip().lower()
        if data_source not in ["kepler", "tess"]:
            return "unknown" # Fallback
        return data_source
    except Exception as e:
        print(f"Error al llamar a Gemini: {e}")
        return "unknown"

@functions_framework.http
def orchestrator_function(request: Request):
    """Cloud Function #1: El Orquestador Inteligente."""
    if 'file' not in request.files:
        return "Error: No se encontró el archivo en la solicitud.", 400

    file = request.files['file']
    params_str = request.form.get("params", "{}")
    params = json.loads(params_str)
    algorithm = params.get("algorithm", "random_forest")

    # 1. Generar un ID único para este trabajo
    job_id = str(uuid.uuid4())

    # 2. Subir el archivo original a Cloud Storage
    blob = storage_client.bucket(UPLOAD_BUCKET_NAME).blob(f"raw-uploads/{job_id}_{file.filename}")
    blob.upload_from_file(file)
    gcs_uri = f"gs://{UPLOAD_BUCKET_NAME}/{blob.name}"
    print(f"INFO: Archivo subido a {gcs_uri}")

    # 3. Identificar la fuente de datos con Gemini
    file.seek(0) # Regresa el cursor al inicio del archivo para leerlo
    headers = file.readline().decode('utf-8').strip()
    data_source = get_data_source_from_headers(headers)
    print(f"INFO: Gemini identificó la fuente como: '{data_source}'")

    # 4. Crear una tarea asíncrona para invocar al Entrenador (CF2)
    task_payload = {
        "job_id": job_id,
        "gcs_input_uri": gcs_uri,
        "data_source": data_source,
        "algorithm": algorithm
    }

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": TRAINER_FUNCTION_URL,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(task_payload).encode(),
        }
    }

    parent = tasks_client.queue_path(GCP_PROJECT, GCP_LOCATION, TASKS_QUEUE)
    tasks_client.create_task(parent=parent, task=task)
    print(f"INFO: Tarea para el job {job_id} encolada en {TASKS_QUEUE}")

    # 5. Responder inmediatamente al cliente
    return jsonify({"status": "processing", "job_id": job_id}), 202