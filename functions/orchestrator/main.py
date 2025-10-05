import functions_framework
from flask import Request, jsonify
from google.cloud import firestore
import os
import uuid
import json
import hashlib 

import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud import storage, tasks_v2

# --- INICIALIZACIÓN DE CLIENTES (GLOBALES) ---
# Se inicializarán de forma perezosa (solo la primera vez que se necesiten)
storage_client = None
tasks_client = None
gemini_model = None
firestore_client = None

# --- CONSTANTES ---
UPLOAD_BUCKET_NAME = "exoplanets-nasa-models" 

# --- FUNCIONES AUXILIARES (Sin cambios) ---
def get_gemini_model():
    # ... (Tu función get_gemini_model sin cambios)
    global gemini_model
    if gemini_model is None:
        gcp_project = os.environ.get("GCP_PROJECT")
        gcp_location = os.environ.get("GCP_LOCATION", "us-central1")
        if not gcp_project:
            raise RuntimeError("La variable de entorno GCP_PROJECT no está configurada.")
        vertexai.init(project=gcp_project, location=gcp_location)
        gemini_model = GenerativeModel("gemini-2.5-flash")
    return gemini_model

def get_data_source_from_headers(headers: str) -> str:
    # ... (Tu función get_data_source_from_headers sin cambios)
    prompt = f"""
    Eres un experto en ciencia de datos especializado en astronomía. Tu tarea es identificar el origen de un dataset de exoplanetas (Kepler, TESS, o K2) basándote en una lista de sus nombres de columnas.

    Aquí tienes las "huellas dactilares" de cada dataset:

    1.  **Dataset: Kepler**
        * **Patrón General:** Usa frecuentemente los prefijos `koi_`, `kepoi_`, y `kepler_`.
        * **Columnas Únicas Clave:** Busca la presencia de `kepoi_name`, `koi_disposition`, `koi_score`, o `koi_vet_stat`. La presencia de `koi_` es la señal más fuerte.

    2.  **Dataset: TESS**
        * **Patrón General:** Usa prefijos como `st_` (stellar), `pl_` (planetary) y `loc_`.
        * **Columnas Únicas Clave:** La señal más fuerte es la presencia de `toi` (TESS Object of Interest), `tid` (TESS Input Catalog ID), o `tfopwg_disp`.

    3.  **Dataset: K2**
        * **Patrón General:** Similar a TESS, puede usar `pl_` y `st_`, pero también `sy_` (system).
        * **Columnas Únicas Clave:** La señal más fuerte y distintiva es la presencia de columnas sobre el método de descubrimiento, como `discoverymethod`, `disc_year`, y `disc_facility`. Si ves estas columnas, es casi seguro que es K2.

    **Nota sobre columnas comunes:** Columnas como `ra`, `dec`, `pl_orbper`, `pl_rade`, y `st_teff` pueden aparecer en varios datasets, así que úsalas con menos peso.

    **Instrucción:**
    Analiza la siguiente lista de columnas y decide si provienen de 'kepler', 'tess', o 'k2'. Si no puedes determinarlo con certeza, responde 'unknown'.

    **Responde únicamente con una de estas cuatro palabras: kepler, tess, k2, o unknown.**

    **Columnas a analizar:** "{headers}"
    """
    try:
        model = get_gemini_model()
        response = model.generate_content(prompt)
        data_source = response.text.strip().lower()
        if data_source in ["kepler", "tess", "k2", "unknown"]:
            return data_source
        else:
            return "unknown"
    except Exception as e:
        print(f"Error al llamar a Gemini: {e}")
        return "unknown"


@functions_framework.http
def orchestrator_function(request: Request):
    """
    Cloud Function #1: El Orquestador Inteligente e Idempotente con CORS.
    Esta es la versión corregida y unificada.
    """
    # --- CABECERAS CORS ---
    # Se añaden a todas las respuestas para permitir llamadas desde el navegador.
    cors_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
    }

    # Manejar solicitud PREFLIGHT de CORS (el navegador la envía antes del POST)
    if request.method == 'OPTIONS':
        return ('', 204, cors_headers)

    # --- INICIALIZACIÓN PEREZOSA DE CLIENTES ---
    global storage_client, tasks_client, firestore_client
    if storage_client is None:
        storage_client = storage.Client()
    if firestore_client is None:
        firestore_client = firestore.Client()
    if tasks_client is None:
        tasks_client = tasks_v2.CloudTasksClient()

    try:
        # --- VALIDACIÓN DE ENTRADA ---
        if 'file' not in request.files:
            return jsonify({"error": "No se encontró el archivo en la solicitud."}), 400, cors_headers

        file = request.files['file']
        
        # --- LÓGICA DE IDEMPOTENCIA ---
        file_content = file.read()
        if not file_content:
             return jsonify({"error": "El archivo enviado está vacío."}), 400, cors_headers

        file_hash = hashlib.sha256(file_content).hexdigest()
        job_id = file_hash
        file.seek(0) # Rebobinar el archivo para poder leerlo de nuevo

        doc_ref = firestore_client.collection("exo_scout_models").document(job_id)
        doc = doc_ref.get()
        if doc.exists:
            print(f"INFO: Job duplicado detectado: {job_id}. Devolviendo estado existente.")
            return jsonify(doc.to_dict()), 200, cors_headers

        # --- EXTRACCIÓN Y VALIDACIÓN DE PARÁMETROS ---
        params = json.loads(request.form.get("params", "{}"))
        algorithm = params.get("algorithm", "gradient_boosting")
        valid_algorithms = ["random_forest", "gradient_boosting", "xgboost"]
        model_name = params.get("model_name", f"model_{algorithm}_{job_id[:8]}")

        if algorithm not in valid_algorithms:
            return jsonify({"error": f"Algoritmo no válido. Opciones: {valid_algorithms}"}), 400, cors_headers

        # --- IDENTIFICACIÓN DE FUENTE CON GEMINI ---
        headers = ""
        for line_bytes in file:
            try:
                line_str = line_bytes.decode('utf-8').strip()
                if line_str and not line_str.startswith('#'):
                    headers = line_str
                    break
            except UnicodeDecodeError:
                continue # Ignorar líneas que no son UTF-8
        
        file.seek(0) # Rebobinar de nuevo para la subida
        if not headers:
            return jsonify({"error": "No se encontró una línea de cabecera válida en el archivo."}), 400, cors_headers
        
        data_source = get_data_source_from_headers(headers)
        if data_source == "unknown":
            return jsonify({"error": "No se pudo determinar la fuente de datos (Kepler, TESS, K2) a partir de las columnas."}), 400, cors_headers

        # --- SUBIDA A GCS Y CREACIÓN DE TAREA ---
        blob = storage_client.bucket(UPLOAD_BUCKET_NAME).blob(f"raw-uploads/{job_id}_{file.filename}")
        blob.upload_from_file(file)
        gcs_uri = f"gs://{UPLOAD_BUCKET_NAME}/{blob.name}"
        
        # Encolar la tarea para la segunda función
        gcp_project = os.environ.get("GCP_PROJECT")
        gcp_location = os.environ.get("GCP_LOCATION", "us-central1")
        trainer_function_url = os.environ.get("TRAINER_FUNCTION_URL")
        tasks_queue = os.environ.get("TASKS_QUEUE", "exo-scout-queue")

        if not all([gcp_project, gcp_location, trainer_function_url, tasks_queue]):
            raise RuntimeError("Faltan variables de entorno para Cloud Tasks.")

        task_payload = {
            "job_id": job_id,
            "gcs_input_uri": gcs_uri,
            "data_source": data_source,
            "algorithm": algorithm,
            "model_name": model_name  
        }
        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": trainer_function_url,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(task_payload).encode(),
            }
        }
        parent = tasks_client.queue_path(gcp_project, gcp_location, tasks_queue)
        tasks_client.create_task(parent=parent, task=task)
        print(f"INFO: Tarea para el job {job_id} encolada en {tasks_queue}")
        
        return jsonify({"status": "processing", "job_id": job_id}), 202, cors_headers

    except Exception as e:
        print(f"ERROR CRÍTICO en la orquestación: {e}")
        return jsonify({"error": "Ocurrió un error interno en el servidor."}), 500, cors_headers