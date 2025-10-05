import functions_framework
from flask import Request, jsonify
import os
import uuid
import json
import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud import storage, tasks_v2

# --- INICIALIZACIÓN PEREZOSA ---
storage_client = None
tasks_client = None
gemini_model = None
gcp_project = None
gcp_location = None
trainer_function_url = None
tasks_queue = None

# --- CONSTANTES ---
UPLOAD_BUCKET_NAME = "exoplanets-nasa-models" 

def get_gemini_model():
    """
    Función que inicializa el cliente de Vertex AI la primera vez que se necesita.
    """
    global gemini_model, gcp_project, gcp_location
    
    if gemini_model is None:
        print("INFO: Inicializando cliente de Vertex AI por primera vez...")
        gcp_project = os.environ.get("GCP_PROJECT")
        gcp_location = os.environ.get("GCP_LOCATION", "us-central1")
        
        if not gcp_project:
            raise RuntimeError("La variable de entorno GCP_PROJECT no está configurada.")
            
        vertexai.init(project=gcp_project, location=gcp_location)
    
        gemini_model = GenerativeModel("gemini-2.5-flash")
        
        print("INFO: Cliente de Vertex AI inicializado.")
        
    return gemini_model

def get_data_source_from_headers(headers: str) -> str:
    """Usa Gemini para identificar la fuente de datos a partir de las columnas del CSV."""
    
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
            print(f"INFO: Gemini identificó la fuente como: '{data_source}' basado en análisis detallado.")
            return data_source
        else:
            print(f"WARN: Gemini devolvió una respuesta inesperada: '{data_source}'. Se usará 'unknown'.")
            return "unknown"
            
    except Exception as e:
        print(f"Error DETALLADO al llamar a Gemini: {e}")
        return "unknown"

@functions_framework.http
def orchestrator_function(request: Request):
    """Cloud Function #1: El Orquestador Inteligente."""
    global storage_client, tasks_client, gcp_project, gcp_location, trainer_function_url, tasks_queue

    # --- Inicialización perezosa de otros clientes ---
    if storage_client is None:
        print("INFO: Inicializando cliente de Storage por primera vez...")
        storage_client = storage.Client()
    if tasks_client is None:
        print("INFO: Inicializando cliente de Tasks por primera vez...")
        tasks_client = tasks_v2.CloudTasksClient()
        gcp_project = os.environ.get("GCP_PROJECT")
        gcp_location = os.environ.get("GCP_LOCATION", "us-central1")
        trainer_function_url = os.environ.get("TRAINER_FUNCTION_URL")
        tasks_queue = os.environ.get("TASKS_QUEUE", "exo-scout-queue")

    # --- Lógica de la función ---
    if 'file' not in request.files:
        return "Error: No se encontró el archivo en la solicitud.", 400

    file = request.files['file']
    params = json.loads(request.form.get("params", "{}"))
    algorithm = params.get("algorithm", "gradient_boosting")
    
    # CORRECCIÓN 2: El nombre correcto del algoritmo es 'xgboost', no 'gx_boost'.
    valid_algorithms = ["random_forest", "gradient_boosting", "xgboost"]
    
    if algorithm not in valid_algorithms:
        # CORRECCIÓN 3: Corregido el typo 'random_fores' en el mensaje de error.
        return "Error: Algoritmo no valido. opciones [random_forest, gradient_boosting, xgboost]", 400
    
    job_id = str(uuid.uuid4())

    blob = storage_client.bucket(UPLOAD_BUCKET_NAME).blob(f"raw-uploads/{job_id}_{file.filename}")
    blob.upload_from_file(file)
    gcs_uri = f"gs://{UPLOAD_BUCKET_NAME}/{blob.name}"
    print(f"INFO: Archivo subido a {gcs_uri}")

    # --- CORRECCIÓN IMPORTANTE AQUÍ ---
    # Buscamos la primera línea que NO sea un comentario para enviarla a Gemini.
    file.seek(0) # Reinicia el cursor del archivo
    headers = ""
    for line_bytes in file:
        line_str = line_bytes.decode('utf-8').strip()
        if not line_str.startswith('#'):
            headers = line_str
            break # Encontramos la cabecera, salimos del bucle
    
    if not headers:
        # Si el archivo está vacío o solo tiene comentarios, manejamos el error.
        return "Error: No se encontró una línea de cabecera válida en el archivo.", 400
    data_source = get_data_source_from_headers(headers)

    task_payload = {
        "job_id": job_id,
        "gcs_input_uri": gcs_uri,
        "data_source": data_source,
        "algorithm": algorithm
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

    return jsonify({"status": "processing", "job_id": job_id}), 202