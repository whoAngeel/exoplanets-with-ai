import functions_framework
from flask import Request, jsonify
from google.cloud import firestore

# --- INICIALIZACIÓN PEREZOSA ---
firestore_client = None

# --- MANEJO DE CORS ---
CORS_HEADERS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
}

def get_firestore_client():
    """Inicializa el cliente de Firestore la primera vez que se necesita."""
    global firestore_client
    if firestore_client is None:
        firestore_client = firestore.Client()
    return firestore_client

@functions_framework.http
def jobs_crud(request: Request):
    """
    Función CRUD para gestionar los trabajos de entrenamiento en Firestore.
    """
    if request.method == 'OPTIONS':
        return ('', 204, CORS_HEADERS)

    try:
        client = get_firestore_client()
        collection_ref = client.collection("exo_scout_models")
        
        path_parts = request.path.strip('/').split('/')

        # RUTA: /jobs (Listar todos los trabajos)
        if len(path_parts) == 1 and path_parts[0] == 'jobs' and request.method == 'GET':
            
            # --- CORRECCIÓN AQUÍ ---
            # Añadimos .order_by() para ordenar por el campo 'created_at' en orden descendente.
            query = collection_ref.order_by("created_at", direction=firestore.Query.DESCENDING)
            
            all_jobs = []
            # Usamos la nueva 'query' en lugar de 'collection_ref' directamente.
            for doc in query.stream():
                job_data = doc.to_dict()
                job_data['job_id'] = doc.id
                all_jobs.append(job_data)
                
            return (jsonify(all_jobs), 200, CORS_HEADERS)

        # RUTA: /jobs/{job_id} (Obtener o Eliminar un trabajo)
        elif len(path_parts) == 2 and path_parts[0] == 'jobs':
            job_id = path_parts[1]
            doc_ref = collection_ref.document(job_id)

            if request.method == 'GET':
                doc = doc_ref.get()
                if doc.exists:
                    return (jsonify(doc.to_dict()), 200, CORS_HEADERS)
                else:
                    return (jsonify({"error": "Job no encontrado"}), 404, CORS_HEADERS)

            elif request.method == 'DELETE':
                doc_ref.delete()
                return (jsonify({"status": "éxito", "message": f"Job {job_id} eliminado."}), 200, CORS_HEADERS)

        return (jsonify({"error": "Ruta no encontrada"}), 404, CORS_HEADERS)

    except Exception as e:
        print(f"Error interno: {e}")
        return (jsonify({"error": "Ocurrió un error interno en el servidor"}), 500, CORS_HEADERS)