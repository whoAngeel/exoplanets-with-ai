import functions_framework
from flask import Request, jsonify
from google.cloud import firestore

firestore_client = None

# Cabeceras CORS para permitir el acceso desde cualquier origen
CORS_HEADERS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
}

@functions_framework.http
def save_exoplanet(request: Request):
    """
    Guarda un objeto JSON en la colección 'exoplanetas' de Firestore.
    """
    global firestore_client

    # Manejar la petición PREFLIGHT de CORS
    if request.method == 'OPTIONS':
        return ('', 204, CORS_HEADERS)

    # Solo permitir peticiones POST
    if request.method != 'POST':
        return (jsonify({"error": "Método no permitido"}), 405, CORS_HEADERS)

    try:
        # Inicialización perezosa del cliente de Firestore
        if firestore_client is None:
            firestore_client = firestore.Client()

        # Obtener el JSON del cuerpo de la petición
        data = request.get_json(silent=True)
        if not data:
            return (jsonify({"error": "No se proporcionó un JSON válido en el cuerpo de la petición."}), 400, CORS_HEADERS)

        # Añadir el documento a la colección 'exoplanetas' (Firestore genera el ID)
        update_time, doc_ref = firestore_client.collection('exoplanetas').add(data)
        
        print(f"Documento {doc_ref.id} creado en la colección 'exoplanetas'.")
        
        # Devolver una respuesta exitosa
        return (jsonify({"status": "éxito", "id": doc_ref.id, "data": data}), 201, CORS_HEADERS)

    except Exception as e:
        print(f"Error al guardar en Firestore: {e}")
        return (jsonify({"error": "Ocurrió un error interno al guardar los datos."}), 500, CORS_HEADERS)

