import functions_framework
from flask import Request, jsonify
from google.cloud import firestore

firestore_client = None

# Cabeceras CORS para permitir el acceso desde cualquier origen
CORS_HEADERS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
}

@functions_framework.http
def get_exoplanets(request: Request):
    """
    Consulta documentos de la colección 'exoplanetas' de Firestore.
    - GET /: Lista todos los documentos.
    - GET /{doc_id}: Obtiene un documento específico.
    """
    global firestore_client

    if request.method == 'OPTIONS':
        return ('', 204, CORS_HEADERS)

    if request.method != 'GET':
        return (jsonify({"error": "Método no permitido"}), 405, CORS_HEADERS)

    try:
        if firestore_client is None:
            firestore_client = firestore.Client()
        
        # Extraer la ruta de la URL para decidir si listar todo o buscar uno
        path_parts = request.path.strip('/').split('/')
        collection_ref = firestore_client.collection('exoplanetas')

        # Si la URL es solo la base (ej. /get-exoplanets), listar todo
        if len(path_parts) == 1:
            all_docs = []
            for doc in collection_ref.stream():
                doc_data = doc.to_dict()
                doc_data['id'] = doc.id # Añadir el ID al resultado
                all_docs.append(doc_data)
            return (jsonify(all_docs), 200, CORS_HEADERS)
        
        # Si la URL tiene un ID (ej. /get-exoplanets/xyz), buscar ese documento
        elif len(path_parts) == 2:
            doc_id = path_parts[1]
            doc_ref = collection_ref.document(doc_id)
            doc = doc_ref.get()
            if doc.exists:
                return (jsonify(doc.to_dict()), 200, CORS_HEADERS)
            else:
                return (jsonify({"error": "Documento no encontrado"}), 404, CORS_HEADERS)
        
        # Ruta no válida
        return (jsonify({"error": "Ruta no válida"}), 400, CORS_HEADERS)

    except Exception as e:
        print(f"Error al consultar Firestore: {e}")
        return (jsonify({"error": "Ocurrió un error interno al consultar los datos."}), 500, CORS_HEADERS)