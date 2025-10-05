# common/gcp_utils.py

import pickle
from google.cloud import storage, firestore
from datetime import datetime

def save_artifacts_to_gcs(bucket_name, job_id, artifacts):
    """Sube el diccionario de artefactos a GCS."""
    print("💾 Guardando artefactos del modelo en Cloud Storage...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    gcs_path = f"models/{job_id}/artifacts.pkl"
    blob = bucket.blob(gcs_path)

    with blob.open("wb") as f:
        pickle.dump(artifacts, f)
        
    gcs_uri = f"gs://{bucket_name}/{blob.name}"
    print(f"✓ Artefactos guardados en: {gcs_uri}")
    return gcs_uri

def update_firestore_metadata(job_id, gcs_uri, metadata):
    """Actualiza el documento de un job en Firestore con los resultados."""
    print("📝 Guardando metadatos en Firestore...")
    firestore_client = firestore.Client()
    doc_ref = firestore_client.collection("exo_scout_models").document(job_id)
    
    final_metadata = {
        "status": "completed",
        "completed_at": datetime.now(),
        "results": {
            "gcs_artifacts_path": gcs_uri,
            **metadata
        }
    }
    doc_ref.update(final_metadata)
    print(f"✓ Metadatos actualizados para el job: {job_id}")
    return final_metadata