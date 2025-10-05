# common/config.py

# class ModelConfig:
#     """Configuración de parámetros del modelo."""
#     test_size = 0.2
#     random_state = 42
#     imputation_strategy = 'median'
#     rf_n_estimators = 200
#     rf_max_depth = 20
#     rf_class_weight = 'balanced'
#     gb_n_estimators = 200
#     gb_max_depth = 5
#     gb_learning_rate = 0.1
#     xgb_n_estimators = 200
#     xgb_max_depth = 6
#     xgb_learning_rate = 0.1
#     cv_splits = 5
#     top_features_to_show = 20


class ModelConfig:
    """Configuración de parámetros del modelo."""
    # === PARÁMETROS DE DATOS ===
    test_size = 0.2
    random_state = 42
    imputation_strategy = 'median'
    
    # --- ¡NUEVO! Parámetros específicos de K2 ---
    min_valid_features = 5       # Mínimo de features no-nulas para incluir una fila
    remove_controversial = False # Si True, elimina planetas controversiales
    
    # === PARÁMETROS DE MODELOS ===
    rf_n_estimators = 200
    rf_max_depth = 20
    rf_class_weight = 'balanced'
    gb_n_estimators = 200
    gb_max_depth = 5
    gb_learning_rate = 0.1
    xgb_n_estimators = 200
    xgb_max_depth = 6
    xgb_learning_rate = 0.1
    
    # === PARÁMETROS DE EVALUACIÓN ===
    cv_splits = 5
    top_features_to_show = 20