import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Importamos la clase base para heredar su funcionalidad
from .base_pipeline import BaseTrainingPipeline

class K2TrainingPipeline(BaseTrainingPipeline):
    """
    Pipeline de entrenamiento específico para los datos de K2.
    """
    def select_features(self):
        print("PASO 1: SELECCIÓN DE FEATURES K2")
        k2_feature_groups = {
            'planeta': ['pl_orbper', 'pl_rade', 'pl_radj', 'pl_bmasse', 'pl_bmassj', 'pl_orbeccen', 'pl_orbsmax', 'pl_insol', 'pl_eqt'],
            'estrella': ['st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg'],
            'flags_calidad': ['pl_controv_flag', 'ttv_flag']
        }
        selected_features = [f for group in k2_feature_groups.values() for f in group]
        available_features = [f for f in selected_features if f in self.df.columns]
        
        # Filtramos filas con demasiados valores nulos
        valid_counts = self.df[available_features].notna().sum(axis=1)
        df_filtered = self.df[valid_counts >= self.config.min_valid_features].copy()
        
        self.X = df_filtered[available_features].copy()
        self.y = df_filtered['disposition'].copy()
        print(f"✓ Usando {self.X.shape[1]} features disponibles. Dataset final: {self.X.shape[0]} filas.")

    def engineer_features(self):
        print("PASO 2: FEATURE ENGINEERING K2")
        # Usamos self.X que fue creado en el paso anterior
        if 'pl_rade' in self.X.columns and 'st_rad' in self.X.columns:
            self.X['planet_star_ratio'] = self.X['pl_rade'] / (self.X['st_rad'] * 109.1)
        if 'pl_rade' in self.X.columns and 'pl_orbper' in self.X.columns:
            self.X['density_proxy'] = self.X['pl_rade'] / (self.X['pl_orbper'] ** (1/3))
        print("✓ Features de ingeniería para K2 creadas.")

    def preprocess_data(self):
        print("PASO 3: PREPROCESAMIENTO K2")
        if len(self.X) < self.config.cv_splits:
            raise ValueError(f"Datos insuficientes ({len(self.X)} filas) para continuar con el entrenamiento.")
            
        le = LabelEncoder()
        self.y_encoded = le.fit_transform(self.y)
        
        imputer = SimpleImputer(strategy=self.config.imputation_strategy)
        X_imputed = pd.DataFrame(imputer.fit_transform(self.X), columns=self.X.columns)
        
        scaler = StandardScaler()
        self.X_processed = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)
        
        # Guardamos los artefactos en el diccionario de la clase base
        self.artifacts['label_encoder'] = le
        self.artifacts['imputer'] = imputer
        self.artifacts['scaler'] = scaler
        self.artifacts['feature_names'] = self.X_processed.columns.tolist()
        print("✓ Preprocesamiento K2 completo.")