# pipelines/kepler_pipeline.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from .base_pipeline import BaseTrainingPipeline

class KeplerTrainingPipeline(BaseTrainingPipeline):
    """Pipeline de entrenamiento específico para datos de Kepler."""

    def select_features(self):
        print("PASO 1 (Kepler): SELECCIÓN DE FEATURES")
        feature_groups = {
            'flags': ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_score'],
            'planeta': ['koi_period', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_sma', 'koi_eccen', 'koi_incl'],
            'transito': ['koi_duration', 'koi_depth', 'koi_ror', 'koi_impact', 'koi_model_snr'],
            'estrella': ['koi_steff', 'koi_slogg', 'koi_srad', 'koi_smass', 'koi_smet'],
            'calidad': ['koi_count', 'koi_num_transits']
        }
        selected_features = [f for group in feature_groups.values() for f in group]
        available_features = [f for f in selected_features if f in self.df.columns]
        
        self.X = self.df[available_features].copy()
        self.y = self.df['koi_disposition'].copy()
        print(f"✓ Usando {self.X.shape[1]} features de Kepler.")

    def engineer_features(self):
        print("PASO 2 (Kepler): FEATURE ENGINEERING")
        X_eng = self.X.copy()
        if 'koi_prad' in self.X.columns and 'koi_srad' in self.X.columns:
            X_eng['planet_star_ratio'] = X_eng['koi_prad'] / (X_eng['koi_srad'] * 109.1)
        if 'koi_prad' in self.X.columns and 'koi_period' in self.X.columns:
            X_eng['density_proxy'] = X_eng['koi_prad'] / (X_eng['koi_period'] ** (1/3))
        # ... más features ...
        self.X = X_eng
        print("✓ Features de Kepler creadas.")
    
    def preprocess_data(self):
        print("PASO 3 (Kepler): PREPROCESAMIENTO")
        le = LabelEncoder()
        self.y_encoded = le.fit_transform(self.y)
        
        imputer = SimpleImputer(strategy=self.config.imputation_strategy)
        X_imputed = pd.DataFrame(imputer.fit_transform(self.X), columns=self.X.columns)
        
        scaler = StandardScaler()
        self.X_processed = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)
        
        self.artifacts['label_encoder'] = le
        self.artifacts['imputer'] = imputer
        self.artifacts['scaler'] = scaler
        print("✓ Preprocesamiento de Kepler completo.")