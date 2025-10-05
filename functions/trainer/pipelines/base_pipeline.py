# pipelines/base_pipeline.py

import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report

from common.config import ModelConfig

class BaseTrainingPipeline(ABC):
    """
    Clase base para todos los pipelines de entrenamiento.
    Define la estructura y contiene la lógica común.
    """
    def __init__(self, df, algorithm):
        self.df = df
        self.algorithm = algorithm
        self.config = ModelConfig()
        self.artifacts = {}
        self.metadata = {}

    @abstractmethod
    def select_features(self):
        """Método abstracto para seleccionar features. Debe ser implementado por cada subclase."""
        pass

    @abstractmethod
    def engineer_features(self):
        """Método abstracto para crear features. Debe ser implementado por cada subclase."""
        pass

    @abstractmethod
    def preprocess_data(self):
        """Método abstracto para preprocesar. Debe ser implementado por cada subclase."""
        pass

    def _train_and_evaluate(self):
        """Lógica de entrenamiento y evaluación, común para todos."""
        print(f"PASO 4: ENTRENANDO MODELO: {self.algorithm}")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_processed, self.y_encoded,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=self.y_encoded
        )
        
        models = {
            'random_forest': RandomForestClassifier(n_estimators=self.config.rf_n_estimators, max_depth=self.config.rf_max_depth, class_weight=self.config.rf_class_weight, random_state=self.config.random_state, n_jobs=-1),
            'xgboost': xgb.XGBClassifier(n_estimators=self.config.xgb_n_estimators, max_depth=self.config.xgb_max_depth, learning_rate=self.config.xgb_learning_rate, random_state=self.config.random_state, eval_metric='mlogloss'),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=self.config.gb_n_estimators, max_depth=self.config.gb_max_depth, learning_rate=self.config.gb_learning_rate, random_state=self.config.random_state)
        }
        
        model = models.get(self.algorithm)
        if model is None:
            raise ValueError(f"Algoritmo '{self.algorithm}' no soportado.")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        self.artifacts['model'] = model
        self.metadata['f1_score'] = f1_score(y_test, y_pred, average='weighted')
        self.metadata['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        if hasattr(model, 'feature_importances_'):
            self.metadata['feature_importance'] = pd.DataFrame({
                'Feature': self.X_processed.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(self.config.top_features_to_show).to_dict('records')
            
        print(f"✓ Entrenamiento completo. F1-Score: {self.metadata['f1_score']:.4f}")

    def run(self):
        """Ejecuta el pipeline completo en orden."""
        self.select_features()
        self.engineer_features()
        self.preprocess_data()
        self._train_and_evaluate()
        return self.artifacts, self.metadata