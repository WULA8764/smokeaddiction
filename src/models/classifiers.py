"""EEG classification models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

logger = logging.getLogger(__name__)


class EEGClassifier:
    """EEG classification model with multiple algorithms."""
    
    def __init__(self, config: Dict):
        """Initialize the classifier.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_score = 0.0
        self.feature_names = None
        self.class_names = None
        
    def train_models(self, X: pd.DataFrame, y: pd.Series, feature_names: Optional[List[str]] = None) -> Dict:
        """Train multiple classification models.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            feature_names: List of feature names
            
        Returns:
            Dictionary with training results
        """
        logger.info("Training classification models")
        
        self.feature_names = feature_names if feature_names else list(X.columns)
        
        # Get unique class names
        self.class_names = sorted(y.unique())
        
        # Convert to numpy arrays
        X_array = X.values
        y_array = y.values
        
        results = {}
        
        # Train each model type
        for model_name, model_config in self.config.get('classifiers', {}).items():
            try:
                logger.info(f"Training {model_name}")
                model_result = self._train_single_model(
                    X_array, y_array, model_name, model_config
                )
                results[model_name] = model_result
                
                # Update best model
                if model_result['cv_score'] > self.best_score:
                    self.best_score = model_result['cv_score']
                    self.best_model = model_result['model']
                    
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def _train_single_model(self, X: np.ndarray, y: np.ndarray, model_name: str, 
                           model_config: Dict) -> Dict:
        """Train a single classification model.
        
        Args:
            X: Feature array
            y: Target array
            model_name: Name of the model
            model_config: Model configuration
            
        Returns:
            Dictionary with training results
        """
        # Initialize model
        model = self._create_model(model_name, model_config)
        
        # Perform grid search if parameters are provided
        if 'params' in model_config:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                model, model_config['params'], 
                cv=cv, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X, y)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_
        else:
            # Simple cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            best_model = model.fit(X, y)
            best_params = {}
            cv_score = cv_scores.mean()
        
        # Store model
        self.models[model_name] = best_model
        
        # Calculate additional metrics
        y_pred = best_model.predict(X)
        y_pred_proba = best_model.predict_proba(X) if hasattr(best_model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None and len(self.class_names) == 2:
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba[:, 1])
        
        return {
            'model': best_model,
            'best_params': best_params,
            'cv_score': cv_score,
            'metrics': metrics,
            'feature_importance': self._get_feature_importance(best_model)
        }
    
    def _create_model(self, model_name: str, model_config: Dict) -> Any:
        """Create a model instance.
        
        Args:
            model_name: Name of the model
            model_config: Model configuration
            
        Returns:
            Model instance
        """
        if model_name == 'svm':
            return SVC(
                kernel=model_config.get('kernel', 'rbf'),
                C=model_config.get('C', 1.0),
                probability=True,
                random_state=42
            )
        elif model_name == 'random_forest':
            return RandomForestClassifier(
                n_estimators=model_config.get('n_estimators', 100),
                max_depth=model_config.get('max_depth', None),
                random_state=42
            )
        elif model_name == 'knn':
            return KNeighborsClassifier(
                n_neighbors=model_config.get('n_neighbors', 5),
                weights=model_config.get('weights', 'uniform')
            )
        elif model_name == 'logistic_regression':
            return LogisticRegression(
                C=model_config.get('C', 1.0),
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def _get_feature_importance(self, model: Any) -> Optional[Dict]:
        """Get feature importance from model.
        
        Args:
            model: Trained model
            
        Returns:
            Dictionary with feature importance
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return None
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, feature in enumerate(self.feature_names):
            feature_importance[feature] = importance[i]
        
        return feature_importance
    
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """Make predictions using a trained model.
        
        Args:
            X: Feature DataFrame
            model_name: Name of the model to use (if None, use best model)
            
        Returns:
            Predicted labels
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No trained model available")
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            model = self.models[model_name]
        
        return model.predict(X.values)
    
    def predict_proba(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """Get prediction probabilities.
        
        Args:
            X: Feature DataFrame
            model_name: Name of the model to use (if None, use best model)
            
        Returns:
            Prediction probabilities
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No trained model available")
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            model = self.models[model_name]
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X.values)
        else:
            raise ValueError(f"Model {model_name} does not support probability predictions")
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series, model_name: Optional[str] = None) -> Dict:
        """Evaluate a trained model.
        
        Args:
            X: Feature DataFrame
            y: True labels
            model_name: Name of the model to evaluate (if None, use best model)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No trained model available")
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            model = self.models[model_name]
        
        # Make predictions
        y_pred = model.predict(X.values)
        y_pred_proba = model.predict_proba(X.values) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None and len(self.class_names) == 2:
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba[:, 1])
        
        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'metrics': metrics
        }
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> Optional[Dict]:
        """Get feature importance for a model.
        
        Args:
            model_name: Name of the model (if None, use best model)
            
        Returns:
            Dictionary with feature importance
        """
        if model_name is None:
            if self.best_model is None:
                return None
            model = self.best_model
        else:
            if model_name not in self.models:
                return None
            model = self.models[model_name]
        
        return self._get_feature_importance(model)
    
    def save_models(self, output_dir: str):
        """Save trained models to disk.
        
        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for model_name, model in self.models.items():
            model_path = output_path / f"{model_name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} model: {model_path}")
        
        # Save best model
        if self.best_model is not None:
            best_model_path = output_path / "best_model.pkl"
            joblib.dump(self.best_model, best_model_path)
            logger.info(f"Saved best model: {best_model_path}")
        
        # Save model metadata
        metadata = {
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'best_score': self.best_score
        }
        metadata_path = output_path / "model_metadata.pkl"
        joblib.dump(metadata, metadata_path)
        logger.info(f"Saved model metadata: {metadata_path}")
    
    def load_models(self, input_dir: str):
        """Load trained models from disk.
        
        Args:
            input_dir: Input directory
        """
        input_path = Path(input_dir)
        
        # Load model metadata
        metadata_path = input_path / "model_metadata.pkl"
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.feature_names = metadata['feature_names']
            self.class_names = metadata['class_names']
            self.best_score = metadata['best_score']
        
        # Load best model
        best_model_path = input_path / "best_model.pkl"
        if best_model_path.exists():
            self.best_model = joblib.load(best_model_path)
            logger.info(f"Loaded best model: {best_model_path}")
        
        # Load individual models
        for model_file in input_path.glob("*_model.pkl"):
            if model_file.name != "best_model.pkl":
                model_name = model_file.stem.replace("_model", "")
                self.models[model_name] = joblib.load(model_file)
                logger.info(f"Loaded {model_name} model: {model_file}")
    
    def get_model_summary(self) -> Dict:
        """Get summary of trained models.
        
        Returns:
            Dictionary with model summary
        """
        summary = {
            'n_models': len(self.models),
            'model_names': list(self.models.keys()),
            'best_model_score': self.best_score,
            'feature_names': self.feature_names,
            'class_names': self.class_names
        }
        
        return summary

