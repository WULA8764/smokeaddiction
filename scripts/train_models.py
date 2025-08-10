#!/usr/bin/env python3
"""Model training script."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from models.classifiers import EEGClassifier

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """Main model training function."""
    
    # Setup logging
    logging.basicConfig(level=getattr(logging, cfg.log_level))
    
    logger.info("Starting model training")
    
    # Load fused features
    features_path = Path(cfg.output_dir) / "features" / "fused_features.csv"
    if not features_path.exists():
        logger.error(f"Fused features not found: {features_path}")
        return
    
    features_df = pd.read_csv(features_path)
    logger.info(f"Loaded features with shape: {features_df.shape}")
    
    # Load questionnaire data for labels
    # This is a placeholder - you would need to implement actual label loading
    # based on your specific data structure
    labels = _load_labels(features_df['subject'].unique())
    
    if labels is None:
        logger.error("Could not load labels, cannot proceed with training")
        return
    
    # Prepare features and labels
    feature_cols = [col for col in features_df.columns if col not in ['subject', 'event_type', 'component', 'n_epochs']]
    X = features_df[feature_cols].copy()
    y = pd.Series([labels.get(subject, 0) for subject in features_df['subject']], index=features_df.index)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.models.evaluation.test_size, 
        random_state=cfg.models.evaluation.random_state, stratify=y
    )
    
    logger.info(f"Training set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Initialize classifier
    classifier = EEGClassifier(cfg.models)
    
    # Train models
    training_results = classifier.train_models(X_train, y_train, feature_names=feature_cols)
    
    # Print training results
    for model_name, result in training_results.items():
        if 'error' not in result:
            logger.info(f"{model_name}: CV Score = {result['cv_score']:.3f}")
            logger.info(f"{model_name}: Best Params = {result['best_params']}")
    
    # Evaluate on test set
    test_results = classifier.evaluate_model(X_test, y_test)
    logger.info(f"Test set accuracy: {test_results['metrics']['accuracy']:.3f}")
    logger.info(f"Test set F1 score: {test_results['metrics']['f1']:.3f}")
    
    # Save models
    models_dir = Path(cfg.output_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    classifier.save_models(str(models_dir))
    
    # Save test results
    test_results_df = pd.DataFrame([test_results['metrics']])
    test_results_path = models_dir / "test_results.csv"
    test_results_df.to_csv(test_results_path, index=False)
    logger.info(f"Saved test results: {test_results_path}")
    
    # Save feature importance
    feature_importance = classifier.get_feature_importance()
    if feature_importance:
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': importance}
            for feature, importance in feature_importance.items()
        ]).sort_values('importance', ascending=False)
        
        importance_path = models_dir / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Saved feature importance: {importance_path}")
    
    logger.info("Model training completed")


def _load_labels(subjects):
    """Load labels for subjects.
    
    This is a placeholder function. You would need to implement this based on your
    specific data structure and how you want to define the target variable.
    
    Args:
        subjects: List of subject IDs
        
    Returns:
        Dictionary mapping subject IDs to labels
    """
    # Placeholder implementation
    # In practice, you would load this from questionnaire data or other sources
    
    # Example: Load FTND scores and create binary labels
    # ftnd_scores = {}
    # for subject in subjects:
    #     try:
    #         ftnd_data = pd.read_csv(f"data/raw/questionnaires/sub-{subject}_ftnd.csv")
    #         ftnd_scores[subject] = ftnd_data['total_score'].iloc[0]
    #     except:
    #         ftnd_scores[subject] = 0
    
    # # Create binary labels (example: high vs low addiction)
    # labels = {subject: 1 if score > 5 else 0 for subject, score in ftnd_scores.items()}
    
    # For now, return random labels for demonstration
    np.random.seed(42)
    labels = {subject: np.random.choice([0, 1]) for subject in subjects}
    
    return labels


if __name__ == "__main__":
    main()

