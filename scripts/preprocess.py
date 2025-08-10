#!/usr/bin/env python3
"""EEG data preprocessing script."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import hydra
from omegaconf import DictConfig
import pandas as pd

from data.loader import EEGDataLoader
from data.preprocessor import EEGPreprocessor

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """Main preprocessing function."""
    
    # Setup logging
    logging.basicConfig(level=getattr(logging, cfg.log_level))
    
    logger.info("Starting EEG data preprocessing")
    
    # Initialize data loader and preprocessor
    data_loader = EEGDataLoader(cfg.data_dir)
    preprocessor = EEGPreprocessor(cfg.preprocessing)
    
    # Get list of subjects
    subjects = data_loader.list_subjects()
    logger.info(f"Found {len(subjects)} subjects")
    
    # Process each subject
    for subject in subjects:
        try:
            logger.info(f"Processing subject {subject}")
            
            # Load raw data
            raw = data_loader.load_raw_data(subject, "resting_state")
            
            # Preprocess data
            processed_raw = preprocessor.preprocess_raw(raw, subject)
            
            # Save preprocessed data
            output_dir = Path(cfg.output_dir) / "processed"
            output_dir.mkdir(parents=True, exist_ok=True)
            preprocessor.save_preprocessed_data(processed_raw, subject, "resting_state", str(output_dir))
            
            # Create epochs if events are available
            try:
                events = raw.annotations.to_events()
                epochs = preprocessor.create_epochs(processed_raw, events, subject)
                epochs_clean = preprocessor.preprocess_epochs(epochs, subject)
                preprocessor.save_epochs(epochs_clean, subject, "resting_state", str(output_dir))
            except Exception as e:
                logger.warning(f"Could not create epochs for subject {subject}: {e}")
            
            logger.info(f"Completed processing for subject {subject}")
            
        except Exception as e:
            logger.error(f"Failed to process subject {subject}: {e}")
    
    logger.info("Preprocessing completed")


if __name__ == "__main__":
    main()

