"""ERP feature extraction for EEG analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

import mne
from scipy import signal
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


class ERPFeatureExtractor:
    """Extract ERP features from EEG epochs."""
    
    def __init__(self, config: Dict):
        """Initialize the ERP feature extractor.
        
        Args:
            config: Configuration dictionary with ERP parameters
        """
        self.config = config
        self.components = config.get('components', ['N1', 'P2', 'LPP'])
        self.time_windows = config.get('time_windows', {
            'N1': [0.08, 0.12],
            'P2': [0.15, 0.25],
            'LPP': [0.4, 0.8]
        })
        self.channels = config.get('channels', ['Fz', 'Cz', 'Pz'])
        
    def extract_features(self, epochs: mne.Epochs, subject: str) -> pd.DataFrame:
        """Extract ERP features from epochs.
        
        Args:
            epochs: MNE Epochs object
            subject: Subject ID
            
        Returns:
            DataFrame with ERP features
        """
        logger.info(f"Extracting ERP features for subject {subject}")
        
        features_list = []
        
        # Get event types
        event_types = np.unique(epochs.events[:, 2])
        
        for event_type in event_types:
            # Select epochs for this event type
            event_epochs = epochs[event_type]
            
            if len(event_epochs) == 0:
                continue
                
            # Extract features for each component
            for component in self.components:
                component_features = self._extract_component_features(
                    event_epochs, component, subject, event_type
                )
                features_list.append(component_features)
        
        if features_list:
            return pd.concat(features_list, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _extract_component_features(self, epochs: mne.Epochs, component: str, 
                                  subject: str, event_type: int) -> pd.DataFrame:
        """Extract features for a specific ERP component.
        
        Args:
            epochs: Epochs for specific event type
            component: ERP component name
            subject: Subject ID
            event_type: Event type
            
        Returns:
            DataFrame with component features
        """
        features = {}
        
        # Get time window for component
        time_window = self.time_windows[component]
        tmin, tmax = time_window
        
        # Get time indices
        time_mask = (epochs.times >= tmin) & (epochs.times <= tmax)
        times = epochs.times[time_mask]
        
        # Extract features for each channel
        for ch_name in self.channels:
            if ch_name not in epochs.ch_names:
                continue
                
            # Get channel data
            ch_data = epochs.get_data()[:, epochs.ch_names.index(ch_name), time_mask]
            
            # Calculate ERP features
            ch_features = self._calculate_erp_features(ch_data, times, ch_name, component)
            features.update(ch_features)
        
        # Global features
        global_features = self._calculate_global_erp_features(epochs, time_mask, component)
        features.update(global_features)
        
        # Add metadata
        features['subject'] = subject
        features['event_type'] = event_type
        features['component'] = component
        features['n_epochs'] = len(epochs)
        
        return pd.DataFrame([features])
    
    def _calculate_erp_features(self, data: np.ndarray, times: np.ndarray, 
                              ch_name: str, component: str) -> Dict:
        """Calculate ERP features for a specific channel and component.
        
        Args:
            data: Channel data (epochs x time)
            times: Time points
            ch_name: Channel name
            component: ERP component name
            
        Returns:
            Dictionary with ERP features
        """
        features = {}
        
        # Average across epochs
        erp = np.mean(data, axis=0)
        
        # Peak amplitude
        if component in ['N1', 'P2']:
            # For N1 and P2, find the most negative/positive peak
            if component == 'N1':
                peak_idx = np.argmin(erp)
                peak_amplitude = erp[peak_idx]
            else:  # P2
                peak_idx = np.argmax(erp)
                peak_amplitude = erp[peak_idx]
        else:  # LPP - mean amplitude in window
            peak_amplitude = np.mean(erp)
            peak_idx = np.argmax(np.abs(erp))
        
        features[f'{ch_name}_{component}_amplitude'] = peak_amplitude
        features[f'{ch_name}_{component}_latency'] = times[peak_idx]
        
        # Mean amplitude in window
        features[f'{ch_name}_{component}_mean_amplitude'] = np.mean(erp)
        
        # Peak-to-peak amplitude
        features[f'{ch_name}_{component}_peak_to_peak'] = np.max(erp) - np.min(erp)
        
        # Area under curve
        features[f'{ch_name}_{component}_area'] = np.trapz(erp, times)
        
        # Standard deviation
        features[f'{ch_name}_{component}_std'] = np.std(erp)
        
        # Inter-trial variability
        features[f'{ch_name}_{component}_itv'] = np.std(data, axis=0).mean()
        
        return features
    
    def _calculate_global_erp_features(self, epochs: mne.Epochs, time_mask: np.ndarray, 
                                     component: str) -> Dict:
        """Calculate global ERP features.
        
        Args:
            epochs: Epochs object
            time_mask: Time window mask
            component: ERP component name
            
        Returns:
            Dictionary with global features
        """
        features = {}
        
        # Get data in time window
        data = epochs.get_data()[:, :, time_mask]
        
        # Global mean amplitude
        features[f'global_{component}_mean_amplitude'] = np.mean(data)
        
        # Global peak amplitude
        if component in ['N1', 'P2']:
            if component == 'N1':
                features[f'global_{component}_peak_amplitude'] = np.min(data)
            else:
                features[f'global_{component}_peak_amplitude'] = np.max(data)
        else:
            features[f'global_{component}_peak_amplitude'] = np.mean(data)
        
        # Topographic features
        if data.shape[1] > 1:  # Multiple channels
            # Anterior-posterior gradient
            anterior_channels = [i for i, ch in enumerate(epochs.ch_names) 
                               if any(region in ch for region in ['F', 'AF'])]
            posterior_channels = [i for i, ch in enumerate(epochs.ch_names) 
                                if any(region in ch for region in ['P', 'O'])]
            
            if anterior_channels and posterior_channels:
                anterior_mean = np.mean(data[:, anterior_channels, :])
                posterior_mean = np.mean(data[:, posterior_channels, :])
                features[f'{component}_anterior_posterior_gradient'] = anterior_mean - posterior_mean
        
        return features
    
    def extract_component_specific_features(self, epochs: mne.Epochs, subject: str) -> Dict[str, pd.DataFrame]:
        """Extract component-specific features.
        
        Args:
            epochs: MNE Epochs object
            subject: Subject ID
            
        Returns:
            Dictionary mapping component names to feature DataFrames
        """
        component_features = {}
        
        for component in self.components:
            # Create epochs for this component
            component_epochs = epochs.copy()
            
            # Extract features
            features = self.extract_features(component_epochs, subject)
            component_features[component] = features
        
        return component_features
    
    def calculate_difference_waves(self, epochs: mne.Epochs, condition1: int, 
                                 condition2: int, subject: str) -> pd.DataFrame:
        """Calculate difference wave features between two conditions.
        
        Args:
            epochs: MNE Epochs object
            condition1: First condition event type
            condition2: Second condition event type
            subject: Subject ID
            
        Returns:
            DataFrame with difference wave features
        """
        logger.info(f"Calculating difference wave for subject {subject}")
        
        # Get epochs for each condition
        epochs1 = epochs[condition1]
        epochs2 = epochs[condition2]
        
        if len(epochs1) == 0 or len(epochs2) == 0:
            return pd.DataFrame()
        
        # Calculate difference wave
        erp1 = epochs1.average()
        erp2 = epochs2.average()
        diff_wave = erp1 - erp2
        
        features = {}
        
        # Extract features from difference wave
        for component in self.components:
            time_window = self.time_windows[component]
            tmin, tmax = time_window
            
            # Get time indices
            time_mask = (diff_wave.times >= tmin) & (diff_wave.times <= tmax)
            times = diff_wave.times[time_mask]
            
            # Extract features for each channel
            for ch_name in self.channels:
                if ch_name not in diff_wave.ch_names:
                    continue
                    
                ch_idx = diff_wave.ch_names.index(ch_name)
                ch_data = diff_wave.data[ch_idx, time_mask]
                
                # Calculate difference wave features
                ch_features = self._calculate_erp_features(
                    ch_data.reshape(1, -1), times, ch_name, f"{component}_diff"
                )
                features.update(ch_features)
        
        # Add metadata
        features['subject'] = subject
        features['condition1'] = condition1
        features['condition2'] = condition2
        
        return pd.DataFrame([features])
    
    def save_features(self, features: pd.DataFrame, subject: str, task: str, output_dir: str):
        """Save extracted ERP features.
        
        Args:
            features: Features DataFrame
            subject: Subject ID
            task: Task name
            output_dir: Output directory
        """
        output_path = Path(output_dir) / f"sub-{subject}_task-{task}_erp_features.csv"
        features.to_csv(output_path, index=False)
        logger.info(f"Saved ERP features: {output_path}")
    
    def plot_erp_components(self, epochs: mne.Epochs, subject: str, output_dir: str):
        """Plot ERP components for visualization.
        
        Args:
            epochs: MNE Epochs object
            subject: Subject ID
            output_dir: Output directory
        """
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, axes = plt.subplots(len(self.components), 1, figsize=(12, 8))
        if len(self.components) == 1:
            axes = [axes]
        
        # Plot each component
        for i, component in enumerate(self.components):
            time_window = self.time_windows[component]
            tmin, tmax = time_window
            
            # Average across epochs
            evoked = epochs.average()
            
            # Plot for each channel
            for ch_name in self.channels:
                if ch_name in evoked.ch_names:
                    ch_idx = evoked.ch_names.index(ch_name)
                    axes[i].plot(evoked.times, evoked.data[ch_idx], label=ch_name)
            
            axes[i].set_title(f'{component} Component')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Amplitude (μV)')
            axes[i].axvspan(tmin, tmax, alpha=0.3, color='gray')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / f"sub-{subject}_erp_components.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved ERP plot: {output_path}")

