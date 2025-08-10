"""Microstate feature extraction for EEG analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

import mne
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import entropy

logger = logging.getLogger(__name__)


class MicrostateFeatureExtractor:
    """Extract microstate features from EEG data."""
    
    def __init__(self, config: Dict):
        """Initialize the microstate feature extractor.
        
        Args:
            config: Configuration dictionary with microstate parameters
        """
        self.config = config
        self.n_states = config.get('n_states', 4)
        self.algorithm = config.get('algorithm', 'kmeans')
        self.max_iter = config.get('max_iter', 1000)
        self.random_state = config.get('random_state', 42)
        self.min_segment_length = config.get('min_segment_length', 10)  # samples
        
    def extract_features(self, raw: mne.io.Raw, subject: str) -> pd.DataFrame:
        """Extract microstate features from raw EEG data.
        
        Args:
            raw: MNE Raw object
            subject: Subject ID
            
        Returns:
            DataFrame with microstate features
        """
        logger.info(f"Extracting microstate features for subject {subject}")
        
        # Get data
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        
        # Preprocess data for microstate analysis
        processed_data = self._preprocess_for_microstates(data, sfreq)
        
        # Perform microstate analysis
        microstates, labels, segments = self._perform_microstate_analysis(processed_data)
        
        # Extract features
        features = self._extract_microstate_features(
            processed_data, microstates, labels, segments, sfreq, subject
        )
        
        return pd.DataFrame([features])
    
    def _preprocess_for_microstates(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """Preprocess data for microstate analysis.
        
        Args:
            data: Raw EEG data
            sfreq: Sampling frequency
            
        Returns:
            Preprocessed data
        """
        # Apply bandpass filter (2-20 Hz for microstates)
        from scipy import signal
        
        nyquist = sfreq / 2
        low_norm = 2.0 / nyquist
        high_norm = 20.0 / nyquist
        
        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=1)
        
        # Normalize data
        normalized_data = (filtered_data - np.mean(filtered_data, axis=1, keepdims=True)) / np.std(filtered_data, axis=1, keepdims=True)
        
        return normalized_data
    
    def _perform_microstate_analysis(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List]:
        """Perform microstate analysis.
        
        Args:
            data: Preprocessed EEG data
            
        Returns:
            Tuple of (microstates, labels, segments)
        """
        # Transpose data for clustering (time x channels)
        data_t = data.T
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(self.n_states * 2, data_t.shape[1]))
        data_pca = pca.fit_transform(data_t)
        
        # Perform clustering
        if self.algorithm == 'kmeans':
            kmeans = KMeans(
                n_clusters=self.n_states,
                max_iter=self.max_iter,
                random_state=self.random_state,
                n_init=10
            )
            labels = kmeans.fit_predict(data_pca)
            microstates = kmeans.cluster_centers_
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # Convert back to original space
        microstates_original = pca.inverse_transform(microstates)
        
        # Find segments
        segments = self._find_segments(labels)
        
        return microstates_original, labels, segments
    
    def _find_segments(self, labels: np.ndarray) -> List[Dict]:
        """Find microstate segments.
        
        Args:
            labels: Microstate labels
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        current_label = labels[0]
        start_idx = 0
        length = 1
        
        for i in range(1, len(labels)):
            if labels[i] == current_label:
                length += 1
            else:
                if length >= self.min_segment_length:
                    segments.append({
                        'state': current_label,
                        'start': start_idx,
                        'end': i - 1,
                        'length': length
                    })
                current_label = labels[i]
                start_idx = i
                length = 1
        
        # Add last segment
        if length >= self.min_segment_length:
            segments.append({
                'state': current_label,
                'start': start_idx,
                'end': len(labels) - 1,
                'length': length
            })
        
        return segments
    
    def _extract_microstate_features(self, data: np.ndarray, microstates: np.ndarray, 
                                   labels: np.ndarray, segments: List[Dict], 
                                   sfreq: float, subject: str) -> Dict:
        """Extract microstate features.
        
        Args:
            data: Preprocessed EEG data
            microstates: Microstate templates
            labels: Microstate labels
            segments: Microstate segments
            sfreq: Sampling frequency
            subject: Subject ID
            
        Returns:
            Dictionary with microstate features
        """
        features = {}
        
        # Basic microstate features
        features['subject'] = subject
        features['n_microstates'] = self.n_states
        features['total_segments'] = len(segments)
        
        # Duration features
        segment_lengths = [seg['length'] for seg in segments]
        segment_durations = [length / sfreq for length in segment_lengths]
        
        features['mean_segment_duration'] = np.mean(segment_durations)
        features['std_segment_duration'] = np.std(segment_durations)
        features['min_segment_duration'] = np.min(segment_durations)
        features['max_segment_duration'] = np.max(segment_durations)
        
        # Coverage features
        total_samples = len(labels)
        for state in range(self.n_states):
            state_samples = np.sum(labels == state)
            coverage = state_samples / total_samples
            features[f'state_{state}_coverage'] = coverage
        
        # Occurrence frequency
        for state in range(self.n_states):
            state_segments = [seg for seg in segments if seg['state'] == state]
            frequency = len(state_segments) / (total_samples / sfreq)  # per second
            features[f'state_{state}_frequency'] = frequency
        
        # Transition features
        transition_matrix = self._calculate_transition_matrix(labels, self.n_states)
        features.update(self._extract_transition_features(transition_matrix))
        
        # Global field power features
        gfp_features = self._extract_gfp_features(data, labels, segments)
        features.update(gfp_features)
        
        # Topographic features
        topo_features = self._extract_topographic_features(microstates)
        features.update(topo_features)
        
        return features
    
    def _calculate_transition_matrix(self, labels: np.ndarray, n_states: int) -> np.ndarray:
        """Calculate transition matrix between microstates.
        
        Args:
            labels: Microstate labels
            n_states: Number of microstates
            
        Returns:
            Transition matrix
        """
        transition_matrix = np.zeros((n_states, n_states))
        
        for i in range(len(labels) - 1):
            current_state = labels[i]
            next_state = labels[i + 1]
            transition_matrix[current_state, next_state] += 1
        
        # Normalize
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        transition_matrix = np.nan_to_num(transition_matrix, 0.0)
        
        return transition_matrix
    
    def _extract_transition_features(self, transition_matrix: np.ndarray) -> Dict:
        """Extract features from transition matrix.
        
        Args:
            transition_matrix: Microstate transition matrix
            
        Returns:
            Dictionary with transition features
        """
        features = {}
        
        # Transition entropy
        for i in range(transition_matrix.shape[0]):
            row = transition_matrix[i, :]
            row = row[row > 0]  # Remove zero probabilities
            if len(row) > 0:
                entropy_val = entropy(row)
                features[f'state_{i}_transition_entropy'] = entropy_val
            else:
                features[f'state_{i}_transition_entropy'] = 0.0
        
        # Global transition entropy
        flat_transitions = transition_matrix.flatten()
        flat_transitions = flat_transitions[flat_transitions > 0]
        if len(flat_transitions) > 0:
            features['global_transition_entropy'] = entropy(flat_transitions)
        else:
            features['global_transition_entropy'] = 0.0
        
        # Self-transition probability
        for i in range(transition_matrix.shape[0]):
            features[f'state_{i}_self_transition'] = transition_matrix[i, i]
        
        return features
    
    def _extract_gfp_features(self, data: np.ndarray, labels: np.ndarray, 
                            segments: List[Dict]) -> Dict:
        """Extract Global Field Power features.
        
        Args:
            data: EEG data
            labels: Microstate labels
            segments: Microstate segments
            
        Returns:
            Dictionary with GFP features
        """
        features = {}
        
        # Calculate GFP for each time point
        gfp = np.std(data, axis=0)
        
        # GFP features for each microstate
        for state in range(self.n_states):
            state_mask = labels == state
            if np.any(state_mask):
                state_gfp = gfp[state_mask]
                features[f'state_{state}_mean_gfp'] = np.mean(state_gfp)
                features[f'state_{state}_std_gfp'] = np.std(state_gfp)
                features[f'state_{state}_max_gfp'] = np.max(state_gfp)
        
        # Overall GFP statistics
        features['mean_gfp'] = np.mean(gfp)
        features['std_gfp'] = np.std(gfp)
        features['max_gfp'] = np.max(gfp)
        
        return features
    
    def _extract_topographic_features(self, microstates: np.ndarray) -> Dict:
        """Extract topographic features from microstate templates.
        
        Args:
            microstates: Microstate templates
            
        Returns:
            Dictionary with topographic features
        """
        features = {}
        
        for i, microstate in enumerate(microstates):
            # Amplitude features
            features[f'state_{i}_mean_amplitude'] = np.mean(microstate)
            features[f'state_{i}_std_amplitude'] = np.std(microstate)
            features[f'state_{i}_max_amplitude'] = np.max(microstate)
            features[f'state_{i}_min_amplitude'] = np.min(microstate)
            
            # Topographic complexity (spatial entropy)
            normalized_state = (microstate - np.min(microstate)) / (np.max(microstate) - np.min(microstate))
            normalized_state = normalized_state / np.sum(normalized_state)
            features[f'state_{i}_topographic_entropy'] = entropy(normalized_state)
        
        return features
    
    def extract_features_from_epochs(self, epochs: mne.Epochs, subject: str) -> pd.DataFrame:
        """Extract microstate features from epochs.
        
        Args:
            epochs: MNE Epochs object
            subject: Subject ID
            
        Returns:
            DataFrame with microstate features per epoch
        """
        logger.info(f"Extracting microstate features from epochs for subject {subject}")
        
        features_list = []
        
        for i, epoch in enumerate(epochs):
            # Convert epoch to raw-like format
            epoch_data = epoch.get_data()[0, :, :]  # Take first epoch
            
            # Create temporary raw object
            temp_raw = mne.io.RawArray(epoch_data, epochs.info)
            
            # Extract features
            epoch_features = self.extract_features(temp_raw, f"{subject}_epoch_{i}")
            features_list.append(epoch_features)
        
        return pd.concat(features_list, ignore_index=True)
    
    def save_features(self, features: pd.DataFrame, subject: str, task: str, output_dir: str):
        """Save extracted microstate features.
        
        Args:
            features: Features DataFrame
            subject: Subject ID
            task: Task name
            output_dir: Output directory
        """
        output_path = Path(output_dir) / f"sub-{subject}_task-{task}_microstate_features.csv"
        features.to_csv(output_path, index=False)
        logger.info(f"Saved microstate features: {output_path}")
    
    def plot_microstates(self, microstates: np.ndarray, subject: str, output_dir: str):
        """Plot microstate templates.
        
        Args:
            microstates: Microstate templates
            subject: Subject ID
            output_dir: Output directory
        """
        import matplotlib.pyplot as plt
        
        n_states = len(microstates)
        fig, axes = plt.subplots(1, n_states, figsize=(4 * n_states, 4))
        
        if n_states == 1:
            axes = [axes]
        
        for i, microstate in enumerate(microstates):
            axes[i].plot(microstate)
            axes[i].set_title(f'Microstate {i}')
            axes[i].set_xlabel('Channel')
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / f"sub-{subject}_microstates.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved microstate plot: {output_path}")

