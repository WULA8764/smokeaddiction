"""Functional connectivity feature extraction for EEG analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

import mne
from scipy import signal
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
import networkx as nx

logger = logging.getLogger(__name__)


class ConnectivityFeatureExtractor:
    """Extract functional connectivity features from EEG data."""
    
    def __init__(self, config: Dict):
        """Initialize the connectivity feature extractor.
        
        Args:
            config: Configuration dictionary with connectivity parameters
        """
        self.config = config
        self.method = config.get('method', 'plv')
        self.bands = config.get('bands', ['theta', 'alpha', 'beta'])
        self.band_freqs = {
            'delta': [0.5, 4.0],
            'theta': [4.0, 8.0],
            'alpha': [8.0, 13.0],
            'beta': [13.0, 30.0],
            'gamma': [30.0, 100.0]
        }
        self.window_length = config.get('window_length', 1.0)
        self.overlap = config.get('overlap', 0.5)
        self.threshold = config.get('threshold', 0.5)
        
    def extract_features(self, raw: mne.io.Raw, subject: str) -> pd.DataFrame:
        """Extract connectivity features from raw EEG data.
        
        Args:
            raw: MNE Raw object
            subject: Subject ID
            
        Returns:
            DataFrame with connectivity features
        """
        logger.info(f"Extracting connectivity features for subject {subject}")
        
        features = {}
        
        # Extract features for each frequency band
        for band in self.bands:
            band_features = self._extract_band_connectivity(raw, band, subject)
            features.update(band_features)
        
        # Global connectivity features
        global_features = self._extract_global_connectivity(raw, subject)
        features.update(global_features)
        
        # Add subject information
        features['subject'] = subject
        
        return pd.DataFrame([features])
    
    def _extract_band_connectivity(self, raw: mne.io.Raw, band: str, subject: str) -> Dict:
        """Extract connectivity features for a specific frequency band.
        
        Args:
            raw: MNE Raw object
            band: Frequency band name
            subject: Subject ID
            
        Returns:
            Dictionary with band-specific connectivity features
        """
        features = {}
        
        # Get frequency range
        low_freq, high_freq = self.band_freqs[band]
        
        # Filter data for the band
        filtered_data = self._bandpass_filter(raw, low_freq, high_freq)
        
        # Calculate connectivity matrix
        connectivity_matrix = self._calculate_connectivity_matrix(filtered_data)
        
        # Extract features from connectivity matrix
        matrix_features = self._extract_matrix_features(connectivity_matrix, band)
        features.update(matrix_features)
        
        # Network features
        network_features = self._extract_network_features(connectivity_matrix, band)
        features.update(network_features)
        
        return features
    
    def _bandpass_filter(self, raw: mne.io.Raw, low_freq: float, high_freq: float) -> np.ndarray:
        """Apply bandpass filter to raw data.
        
        Args:
            raw: MNE Raw object
            low_freq: Low frequency cutoff
            high_freq: High frequency cutoff
            
        Returns:
            Filtered data
        """
        sfreq = raw.info['sfreq']
        nyquist = sfreq / 2
        
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Design filter
        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        
        # Apply filter
        data = raw.get_data()
        filtered_data = signal.filtfilt(b, a, data, axis=1)
        
        return filtered_data
    
    def _calculate_connectivity_matrix(self, data: np.ndarray) -> np.ndarray:
        """Calculate connectivity matrix using specified method.
        
        Args:
            data: Filtered EEG data
            
        Returns:
            Connectivity matrix
        """
        n_channels = data.shape[0]
        connectivity_matrix = np.zeros((n_channels, n_channels))
        
        if self.method == 'plv':
            connectivity_matrix = self._calculate_plv(data)
        elif self.method == 'correlation':
            connectivity_matrix = self._calculate_correlation(data)
        elif self.method == 'coherence':
            connectivity_matrix = self._calculate_coherence(data)
        elif self.method == 'mutual_info':
            connectivity_matrix = self._calculate_mutual_information(data)
        else:
            raise ValueError(f"Unsupported connectivity method: {self.method}")
        
        return connectivity_matrix
    
    def _calculate_plv(self, data: np.ndarray) -> np.ndarray:
        """Calculate Phase Locking Value (PLV).
        
        Args:
            data: Filtered EEG data
            
        Returns:
            PLV connectivity matrix
        """
        n_channels = data.shape[0]
        connectivity_matrix = np.zeros((n_channels, n_channels))
        
        # Calculate instantaneous phase using Hilbert transform
        analytic_signal = signal.hilbert(data, axis=1)
        phase = np.angle(analytic_signal)
        
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # Calculate phase difference
                phase_diff = phase[i, :] - phase[j, :]
                
                # Calculate PLV
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                connectivity_matrix[i, j] = plv
                connectivity_matrix[j, i] = plv
        
        return connectivity_matrix
    
    def _calculate_correlation(self, data: np.ndarray) -> np.ndarray:
        """Calculate correlation-based connectivity.
        
        Args:
            data: Filtered EEG data
            
        Returns:
            Correlation connectivity matrix
        """
        n_channels = data.shape[0]
        connectivity_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                corr, _ = pearsonr(data[i, :], data[j, :])
                connectivity_matrix[i, j] = np.abs(corr)
                connectivity_matrix[j, i] = np.abs(corr)
        
        return connectivity_matrix
    
    def _calculate_coherence(self, data: np.ndarray) -> np.ndarray:
        """Calculate coherence-based connectivity.
        
        Args:
            data: Filtered EEG data
            
        Returns:
            Coherence connectivity matrix
        """
        n_channels = data.shape[0]
        connectivity_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # Calculate cross-spectral density
                freqs, csd = signal.csd(data[i, :], data[j, :], fs=1000, nperseg=256)
                
                # Calculate power spectral densities
                _, psd1 = signal.welch(data[i, :], fs=1000, nperseg=256)
                _, psd2 = signal.welch(data[j, :], fs=1000, nperseg=256)
                
                # Calculate coherence
                coherence = np.abs(csd) ** 2 / (psd1 * psd2)
                connectivity_matrix[i, j] = np.mean(coherence)
                connectivity_matrix[j, i] = np.mean(coherence)
        
        return connectivity_matrix
    
    def _calculate_mutual_information(self, data: np.ndarray) -> np.ndarray:
        """Calculate mutual information-based connectivity.
        
        Args:
            data: Filtered EEG data
            
        Returns:
            Mutual information connectivity matrix
        """
        n_channels = data.shape[0]
        connectivity_matrix = np.zeros((n_channels, n_channels))
        
        # Discretize data for mutual information calculation
        n_bins = 20
        for i in range(n_channels):
            data[i, :] = np.digitize(data[i, :], bins=np.linspace(data[i, :].min(), data[i, :].max(), n_bins))
        
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                mi = mutual_info_score(data[i, :], data[j, :])
                connectivity_matrix[i, j] = mi
                connectivity_matrix[j, i] = mi
        
        return connectivity_matrix
    
    def _extract_matrix_features(self, connectivity_matrix: np.ndarray, band: str) -> Dict:
        """Extract features from connectivity matrix.
        
        Args:
            connectivity_matrix: Connectivity matrix
            band: Frequency band name
            
        Returns:
            Dictionary with matrix features
        """
        features = {}
        
        # Basic statistics
        features[f'{band}_mean_connectivity'] = np.mean(connectivity_matrix)
        features[f'{band}_std_connectivity'] = np.std(connectivity_matrix)
        features[f'{band}_max_connectivity'] = np.max(connectivity_matrix)
        features[f'{band}_min_connectivity'] = np.min(connectivity_matrix)
        
        # Strength features
        node_strengths = np.sum(connectivity_matrix, axis=1)
        features[f'{band}_mean_node_strength'] = np.mean(node_strengths)
        features[f'{band}_std_node_strength'] = np.std(node_strengths)
        features[f'{band}_max_node_strength'] = np.max(node_strengths)
        
        # Density
        n_edges = np.sum(connectivity_matrix > self.threshold)
        n_possible_edges = connectivity_matrix.shape[0] * (connectivity_matrix.shape[0] - 1) / 2
        features[f'{band}_density'] = n_edges / n_possible_edges
        
        # Clustering coefficient
        features[f'{band}_clustering_coefficient'] = self._calculate_clustering_coefficient(connectivity_matrix)
        
        return features
    
    def _extract_network_features(self, connectivity_matrix: np.ndarray, band: str) -> Dict:
        """Extract network topology features.
        
        Args:
            connectivity_matrix: Connectivity matrix
            band: Frequency band name
            
        Returns:
            Dictionary with network features
        """
        features = {}
        
        # Create network graph
        G = nx.from_numpy_array(connectivity_matrix)
        
        # Remove self-loops
        G.remove_edges_from(nx.selfloop_edges(G))
        
        # Apply threshold
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < self.threshold]
        G.remove_edges_from(edges_to_remove)
        
        if len(G.nodes()) > 0:
            # Network metrics
            features[f'{band}_average_clustering'] = nx.average_clustering(G)
            features[f'{band}_average_shortest_path'] = nx.average_shortest_path_length(G) if nx.is_connected(G) else np.nan
            features[f'{band}_diameter'] = nx.diameter(G) if nx.is_connected(G) else np.nan
            features[f'{band}_density'] = nx.density(G)
            features[f'{band}_number_of_edges'] = G.number_of_edges()
            features[f'{band}_number_of_nodes'] = G.number_of_nodes()
            
            # Centrality measures
            if len(G.nodes()) > 1:
                degree_centrality = nx.degree_centrality(G)
                features[f'{band}_mean_degree_centrality'] = np.mean(list(degree_centrality.values()))
                features[f'{band}_std_degree_centrality'] = np.std(list(degree_centrality.values()))
                
                betweenness_centrality = nx.betweenness_centrality(G)
                features[f'{band}_mean_betweenness_centrality'] = np.mean(list(betweenness_centrality.values()))
                features[f'{band}_std_betweenness_centrality'] = np.std(list(betweenness_centrality.values()))
            else:
                features[f'{band}_mean_degree_centrality'] = 0.0
                features[f'{band}_std_degree_centrality'] = 0.0
                features[f'{band}_mean_betweenness_centrality'] = 0.0
                features[f'{band}_std_betweenness_centrality'] = 0.0
        else:
            # Empty graph
            features[f'{band}_average_clustering'] = 0.0
            features[f'{band}_average_shortest_path'] = np.nan
            features[f'{band}_diameter'] = np.nan
            features[f'{band}_density'] = 0.0
            features[f'{band}_number_of_edges'] = 0
            features[f'{band}_number_of_nodes'] = 0
            features[f'{band}_mean_degree_centrality'] = 0.0
            features[f'{band}_std_degree_centrality'] = 0.0
            features[f'{band}_mean_betweenness_centrality'] = 0.0
            features[f'{band}_std_betweenness_centrality'] = 0.0
        
        return features
    
    def _calculate_clustering_coefficient(self, connectivity_matrix: np.ndarray) -> float:
        """Calculate clustering coefficient.
        
        Args:
            connectivity_matrix: Connectivity matrix
            
        Returns:
            Clustering coefficient
        """
        n_nodes = connectivity_matrix.shape[0]
        total_clustering = 0.0
        valid_nodes = 0
        
        for i in range(n_nodes):
            # Find neighbors
            neighbors = np.where(connectivity_matrix[i, :] > self.threshold)[0]
            neighbors = neighbors[neighbors != i]  # Remove self
            
            if len(neighbors) >= 2:
                # Count triangles
                triangles = 0
                possible_triangles = len(neighbors) * (len(neighbors) - 1) / 2
                
                for j in range(len(neighbors)):
                    for k in range(j + 1, len(neighbors)):
                        if connectivity_matrix[neighbors[j], neighbors[k]] > self.threshold:
                            triangles += 1
                
                if possible_triangles > 0:
                    clustering = triangles / possible_triangles
                    total_clustering += clustering
                    valid_nodes += 1
        
        return total_clustering / valid_nodes if valid_nodes > 0 else 0.0
    
    def _extract_global_connectivity(self, raw: mne.io.Raw, subject: str) -> Dict:
        """Extract global connectivity features.
        
        Args:
            raw: MNE Raw object
            subject: Subject ID
            
        Returns:
            Dictionary with global connectivity features
        """
        features = {}
        
        # Cross-frequency coupling
        for band1 in self.bands:
            for band2 in self.bands:
                if band1 != band2:
                    coupling = self._calculate_cross_frequency_coupling(raw, band1, band2)
                    features[f'{band1}_{band2}_coupling'] = coupling
        
        # Global synchronization index
        features['global_synchronization'] = self._calculate_global_synchronization(raw)
        
        return features
    
    def _calculate_cross_frequency_coupling(self, raw: mne.io.Raw, band1: str, band2: str) -> float:
        """Calculate cross-frequency coupling.
        
        Args:
            raw: MNE Raw object
            band1: First frequency band
            band2: Second frequency band
            
        Returns:
            Cross-frequency coupling value
        """
        # Get frequency ranges
        low1, high1 = self.band_freqs[band1]
        low2, high2 = self.band_freqs[band2]
        
        # Filter data for both bands
        data1 = self._bandpass_filter(raw, low1, high1)
        data2 = self._bandpass_filter(raw, low2, high2)
        
        # Calculate phase coupling
        analytic1 = signal.hilbert(data1, axis=1)
        analytic2 = signal.hilbert(data2, axis=1)
        
        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)
        
        # Calculate phase coupling across all channels
        coupling_values = []
        for i in range(data1.shape[0]):
            for j in range(data2.shape[0]):
                if i != j:
                    phase_diff = phase1[i, :] - phase2[j, :]
                    coupling = np.abs(np.mean(np.exp(1j * phase_diff)))
                    coupling_values.append(coupling)
        
        return np.mean(coupling_values) if coupling_values else 0.0
    
    def _calculate_global_synchronization(self, raw: mne.io.Raw) -> float:
        """Calculate global synchronization index.
        
        Args:
            raw: MNE Raw object
            
        Returns:
            Global synchronization value
        """
        data = raw.get_data()
        
        # Calculate instantaneous phase
        analytic = signal.hilbert(data, axis=1)
        phase = np.angle(analytic)
        
        # Calculate global phase synchronization
        n_channels = data.shape[0]
        synchronization_values = []
        
        for t in range(data.shape[1]):
            phases_at_t = phase[:, t]
            synchronization = np.abs(np.mean(np.exp(1j * phases_at_t)))
            synchronization_values.append(synchronization)
        
        return np.mean(synchronization_values)
    
    def extract_features_from_epochs(self, epochs: mne.Epochs, subject: str) -> pd.DataFrame:
        """Extract connectivity features from epochs.
        
        Args:
            epochs: MNE Epochs object
            subject: Subject ID
            
        Returns:
            DataFrame with connectivity features per epoch
        """
        logger.info(f"Extracting connectivity features from epochs for subject {subject}")
        
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
        """Save extracted connectivity features.
        
        Args:
            features: Features DataFrame
            subject: Subject ID
            task: Task name
            output_dir: Output directory
        """
        output_path = Path(output_dir) / f"sub-{subject}_task-{task}_connectivity_features.csv"
        features.to_csv(output_path, index=False)
        logger.info(f"Saved connectivity features: {output_path}")
    
    def plot_connectivity_matrix(self, connectivity_matrix: np.ndarray, subject: str, 
                               band: str, output_dir: str):
        """Plot connectivity matrix.
        
        Args:
            connectivity_matrix: Connectivity matrix
            subject: Subject ID
            band: Frequency band
            output_dir: Output directory
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(connectivity_matrix, cmap='viridis', square=True)
        plt.title(f'Connectivity Matrix - {band} band - Subject {subject}')
        plt.xlabel('Channels')
        plt.ylabel('Channels')
        
        # Save plot
        output_path = Path(output_dir) / f"sub-{subject}_{band}_connectivity_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved connectivity matrix plot: {output_path}")

