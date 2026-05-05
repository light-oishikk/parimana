"""
Computational Ghost Imaging Simulator
======================================
Simulates the computational ghost imaging (CGI) process:
1. Generate random illumination patterns (speckle patterns)
2. Illuminate the object with each pattern
3. Record bucket detector signal (total transmitted intensity)
4. Reconstruct using traditional correlation or DL methods

Reference: Shapiro, "Computational ghost imaging", PRA 78, 061802 (2008)
"""

import numpy as np
from typing import Tuple, Optional


class GhostImagingSimulator:
    """Simulates computational ghost imaging measurements.
    
    In CGI, a spatial light modulator (SLM) generates known random patterns.
    The object is illuminated by each pattern, and a single-pixel (bucket)
    detector records the total transmitted/reflected intensity.
    
    Args:
        image_size: Size of the image (assumes square images).
        pattern_type: Type of random illumination pattern ('gaussian', 'binary', 'hadamard').
        noise_level: Standard deviation of Gaussian noise added to bucket signals.
        seed: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        image_size: int = 28,
        pattern_type: str = 'gaussian',
        noise_level: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.image_size = image_size
        self.n_pixels = image_size ** 2
        self.pattern_type = pattern_type
        self.noise_level = noise_level
        self.rng = np.random.RandomState(seed)
    
    def generate_patterns(self, n_patterns: int) -> np.ndarray:
        """Generate random illumination patterns.
        
        Args:
            n_patterns: Number of patterns to generate (M measurements).
            
        Returns:
            Array of shape (n_patterns, image_size, image_size).
        """
        if self.pattern_type == 'gaussian':
            patterns = self.rng.randn(n_patterns, self.image_size, self.image_size)
        elif self.pattern_type == 'binary':
            patterns = self.rng.choice([0, 1], size=(n_patterns, self.image_size, self.image_size)).astype(np.float64)
        elif self.pattern_type == 'hadamard':
            from scipy.linalg import hadamard
            H = hadamard(self.n_pixels)
            if n_patterns > self.n_pixels:
                raise ValueError(f"Hadamard: max {self.n_pixels} patterns for image_size={self.image_size}")
            patterns = H[:n_patterns].reshape(n_patterns, self.image_size, self.image_size).astype(np.float64)
        else:
            raise ValueError(f"Unknown pattern type: {self.pattern_type}")
        
        return patterns
    
    def measure(self, obj: np.ndarray, patterns: np.ndarray) -> np.ndarray:
        """Simulate bucket detector measurements.
        
        The bucket signal for pattern m is: B_m = sum(phi_m * obj) + noise
        
        Args:
            obj: Ground truth object of shape (image_size, image_size), values in [0, 1].
            patterns: Illumination patterns of shape (n_patterns, image_size, image_size).
            
        Returns:
            Bucket detector signals of shape (n_patterns,).
        """
        # Bucket signal = inner product of pattern and object
        bucket_signals = np.sum(patterns * obj[np.newaxis, :, :], axis=(1, 2))
        
        # Add measurement noise
        if self.noise_level > 0:
            bucket_signals += self.rng.randn(len(bucket_signals)) * self.noise_level
        
        return bucket_signals
    
    def traditional_reconstruct(
        self, patterns: np.ndarray, bucket_signals: np.ndarray
    ) -> np.ndarray:
        """Traditional correlation-based ghost imaging reconstruction.
        
        I(r) = (1/M) * sum_m (B_m - <B>) * phi_m(r)
        
        Reference: Erkmen & Shapiro, Adv. Opt. Photon. 2, 405 (2010)
        
        Args:
            patterns: Illumination patterns of shape (M, H, W).
            bucket_signals: Bucket detector signals of shape (M,).
            
        Returns:
            Reconstructed image of shape (H, W).
        """
        M = len(bucket_signals)
        mean_bucket = np.mean(bucket_signals)
        
        # Differential ghost imaging
        fluctuations = bucket_signals - mean_bucket  # (M,)
        reconstruction = np.mean(
            fluctuations[:, np.newaxis, np.newaxis] * patterns, axis=0
        )
        
        return reconstruction
    
    def simulate_full(
        self, obj: np.ndarray, n_measurements: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run a full CGI simulation: generate patterns, measure, reconstruct.
        
        Args:
            obj: Ground truth object of shape (image_size, image_size).
            n_measurements: Number of measurements M.
            
        Returns:
            Tuple of (patterns, bucket_signals, traditional_reconstruction).
        """
        patterns = self.generate_patterns(n_measurements)
        bucket_signals = self.measure(obj, patterns)
        recon = self.traditional_reconstruct(patterns, bucket_signals)
        return patterns, bucket_signals, recon


def compute_sampling_ratio(n_measurements: int, image_size: int) -> float:
    """Compute the sampling ratio M / N^2."""
    return n_measurements / (image_size ** 2)
