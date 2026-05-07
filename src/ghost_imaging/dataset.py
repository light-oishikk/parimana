"""
Ghost Imaging Dataset for PyTorch
==================================
Generates paired (bucket_signals, ground_truth_image) data for training
deep learning reconstruction models.

Uses MNIST as the default object dataset (28x28, matching Lyu et al. 2017).
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from typing import Optional, Tuple

from src.ghost_imaging.simulator import GhostImagingSimulator


class GhostImagingDataset(Dataset):
    """PyTorch dataset for ghost imaging reconstruction.
    
    For each image in the base dataset, simulates CGI measurements
    and returns (bucket_signals, ground_truth_image) pairs.
    
    Args:
        root: Root directory for the base image dataset.
        n_measurements: Number of CGI measurements per image.
        image_size: Image resolution (square).
        pattern_type: Type of illumination patterns.
        noise_level: Measurement noise standard deviation.
        train: If True, use training split.
        seed: Random seed for pattern generation.
        precompute_patterns: If True, generate patterns once and reuse.
    """
    
    def __init__(
        self,
        root: str = './data',
        n_measurements: int = 98,  # ~12.5% of 784 (28x28), matching Lyu et al.
        image_size: int = 28,
        pattern_type: str = 'gaussian',
        noise_level: float = 0.01,
        train: bool = True,
        seed: int = 42,
        precompute_patterns: bool = True,
    ):
        self.n_measurements = n_measurements
        self.image_size = image_size
        self.noise_level = noise_level
        
        # Load MNIST as object images
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self.base_dataset = datasets.MNIST(
            root=root, train=train, download=True, transform=transform
        )
        
        # Initialize simulator
        self.simulator = GhostImagingSimulator(
            image_size=image_size,
            pattern_type=pattern_type,
            noise_level=noise_level,
            seed=seed,
        )
        
        # Pre-generate patterns (shared across all images, as in real CGI)
        self.precompute_patterns = precompute_patterns
        if precompute_patterns:
            self.patterns = self.simulator.generate_patterns(n_measurements)
            self.patterns_tensor = torch.FloatTensor(self.patterns)
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            bucket_signals: Tensor of shape (n_measurements,) — the CGI measurements.
            ground_truth: Tensor of shape (1, image_size, image_size) — the target image.
        """
        image, _ = self.base_dataset[idx]  # (1, H, W)
        obj = image.squeeze().numpy()  # (H, W)
        
        if self.precompute_patterns:
            patterns = self.patterns
        else:
            patterns = self.simulator.generate_patterns(self.n_measurements)
        
        # Simulate bucket detector measurements
        bucket_signals = self.simulator.measure(obj, patterns)
        
        bucket_tensor = torch.FloatTensor(bucket_signals)
        ground_truth = image  # (1, H, W)
        
        return bucket_tensor, ground_truth


class PrecomputedGIDataset(Dataset):
    """Memory-efficient dataset that pre-computes all measurements.
    
    Useful for faster training when dataset fits in memory.
    
    Args:
        root: Root directory for base dataset.
        n_measurements: Number of CGI measurements.
        image_size: Image size.
        noise_level: Measurement noise.
        train: Training or test split.
        max_samples: Maximum samples to precompute (None = all).
        seed: Random seed.
    """
    
    def __init__(
        self,
        root: str = './data',
        n_measurements: int = 98,
        image_size: int = 28,
        noise_level: float = 0.01,
        train: bool = True,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        base = datasets.MNIST(root=root, train=train, download=True, transform=transform)
        
        n_samples = len(base) if max_samples is None else min(max_samples, len(base))
        
        simulator = GhostImagingSimulator(
            image_size=image_size, noise_level=noise_level, seed=seed
        )
        patterns = simulator.generate_patterns(n_measurements)
        
        # Precompute all measurements
        self.bucket_signals = torch.zeros(n_samples, n_measurements)
        self.ground_truths = torch.zeros(n_samples, 1, image_size, image_size)
        
        for i in range(n_samples):
            img, _ = base[i]
            obj = img.squeeze().numpy()
            self.bucket_signals[i] = torch.FloatTensor(simulator.measure(obj, patterns))
            self.ground_truths[i] = img
        
        self.patterns = torch.FloatTensor(patterns)
    
    def __len__(self) -> int:
        return len(self.ground_truths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.bucket_signals[idx], self.ground_truths[idx]
