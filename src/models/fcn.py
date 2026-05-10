"""
Fully Connected Network for Ghost Imaging Reconstruction
==========================================================
Reproduces the approach of Lyu et al., "Deep-learning-based ghost imaging",
Scientific Reports 7, 17865 (2017).

Maps bucket detector signals directly to reconstructed images via
fully connected layers.
"""

import torch
import torch.nn as nn


class FCN(nn.Module):
    """Fully connected reconstruction network (Lyu et al. 2017 style).
    
    Architecture: bucket_signals -> FC layers -> reconstructed image
    
    Args:
        n_measurements: Number of input bucket measurements (M).
        image_size: Output image size (assumes square).
        hidden_dims: List of hidden layer dimensions.
        dropout: Dropout rate for regularization.
    """
    
    def __init__(
        self,
        n_measurements: int = 98,
        image_size: int = 28,
        hidden_dims: list = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.image_size = image_size
        n_pixels = image_size ** 2
        
        if hidden_dims is None:
            hidden_dims = [512, 1024, 1024]
        
        layers = []
        in_dim = n_measurements
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, n_pixels))
        layers.append(nn.Sigmoid())  # Output in [0, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, bucket_signals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bucket_signals: (batch, n_measurements)
        Returns:
            Reconstructed image: (batch, 1, image_size, image_size)
        """
        out = self.network(bucket_signals)
        return out.view(-1, 1, self.image_size, self.image_size)
