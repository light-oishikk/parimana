"""
CNN-Based Ghost Imaging Reconstruction
========================================
Inspired by He et al., "Ghost imaging based on deep learning",
Scientific Reports 8, 6469 (2018).

First reshapes bucket signals into a 2D feature map, then applies
convolutional layers for spatial reconstruction.
"""

import torch
import torch.nn as nn
import math


class GICNN(nn.Module):
    """CNN reconstruction network for ghost imaging (He et al. 2018 style).
    
    Maps bucket signals -> intermediate feature map -> conv layers -> image.
    
    Args:
        n_measurements: Number of input bucket measurements.
        image_size: Output image size.
        base_channels: Base number of convolutional filters.
    """
    
    def __init__(
        self,
        n_measurements: int = 98,
        image_size: int = 28,
        base_channels: int = 64,
    ):
        super().__init__()
        self.image_size = image_size
        
        # Determine intermediate spatial size for reshaping
        self.feat_size = 7  # 7x7 intermediate feature map
        feat_pixels = self.feat_size ** 2
        n_feat_channels = max(1, n_measurements // feat_pixels)
        fc_out = n_feat_channels * feat_pixels
        
        self.n_feat_channels = n_feat_channels
        
        # FC layer to reshape measurements into spatial feature map
        self.fc = nn.Sequential(
            nn.Linear(n_measurements, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, fc_out),
            nn.BatchNorm1d(fc_out),
            nn.ReLU(inplace=True),
        )
        
        # Convolutional reconstruction head
        self.conv = nn.Sequential(
            # Upsample 7x7 -> 14x14
            nn.ConvTranspose2d(n_feat_channels, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # Upsample 14x14 -> 28x28
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            
            # Refine
            nn.Conv2d(base_channels // 2, base_channels // 4, 3, padding=1),
            nn.BatchNorm2d(base_channels // 4),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels // 4, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        
        # Adaptive output to handle arbitrary image sizes
        self.adapt = nn.AdaptiveAvgPool2d(image_size)
    
    def forward(self, bucket_signals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bucket_signals: (batch, n_measurements)
        Returns:
            Reconstructed image: (batch, 1, image_size, image_size)
        """
        x = self.fc(bucket_signals)
        x = x.view(-1, self.n_feat_channels, self.feat_size, self.feat_size)
        x = self.conv(x)
        x = self.adapt(x)
        return x
