"""
U-Net for Ghost Imaging Reconstruction
========================================
Adapts the U-Net architecture (Ronneberger et al., MICCAI 2015) for
ghost imaging reconstruction with skip connections.

Maps bucket signals -> initial spatial map -> encoder-decoder with skips.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two conv-BN-ReLU blocks."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)


class GIUNet(nn.Module):
    """U-Net reconstruction for ghost imaging.
    
    Architecture:
        bucket_signals -> FC -> reshape to (C, H, W) -> U-Net encoder-decoder -> image
    
    Args:
        n_measurements: Number of bucket detector measurements.
        image_size: Output image size (28 or 64).
        base_channels: Base channel count for U-Net.
    """
    
    def __init__(
        self,
        n_measurements: int = 98,
        image_size: int = 28,
        base_channels: int = 32,
    ):
        super().__init__()
        self.image_size = image_size
        c = base_channels
        
        # Project bucket signals to initial spatial feature map
        init_channels = c
        self.projection = nn.Sequential(
            nn.Linear(n_measurements, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, init_channels * image_size * image_size),
            nn.ReLU(inplace=True),
        )
        self.init_channels = init_channels
        
        # Encoder
        self.enc1 = DoubleConv(init_channels, c)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(c, c * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(c * 2, c * 4)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)  # concat with enc2
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)  # concat with enc1
        
        # Output
        self.out_conv = nn.Sequential(
            nn.Conv2d(c, 1, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, bucket_signals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bucket_signals: (batch, n_measurements)
        Returns:
            Reconstructed image: (batch, 1, image_size, image_size)
        """
        B = bucket_signals.shape[0]
        
        # Project to spatial domain
        x = self.projection(bucket_signals)
        x = x.view(B, self.init_channels, self.image_size, self.image_size)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        # Bottleneck
        b = self.bottleneck(self.pool2(e2))
        
        # Decoder with skip connections
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.out_conv(d1)
