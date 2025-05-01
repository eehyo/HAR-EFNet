import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .base import EncoderBase

# TODO: regressor_type linear 또는 mlp 선택 가능하도록 수정

class ConvBlock(nn.Module):
    """
    A single 1D convolution block
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1, dropout_rate: float = 0.3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2  # 동일한 출력 크기 유지
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class CNNEncoder(EncoderBase):
    """
    CNN-based encoder for ECDF feature regression
    Uses multiple convolution blocks and a regression head
    """
    def __init__(self, args: dict):
        super(CNNEncoder, self).__init__(args)
        
        self.dropout_rate = args.get('dropout_rate', 0.3)
        self.encoder_type = 'cnn'

        if isinstance(self.output_size, tuple) and len(self.output_size) == 2:
            self.axis_dim, self.feat_per_axis = self.output_size
            self.flat_output_size = self.axis_dim * self.feat_per_axis  # 3 * 78 = 234
        else:
            raise ValueError(f"Expected output_size to be a tuple (3, 78), got {self.output_size}")
        
        self.conv_blocks = nn.ModuleList([
            ConvBlock(
                in_channels=self.input_channels, 
                out_channels=64, 
                kernel_size=3, 
                dropout_rate=self.dropout_rate
            ),
            ConvBlock(
                in_channels=64, 
                out_channels=128, 
                kernel_size=5, 
                dropout_rate=self.dropout_rate
            ),
            ConvBlock(
                in_channels=128, 
                out_channels=256, 
                kernel_size=7, 
                dropout_rate=self.dropout_rate
            )
        ])
        
        # Max pooling
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(256, 512),  
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, self.flat_output_size)
        )
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings without applying regression head
        
        Args:
            x: Input data [batch_size, window_size, input_channels]
            
        Returns:
            Extracted feature embeddings [batch_size, 256]
        """
        # Rearrange input to match Conv1d requirement: [B, C, T]
        # (128, 168, 9) -> (128, 9, 168)
        x = x.permute(0, 2, 1)
        
        # Pass through convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Global pooling
        x = self.max_pool(x).squeeze(-1)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN encoder
        
        Args:
            x: Input data [batch_size, window_size, input_channels]
            
        Returns:
            encoder output [batch_size, 3, 78] - ECDF feature predictions
        """
        # Get embeddings from the feature extractor
        features = self.get_embedding(x)
        
        # Apply regression head to predict ECDF features (flat)
        x = self.regressor(features)
        
        # Reshape from [batch_size, 234] to [batch_size, 3, 78]
        batch_size = x.size(0)
        x = x.view(batch_size, self.axis_dim, self.feat_per_axis)
        
        return x 