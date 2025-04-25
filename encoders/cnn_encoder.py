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

        # Enforce expected output size (234-dimensional ECDF feature)
        assert self.output_size == 234, \
            f"Expected output_size = 234, got {self.output_size}"
        
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
            nn.Linear(256, self.output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN encoder
        
        Args:
            x: Input data [batch_size, window_size, input_channels]
            
        Returns:
            encoder output [batch_size, output_size] - ECDF feature predictions
        """
        # Rearrange input to match Conv1d requirement: [B, C, T]
        # (128, 168, 9) -> (128, 9, 168)
        x = x.permute(0, 2, 1)

        # dim mismatch
        # # 4차원 입력 [B, 1, T, C] -> [B, C, T] 변환
        # if len(x.shape) == 4:
        #     # [batch_size, 1, window_size, channels] -> [batch_size, channels, window_size]
        #     x = x.squeeze(1).permute(0, 2, 1)
        # else:
        #     # 3차원 입력 [B, T, C] -> [B, C, T]
        #     x = x.permute(0, 2, 1)
        
        # Pass through convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Global pooling
        x = self.max_pool(x).squeeze(-1)
        
        # Regression head
        x = self.regressor(x)
        
        return x 