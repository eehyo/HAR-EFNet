import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import EncoderBase

# TODO: regressor_type linear 또는 mlp 선택 가능하도록 수정

class ConvBlock(nn.Module):
    """
    A single 1D convolution block
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, dropout_rate=0.3):
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

    def forward(self, x):
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
    def __init__(self, args):
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
    
    def forward(self, x):
        """
        forward pass
        
        Args:
            x: input data [batch_size, window_size, input_channels]
            
        Returns:
            encoder output [batch_size, output_size]
        """
        # 디버깅시 출력 형태 확인
        print(f"Input shape: {x.shape}")

        # Rearrange input to match Conv1d requirement: [B, C, T]
        x = x.permute(0, 2, 1)
        
        print(f"Transformed input shape: {x.shape}")
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        x = self.max_pool(x).squeeze(-1)
        
        x = self.regressor(x)
        
        return x 