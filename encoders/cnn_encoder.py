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
        
        # 각 ECDF 특성에 대한 독립적인 FC 레이어 체인 생성 (기존 regressor 구조와 동일)
        self.feature_predictors = nn.ModuleList()
        for i in range(self.feat_per_axis):
            # 기존 regressor 구조와 동일하게 구성 (512->256->3)
            fc1 = nn.Linear(256, 512)
            bn1 = nn.BatchNorm1d(512)
            fc2 = nn.Linear(512, 256)
            bn2 = nn.BatchNorm1d(256)
            fc3 = nn.Linear(256, self.axis_dim)
            
            self.feature_predictors.append(nn.ModuleList([fc1, bn1, fc2, bn2, fc3]))
    
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
        
        # 각 특성별로 독립적인 FC 레이어 체인 적용
        outputs = []
        for i in range(self.feat_per_axis):
            # 레이어 가져오기
            fc1, bn1, fc2, bn2, fc3 = self.feature_predictors[i]
            
            # 첫 번째 블록
            x_feature = fc1(features)
            x_feature = bn1(x_feature)
            x_feature = F.relu(x_feature)
            x_feature = F.dropout(x_feature, p=self.dropout_rate, training=self.training)
            
            # 두 번째 블록
            x_feature = fc2(x_feature)
            x_feature = bn2(x_feature)
            x_feature = F.relu(x_feature)
            x_feature = F.dropout(x_feature, p=self.dropout_rate, training=self.training)
            
            # 출력 레이어
            x_feature = fc3(x_feature)
            
            outputs.append(x_feature)
        
        # 모든 출력을 결합하여 [batch_size, 3, 78] 형태로 생성
        x = torch.stack(outputs, dim=2)  # [batch_size, 3, 78]
        
        return x
    
    def calculate_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss for each ECDF feature independently
        
        Args:
            predictions: Predicted ECDF features [batch_size, 3, 78]
            targets: Target ECDF features [batch_size, 3, 78]
            
        Returns:
            Total loss and per-feature losses
        """
        total_loss = 0
        feature_losses = []
        
        for i in range(self.feat_per_axis):
            feature_pred = predictions[:, :, i]  # [batch_size, 3]
            feature_target = targets[:, :, i]    # [batch_size, 3]
            feature_loss = F.mse_loss(feature_pred, feature_target)
            feature_losses.append(feature_loss)
            total_loss += feature_loss
        
        return total_loss, feature_losses 