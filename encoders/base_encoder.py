import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union

from .base import EncoderBase

class BaseEncoder(EncoderBase):
    """
    Simple baseline encoder implementation
    """
    def __init__(self, args: Dict[str, Any]):
        super(BaseEncoder, self).__init__(args)
        
        self.encoder_type = 'base_encoder'
        
        # Encoding through simple linear layers
        self.flatten = nn.Flatten()
        
        # Calculate input dimension: window_size * channels
        input_dim = self.window_size * self.input_channels
        
        # Use single simple linear layer
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the baseline encoder
        
        Args:
            x: Input data [batch_size, window_size, input_channels]
            
        Returns:
            encoder output [batch_size, output_size] - ECDF feature predictions
        """
        # [batch_size, window_size, input_channels] -> [batch_size, window_size * input_channels]
        x = self.flatten(x)
        
        # Pass through linear layers
        x = self.linear(x)
        
        return x 