import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

from ..base.deepconvlstm_encoder import DeepConvLSTMEncoder


class SimCLRDeepConvLSTMEncoder(nn.Module):
    """
    SimCLR DeepConvLSTM Encoder class
    Add projection head to the existing DeepConvLSTM encoder for contrastive learning
    """
    def __init__(self, args: Dict[str, Any]):
        """
        Initialize SimCLR DeepConvLSTM encoder
        
        Args:
            args: Model configuration parameters (Dict)
        """
        super(SimCLRDeepConvLSTMEncoder, self).__init__()
        
        # Configure base encoder
        self.encoder = DeepConvLSTMEncoder(args)
        
        # LSTM output size (hidden_dim set in DeepConvLSTM)
        self.hidden_size = self.encoder.embedding_dim
        
        # Device configuration
        self.device = args.get('device', 'cpu')
        
        # SimCLR projection head parameters
        self.projection_hidden_dim1 = args.get('projection_hidden_dim1', 256)  
        self.projection_hidden_dim2 = args.get('projection_hidden_dim2', 128) 
        self.projection_dim = args.get('projection_dim', 50) 
        
        # Projection head for SimCLR
        # f(.) -> g(.) projection
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.projection_hidden_dim1),
            nn.ReLU(),
            nn.Linear(self.projection_hidden_dim1, self.projection_hidden_dim2),
            nn.ReLU(),
            nn.Linear(self.projection_hidden_dim2, self.projection_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize weights for projection head
        """
        for layer in self.projection_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SimCLR training
        
        Args:
            x: Input time series data [batch_size, window_size, channels]
            
        Returns:
            Projected features for contrastive learning [batch_size, projection_dim]
        """
        # Get encoder features
        features = self.encoder.extract_features(x)  # [batch_size, hidden_size]
        
        # Apply projection head
        projected = self.projection_head(features)  # [batch_size, projection_dim]
        
        # L2 normalize for contrastive learning
        projected = F.normalize(projected, dim=1)
        
        return projected
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without projection (for downstream tasks)
        
        Args:
            x: Input time series data [batch_size, window_size, channels]
            
        Returns:
            Encoder features [batch_size, hidden_size]
        """
        return self.encoder.extract_features(x)
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of the encoder's feature embedding
        
        Returns:
            Feature embedding dimension
        """
        return self.hidden_size
    
    def freeze_encoder(self):
        """
        Freeze encoder parameters for fine-tuning
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """
        Unfreeze encoder parameters for full fine-tuning
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def freeze_projection_head(self):
        """
        Freeze projection head parameters
        """
        for param in self.projection_head.parameters():
            param.requires_grad = False
    
    def unfreeze_projection_head(self):
        """
        Unfreeze projection head parameters
        """
        for param in self.projection_head.parameters():
            param.requires_grad = True 