import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union

class EncoderBase(nn.Module):
    """
    Base class for all encoder models
    
    This abstract class provides common functionality and interface
    for different encoder implementations.
    """
    def __init__(self, args: Dict[str, Any]):
        """
        Initialize base encoder
        
        Args:
            args: Dictionary containing encoder configuration:
                - input_channels: Number of input channels
                - window_size: Size of the input window
                - output_size: Dimension of the output ECDF features
                - device: Device for computation ('cpu' or 'cuda')
        """
        super(EncoderBase, self).__init__()
        
        self.input_channels = args['input_channels']     
        self.window_size = args['window_size']           
        self.output_size = args['output_size']           # ECDF feature dimension
        self.device = args['device']
        
        self.encoder_type = 'base'
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (to be implemented by subclasses)
        
        Args:
            x: Input data [batch_size, window_size, input_channels]
            
        Returns:
            Encoder output [batch_size, output_size]
            
        Raises:
            NotImplementedError: If subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement forward method") 
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of the encoder's output embedding
        
        Returns:
            Output embedding dimension
        """
        return self.output_size 