import torch
from torch import nn
from ..base.deepconvlstm_attn_encoder import DeepConvLSTMAttnEncoder
from .decoder import MaskedReconstruction


class MaskedDeepConvLSTMAttnEncoder(nn.Module):
    """
    Masked Reconstruction wrapper for DeepConvLSTMAttn encoder
    """
    def __init__(self, config):
        super(MaskedDeepConvLSTMAttnEncoder, self).__init__()
        
        # Create base encoder (without ECDF prediction heads)
        self.base_encoder = DeepConvLSTMAttnEncoder(config)
        
        # Store configuration
        self.window_size = config.get('window_size', 168)
        self.input_channels = config.get('input_channels', 9)
        self.device = config.get('device', 'cpu')
        
        # Create masked reconstruction model
        self.masked_model = MaskedReconstruction(
            encoder=self.base_encoder,
            window_size=self.window_size
        )
    
    def forward(self, x):
        """
        Forward pass for masked reconstruction
        
        Args:
            x: Input tensor [batch_size, window_size, input_channels]
            
        Returns:
            Reconstructed tensor [batch_size, window_size, input_channels]
        """
        return self.masked_model(x)
    
    def get_embedding(self, x):
        """
        Get feature embedding from base encoder
        
        Args:
            x: Input tensor [batch_size, window_size, input_channels]
            
        Returns:
            Feature embedding [batch_size, embedding_dim]
        """
        return self.base_encoder.get_embedding(x)
    
    def get_embedding_dim(self):
        """
        Get the dimension of the encoder's feature embedding
        
        Returns:
            Feature embedding dimension
        """
        return self.base_encoder.get_embedding_dim()
    
    def extract_features(self, x):
        """
        Extract features for classification or other downstream tasks
        
        Args:
            x: Input tensor [batch_size, window_size, input_channels]
            
        Returns:
            Feature embedding [batch_size, embedding_dim]
        """
        return self.base_encoder.extract_features(x)
    
    def get_loss(self, input_target, validation=False):
        """
        Calculate masked reconstruction loss
        
        Args:
            input_target: Tuple of (x_original, x_target, choice_mask)
            validation: Whether in validation mode
            
        Returns:
            MSE loss between masked reconstruction and target
        """
        return self.masked_model.get_loss(input_target, validation)
    
    def freeze_all(self):
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True 