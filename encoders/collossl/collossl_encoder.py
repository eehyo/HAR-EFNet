import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from ..base import DeepConvLSTMEncoder, DeepConvLSTMAttnEncoder, SAHAREncoder


class ColloSSLEncoder(nn.Module):
    """
    ColloSSL encoder wrapper for multi-device contrastive learning
    Wraps base encoder architectures (DeepConvLSTM, DeepConvLSTM-Attn, SA-HAR)
    """
    def __init__(self, args: Dict[str, Any]):
        """
        Initialize ColloSSL encoder
        
        Args:
            args: Configuration parameters containing:
                - encoder_type: Base encoder type
                - input_channels: Number of input channels per device (3 for x,y,z)
                - window_size: Input window size
                - device: Computation device
                - Other base encoder specific parameters
        """
        super(ColloSSLEncoder, self).__init__()
        
        # Store configuration
        self.encoder_type = args.get('encoder_type', 'sa_har')
        self.device = args.get('device', 'cpu')
        
        # Modify args for single device input (3 channels instead of 9)
        single_device_args = args.copy()
        single_device_args['input_channels'] = 3  # x, y, z for single device
        
        # Create base encoder for single device processing
        self.base_encoder = self._create_base_encoder(single_device_args)
        
        # Store embedding dimension
        self.embedding_dim = self.base_encoder.get_embedding_dim()
        
        # Initialize device for computation
        self.to(self.device)
    
    def _create_base_encoder(self, args: Dict[str, Any]) -> nn.Module:
        """
        Create base encoder based on encoder type
        
        Args:
            args: Configuration parameters
            
        Returns:
            Base encoder instance
        """
        if self.encoder_type == 'deepconvlstm':
            return DeepConvLSTMEncoder(args)
        elif self.encoder_type == 'deepconvlstm_attn':
            return DeepConvLSTMAttnEncoder(args)
        elif self.encoder_type == 'sa_har':
            return SAHAREncoder(args)
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of the encoder's feature embedding
        
        Returns:
            Feature embedding dimension
        """
        return self.embedding_dim
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from single device input
        
        Args:
            x: Input tensor [batch_size, window_size, 3] for single device
            
        Returns:
            Feature embedding [batch_size, embedding_dim]
        """
        return self.base_encoder.extract_features(x)
    
    def forward_single_device(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for single device data
        
        Args:
            x: Single device input [batch_size, window_size, 3]
            
        Returns:
            Feature embeddings [batch_size, embedding_dim]
        """
        return self.extract_features(x)
    
    def forward_multi_device(self, anchor_data: torch.Tensor, 
                           other_devices_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-device ColloSSL data
        
        Args:
            anchor_data: Anchor device data [batch_size, window_size, 3]
            other_devices_data: Dict of other device data [batch_size, window_size, 3]
            
        Returns:
            Dict of device embeddings {device_name: [batch_size, embedding_dim]}
        """
        embeddings = {}
        
        # Process anchor data
        embeddings['anchor'] = self.forward_single_device(anchor_data)
        
        # Process other devices data
        for device_name, device_data in other_devices_data.items():
            embeddings[device_name] = self.forward_single_device(device_data)
        
        return embeddings
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for compatibility with existing training loops
        
        Args:
            x: Input tensor [batch_size, window_size, channels]
            
        Returns:
            Feature embeddings [batch_size, embedding_dim]
        """
        return self.forward_single_device(x)
    
    def freeze_encoder(self, freeze_mode: str = 'all'):
        """
        Freeze encoder parameters for fine-tuning
        
        Args:
            freeze_mode: Freezing mode ('all', 'partial', etc.)
        """
        if hasattr(self.base_encoder, 'freeze_encoder'):
            self.base_encoder.freeze_encoder(freeze_mode)
        else:
            # Fallback: freeze all parameters
            for param in self.base_encoder.parameters():
                param.requires_grad = False


def create_collossl_encoder(args: Any) -> ColloSSLEncoder:
    """
    Create ColloSSL encoder model based on configuration
    
    Args:
        args: Configuration parameters
        
    Returns:
        Created ColloSSL encoder model
    """
    from utils.logger import Logger
    import yaml
    import os
    
    logger = Logger("collossl_encoder_creator")
    
    # Convert args to dict if needed
    if hasattr(args, '__dict__'):
        encoder_args = {
            'encoder_type': args.encoder_type,
            'input_channels': 3,  # Single device: x, y, z
            'window_size': args.window_size,
            'output_size': args.output_size,
            'device': args.device
        }
    else:
        encoder_args = args.copy()
        encoder_args['input_channels'] = 3
    
    # Load model configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'configs', 'model.yaml')
    with open(config_path, mode='r') as config_file:
        model_config = yaml.load(config_file, Loader=yaml.FullLoader)
    
    encoder_config = model_config['efnet_encoder']
    
    # Update encoder args with model-specific config
    if args.encoder_type in encoder_config:
        encoder_args.update(encoder_config[args.encoder_type])
    
    # Create ColloSSL encoder
    encoder = ColloSSLEncoder(encoder_args)
    
    logger.info(f"Created ColloSSL {args.encoder_type} encoder")
    return encoder 