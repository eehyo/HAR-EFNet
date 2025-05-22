import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union

from .deepconvlstm_encoder import DeepConvLSTMEncoder


class MTLDeepConvLSTMEncoder(nn.Module):
    """
    Multi-Task Learning DeepConvLSTM Encoder class
    Adds MTL heads for Self-Supervised Learning based on the existing DeepConvLSTM encoder
    Performs binary classification tasks in parallel to predict each transformation (augmentation) type
    """
    def __init__(self, args: Dict[str, Any]):
        """
        Initialize MTL DeepConvLSTM encoder
        
        Args:
            args: Model configuration parameters (Dict)
        """
        super(MTLDeepConvLSTMEncoder, self).__init__()
        
        # Configure base encoder
        self.encoder = DeepConvLSTMEncoder(args)
        
        # LSTM output size (hidden_dim set in DeepConvLSTM)
        self.hidden_size = self.encoder.embedding_dim
        
        # Device configuration
        self.device = args.get('device', 'cpu')
        
        # List of supported tasks
        self.task_list = ['jitter', 'scaling', 'time_warp', 'rotation', 'permutation', 
                          'negated', 'horizontal_flip', 'channel_shuffle']
        
        # Task-specific binary classification heads (MTL)
        self.task_heads = nn.ModuleDict({
            'jitter': nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ),
            'scaling': nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ),
            'time_warp': nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ),
            'rotation': nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ),
            'permutation': nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ),
            'negated': nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ),
            'horizontal_flip': nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ),
            'channel_shuffle': nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        })
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize weights for MTL heads
        """
        for head in self.task_heads.values():
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor, task: Optional[str] = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input time series data [batch_size, window_size, channels]
            task: Name of specific task to run (when performing only one task)
            
        Returns:
            Returns output of the specified task if a task is given
            Otherwise returns outputs of all tasks as a dict
        """
        # Get encoder output
        encoder_output = self.encoder.extract_features(x)  # [batch_size, hidden_size]
        
        # When in specific task mode
        if task is not None:
            if task in self.task_heads:
                return self.task_heads[task](encoder_output)
            else:
                raise ValueError(f"Unsupported task: {task}. Available tasks: {list(self.task_heads.keys())}")
        
        # Generate outputs for all tasks
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(encoder_output)
        
        return outputs
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feature extraction function (used in Classifier)
        
        Args:
            x: Input time series data [batch_size, window_size, channels]
            
        Returns:
            Encoded feature vector [batch_size, hidden_size]
        """
        return self.encoder.extract_features(x) 
        
    def freeze_all(self):
        """
        Freeze all encoder parameters including base encoder and MTL heads
        """
        # Freeze base encoder
        self.encoder.freeze_all()
        
        # Freeze MTL heads
        for head in self.task_heads.values():
            for param in head.parameters():
                param.requires_grad = False
        
    def unfreeze_all(self):
        """
        Unfreeze all encoder parameters for full fine-tuning
        """
        # Unfreeze base encoder
        self.encoder.unfreeze_all()
        
        # Unfreeze MTL heads
        for head in self.task_heads.values():
            for param in head.parameters():
                param.requires_grad = True 