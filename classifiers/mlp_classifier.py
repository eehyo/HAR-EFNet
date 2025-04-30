import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union

class MLPClassifier(nn.Module):
    """
    Classifier model based on pretrained encoder
    
    This model combines a pretrained encoder with a classification head
    to perform activity recognition using extracted ECDF features
    """
    def __init__(self, encoder: nn.Module, num_classes: int, config: Dict[str, Any]):
        """
        Initialize classifier
        
        Args:
            encoder: Pretrained encoder model used as feature extractor
            num_classes: Number of activity classes to classify
            config: Configuration dictionary for the classifier with keys:
                - classifier_hidden: List of hidden layer sizes
                - dropout_rate: Dropout probability
                - freeze_encoder: Whether to freeze encoder parameters
        """
        super(MLPClassifier, self).__init__()
        
        self.encoder = encoder
        self.encoder_output_size = encoder.get_embedding_dim()
        
        # Build classification head
        layers = []
        
        hidden_sizes = config.get('classifier_hidden', [256, 128])
        dropout_rate = config.get('dropout_rate', 0.5)
        
        # Input layer
        input_size = self.encoder_output_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Freeze encoder if required
        if config.get('freeze_encoder', True):
            self.freeze_encoder()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input data [batch_size, window_size, input_channels]
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        # Extract features from encoder
        features = self.encoder.get_embedding(x)
        
        # Predict classes with classification head
        logits = self.classifier(features)
        
        return logits
    
    def freeze_encoder(self) -> None:
        """
        Freeze encoder parameters to train only the classification head
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self) -> None:
        """
        Make encoder parameters trainable again
        """
        for param in self.encoder.parameters():
            param.requires_grad = True 