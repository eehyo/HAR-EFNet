import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepConvLSTMAttnClassifier(nn.Module):
    """
    Uses the encoder's internal attention mechanism and applies a simple FC layer for classification
    """
    def __init__(self, encoder, num_classes, config):
        super(DeepConvLSTMAttnClassifier, self).__init__()
        
        # Encoder setup
        self.encoder = encoder
    
        # FC layer for classification
        self.fc = nn.Linear(encoder.get_embedding_dim(), num_classes)
        
    def forward(self, x):
        # Extract features using encoder's internal attention mechanism
        features = self.encoder.extract_features(x)
        
        output = self.fc(features)
        
        return output
    
    def freeze_encoder(self, freeze_mode='cnn_lstm_only'):
        """
        Freeze encoder parameters using specified mode
        
        Args:
            freeze_mode: Mode to freeze encoder ('all', 'cnn_lstm_only')
        """
        if freeze_mode == 'all':
            self.encoder.freeze_all()
        elif freeze_mode == 'cnn_lstm_only':
            self.encoder.freeze_cnn_lstm_only()
        else:
            raise ValueError(f"Unsupported freeze mode: {freeze_mode}. " 
                            "Available modes: 'all', 'cnn_lstm_only'")
            
    def unfreeze_encoder(self):
        """
        Unfreeze all encoder parameters for full fine-tuning
        """
        self.encoder.unfreeze_all() 