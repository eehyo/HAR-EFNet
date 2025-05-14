import torch
import torch.nn as nn
import torch.nn.functional as F

class SAHARClassifier(nn.Module):
    def __init__(self, encoder, num_classes, config):
        super(SAHARClassifier, self).__init__()
        
        # Encoder setup
        self.encoder = encoder
        
        # Parameter setup
        self.dropout_rate = config.get('dropout_rate', 0.2)
        
        # Expands features to 4x num_classes size before final classification
        self.fc1 = nn.Linear(encoder.get_embedding_dim(), 4 * num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc_out = nn.Linear(4 * num_classes, num_classes)
        
    def forward(self, x):
        # Extract features from encoder
        features = self.encoder.extract_features(x)
        
        # First FC layer
        x = self.fc1(features)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output layer
        output = self.fc_out(x)
        
        return output
    
    def freeze_encoder(self, freeze_mode='all'):
        """
        Freeze encoder parameters using specified mode
        
        Args:
            freeze_mode: Mode to freeze encoder ('all')
        """
        if freeze_mode == 'all':
            self.encoder.freeze_all()
        else:
            raise ValueError(f"Unsupported freeze mode for SA-HAR: {freeze_mode}. " 
                           "Available mode: 'all'")
            
    def unfreeze_encoder(self):
        """
        Unfreeze all encoder parameters for full fine-tuning
        """
        self.encoder.unfreeze_all() 