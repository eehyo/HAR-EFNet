import torch
import torch.nn as nn
import torch.nn.functional as F

LAYER_TYPE = '2layer'  # 'single' or '2layer'

class SAHARClassifier(nn.Module):
    def __init__(self, encoder, num_classes, config):
        super(SAHARClassifier, self).__init__()
        
        self.encoder = encoder
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.layer_type = LAYER_TYPE 

        # Single-layer classification model
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(encoder.get_embedding_dim(), num_classes)
        
        # 2-layer classification model
        # Expands features to 4x num_classes size before final classification
        self.fc1 = nn.Linear(encoder.get_embedding_dim(), 4 * num_classes)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4 * num_classes, num_classes)

    def forward(self, x):
        # extract features from encoder
        features = self.encoder.extract_features(x)
        
        if self.layer_type == 'single':
            # Single-layer classification model  
            # apply dropout
            # features = self.dropout(features)
            
            # single FC layer for classification
            output = self.fc(features)
            
        elif self.layer_type == '2layer':
            # 2-layer classification model
            # first FC layer
            x = self.fc1(features)
            x = self.relu(x)
            x = self.dropout(x)
            
            # second FC layer for class prediction
            output = self.fc2(x)
            
        else:
            raise ValueError(f"Unsupported layer_type: {self.layer_type}. "
                           "Available types: 'single', '2layer'")
        
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