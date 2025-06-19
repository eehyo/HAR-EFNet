import torch
import torch.nn as nn

LAYER_TYPE = 'single'  # 'single' or '2layer'

class DeepConvLSTMClassifier(nn.Module):

    def __init__(self, encoder, num_classes, config):
        super(DeepConvLSTMClassifier, self).__init__()
        
        self.encoder = encoder

        self.dropout_rate = config.get('dropout_rate', 0.5)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.layer_type = LAYER_TYPE 

        # Single-layer classification model
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(encoder.get_embedding_dim(), num_classes)

        # 2-layer classification model
        self.fc1 = nn.Linear(encoder.get_embedding_dim(), self.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        # extract features from encoder
        features = self.encoder.extract_features(x) # (128, 128)
        
        if self.layer_type == 'single':
            # Single-layer classification model  
            # apply dropout
            features = self.dropout(features)  # (128, 128)
            
            # Single FC layer for classification
            output = self.fc(features) # (128, 12)
            
        elif self.layer_type == '2layer':
            # 2-layer classification model
            # first FC layer
            x = self.fc1(features) # (batch_size, hidden_dim)
            x = self.relu(x)
            x = self.dropout(x)
            
            # second FC layer for class prediction
            output = self.fc2(x) # (batch_size, num_classes)
            
        else:
            raise ValueError(f"Unsupported layer_type: {self.layer_type}. "
                           "Available types: 'single', '2layer'")
        
        return output
    
    def freeze_encoder(self, freeze_mode='all'):
        """
        Freeze encoder parameters using specified mode
        
        Args:
            freeze_mode: Mode to freeze encoder ('all', 'cnn_lstm_only')
        """
        if freeze_mode == 'all':
            self.encoder.freeze_all()
        elif freeze_mode == 'cnn_lstm_only':
            # If this encoder supports selective freezing
            if hasattr(self.encoder, 'freeze_cnn_lstm_only'):
                self.encoder.freeze_cnn_lstm_only()
            else:
                self.encoder.freeze_all()
        else:
            raise ValueError(f"Unsupported freeze mode: {freeze_mode}. " 
                           "Available modes: 'all', 'cnn_lstm_only'")
            
    def unfreeze_encoder(self):
        """
        Unfreeze all encoder parameters for full fine-tuning
        """
        self.encoder.unfreeze_all() 