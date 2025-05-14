import torch
import torch.nn as nn

class DeepConvLSTMClassifier(nn.Module):

    def __init__(self, encoder, num_classes, config):
        super(DeepConvLSTMClassifier, self).__init__()
        
        # encoder
        self.encoder = encoder
        
        # dropout rate
        self.dropout_rate = config.get('dropout_rate', 0.5)
        
        # model
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(encoder.get_embedding_dim(), num_classes)
        
    def forward(self, x):
        # extract features from encoder
        features = self.encoder.extract_features(x) # (128, 128)
        
        # apply dropout
        features = self.dropout(features) # (128, 128)
        
        # linear classifier for class prediction
        output = self.fc(features) # (128, 12)
        
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