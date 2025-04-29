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
        self.fc = nn.Linear(encoder.output_size, num_classes)
        
    def forward(self, x):
        # extract features from encoder
        features = self.encoder(x)
        
        # apply dropout
        features = self.dropout(features)
        
        # linear classifier for class prediction
        output = self.fc(features)
        
        return output
    
    def freeze_encoder(self):
        """freeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        """unfreeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = True 