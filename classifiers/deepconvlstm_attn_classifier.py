import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepConvLSTMAttnClassifier(nn.Module):
    """
    Applies attention mechanism to LSTM sequence outputs from encoder
    """
    def __init__(self, encoder, num_classes, config):
        super(DeepConvLSTMAttnClassifier, self).__init__()
        
        # Encoder setup
        self.encoder = encoder
        
        # Parameter setup
        self.dropout_rate = config.get('dropout_rate', 0.5)
        
        # Attention mechanism
        self.linear_1 = nn.Linear(encoder.nb_units_lstm, encoder.nb_units_lstm)
        self.tanh = nn.Tanh()
        self.dropout_2 = nn.Dropout(0.2)
        self.linear_2 = nn.Linear(encoder.nb_units_lstm, 1, bias=False)
        
        # FC layer - converts ECDF features to classes
        self.fc = nn.Linear(encoder.nb_units_lstm, num_classes)
        
    def forward(self, x):
        # Get LSTM sequence features from encoder
        # Shape: [batch_size, seq_len, hidden_dim]
        # (128, 36, 128)
        lstm_sequence = self.encoder.forward(x, return_sequences=True)
        
        # Apply attention mechanism
        # (128, 35, 128)
        context = lstm_sequence[:, :-1, :]  # All sequences except last timestep
        # (128, 128)
        out = lstm_sequence[:, -1, :]      # Output of last timestep
        
        # Calculate attention weights
        uit = self.linear_1(context) # (128, 35, 128)
        uit = self.tanh(uit) 
        uit = self.dropout_2(uit) 
        ait = self.linear_2(uit) # (128, 35, 1)
        
        # Apply attention
        attn_weights = F.softmax(ait, dim=1).transpose(-1, -2) # (128, 1, 35)
        # (128, 1, 35) × (128, 35, 128) → (128, 1, 128) → (128, 128)
        attn = torch.matmul(attn_weights, context).squeeze(-2) 
        
        # Combine attention output with last hidden state
        combined = out + attn # (128, 128)
        
        # Class prediction
        output = self.fc(combined)
        
        return output
    
    def freeze_encoder(self):
        """freeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        """unfreeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = True 