import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import EncoderBase


# TODO: Attention 공부 후 나중에 다시 모델 구조 수정

class Attention(nn.Module):
    """
    Attention mechanism for sequence data
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        
        # Layers to compute attention scores
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, lstm_output):
        # lstm_output shape: [batch, seq_len, hidden_size]
        
        # Compute attention scores
        attention_weights = self.attention(lstm_output).squeeze(-1)  # [batch, seq_len]
        
        # Normalize scores with softmax
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Compute weighted sum of LSTM outputs
        # [batch, seq_len, 1] * [batch, seq_len, hidden_size] -> [batch, hidden_size]
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1), 
            lstm_output
        ).squeeze(1)
        
        return context_vector, attention_weights

class LSTMEncoder(EncoderBase):
    """
    LSTM-based encoder for ECDF feature regression
    """
    def __init__(self, args):
        super(LSTMEncoder, self).__init__(args)
        
        self.hidden_size = args.get('hidden_size', 128)
        self.num_layers = args.get('num_layers', 2)
        self.bidirectional = args.get('bidirectional', True)
        self.dropout_rate = args.get('dropout_rate', 0.3)
        
        self.encoder_type = 'lstm'
        
        self.directions = 2 if self.bidirectional else 1
        self.lstm_output_dim = self.hidden_size * self.directions
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Add attention mechanism
        self.attention = Attention(self.lstm_output_dim)
        
        # Regression head with LayerNorm
        self.regressor = nn.Sequential(
            nn.Linear(self.lstm_output_dim * 2, 512),  # 어텐션 + 마지막 히든 스테이트 결합
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, self.output_size)
        )
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings without applying regression head
        
        Args:
            x: Input tensor of shape [batch_size, window_size, input_channels]
        
        Returns:
            Extracted feature embeddings [batch_size, lstm_output_dim * 2]
        """
        lstm_output, (h_n, c_n) = self.lstm(x)
        
        # Extract the final hidden state from the last layer
        if self.bidirectional:
            last_forward = h_n[2 * self.num_layers - 2]
            last_backward = h_n[2 * self.num_layers - 1]
            last_hidden = torch.cat([last_forward, last_backward], dim=1)
        else:
            last_hidden = h_n[self.num_layers - 1]
        
        # Apply attention mechanism
        context_vector, _ = self.attention(lstm_output)
        
        # Concatenate context vector with final hidden state
        combined = torch.cat([context_vector, last_hidden], dim=1)
        
        return combined
    
    def forward(self, x):
        """
        Forward pass of the LSTM encoder.

        Args:
            x: Input tensor of shape [batch_size, window_size, input_channels]
        
        Returns:
            Predicted ECDF features: [batch_size, output_size]
        """
        # Get embeddings
        features = self.get_embedding(x)
        
        # Apply regression head
        output = self.regressor(features)
        
        return output 