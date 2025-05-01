import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import EncoderBase

# same as deepconvlstm_encoder.py
class ConvBlock(nn.Module):
    """
    Normal convolution block
    """
    def __init__(self, filter_width, input_filters, nb_filters, dilation, batch_norm):
        super(ConvBlock, self).__init__()
        self.filter_width = filter_width
        self.input_filters = input_filters
        self.nb_filters = nb_filters
        self.dilation = dilation
        self.batch_norm = batch_norm 

        self.conv1 = nn.Conv2d(self.input_filters, self.nb_filters, (self.filter_width, 1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.filter_width, 1), stride=(2,1))
        if self.batch_norm:
            self.norm1 = nn.BatchNorm2d(self.nb_filters)
            self.norm2 = nn.BatchNorm2d(self.nb_filters)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm1(out)

        out = self.conv2(out)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm2(out)

        return out


class DeepConvLSTMAttnEncoder(EncoderBase):
    """
    DeepConvLSTMAttnEncoder encoder
    - Pretraining: Uses last LSTM hidden state to predict ECDF features
    - Classification: Provides full LSTM sequence output for attention mechanism
    """
    def __init__(self, config):
        super(DeepConvLSTMAttnEncoder, self).__init__(config)
        
        # Model specific parameters
        self.nb_conv_blocks = config.get('nb_conv_blocks', 2)
        self.nb_filters = config.get('nb_filters', 64)
        self.dilation = config.get('dilation', 1)
        self.batch_norm = config.get('batch_norm', True)
        self.filter_width = config.get('filter_width', 5)
        self.nb_layers_lstm = config.get('nb_layers_lstm', 2)
        self.drop_prob = config.get('drop_prob', 0.5)
        self.nb_units_lstm = config.get('nb_units_lstm', 128)
        
        if isinstance(self.output_size, tuple) and len(self.output_size) == 2:
            self.axis_dim, self.feat_per_axis = self.output_size
            self.flat_output_size = self.axis_dim * self.feat_per_axis  # 3 * 78 = 234
        else:
            raise ValueError(f"Expected output_size to be a tuple (3, 78), got {self.output_size}")
        
        # Define convolutional blocks
        self.conv_blocks = []
        for i in range(self.nb_conv_blocks):
            if i == 0:
                input_filters = 1  # Initial input channel
            else:
                input_filters = self.nb_filters
            
            self.conv_blocks.append(ConvBlock(
                self.filter_width, 
                input_filters, 
                self.nb_filters, 
                self.dilation, 
                self.batch_norm
            ))
        
        self.conv_blocks = nn.ModuleList(self.conv_blocks)
        
        # Define LSTM layers
        self.lstm_layers = []
        for i in range(self.nb_layers_lstm):
            if i == 0:
                self.lstm_layers.append(nn.LSTM(
                    self.input_channels * self.nb_filters, 
                    self.nb_units_lstm, 
                    batch_first=True
                ))
            else:
                self.lstm_layers.append(nn.LSTM(
                    self.nb_units_lstm, 
                    self.nb_units_lstm, 
                    batch_first=True
                ))
                
        self.lstm_layers = nn.ModuleList(self.lstm_layers)
        # [batch_size, seq_len, nb_units_lstm]
        
        # Define dropout layer
        self.dropout = nn.Dropout(self.drop_prob)
        
        # Output layer for ECDF features
        self.fc = nn.Linear(self.nb_units_lstm, self.flat_output_size)
        
        # Store output size of feature extractor for get_embedding_dim
        self.embedding_dim = self.nb_units_lstm
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of the encoder's feature embedding
        
        Returns:
            Feature embedding dimension
        """
        return self.embedding_dim
    
    def get_embedding(self, x, return_sequences=False):
        """
        Extract feature embeddings without applying regression head
        
        Args:
            x: Input tensor [batch_size, window_size, input_channels]
            return_sequences: If True, returns full LSTM sequence output,
                            otherwise returns last hidden state (default: False)
            
        Returns:
            If return_sequences=True: LSTM outputs [batch_size, seq_len, nb_units_lstm]
            If return_sequences=False: Last hidden state [batch_size, nb_units_lstm]
        """
        batch_size = x.size(0)
        
        # Reshape input for 2D convolution: [batch_size, 1, window_size, input_channels]
        x = x.unsqueeze(1)
        
        # Apply convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Get final sequence length
        final_seq_len = x.shape[2]
        
        # Reshape for LSTM: [batch, seq_len, features]
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, final_seq_len, self.nb_filters * self.input_channels)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply LSTM layers
        for i, lstm_layer in enumerate(self.lstm_layers):
            if i < len(self.lstm_layers) - 1:
                # For intermediate layers, use the full sequence
                x, _ = lstm_layer(x)
            else:
                # For the last layer, we need both the output sequence and the hidden state
                lstm_out, (h_n, _) = lstm_layer(x)
        
        if return_sequences:
            # Return full sequence output
            return lstm_out
        else:
            # Return the last hidden state
            last_hidden = h_n[-1]  # [batch_size, hidden_size]
            return last_hidden
    
    def forward(self, x, return_sequences=False):
        """
        Forward pass through encoder
        
        Args:
            x: Input tensor [batch_size, window_size, input_channels]
            return_sequences: If True, returns full LSTM sequence output (for classification),
                            otherwise returns ECDF features (default: False)
            
        Returns:
            If return_sequences=True: LSTM outputs [batch_size, seq_len, nb_units_lstm]
            If return_sequences=False: ECDF features [batch_size, 3, 78]
        """
        # Get embeddings - either sequence or last hidden state
        features = self.get_embedding(x, return_sequences)
        
        if return_sequences:
            # For classification: return full sequence output
            return features # (128, 128)
        else:
            # For ECDF prediction: project to output size and reshape
            x = self.fc(features)
            
            # Reshape from [batch_size, 234] to [batch_size, 3, 78]
            batch_size = x.size(0)
            x = x.view(batch_size, self.axis_dim, self.feat_per_axis)
            
            return x # (128, 3, 78)