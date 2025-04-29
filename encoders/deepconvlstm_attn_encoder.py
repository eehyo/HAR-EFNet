import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseEncoderModule

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


class DeepConvLSTMAttnEncoder(BaseEncoderModule):
    """
    DeepConvLSTMAttnEncoder encoder without attention mechanism
    """
    def __init__(self, config):
        super(DeepConvLSTMAttnEncoder, self).__init__()
        
        # Parse configuration
        self.input_channels = config['input_channels']
        self.window_size = config['window_size']
        self.output_size = config['output_size']
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model specific parameters
        self.nb_conv_blocks = config.get('nb_conv_blocks', 2)
        self.nb_filters = config.get('nb_filters', 64)
        self.dilation = config.get('dilation', 1)
        self.batch_norm = config.get('batch_norm', True)

        self.filter_width = config.get('filter_width', 5)
        self.nb_layers_lstm = config.get('nb_layers_lstm', 2)
        self.drop_prob = config.get('drop_prob', 0.5)
        self.nb_units_lstm = config.get('nb_units_lstm', 128)
        
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
        
        # # Attention mechanism
        # self.linear_1 = nn.Linear(self.nb_units_lstm, self.nb_units_lstm)
        # self.tanh = nn.Tanh()
        # self.dropout_2 = nn.Dropout(0.2)
        # self.linear_2 = nn.Linear(self.nb_units_lstm, 1, bias=False)
        
        # Output layer for ECDF features
        self.fc = nn.Linear(self.nb_units_lstm, self.output_size)
    
    def forward(self, x):
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
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)
        
        # # Apply attention mechanism
        # # [batch_size, sequence_length, hidden_dim]
        # context = x[:, :-1, :]  # All but the last time step
        # out = x[:, -1, :]      # Last time step
        
        # # Attention weights
        # uit = self.linear_1(context)
        # uit = self.tanh(uit)
        # uit = self.dropout_2(uit)
        # ait = self.linear_2(uit)
        
        # # Apply attention
        # attn = torch.matmul(F.softmax(ait, dim=1).transpose(-1, -2), context).squeeze(-2)
        
        # # Combine attention output with last hidden state
        # combined = out + attn
        
        # # Project to output size (ECDF features)
        # x = self.fc(combined)
        
        return x 