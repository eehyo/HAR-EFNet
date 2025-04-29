import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import EncoderBase

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

        # self.conv1 = nn.Conv2d(self.input_filters, self.nb_filters, (self.filter_width, 1), dilation=(self.dilation, 1))
        self.conv1 = nn.Conv2d(self.input_filters, self.nb_filters, (self.filter_width, 1))
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.filter_width, 1), dilation=(self.dilation, 1), stride=(2,1))
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


class DeepConvLSTMEncoder(EncoderBase):
    """
    DeepConvLSTM encoder model based on architecture by Ordonez and Roggen
    Adapted to output ECDF features
    """
    def __init__(self, config):
        super(DeepConvLSTMEncoder, self).__init__(config)
        
        # Model specific parameters
        self.nb_conv_blocks = config.get('nb_conv_blocks', 2)
        self.nb_filters = config.get('nb_filters', 64)
        self.dilation = config.get('dilation', 1)
        self.batch_norm = config.get('batch_norm', True)
        self.filter_width = config.get('filter_width', 5)
        self.nb_layers_lstm = config.get('nb_layers_lstm', 1)
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
        
        # Define dropout layer
        self.dropout = nn.Dropout(self.drop_prob)
        
        # Output layer for ECDF features
        self.fc = nn.Linear(self.nb_units_lstm, self.output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape input for 2D convolution: [batch_size, 1, window_size, input_channels]
        x = x.unsqueeze(1)
        
        # Apply convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        # [batch_size, nb_filters, reduced_length, input_channels]
        # (B, nb_filters, L, C)
        # Get final sequence length
        final_seq_len = x.shape[2]
        
        # Reshape for LSTM: [batch_size, final_seq_len, nb_filters * input_channels]
        x = x.permute(0, 2, 1, 3) # (B, L, nb_filters, C)
        # [batch_size, reduced_length, nb_filters*input_channels]
        x = x.reshape(batch_size, final_seq_len, self.nb_filters * self.input_channels)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply LSTM layers
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)
        
        # Get last LSTM output
        x = x[:, -1, :]
        
        # Project to output size (ECDF features)
        x = self.fc(x)
        
        return x 