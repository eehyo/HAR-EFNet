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

        self.conv1 = nn.Conv2d(self.input_filters, self.nb_filters, (self.filter_width, 1), dilation=(self.dilation, 1))
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.filter_width, 1), dilation=(self.dilation, 1), stride=(2,1))
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
        
        # Define dropout layer
        self.dropout = nn.Dropout(self.drop_prob)
        
        # 각 ECDF 특성에 대한 독립적인 FC 레이어 생성
        self.feature_predictors = nn.ModuleList()
        for i in range(self.feat_per_axis):
            # 간단한 단일 FC 레이어로 각 특성에 대한 예측
            fc = nn.Linear(self.nb_units_lstm, self.axis_dim)
            self.feature_predictors.append(fc)
        
        # Store output size of feature extractor for get_embedding_dim
        self.embedding_dim = self.nb_units_lstm
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of the encoder's feature embedding
        
        Returns:
            Feature embedding dimension
        """
        return self.embedding_dim
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for classification or other downstream tasks
        
        Args:
            x: Input tensor [batch_size, window_size, input_channels]
            
        Returns:
            Feature embedding [batch_size, embedding_dim]
        """
        return self.get_embedding(x)
    
    def get_embedding(self, x):
        """
        Extract feature embeddings without applying regression head
        
        Args:
            x: Input tensor of shape [batch_size, window_size, input_channels]
            
        Returns:
            Extracted feature embeddings [batch_size, nb_units_lstm]
        """
        batch_size = x.size(0)
        
        # Reshape input for 2D convolution: [batch_size, 1, window_size, input_channels]
        # (128, 168, 9) -> (128, 1, 168, 9)
        x = x.unsqueeze(1)
        
        # Apply convolutional blocks
        # (128, 1, 168, 9) -> (128, 64, 80, 9) -> (128, 64, 36, 9)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        # [batch_size, nb_filters, reduced_length, input_channels]
        # (B, nb_filters, L, C)
        # Get final sequence length
        final_seq_len = x.shape[2]
        
        # Reshape for LSTM: [batch_size, final_seq_len, nb_filters * input_channels]
        # (128, 64, 36, 9) -> (128, 36, 64, 9)
        x = x.permute(0, 2, 1, 3) 
        x = x.reshape(batch_size, final_seq_len, self.nb_filters * self.input_channels)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply LSTM layers
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)
        
        # Get last LSTM output
        # (128, 36, 128) -> (128, 128)
        x = x[:, -1, :]
        
        return x
    
    def forward(self, x):
        """
        Forward pass through encoder
        
        Args:
            x: Input tensor [batch_size, window_size, input_channels]
            
        Returns:
            ECDF features [batch_size, 3, 78]
        """
        # Get feature embeddings
        features = self.get_embedding(x)
        
        # 각 특성별로 독립적인 FC 레이어 적용
        outputs = []
        for i in range(self.feat_per_axis):
            fc = self.feature_predictors[i]
            x_feature = fc(features)
            outputs.append(x_feature)
        
        # 모든 출력을 결합하여 [batch_size, 3, 78] 형태로 생성
        x = torch.stack(outputs, dim=2)  # [batch_size, 3, 78]
        
        return x
    
    def calculate_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss for each ECDF feature independently
        
        Args:
            predictions: Predicted ECDF features [batch_size, 3, 78]
            targets: Target ECDF features [batch_size, 3, 78]
            
        Returns:
            Total loss and per-feature losses
        """
        total_loss = 0
        feature_losses = []
        
        for i in range(self.feat_per_axis):
            feature_pred = predictions[:, :, i]  # [batch_size, 3]
            feature_target = targets[:, :, i]    # [batch_size, 3]
            feature_loss = F.mse_loss(feature_pred, feature_target)
            feature_losses.append(feature_loss)
            total_loss += feature_loss
        
        return total_loss, feature_losses
    
    def freeze_all(self):
        """
        Freeze all encoder parameters
        Useful for transfer learning when you want to use the encoder as a fixed feature extractor
        """
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """
        Unfreeze all encoder parameters for full fine-tuning
        """
        for param in self.parameters():
            param.requires_grad = True 