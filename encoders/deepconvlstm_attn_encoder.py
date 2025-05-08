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
        self.filter_width = filter_width  #kernel
        self.input_filters = input_filters
        self.nb_filters = nb_filters
        self.dilation = dilation
        self.batch_norm = batch_norm 

        self.conv1 = nn.Conv2d(self.input_filters, self.nb_filters, (self.filter_width, 1))
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.filter_width, 1), stride=(2,1))
        if self.batch_norm:
            self.norm1 = nn.BatchNorm2d(self.nb_filters)
            self.norm2 = nn.BatchNorm2d(self.nb_filters)

    def forward(self, x): # (128, 1, 168, 9) # (128, 64, 80, 9)
        out = self.conv1(x) # (128, 64, ((168-5)/1+1)=164, 9) # (128, 64, ((80-5)/1+1)=76, 9)
        out = self.relu(out) # (128, 64, 164, 9) # (128, 64, 76, 9)
        if self.batch_norm:
            out = self.norm1(out) 

        out = self.conv2(out) # (128, 64, (164-5)/2+1=80, 9) # (128, 64, (76-5)/2+1=36, 9)
        out = self.relu(out) # (128, 64, 80, 9) # (128, 64, 36, 9)
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
            self.axis_dim, self.feat_per_axis = self.output_size  # 3, 78
            self.flat_output_size = self.axis_dim * self.feat_per_axis  # 3 * 78 = 234
        else:
            raise ValueError(f"Expected output_size to be a tuple (3, 78), got {self.output_size}")
        
        # Define convolutional blocks
        self.conv_blocks = []
        for i in range(self.nb_conv_blocks):  # 2
            if i == 0:  # 1 → 64
                input_filters = 1  # Initial input channel
            else:  # 64 → 64
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
        for i in range(self.nb_layers_lstm): # 2
            if i == 0:
                self.lstm_layers.append(nn.LSTM(
                    self.input_channels * self.nb_filters,  # 9 * 64 = 576
                    self.nb_units_lstm, # 128
                    batch_first=True
                ))
            else:
                self.lstm_layers.append(nn.LSTM(
                    self.nb_units_lstm, # 128
                    self.nb_units_lstm, # 128
                    batch_first=True
                ))
                
        self.lstm_layers = nn.ModuleList(self.lstm_layers)
        # [batch_size, seq_len, nb_units_lstm]
        
        # Define dropout layer
        self.dropout = nn.Dropout(self.drop_prob)
        
        # attention
        self.linear_1 = nn.Linear(self.nb_units_lstm, self.nb_units_lstm) # (128 → 128)
        self.tanh = nn.Tanh()
        self.dropout_2 = nn.Dropout(0.2)
        self.linear_2 = nn.Linear(self.nb_units_lstm, 1, bias=False) # (128 → 1)
        
        # 각 ECDF 특성에 대한 독립적인 FC 레이어 생성
        self.feature_predictors = nn.ModuleList()
        for i in range(self.feat_per_axis):
            # 간단한 단일 FC 레이어로 각 특성에 대한 예측
            fc = nn.Linear(self.nb_units_lstm, self.axis_dim) # (128 → 3)
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
    
    def get_embedding(self, x, return_sequences=False):
        """
        Extract feature embeddings without applying regression head
        
        Args:
            x: Input tensor [batch_size, window_size, input_channels] (128, 168, 9)
            return_sequences: If True, returns full LSTM sequence output,
                            otherwise returns last hidden state (default: False)
            
        Returns:
            If return_sequences=True: LSTM outputs [batch_size, seq_len, nb_units_lstm] (128, 168, 128)
            If return_sequences=False: Last hidden state [batch_size, nb_units_lstm] (128, 128)
        """
        batch_size = x.size(0)
        
        # Reshape input for 2D convolution: [batch_size, 1, window_size, input_channels] 
        #  (128, 168, 9) -> (128, 1, 168, 9)
        x = x.unsqueeze(1)
        
        # Apply convolutional blocks
        # (128, 1, 168, 9) -> (128, 64, 80, 9) -> (128, 64, 36, 9)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Get final sequence length
        final_seq_len = x.shape[2] # 36
        
        # Reshape for LSTM: [batch, seq_len, features]
        x = x.permute(0, 2, 1, 3) # (128, 64, 36, 9) -> (128, 36, 64, 9)
        x = x.reshape(batch_size, final_seq_len, self.nb_filters * self.input_channels) # (128, 36, 64*9=576)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply LSTM layers
        # (128, 36, 576) → (128, 36, 128) → (128, 36, 128)
        for i, lstm_layer in enumerate(self.lstm_layers):
            if i < len(self.lstm_layers) - 1:
                # For intermediate layers, use the full sequence
                x, _ = lstm_layer(x)
            else:
                # For the last layer, we need both the output sequence and the hidden state
                lstm_out, (h_n, _) = lstm_layer(x)
        
        if return_sequences:
            # Return full sequence output
            return lstm_out # [B, T, H] (128, 36, 128)
        else:
            # attention - shape: [batch_size, sequence_length, hidden_dim]
            context = lstm_out[:, :-1, :]  # [B, T-1, H] (128, 35, 128)
            out = lstm_out[:, -1, :]       # [B, H] (128, 128)
            
            uit = self.linear_1(context) # [B, T-1, H] (128, 35, 128)
            uit = self.tanh(uit) # (128, 35, 128)
            uit = self.dropout_2(uit) # (128, 35, 128)
            ait = self.linear_2(uit) # [B, T-1, 1] (128, 35, 1)
            attn = torch.matmul(F.softmax(ait, dim=1).transpose(-1, -2), context).squeeze(-2)
            
            # out: 마지막 타임스텝의 정보
            # attn: 중요한 시간 정보만 요약한 벡터
            return out + attn # [B, H]
    
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
        # x (128, 168, 9)
        features = self.get_embedding(x, return_sequences) # (128, 128)
        
        if return_sequences:
            # For classification: return full sequence output
            return features # (128, 128)
        else:
            # 각 특성별로 독립적인 FC 레이어 적용
            outputs = []
            for i in range(self.feat_per_axis):
                fc = self.feature_predictors[i]
                x_feature = fc(features) # (128, 3)
                outputs.append(x_feature)
            
            # 모든 출력을 결합하여 [batch_size, 3, 78] 형태로 생성
            x = torch.stack(outputs, dim=2)  # [batch_size, 3, 78] (128, 3, 78)
            
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
