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
        
        if isinstance(self.output_size, tuple) and len(self.output_size) == 2:
            self.axis_dim, self.feat_per_axis = self.output_size
            self.flat_output_size = self.axis_dim * self.feat_per_axis  # 3 * 78 = 234
        else:
            raise ValueError(f"Expected output_size to be a tuple (3, 78), got {self.output_size}")
        
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
        
        # 각 ECDF 특성에 대한 독립적인 FC 레이어 체인 생성 (기존 regressor 구조와 동일)
        self.feature_predictors = nn.ModuleList()
        for i in range(self.feat_per_axis):
            # 기존 regressor 구조와 동일하게 구성 (512->256->3)
            fc1 = nn.Linear(self.lstm_output_dim * 2, 512)  # 어텐션 + 마지막 히든 스테이트 결합
            ln1 = nn.LayerNorm(512)
            fc2 = nn.Linear(512, 256)
            ln2 = nn.LayerNorm(256)
            fc3 = nn.Linear(256, self.axis_dim)
            
            self.feature_predictors.append(nn.ModuleList([fc1, ln1, fc2, ln2, fc3]))
    
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
            Predicted ECDF features: [batch_size, 3, 78]
        """
        # Get embeddings
        features = self.get_embedding(x)
        
        # 각 특성별로 독립적인 FC 레이어 체인 적용
        outputs = []
        for i in range(self.feat_per_axis):
            # 레이어 가져오기
            fc1, ln1, fc2, ln2, fc3 = self.feature_predictors[i]
            
            # 첫 번째 블록
            x_feature = fc1(features)
            x_feature = ln1(x_feature)
            x_feature = F.relu(x_feature)
            x_feature = F.dropout(x_feature, p=self.dropout_rate, training=self.training)
            
            # 두 번째 블록
            x_feature = fc2(x_feature)
            x_feature = ln2(x_feature)
            x_feature = F.relu(x_feature)
            x_feature = F.dropout(x_feature, p=self.dropout_rate, training=self.training)
            
            # 출력 레이어
            x_feature = fc3(x_feature)
            
            outputs.append(x_feature)
        
        # 모든 출력을 결합하여 [batch_size, 3, 78] 형태로 생성
        output = torch.stack(outputs, dim=2)  # [batch_size, 3, 78]
        
        return output
    
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