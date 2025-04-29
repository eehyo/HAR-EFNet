import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import EncoderBase

class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(AttentionLayer, self).__init__()
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=True)
        self.value_projection = nn.Linear(d_model, d_model, bias=True)
        self.out_projection = nn.Linear(d_model, d_model, bias=True)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, L, H, -1)
        values = self.value_projection(values).view(B, L, H, -1)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        _, _, _, E = queries.shape
        scale = 1./torch.sqrt(torch.tensor(E, dtype=torch.float))
        Attn = torch.softmax(scale * scores, dim=-1)
        V = torch.einsum("bhls,bshd->blhd", Attn, values).contiguous()

        out = V.view(B, L, -1)
        out = self.out_projection(out)
        return out, Attn


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = AttentionLayer(d_model, n_heads)
        self.dropout1 = nn.Dropout(p=dropout)
        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)    
        
        d_ff = d_ff or 4*d_model
        self.ffn1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.ffn2 = nn.Linear(d_ff, d_model, bias=True)
         
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)               
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        attn_output, attn = self.attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        # Shape: (B, L, C); C = nb_units = d_model
        
        ffn_output = self.ffn2(self.relu(self.ffn1(out1)))
        # Shape: (B, L, d_ff) -> (B, L, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class AttentionWithContext(nn.Module):
    def __init__(self, hidden_dim, act_fn="tanh"):
        super(AttentionWithContext, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim) # hidden_dim = nb_units
        
        if act_fn == "tanh":
            self.activation = nn.Tanh() 
        elif act_fn == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            raise NotImplementedError
        
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Shape: (B, L, C); C = nb_units
        uit = self.activation(self.fc1(x))
        # uit shape: (B, L, C) C = nb_units
        ait = self.fc2(uit)
         # ait shape: (B, L, 1) 

        attn_weights = F.softmax(ait, dim=1).transpose(-1, -2)
        # Shape: (B, 1, L)

        out = torch.matmul(attn_weights, x).squeeze(-2)
        # Shape: (B, C); C = nb_units
        return out


class ConvBlock(nn.Module):
    """
    Convolution block for SA-HAR
    """
    def __init__(self, filter_width, input_filters, nb_units, dilation, batch_norm):
        super(ConvBlock, self).__init__()
        self.filter_width = filter_width
        self.input_filters = input_filters
        self.nb_units = nb_units
        self.dilation = dilation
        self.batch_norm = batch_norm

        self.conv1 = nn.Conv2d(self.input_filters, self.nb_units, (self.filter_width, 1), 
                              dilation=(self.dilation, 1), padding='same')
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.nb_units, 1, (self.filter_width, 1), 
                              dilation=(self.dilation, 1), stride=(1,1), padding='same')
        if self.batch_norm:
            self.norm1 = nn.BatchNorm2d(self.nb_units)
            self.norm2 = nn.BatchNorm2d(1)

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


class SensorAttention(nn.Module):
    def __init__(self, input_shape, nb_units):
        super(SensorAttention, self).__init__()
        self.ln = nn.LayerNorm(input_shape[2])
        
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=nb_units, kernel_size=3, 
                               dilation=2, padding='same')
        self.conv_f = nn.Conv2d(in_channels=nb_units, out_channels=1, kernel_size=1, 
                               padding='same')
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=3)

    def forward(self, inputs):
        '''
        input: [batch * length * channel]
        output: [batch, length, d]
        '''
        inputs = self.ln(inputs)               
        x = inputs.unsqueeze(1)                
        # [batch, 1, length, channel]
        x = self.conv_1(x)              
        x = self.relu(x)  
        # [batch, nb_units, length, channel]
        x = self.conv_f(x)               
        # [batch, 1, length, channel]
        x = self.softmax(x)
        x = x.squeeze(1)
        # [batch, length, channel]
        return torch.mul(inputs, x), x


class SAHAREncoder(EncoderBase):
    """
    Sensor Attention HAR Encoder
    Adapted to output ECDF features
    """
    def __init__(self, config):
        super(SAHAREncoder, self).__init__(config)
        
        # Model specific parameters
        self.nb_units = config.get('nb_units', 64)
        self.n_heads = config.get('n_heads', 4)
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.batch_norm = config.get('batch_norm', True)
        if not isinstance(self.batch_norm, bool):
            self.batch_norm = bool(self.batch_norm)
        
        # Initial convolution block
        self.first_conv = ConvBlock(
            filter_width=5,
            input_filters=1,
            nb_units=self.nb_units,
            dilation=1,
            batch_norm=self.batch_norm
        )
        
        # Sensor attention module
        self.sensor_attention = SensorAttention(
            input_shape=(None, self.window_size, self.input_channels),
            nb_units=self.nb_units
        )
        
        # 1D convolution for feature transformation
        self.conv1d = nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=self.nb_units,
            kernel_size=1
        )
        
        # 활성화 함수와 드롭아웃 정의
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate)
        
        # Transformer encoder layers
        self.encoder_layer1 = EncoderLayer(
            d_model=self.nb_units,
            n_heads=self.n_heads,
            d_ff=self.nb_units*4,
            dropout=self.dropout_rate
        )
        
        self.encoder_layer2 = EncoderLayer(
            d_model=self.nb_units,
            n_heads=self.n_heads,
            d_ff=self.nb_units*4,
            dropout=self.dropout_rate
        )
        
        # Global temporal attention
        self.attention_with_context = AttentionWithContext(self.nb_units)
        
        self.fc1 = nn.Linear(self.nb_units, 4*self.output_size)
        self.fc_out = nn.Linear(4*self.output_size, self.output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape for 2D convolution: [batch, 1, window_size, channels]
        x = x.unsqueeze(1)
        
        # Initial convolution
        x = self.first_conv(x)
        x = x.squeeze(1)
        # Shape: [batch, window_size, channels]
        
        # Apply sensor attention
        si, _ = self.sensor_attention(x)
        
        # Feature transformation with 1D convolution
        x = self.conv1d(si.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu(x)
        
        # Apply transformer encoder layers
        x = self.encoder_layer1(x) # batch * len * d_dim
        x = self.encoder_layer2(x) # batch * len * d_dim
        
        # Apply global temporal attention
        x = self.attention_with_context(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        
        return x 