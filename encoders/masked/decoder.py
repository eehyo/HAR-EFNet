import torch
from torch import nn


class Decoder(nn.Module):
    """
    Decoder for masked reconstruction task
    Reconstructs original time series data from encoder representations
    """
    def __init__(self, repr_size, in_size, window_size):
        super(Decoder, self).__init__()
        hiddens = 256, 128
        dropout = 0.2
        
        self.layer1 = nn.Sequential(
            nn.Linear(repr_size, hiddens[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hiddens[0])
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hiddens[0], hiddens[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hiddens[1])
        )

        self.layer3 = nn.Sequential(
            nn.Linear(hiddens[1], in_size * window_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(in_size * window_size)
        )
        
        self.in_size = in_size
        self.window_size = window_size

    def forward(self, x):
        """
        Forward pass through decoder
        
        Args:
            x: Encoded representation [batch_size, repr_size]
            
        Returns:
            Reconstructed time series [batch_size, window_size, in_size]
        """
        # Flatten input representation
        x = x.view(x.shape[0], -1)
        
        # Apply decoder layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Reshape to original time series format
        return x.view(x.shape[0], self.window_size, self.in_size)


class MaskedReconstruction(nn.Module):
    """
    Masked Reconstruction model combining encoder and decoder
    """
    def __init__(self, encoder, window_size, **kwargs):
        super(MaskedReconstruction, self).__init__()
        self.encoder = encoder
        self.window_size = window_size
        
        # Get representation size from encoder
        repr_size = encoder.get_embedding_dim()
        in_size = encoder.input_channels
        
        self.decoder = Decoder(repr_size=repr_size, in_size=in_size, window_size=window_size)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        """
        Forward pass through encoder and decoder
        
        Args:
            x: Input time series [batch_size, window_size, input_channels]
            
        Returns:
            Reconstructed time series [batch_size, window_size, input_channels]
        """
        # Get representation from encoder
        x_repr = self.encoder.get_embedding(x)
        
        # Reconstruct from representation
        return self.decoder(x_repr)

    def forward_from_repr(self, x_repr):
        """
        Forward pass from representation through decoder only
        
        Args:
            x_repr: Encoded representation [batch_size, repr_size]
            
        Returns:
            Reconstructed time series [batch_size, window_size, input_channels]
        """
        return self.decoder(x_repr)

    def get_loss(self, input_target, validation=False):
        """
        Calculate masked reconstruction loss
        
        Args:
            input_target: Tuple of (x_original, x_target, choice_mask)
            validation: Whether in validation mode
            
        Returns:
            MSE loss between masked reconstruction and target
        """
        x_original, x_target, choice_mask = input_target
        
        # Forward pass through encoder-decoder
        reconstructed = self.forward(x_original)
        
        # Apply mask and calculate loss
        # L(F, M; g, h) = ||(1 − M) ⊙ [X − h(g(M ⊙ X ))]||^2_Fr
        # Extract only masked positions for loss calculation
        # choice_mask * reconstructed  -> Extract reconstruction values only at masked positions
        # choice_mask * x_target       -> Extract target values only at masked positions
        return self.criterion(choice_mask * reconstructed, choice_mask * x_target) 