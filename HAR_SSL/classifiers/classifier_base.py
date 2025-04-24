import torch
import torch.nn as nn

class ClassifierModel(nn.Module):
    """
    Classifier model based on pretrained encoder
    
    This model combines a pretrained encoder with a classification head
    """
    def __init__(self, encoder, num_classes, config):
        """
        Initialize classifier
        
        Args:
            encoder: Pretrained encoder
            num_classes: Number of classes to classify
            config: Configuration for the classifier
        """
        super(ClassifierModel, self).__init__()
        
        self.encoder = encoder
        self.encoder_output_size = encoder.get_embedding_dim()
        
        # Build classification head
        layers = []
        
        hidden_sizes = config.get('classifier_hidden', [256, 128])
        dropout_rate = config.get('dropout_rate', 0.5)
        
        # Input layer
        input_size = self.encoder_output_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Freeze encoder if required
        if config.get('freeze_encoder', True):
            self.freeze_encoder()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input data [batch_size, window_size, input_channels]
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        # Extract features from encoder
        features = self.encoder(x)
        
        # Predict classes with classification head
        logits = self.classifier(features)
        
        return logits
    
    def freeze_encoder(self):
        """
        Freeze encoder parameters to train only the classification head
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """
        Make encoder parameters trainable again
        """
        for param in self.encoder.parameters():
            param.requires_grad = True 