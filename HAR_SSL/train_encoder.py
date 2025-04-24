import os
import time
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import yaml

from encoders import CNNEncoder, LSTMEncoder
from dataloaders.data_utils import compute_ecdf_features, compute_batch_ecdf_features
from utils.training_utils import EarlyStopping, adjust_learning_rate, set_seed
from utils.logger import Logger

class EncoderTrainer:
    """
    ECDF feature prediction encoder training class
    """
    def __init__(self, args, model, save_path):
        """
        Initialize the encoder trainer
        
        Args:
            args: configuration parameters
            model: encoder model to train
            save_path: model save path
        """
        self.model = model
        self.device = torch.device("cuda" if args.use_gpu else "cpu")
        self.model.to(self.device)
        
        # Initialize logger
        self.logger = Logger(f"encoder_{args.encoder_type}")
        
        # Use MSE Loss
        self.criterion = nn.MSELoss()
        
        # Setup optimizer
        if args.optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate)
        
        # Save path and logging setup
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # Training settings
        self.epochs = args.train_epochs
        
        # Early stopping and learning rate adjustment
        self.early_stopping = EarlyStopping(patience=args.early_stop_patience, verbose=True)
        self.learning_rate_adapter = adjust_learning_rate(args, verbose=True)
    
    def train_epoch(self, train_loader):
        """
        Train one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Training loss, epoch time
        """
        self.model.train()
        train_loss = []
        epoch_time = time.time()
        # batch_x.shape: (128, 1, 168, 9)
        # batch_x.shape: (64, 1, 168, 9)
        for batch_x, _ in train_loader:  # label is not needed for ECDF features
            # Process input data
            batch_x = batch_x.float().to(self.device)
            
            # Compute ECDF features (Ground Truth)
            batch_ecdf = compute_batch_ecdf_features(batch_x.cpu().numpy())
            batch_ecdf = torch.tensor(batch_ecdf).float().to(self.device)
            
            # Predict ECDF features
            predicted_ecdf = self.model(batch_x)
            
            loss = self.criterion(predicted_ecdf, batch_ecdf)
            
            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss.append(loss.item())
        
        epoch_time = time.time() - epoch_time
        train_loss = np.average(train_loss)
        
        return train_loss, epoch_time
    
    def validate(self, valid_loader):
        """
        Validate model
        
        Args:
            valid_loader: Validation data loader
            
        Returns:
            Validation loss
        """
        self.model.eval()
        valid_loss = []
        
        with torch.no_grad():
            for batch_x, _ in valid_loader:  
                batch_x = batch_x.float().to(self.device)
                
                batch_ecdf = compute_batch_ecdf_features(batch_x.cpu().numpy())
                batch_ecdf = torch.tensor(batch_ecdf).float().to(self.device)
                
                predicted_ecdf = self.model(batch_x)
                
                loss = self.criterion(predicted_ecdf, batch_ecdf)
                
                valid_loss.append(loss.item())
        
        valid_loss = np.average(valid_loss)
        
        return valid_loss
    
    def train(self, train_loader, valid_loader):
        """
        Complete training process
        
        Args:
            train_loader: Training data loader
            valid_loader: Validation data loader
            
        Returns:
            Trained model
        """
        self.logger.info(f"Starting encoder training, saving logs to: {self.save_path}")
        
        # Save the best performance
        best_valid_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Training phase
            train_loss, epoch_time = self.train_epoch(train_loader)
            
            log_message = f"Epoch: {epoch+1}, train_loss: {train_loss:.7f}, time: {epoch_time:.2f}s"
            self.logger.info(log_message)
            
            # Validation phase
            valid_loss = self.validate(valid_loader)
            
            # Log validation results
            log_message = f"Validation: Epoch: {epoch+1}, Train Loss: {train_loss:.7f}, Valid Loss: {valid_loss:.7f}"
            self.logger.info(log_message)
            
            # Early stopping check
            self.early_stopping(valid_loss, self.model, self.save_path, log=None)
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping triggered")
                break
            
            # Learning rate adjustment
            self.learning_rate_adapter(self.optimizer, valid_loss)
        
        # Training complete
        self.logger.info("Encoder training completed")
        return self.model

def create_encoder(args):
    """
    Create encoder model based on configuration
    
    Args:
        args: Configuration parameters
        
    Returns:
        Created encoder model
    """
    # Convert args to dict
    encoder_args = {
        'input_channels': args.input_channels,
        'window_size': args.window_size,
        'output_size': args.output_size,
        'device': args.device
    }
    
    # Load model configuration
    config_file = open('HAR_SSL/configs/model.yaml', mode='r')
    model_config = yaml.load(config_file, Loader=yaml.FullLoader)
    encoder_config = model_config['ssl_encoder']
    
    # Add parameters based on encoder type
    if args.encoder_type == 'cnn':
        encoder_args.update(encoder_config['cnn'])
        encoder = CNNEncoder(encoder_args)
    elif args.encoder_type == 'lstm':
        encoder_args.update(encoder_config['lstm'])
        encoder = LSTMEncoder(encoder_args)
    else:
        raise ValueError(f"Unsupported encoder type: {args.encoder_type}")
    
    return encoder

def load_pretrained_encoder(encoder, path):
    """
    Load pretrained encoder weights
    
    Args:
        encoder: Encoder model
        path: Path to weights file
        
    Returns:
        Encoder with loaded weights
    """
    logger = Logger("encoder_loader")
    logger.info(f"Loading pretrained encoder from: {path}")
    
    checkpoint = torch.load(path, map_location=encoder.device)
    encoder.load_state_dict(checkpoint['model_state_dict'])
    
    return encoder 