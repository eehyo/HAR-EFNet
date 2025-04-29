import os
import time
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import yaml
from typing import Tuple, Dict, List, Optional, Any, Union

from encoders import CNNEncoder, LSTMEncoder, DeepConvLSTMEncoder, DeepConvLSTMAttnEncoder, SAHAREncoder
from dataloaders.data_utils import compute_ecdf_features, compute_batch_ecdf_features
from utils.training_utils import EarlyStopping, adjust_learning_rate, set_seed
from utils.logger import Logger

# Initialize global logger
Logger.initialize(log_dir='logs')

class EncoderTrainer:
    """
    ECDF feature prediction encoder training class
    """
    def __init__(self, args: Any, model: nn.Module, save_path: str):
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
        self.logger.info(f"Using device: {self.device}")
        
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
        self.early_stopping = EarlyStopping(patience=args.early_stop_patience, verbose=True, 
                                          logger_name=f"es_encoder_{args.encoder_type}")
        self.learning_rate_adapter = adjust_learning_rate(args, verbose=True, 
                                                      logger_name=f"lr_encoder_{args.encoder_type}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Training loss, epoch time in seconds
        """
        self.model.train()
        train_loss = []
        epoch_time = time.time()
        batch_count = 0
        # batch_x.shape: (128, 168, 9)
        for batch_x, _ in train_loader:  # label is not needed for ECDF features
            batch_count += 1
            self.logger.debug(f"Processing batch #{batch_count} in train epoch")
            
            # Process input data
            batch_x = batch_x.float().to(self.device)
            
            # Compute ECDF features (Ground Truth)
            batch_ecdf = torch.tensor(compute_batch_ecdf_features(batch_x), 
                                    dtype=torch.float32).to(self.device)
            
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
        self.logger.info(f"Completed epoch with {batch_count} batches")
        
        return train_loss, epoch_time
    
    def validate(self, valid_loader: DataLoader) -> float:
        """
        Validate model
        
        Args:
            valid_loader: Validation data loader
            
        Returns:
            Validation loss
        """
        self.model.eval()
        valid_loss = []
        batch_count = 0
        
        with torch.no_grad():
            for batch_x, _ in valid_loader:  
                batch_count += 1
                self.logger.debug(f"Processing batch #{batch_count} in validation")
                
                batch_x = batch_x.float().to(self.device)
                
                batch_ecdf = torch.tensor(compute_batch_ecdf_features(batch_x), 
                                        dtype=torch.float32).to(self.device)
                
                predicted_ecdf = self.model(batch_x)
                
                loss = self.criterion(predicted_ecdf, batch_ecdf)
                
                valid_loss.append(loss.item())
        
        valid_loss = np.average(valid_loss)
        self.logger.info(f"Completed validation with {batch_count} batches")
        
        return valid_loss
    
    def train(self, train_loader: DataLoader, valid_loader: DataLoader) -> nn.Module:
        """
        Complete training process
        
        Args:
            train_loader: Training data loader
            valid_loader: Validation data loader
            
        Returns:
            Trained model
        """
        self.logger.info(f"Starting encoder training, saving to: {self.save_path}")
        
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
            self.early_stopping(valid_loss, self.model, self.save_path, None)
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping triggered")
                break
            
            # Learning rate adjustment
            self.learning_rate_adapter(self.optimizer, valid_loss)
        
        # Training complete
        self.logger.info("Encoder training completed")
        return self.model


def create_encoder(args: Any) -> nn.Module:
    """
    Create encoder model based on configuration
    
    Args:
        args: Configuration parameters
        
    Returns:
        Created encoder model
    """
    logger = Logger("encoder_creator")
    
    # Convert args to dict
    encoder_args = {
        'input_channels': args.input_channels,
        'window_size': args.window_size,
        'output_size': args.output_size,
        'device': args.device
    }
    
    # Load model configuration - use relative path for flexibility
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'model.yaml')
    with open(config_path, mode='r') as config_file:
        model_config = yaml.load(config_file, Loader=yaml.FullLoader)
    
    encoder_config = model_config['efnet_encoder']
    
    if args.encoder_type == 'cnn':
        encoder_args.update(encoder_config['cnn'])
        model_class = CNNEncoder
        logger.info(f"Using CNN encoder configuration")

    elif args.encoder_type == 'lstm':
        encoder_args.update(encoder_config['lstm'])
        model_class = LSTMEncoder
        logger.info(f"Using LSTM encoder configuration")

    elif args.encoder_type == 'deepconvlstm':
        encoder_args.update(encoder_config.get('deepconvlstm', {}))
        model_class = DeepConvLSTMEncoder
        logger.info(f"Using DeepConvLSTM encoder configuration")

    elif args.encoder_type == 'deepconvlstm_attn':
        encoder_args.update(encoder_config.get('deepconvlstm_attn', {}))
        model_class = DeepConvLSTMAttnEncoder
        logger.info(f"Using DeepConvLSTM with Attention encoder configuration")

    elif args.encoder_type == 'sa_har':
        encoder_args.update(encoder_config.get('sa_har', {}))
        model_class = SAHAREncoder
        logger.info(f"Using SA-HAR encoder configuration")
        
    else:
        logger.error(f"Unsupported encoder type: {args.encoder_type}")
        raise ValueError(f"Unsupported encoder type: {args.encoder_type}")
    
    # Create selected encoder model
    encoder = model_class(encoder_args)
    
    logger.info(f"Created {args.encoder_type} encoder")
    return encoder

def load_pretrained_encoder(encoder: nn.Module, path: str) -> nn.Module:
    """
    Load pretrained encoder weights from checkpoint file
    
    Args:
        encoder: Encoder model instance
        path: Path to checkpoint file
        
    Returns:
        Encoder with loaded weights
        
    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist
        RuntimeError: If there's an error loading the state dict
    """
    logger = Logger("encoder_loader")
    logger.info(f"Loading pretrained encoder from: {path}")
    
    if not os.path.exists(path):
        logger.error(f"Checkpoint file not found: {path}")
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    
    try:
        checkpoint = torch.load(path, map_location=encoder.device, weights_only=False)
        encoder.load_state_dict(checkpoint['model_state_dict'])
        val_loss = checkpoint.get('val_loss', 'N/A')
        logger.info(f"Successfully loaded model with validation loss: {val_loss}")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
    
    return encoder 