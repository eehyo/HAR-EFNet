import os
import time
import torch
import numpy as np
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Optional, Any, Union
import itertools

from encoders import SimCLRDeepConvLSTMEncoder, SimCLRDeepConvLSTMAttnEncoder, SimCLRSAHAREncoder
from dataloaders.simclr_transformations import apply_random_transform_pair, apply_transform_combinations
from utils.training_utils import EarlyStopping, adjust_learning_rate, set_seed
from utils.logger import Logger
from lightly.loss import NTXentLoss

# Initialize global logger
Logger.initialize(log_dir='logs')


class SimCLREncoderTrainer:
    """
    SimCLR Contrastive Learning Trainer class
    """
    def __init__(self, args: Any, model: nn.Module, save_path: str):
        """
        Initialize SimCLR trainer
        
        Args:
            args: Configuration parameters
            model: SimCLR model to train
            save_path: Model save path
        """
        self.model = model
        self.device = args.device
        self.model.to(self.device)
        
        # Initialize logger
        self.logger = Logger(f"simclr_{args.encoder_type}")
        self.logger.info(f"Using device: {self.device}")
        
        # NT-Xent Loss
        self.temperature = getattr(args, 'temperature', 0.1)
        self.criterion = NTXentLoss(temperature=self.temperature)
        
        # Configure optimizer with weight decay
        if args.optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), 
                                      lr=args.learning_rate, 
                                      weight_decay=args.weight_decay)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), 
                                     lr=args.learning_rate, 
                                     weight_decay=args.weight_decay,
                                     momentum=0.9)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.train_epochs
        )
        
        # Save path and logging setup
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # Training settings
        self.epochs = args.train_epochs
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=args.early_stop_patience, verbose=True, 
                                           logger_name=f"es_simclr_{args.encoder_type}")
        
        # Validation mode: 'random' or 'combinations'
        self.validation_mode = getattr(args, 'validation_mode', 'random')
        
    def generate_contrastive_pairs(self, batch_x: torch.Tensor, mode: str = 'random') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate contrastive pairs from input batch
        
        Args:
            batch_x: Original input data [batch_size, window_size, channels]
            mode: 'random' for random transformation pairs, 'combinations' for all combinations
            
        Returns:
            Tuple of (view1, view2) tensors
        """
        batch_size = batch_x.shape[0]
        view1_list = []
        view2_list = []
        
        for i in range(batch_size):
            x_np = batch_x[i].cpu().numpy()  # [window_size, channels]
            
            if mode == 'random':
                # Apply random transformation pair
                x1, x2 = apply_random_transform_pair(x_np)
                view1_list.append(x1)
                view2_list.append(x2)
            else:
                # For validation, use first transformation combination
                pairs = apply_transform_combinations(x_np)
                if pairs:
                    x1, x2 = pairs[0]  # Use first combination
                    view1_list.append(x1)
                    view2_list.append(x2)
                else:
                    # Fallback to random if no pairs available
                    x1, x2 = apply_random_transform_pair(x_np)
                    view1_list.append(x1)
                    view2_list.append(x2)
        
        # Convert to tensors
        view1 = torch.tensor(np.stack(view1_list), dtype=torch.float32).to(self.device)
        view2 = torch.tensor(np.stack(view2_list), dtype=torch.float32).to(self.device)
        
        return view1, view2
    
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
        
        for batch_x, _ in train_loader:  # Labels not needed for SimCLR
            batch_count += 1
            self.logger.debug(f"Processing batch #{batch_count} in train epoch")
            
            # Prepare original data
            batch_x = batch_x.float().to(self.device)
            
            # Generate contrastive pairs
            view1, view2 = self.generate_contrastive_pairs(batch_x, mode='random')
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get projections for both views
            z1 = self.model(view1)  # [batch_size, projection_dim]
            z2 = self.model(view2)  # [batch_size, projection_dim]
            
            # Calculate contrastive loss
            loss = self.criterion(z1, z2)
            
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            
            train_loss.append(loss.item())
        
        # Update learning rate scheduler
        self.scheduler.step()
        
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
                
                # Generate contrastive pairs for validation
                view1, view2 = self.generate_contrastive_pairs(batch_x, mode=self.validation_mode)
                
                # Forward pass
                z1 = self.model(view1)
                z2 = self.model(view2)
                
                # Calculate loss
                loss = self.criterion(z1, z2)
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
        self.logger.info(f"Starting SimCLR encoder training, saving to: {self.save_path}")
        self.logger.info(f"Using temperature: {self.temperature}")
        
        for epoch in range(self.epochs):
            # Training phase
            train_loss, epoch_time = self.train_epoch(train_loader)
            
            # Log training progress
            current_lr = self.scheduler.get_last_lr()[0]
            log_message = f"Epoch: {epoch+1}, train_loss: {train_loss:.7f}, time: {epoch_time:.2f}s, lr: {current_lr:.6f}"
            self.logger.info(log_message)
            
            # Validation phase
            valid_loss = self.validate(valid_loader)
            
            # Log validation results
            log_message = f"Validation: Epoch: {epoch+1}, Train Loss: {train_loss:.7f}, Valid Loss: {valid_loss:.7f}"
            self.logger.info(log_message)
            
            # Check early stopping
            self.early_stopping(valid_loss, self.model, self.save_path, None)
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping triggered")
                break
        
        # Training complete
        self.logger.info("SimCLR encoder training completed")
        return self.model


def create_simclr_encoder(args: Any) -> nn.Module:
    """
    Create SimCLR encoder model
    
    Args:
        args: Configuration parameters
        
    Returns:
        Created SimCLR encoder model
    """
    logger = Logger("simclr_encoder_creator")
    
    # Convert args to dict
    encoder_args = {
        'input_channels': args.input_channels,
        'window_size': args.window_size,
        'output_size': args.output_size,  # (3, 78)
        'device': args.device,
        'projection_dim': getattr(args, 'projection_dim', 50),
        'projection_hidden_dim1': getattr(args, 'projection_hidden_dim1', 256),
        'projection_hidden_dim2': getattr(args, 'projection_hidden_dim2', 128)
    }
    
    # Load model configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'model.yaml')
    with open(config_path, mode='r') as config_file:
        model_config = yaml.load(config_file, Loader=yaml.FullLoader)
    
    encoder_config = model_config['efnet_encoder']
    
    if args.encoder_type == 'deepconvlstm':
        encoder_args.update(encoder_config.get('deepconvlstm', {}))
        model_class = SimCLRDeepConvLSTMEncoder
        logger.info(f"Using SimCLR DeepConvLSTM encoder configuration")

    elif args.encoder_type == 'deepconvlstm_attn':
        encoder_args.update(encoder_config.get('deepconvlstm_attn', {}))
        model_class = SimCLRDeepConvLSTMAttnEncoder
        logger.info(f"Using SimCLR DeepConvLSTM with Attention encoder configuration")

    elif args.encoder_type == 'sa_har':
        encoder_args.update(encoder_config.get('sa_har', {}))
        model_class = SimCLRSAHAREncoder
        logger.info(f"Using SimCLR SA-HAR encoder configuration")
        
    else:
        logger.error(f"Unsupported encoder type: {args.encoder_type}")
        raise ValueError(f"Unsupported encoder type: {args.encoder_type}")
    
    # Create selected SimCLR encoder model
    encoder = model_class(encoder_args)
    
    logger.info(f"Created SimCLR {args.encoder_type} encoder")
    return encoder

def load_pretrained_simclr_encoder(encoder: nn.Module, path: str) -> nn.Module:
    """
    Load pretrained SimCLR encoder weights from checkpoint file
    
    Args:
        encoder: SimCLR encoder model instance
        path: Checkpoint file path
        
    Returns:
        Encoder with loaded weights
        
    Raises:
        FileNotFoundError: If checkpoint file does not exist
        RuntimeError: If state dictionary loading error occurs
    """
    logger = Logger("simclr_encoder_loader")
    logger.info(f"Loading pretrained SimCLR encoder from: {path}")
    
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
