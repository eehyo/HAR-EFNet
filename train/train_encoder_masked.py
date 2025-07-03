import os
import time
import torch
import numpy as np
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Optional, Any, Union

from encoders.masked import MaskedDeepConvLSTMEncoder, MaskedDeepConvLSTMAttnEncoder, MaskedSAHAREncoder
from dataloaders.masked_data_utils import create_masked_dataloader
from utils.training_utils import EarlyStopping, adjust_learning_rate, set_seed
from utils.logger import Logger

# Initialize global logger
Logger.initialize(log_dir='logs')


class MaskedReconstructionTrainer:
    """
    Masked Reconstruction Trainer class
    """
    def __init__(self, args: Any, model: nn.Module, save_path: str):
        """
        Initialize Masked Reconstruction trainer
        
        Args:
            args: Configuration parameters
            model: Masked reconstruction model to train
            save_path: Model save path
        """
        self.model = model
        self.device = args.device
        self.model.to(self.device)
        
        # Initialize logger
        self.logger = Logger(f"masked_{args.encoder_type}")
        self.logger.info(f"Using device: {self.device}")
        
        # Configure optimizer with weight decay
        if args.optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), 
                                      lr=args.learning_rate)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), 
                                     lr=args.learning_rate)
        
        
        # Save path and logging setup
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # Training settings
        self.epochs = args.train_epochs
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=args.early_stop_patience, verbose=True, 
                                           logger_name=f"es_masked_{args.encoder_type}")
        
        # Masking settings
        self.mask_choice = args.mask_choice
        
        self.logger.info(f"Masked reconstruction trainer initialized with mask probability: {self.mask_choice}")
        
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
        
        for batch_data in train_loader:
            batch_count += 1
            self.logger.debug(f"Processing batch #{batch_count} in train epoch")
            
            # Unpack masked reconstruction data
            # batch_data: (x_perturbed, x_target, choice_mask)
            x_perturbed = batch_data[0].float().to(self.device)
            x_target = batch_data[1].float().to(self.device)
            choice_mask = batch_data[2].float().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Calculate masked reconstruction loss
            input_target = (x_perturbed, x_target, choice_mask)
            loss = self.model.get_loss(input_target, validation=False)
            
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            
            train_loss.append(loss.item())
        
        epoch_time = time.time() - epoch_time
        train_loss = np.average(train_loss)
        self.logger.info(f"Completed epoch with {batch_count} batches")
        
        return train_loss, epoch_time
    
    def validate(self, valid_loader: DataLoader) -> float:
        """
        Validate model using masked reconstruction loss
        
        Args:
            valid_loader: Validation data loader
            
        Returns:
            Validation loss
        """
        self.model.eval()
        valid_loss = []
        batch_count = 0
        
        with torch.no_grad():
            for batch_data in valid_loader:
                batch_count += 1
                self.logger.debug(f"Processing batch #{batch_count} in validation")
                
                # Unpack masked reconstruction data
                x_perturbed = batch_data[0].float().to(self.device)
                x_target = batch_data[1].float().to(self.device)
                choice_mask = batch_data[2].float().to(self.device)
                
                # Calculate loss
                input_target = (x_perturbed, x_target, choice_mask)
                loss = self.model.get_loss(input_target, validation=True)
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
        self.logger.info(f"Starting masked reconstruction encoder training, saving to: {self.save_path}")
        
        # Convert dataloaders to masked reconstruction format
        masked_train_loader = create_masked_dataloader(train_loader, mask_choice=self.mask_choice)
        masked_valid_loader = create_masked_dataloader(valid_loader, mask_choice=self.mask_choice)
        
        for epoch in range(self.epochs):
            # Training phase
            train_loss, train_time = self.train_epoch(masked_train_loader)
            
            # Log training progress
            log_message = f"Epoch: {epoch+1}, train_loss: {train_loss:.7f}, time: {train_time:.2f}s"
            self.logger.info(log_message)
            
            # Validation phase
            valid_loss = self.validate(masked_valid_loader)
            
            # Log validation results
            log_message = f"Validation: Epoch: {epoch+1}, Train Loss: {train_loss:.7f}, Valid Loss: {valid_loss:.7f}"
            self.logger.info(log_message)
            
            # Early stopping check
            self.early_stopping(valid_loss, self.model, self.save_path, None)
            
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping triggered")
                break
        
        self.logger.info("Masked reconstruction training completed")
        return self.model


def create_masked_encoder(args: Any) -> nn.Module:
    """
    Create masked reconstruction encoder based on specified encoder type
    
    Args:
        args: Configuration arguments containing encoder_type and model parameters
        
    Returns:
        Masked reconstruction encoder model
    """
    logger = Logger("masked_encoder_creator")
    
    # Convert args to dict - same as train_encoder.py
    encoder_args = {
        'input_channels': args.input_channels,
        'window_size': args.window_size,
        'output_size': args.output_size,
        'device': args.device
    }
    
    # Load model configuration - use relative path for flexibility
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'model.yaml')
    with open(config_path, mode='r') as config_file:
        model_config = yaml.load(config_file, Loader=yaml.FullLoader)
    
    encoder_config = model_config['efnet_encoder']
    
    if args.encoder_type == 'deepconvlstm':
        encoder_args.update(encoder_config.get('deepconvlstm', {}))
        model_class = MaskedDeepConvLSTMEncoder
        logger.info(f"Using DeepConvLSTM masked reconstruction encoder configuration")

    elif args.encoder_type == 'deepconvlstm_attn':
        encoder_args.update(encoder_config.get('deepconvlstm_attn', {}))
        model_class = MaskedDeepConvLSTMAttnEncoder
        logger.info(f"Using DeepConvLSTM with Attention masked reconstruction encoder configuration")

    elif args.encoder_type == 'sa_har':
        encoder_args.update(encoder_config.get('sa_har', {}))
        model_class = MaskedSAHAREncoder
        logger.info(f"Using SA-HAR masked reconstruction encoder configuration")
        
    else:
        logger.error(f"Unsupported encoder type: {args.encoder_type}")
        raise ValueError(f"Unsupported encoder type: {args.encoder_type}")
    
    # Create selected encoder model
    encoder = model_class(encoder_args)
    
    logger.info(f"Created {args.encoder_type} masked reconstruction encoder")
    return encoder


def load_pretrained_masked_encoder(encoder: nn.Module, path: str) -> nn.Module:
    """
    Load pretrained masked reconstruction encoder
    
    Args:
        encoder: Masked reconstruction encoder model instance
        path: Path to checkpoint file
        
    Returns:
        Encoder with loaded weights
        
    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist
        RuntimeError: If there's an error loading the state dict
    """
    logger = Logger("masked_encoder_loader")
    logger.info(f"Loading pretrained masked reconstruction encoder from: {path}")
    
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