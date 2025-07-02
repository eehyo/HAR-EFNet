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
                                      lr=args.learning_rate, 
                                      weight_decay=args.weight_decay)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), 
                                     lr=args.learning_rate, 
                                     weight_decay=args.weight_decay,
                                     momentum=0.9)
        
        # Learning rate scheduler (basic step scheduler as requested)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=args.learning_rate_factor,
            patience=args.learning_rate_patience,
            verbose=True
        )
        
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
        
        # Get run_id from Logger for consistent naming
        run_id = Logger.get_run_id()
        
        # Training loop
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # Train one epoch
            train_loss, train_time = self.train_epoch(masked_train_loader)
            
            # Validate
            valid_loss = self.validate(masked_valid_loader)
            
            # Update learning rate scheduler
            self.scheduler.step(valid_loss)
            
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Valid Loss: {valid_loss:.6f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Early stopping check
            self.early_stopping(valid_loss, self.model)
            
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping triggered")
                break
            
            # Save best model
            if self.early_stopping.save_checkpoint:
                model_path = os.path.join(self.save_path, f"best_model_{run_id}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                }, model_path)
                self.logger.info(f"Model saved to {model_path}")
        
        # Load best model
        best_model_path = os.path.join(self.save_path, f"best_model_{run_id}.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded best model from {best_model_path}")
        
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
        encoder: Encoder model to load weights into
        path: Path to saved model checkpoint
        
    Returns:
        Encoder with loaded weights
    """
    logger = Logger.get_logger("masked_encoder_loader")
    
    if not os.path.exists(path):
        logger.error(f"Checkpoint file not found: {path}")
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model state dict from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            encoder.load_state_dict(checkpoint)
            logger.info("Loaded model state dict (legacy format)")
            
        logger.info(f"Successfully loaded masked reconstruction encoder from {path}")
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise e
    
    return encoder 