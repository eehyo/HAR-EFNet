import os
import time
import torch
import numpy as np
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Dict, List, Optional, Any, Union, Callable

from encoders import MTLDeepConvLSTMEncoder, MTLDeepConvLSTMAttnEncoder, MTLSAHAREncoder
from dataloaders.transformations import (
    noise_transform,
    scaling_transform,
    rotation_transform,
    negate_transform,
    channel_shuffle_transform,
    time_segment_permutation_transform_improved,
    time_warp_transform,
    horizontal_flip_transform
)
from utils.training_utils import EarlyStopping, adjust_learning_rate, set_seed
from utils.logger import Logger

# Initialize global logger
Logger.initialize(log_dir='logs')

class MTLEncoderTrainer:
    """
    Self-Supervised Multi-Task Learning Trainer class
    Learns binary classification tasks that predict whether various time series transformations (augmentations) were applied
    """
    def __init__(self, args: Any, model: nn.Module, transform_funcs: Dict[str, Callable], save_path: str):
        """
        Initialize Self-Supervised MTL Trainer
        
        Args:
            args: Configuration parameters
            model: MTL model to train
            transform_funcs: Dictionary of transformation functions {task_name: transform_function}
            save_path: Model save path
        """
        self.model = model
        self.device = args.device
        self.model.to(self.device)
        
        # Initialize Logger
        self.logger = Logger(f"mtl_{args.encoder_type}")
        self.logger.info(f"Using device: {self.device}")
        
        # Configure transformation functions
        self.transform_funcs = transform_funcs
        self.task_names = list(transform_funcs.keys())
        
        # Task-specific weights
        self.task_weights = args.task_weights
        
        # Validate that all transformation functions have corresponding weights
        for task_name in self.task_names:
            if task_name not in self.task_weights:
                self.logger.warning(f"Missing weight for transformation task '{task_name}'. Using default weight 1.0.")
                self.task_weights[task_name] = 1.0
        
        # Validate that all weights have corresponding transformation functions
        for task_name in self.task_weights:
            if task_name not in self.transform_funcs:
                self.logger.warning(f"Weight specified for '{task_name}' but no transformation function found. This weight will be ignored.")
        
        self.logger.info(f"Using transformations: {sorted(self.task_names)}")
        self.logger.info(f"Task weights: {self.task_weights}")
        
        # Loss function: BCE Loss for binary classification
        self.criterion = nn.BCELoss()
        
        # Configure optimizer with L2 regularization (weight decay) of 0.0001
        if args.optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # Save path and logging setup
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # Training settings
        self.epochs = args.train_epochs
        
        # Early stopping and learning rate adjustment
        self.early_stopping = EarlyStopping(patience=args.early_stop_patience, verbose=True, 
                                           logger_name=f"es_mtl_{args.encoder_type}")
        
    def generate_ssl_batch(self, batch_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate batch data for SSL learning
        
        Args:
            batch_x: Original input data [batch_size, window_size, channels]
            
        Returns:
            Augmented input data and corresponding labels (whether transformation was applied)
        """
        B, T, C = batch_x.shape
        task_num = len(self.task_names)
        
        # Convert to numpy for batch processing
        batch_np = batch_x.cpu().numpy()  # [B, T, C]
        
        # Initialize arrays for augmented data and labels
        X_aug = np.zeros((B, task_num, T, C), dtype=np.float32)  # [B, num_tasks, T, C]
        Y = np.zeros((B, task_num), dtype=np.float32)            # [B, num_tasks]
        
        # Apply each transformation to the entire batch
        for t_idx, tname in enumerate(self.task_names):
            # Get transformation function
            tfunc = self.transform_funcs[tname]
            
            # Generate random mask for each sample in batch (50% probability)
            apply_mask = np.random.rand(B) < 0.5  # [B]
            
            # Create a copy of the original batch for this task
            task_batch = batch_np.copy()  # [B, T, C]
            
            # Apply transformation to samples where mask is True
            if np.any(apply_mask):
                # Apply transformation to selected samples in batch
                samples_to_transform = task_batch[apply_mask]  # [num_selected, T, C]
                if samples_to_transform.shape[0] > 0:
                    transformed_samples = tfunc(samples_to_transform)  # [num_selected, T, C]
                    task_batch[apply_mask] = transformed_samples
            
            # Store the augmented data and labels
            X_aug[:, t_idx] = task_batch  # [B, T, C]
            Y[:, t_idx] = apply_mask.astype(np.float32)  # [B] - 1 if transformed, 0 if not
        
        # Convert numpy -> torch and move to device
        X_aug = torch.tensor(X_aug, dtype=torch.float32).to(self.device)  # [B, num_tasks, T, C]
        Y = torch.tensor(Y, dtype=torch.float32).to(self.device)          # [B, num_tasks]
        
        return X_aug, Y
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Training loss, epoch time (seconds)
        """
        self.model.train()
        train_loss = []
        task_losses = {task: [] for task in self.task_names}
        
        epoch_time = time.time()
        batch_count = 0
        
        for batch_x, _ in train_loader:  # Labels not needed for MTL
            batch_count += 1
            self.logger.debug(f"Processing batch #{batch_count} in train epoch")
            
            # Prepare original data
            batch_x = batch_x.float().to(self.device)  # [B, T, C]
            
            # Generate augmented data and labels for SSL learning
            X_aug, Y = self.generate_ssl_batch(batch_x)  # [B, num_tasks, T, C], [B, num_tasks]
            
            # Batch size and number of tasks
            B = X_aug.shape[0]
            
            # Initialize total loss for all tasks
            total_loss = 0
            self.optimizer.zero_grad()
            
            # Process each task
            for t_idx, tname in enumerate(self.task_names):
                # Extract input data and labels for current task
                x_t = X_aug[:, t_idx]        # [B, T, C]
                y_t = Y[:, t_idx]            # [B]
                
                # Model prediction (for specific task)
                out_t = self.model(x_t, task=tname)  # [B, 1]
                
                # Calculate and accumulate loss
                loss_t = self.task_weights[tname] * self.criterion(out_t.squeeze(), y_t)
                total_loss += loss_t
                
                # Record task-specific loss
                task_losses[tname].append(loss_t.item())
            
            # Backpropagation and optimization
            total_loss.backward()
            self.optimizer.step()
            
            # Record total loss
            train_loss.append(total_loss.item())
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_time
        
        # Calculate average loss
        train_loss = np.average(train_loss)
        
        # Calculate and log task-specific average losses
        for tname in self.task_names:
            task_avg_loss = np.average(task_losses[tname])
            self.logger.debug(f"Task {tname} average loss: {task_avg_loss:.6f}")
        
        self.logger.info(f"Completed epoch with {batch_count} batches")
        
        return train_loss, epoch_time
    
    def validate(self, valid_loader: DataLoader) -> float:
        """
        Model validation
        
        Args:
            valid_loader: Validation data loader
            
        Returns:
            Validation loss
        """
        self.model.eval()
        valid_loss = []
        task_losses = {task: [] for task in self.task_names}
        batch_count = 0
        
        with torch.no_grad():
            for batch_x, _ in valid_loader:
                batch_count += 1
                self.logger.debug(f"Processing batch #{batch_count} in validation")
                
                # Prepare original data
                batch_x = batch_x.float().to(self.device)
                
                # Generate augmented data and labels for SSL validation
                X_aug, Y = self.generate_ssl_batch(batch_x)
                
                # Initialize total loss for all tasks
                total_loss = 0
                
                # Process each task
                for t_idx, tname in enumerate(self.task_names):
                    # Extract input data and labels for current task
                    x_t = X_aug[:, t_idx]
                    y_t = Y[:, t_idx]
                    
                    # Model prediction (for specific task)
                    out_t = self.model(x_t, task=tname)
                    
                    # Calculate and accumulate loss
                    loss_t = self.task_weights[tname] * self.criterion(out_t.squeeze(), y_t)
                    total_loss += loss_t
                    
                    # Record task-specific loss
                    task_losses[tname].append(loss_t.item())
                
                # Record total loss
                valid_loss.append(total_loss.item())
        
        # Calculate average validation loss
        valid_loss = np.average(valid_loss)
        
        # Calculate and log task-specific average losses
        for tname in self.task_names:
            task_avg_loss = np.average(task_losses[tname])
            self.logger.debug(f"Validation task {tname} average loss: {task_avg_loss:.6f}")
        
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
        self.logger.info(f"Starting SSL-MTL encoder training, saving to: {self.save_path}")
        
        for epoch in range(self.epochs):
            # Training phase
            train_loss, epoch_time = self.train_epoch(train_loader)
            
            # Log training progress
            log_message = f"Epoch: {epoch+1}, train_loss: {train_loss:.7f}, time: {epoch_time:.2f}s"
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
        self.logger.info("SSL-MTL encoder training completed")
        return self.model


def create_mtl_encoder(args: Any) -> nn.Module:
    """
    Create MTL encoder model
    
    Args:
        args: Configuration parameters
        
    Returns:
        Created MTL encoder model
    """
    logger = Logger("mtl_encoder_creator")
    
    # Convert args to dict
    encoder_args = {
        'input_channels': args.input_channels,
        'window_size': args.window_size,
        'output_size': args.output_size, # (3, 78)
        'device': args.device
    }
    
    # Load model configuration - use relative path for flexibility
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'model.yaml')
    with open(config_path, mode='r') as config_file:
        model_config = yaml.load(config_file, Loader=yaml.FullLoader)
    
    encoder_config = model_config['efnet_encoder']
    
    if args.encoder_type == 'deepconvlstm':
        encoder_args.update(encoder_config.get('deepconvlstm', {}))
        model_class = MTLDeepConvLSTMEncoder
        logger.info(f"Using MTL DeepConvLSTM encoder configuration")

    elif args.encoder_type == 'deepconvlstm_attn':
        encoder_args.update(encoder_config.get('deepconvlstm_attn', {}))
        model_class = MTLDeepConvLSTMAttnEncoder
        logger.info(f"Using MTL DeepConvLSTM with Attention encoder configuration")

    elif args.encoder_type == 'sa_har':
        encoder_args.update(encoder_config.get('sa_har', {}))
        model_class = MTLSAHAREncoder
        logger.info(f"Using MTL SA-HAR encoder configuration")
        
    else:
        logger.error(f"Unsupported encoder type: {args.encoder_type}")
        raise ValueError(f"Unsupported encoder type: {args.encoder_type}")
    
    # Create selected MTL encoder model
    encoder = model_class(encoder_args)
    
    logger.info(f"Created MTL {args.encoder_type} encoder")
    return encoder

def load_pretrained_mtl_encoder(encoder: nn.Module, path: str) -> nn.Module:
    """
    Load pretrained MTL encoder weights from checkpoint file
    
    Args:
        encoder: MTL encoder model instance
        path: Checkpoint file path
        
    Returns:
        Encoder with loaded weights
        
    Raises:
        FileNotFoundError: If checkpoint file does not exist
        RuntimeError: If state dictionary loading error occurs
    """
    logger = Logger("mtl_encoder_loader")
    logger.info(f"Loading pretrained MTL encoder from: {path}")
    
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

def get_transform_functions() -> Dict[str, Callable]:
    """
    Return time series transformation (augmentation) functions
    
    Returns:
        Dictionary of transformation functions {task_name: transform_function}
    """
    # Define arguments for warping transformations
    time_warp_args = {'sigma': 0.2, 'knot': 4}
    permutation_args = {'nPerm': 4, 'minSegLength': 10}
    
    def rotation_func(x):
        try:
            return rotation_transform(x)
        except Exception as e:
            print(f"Error applying rotation transformation: {str(e)}, input shape: {x.shape}")
            raise
    
    # Construct transformation functions dictionary
    transform_funcs = {
        'noise': lambda x: noise_transform(x, sigma=0.05),
        'scaling': lambda x: scaling_transform(x, sigma=0.1),
        'time_warp': lambda x: time_warp_transform(x, **time_warp_args),
        'rotation': rotation_func,
        'time_segment_permutation': lambda x: time_segment_permutation_transform_improved(x, **permutation_args),
        'negate': negate_transform,
        'horizontal_flip': horizontal_flip_transform,
        'channel_shuffle': channel_shuffle_transform
    }
    
    return transform_funcs 
