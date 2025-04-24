import numpy as np
import random
import torch
import os
import scipy as sp
from scipy.fftpack import fft, fftfreq, ifft

from .logger import Logger

def set_seed(seed):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed to set
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, logger_name="early_stopping"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
  
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.logger = Logger(logger_name)
        
    def __call__(self, val_loss, model, save_path, metric=None, log=None):
        """
        Check for early stopping criteria
        
        Args:
            val_loss: Validation loss
            model: Model to save
            save_path: Path to save model
            metric: Additional metric (optional)
            log: Log file (optional, deprecated)
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path, metric)
        elif score < self.best_score + self.delta:
            self.counter += 1
            
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.logger.info("New best score! Saving model...")
            
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path, metric)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model, save_path, metric=None):
        """
        Save model checkpoint when validation loss decreases
        
        Args:
            val_loss: Validation loss
            model: Model to save
            save_path: Path to save model
            metric: Additional metric (optional)
        """
        if self.verbose:
            self.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        
        # path check
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # save model state
        model_path = os.path.join(save_path, 'best_model.pth')
        
        # save model state and metadata
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss
        }
        
        if metric is not None:
            checkpoint['metric'] = metric
            
        torch.save(checkpoint, model_path)
        self.val_loss_min = val_loss

class adjust_learning_rate:
    """
    Learning rate adjustment class
    
    Adjusts learning rate when validation loss doesn't improve
    """
    def __init__(self, args, verbose=True, logger_name="lr_adjuster"):
        """
        Args:
            args: Training parameters with learning rate settings
            verbose: Whether to print messages
            logger_name: Name for the logger
        """
        self.patience = args.learning_rate_patience
        self.factor = args.learning_rate_factor
        self.learning_rate = args.learning_rate
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.logger = Logger(logger_name)
        
    def __call__(self, optimizer, val_loss):
        """
        Adjust learning rate
        
        Args:
            optimizer: Optimization algorithm
            val_loss: Validation loss
        """
        # val_loss is a positive value, and smaller the better
        # bigger the score, the better
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.counter += 1
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                self.logger.info(f'Learning rate adjusting counter: {self.counter} out of {self.patience}')
        else:
            if self.verbose:
                self.logger.info("New best score!")
            self.best_score = score
            self.counter = 0
            
        if self.counter == self.patience:
            self.learning_rate = self.learning_rate * self.factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                if self.verbose:
                    self.logger.info(f'Updating learning rate to {self.learning_rate}')
            self.counter = 0


# def vn_c_reshape(x, time_length):
#     # For PAMAP only!!

#     # Example input: (batch, time_length, 9)
#     # Original order: [x_hand, y_hand, z_hand, x_chest, y_chest, z_chest, x_ankle, y_ankle, z_ankle]
#     channel_indices = [
#         0, 3, 6,  # x for hand, chest, ankle
#         1, 4, 7,  # y for hand, chest, ankle
#         2, 5, 8   # z for hand, chest, ankle
#     ]
#     batch = x.size(0)

#     # x is your input tensor of shape (batch, 1, time_length, 9)
#     x_reordered = x[:, :, :, channel_indices]  # (batch, 1, time_length, 9)

#     # Now reshape
#     x_reshaped = x_reordered.reshape(batch, time_length, 3, 3)

#     return x_reshaped