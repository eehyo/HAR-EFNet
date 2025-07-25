import os
import time
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import yaml
from typing import Tuple, Dict, List, Optional, Any, Union

from encoders.collossl import ColloSSLEncoder
from encoders.collossl.collossl_encoder import create_collossl_encoder
from utils.collossl_utils import MultiViewContrastiveLoss, ColloSSLLoss
from dataloaders.collossl_data_utils import create_collossl_dataloader
from utils.training_utils import EarlyStopping, adjust_learning_rate, set_seed
from utils.logger import Logger

# Initialize global logger
if Logger._run_id is None:
    Logger.initialize(log_dir='logs')

class EncoderTrainerColloSSL:
    """
    ColloSSL encoder training class for multi-device contrastive learning
    Updated to follow paper specifications exactly
    """
    def __init__(self, args: Any, model: ColloSSLEncoder, save_path: str):
        """
        Initialize the ColloSSL encoder trainer
        
        Args:
            args: configuration parameters
            model: ColloSSL encoder model to train
            save_path: model save path
        """
        self.model = model
        self.device = args.device
        self.model.to(self.device)
        
        # Initialize logger
        self.logger = Logger(f"collossl_encoder_{args.encoder_type}")
        self.logger.info(f"Using device: {self.device}")
        
        # ColloSSL contrastive loss (updated parameters)
        self.criterion = ColloSSLLoss(
            temperature=args.collossl_temperature,
            mmd_kernel=args.mmd_kernel,
            mmd_sigma=args.mmd_sigma,
            neg_sample_size=getattr(args, 'neg_sample_size', 1),
            device_selection_metric=getattr(args, 'device_selection_metric', 'mmd_acc_norm'),
            device_selection_strategy=getattr(args, 'device_selection_strategy', 'hard_negative')
        )
        
        # Setup optimizer
        if args.optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, 
                                       weight_decay=getattr(args, 'weight_decay', 0.0001))
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate,
                                      weight_decay=getattr(args, 'weight_decay', 0.0001))
        
        # Save path and logging setup
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # Training settings
        self.epochs = args.train_epochs
        
        # Store args for device channel mapping
        self.args = args
        
        # Early stopping and learning rate adjustment
        self.early_stopping = EarlyStopping(patience=args.early_stop_patience, verbose=True, 
                                          logger_name=f"es_collossl_encoder_{args.encoder_type}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, Dict]:
        """
        Train one epoch with ColloSSL contrastive learning
        
        Args:
            train_loader: Training data loader (ColloSSL format)
            
        Returns:
            Training loss, epoch time in seconds, epoch info dict
        """
        self.model.train()
        train_losses = []
        epoch_time = time.time()
        batch_count = 0
        
        # Accumulate epoch statistics
        epoch_stats = {
            'positive_device_counts': {},
            'avg_mmd_distances': {},
            'avg_positive_probs': [],
            'total_negative_devices': 0,
            'total_negative_samples': 0
        }
        
        for batch_data in train_loader:
            batch_count += 1
            self.logger.debug(f"Processing batch #{batch_count} in train epoch")
            
            # Extract batch data (new format)
            anchor_data = batch_data['anchor_data'].float().to(self.device)  # [batch_size, window_size, 3]
            sync_samples = batch_data['sync_samples']  # Dict of synchronized samples
            async_samples = batch_data['async_samples']  # Dict of asynchronous samples
            
            # Move data to device
            for device_name in sync_samples:
                sync_samples[device_name] = sync_samples[device_name].float().to(self.device)
                async_samples[device_name] = async_samples[device_name].float().to(self.device)
            
            # Forward pass through encoder for all samples
            anchor_embeddings = self.model.forward_single_device(anchor_data)
            
            # Process synchronized embeddings (for positive sampling)
            sync_embeddings = {}
            for device_name, device_data in sync_samples.items():
                sync_embeddings[device_name] = self.model.forward_single_device(device_data)
            
            # Process asynchronous embeddings (for negative sampling)
            # async_samples[device_name]: [batch_size, neg_sample_size, window_size, 3]
            async_embeddings = {}
            for device_name, device_data in async_samples.items():
                batch_size, neg_sample_size, window_size, channels = device_data.shape
                # Reshape to [batch_size * neg_sample_size, window_size, 3]
                device_data_reshaped = device_data.view(-1, window_size, channels)
                # Get embeddings: [batch_size * neg_sample_size, embedding_dim]
                device_embeddings = self.model.forward_single_device(device_data_reshaped)
                # Reshape back to [batch_size, neg_sample_size, embedding_dim]
                embedding_dim = device_embeddings.shape[1]
                async_embeddings[device_name] = device_embeddings.view(batch_size, neg_sample_size, embedding_dim)
            
            # Compute ColloSSL loss (updated function signature)
            loss, loss_info = self.criterion(
                anchor_embeddings=anchor_embeddings,
                sync_embeddings=sync_embeddings,
                async_embeddings=async_embeddings,
                anchor_data=anchor_data,
                sync_data=sync_samples
            )
            
            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_losses.append(loss.item())
            
            # Accumulate statistics
            positive_device = loss_info.get('positive_device', 'unknown')
            if positive_device not in epoch_stats['positive_device_counts']:
                epoch_stats['positive_device_counts'][positive_device] = 0
            epoch_stats['positive_device_counts'][positive_device] += 1
            
            # Accumulate MMD distances
            mmd_distances = loss_info.get('mmd_distances', {})
            for device, distance in mmd_distances.items():
                if device not in epoch_stats['avg_mmd_distances']:
                    epoch_stats['avg_mmd_distances'][device] = []
                epoch_stats['avg_mmd_distances'][device].append(distance)
            
            epoch_stats['avg_positive_probs'].append(loss_info.get('avg_positive_prob', 0.0))
            epoch_stats['total_negative_devices'] += loss_info.get('num_negative_devices', 0)
            epoch_stats['total_negative_samples'] += loss_info.get('total_negative_samples', 0)
        
        epoch_time = time.time() - epoch_time
        train_loss = np.average(train_losses)
        
        # Compute average statistics
        for device in epoch_stats['avg_mmd_distances']:
            epoch_stats['avg_mmd_distances'][device] = np.mean(epoch_stats['avg_mmd_distances'][device])
        epoch_stats['avg_positive_prob'] = np.mean(epoch_stats['avg_positive_probs'])
        epoch_stats['avg_negative_devices'] = epoch_stats['total_negative_devices'] / batch_count if batch_count > 0 else 0
        epoch_stats['avg_negative_samples'] = epoch_stats['total_negative_samples'] / batch_count if batch_count > 0 else 0
        
        self.logger.info(f"Completed epoch with {batch_count} batches")
        
        return train_loss, epoch_time, epoch_stats
    
    def validate(self, valid_loader: DataLoader) -> Tuple[float, Dict]:
        """
        Validate model with ColloSSL contrastive learning
        
        Args:
            valid_loader: Validation data loader (ColloSSL format)
            
        Returns:
            Validation loss, validation info dict
        """
        self.model.eval()
        valid_losses = []
        batch_count = 0
        
        # Accumulate validation statistics
        valid_stats = {
            'positive_device_counts': {},
            'avg_mmd_distances': {},
            'avg_positive_probs': [],
            'total_negative_devices': 0,
            'total_negative_samples': 0
        }
        
        with torch.no_grad():
            for batch_data in valid_loader:
                batch_count += 1
                self.logger.debug(f"Processing batch #{batch_count} in validation")
                
                # Extract batch data (new format)
                anchor_data = batch_data['anchor_data'].float().to(self.device)
                sync_samples = batch_data['sync_samples']
                async_samples = batch_data['async_samples']
                
                # Move data to device
                for device_name in sync_samples:
                    sync_samples[device_name] = sync_samples[device_name].float().to(self.device)
                    async_samples[device_name] = async_samples[device_name].float().to(self.device)
                
                # Forward pass through encoder for all samples
                anchor_embeddings = self.model.forward_single_device(anchor_data)
                
                # Process synchronized embeddings
                sync_embeddings = {}
                for device_name, device_data in sync_samples.items():
                    sync_embeddings[device_name] = self.model.forward_single_device(device_data)
                
                # Process asynchronous embeddings
                async_embeddings = {}
                for device_name, device_data in async_samples.items():
                    batch_size, neg_sample_size, window_size, channels = device_data.shape
                    # Reshape to [batch_size * neg_sample_size, window_size, 3]
                    device_data_reshaped = device_data.view(-1, window_size, channels)
                    # Get embeddings: [batch_size * neg_sample_size, embedding_dim]
                    device_embeddings = self.model.forward_single_device(device_data_reshaped)
                    # Reshape back to [batch_size, neg_sample_size, embedding_dim]
                    embedding_dim = device_embeddings.shape[1]
                    async_embeddings[device_name] = device_embeddings.view(batch_size, neg_sample_size, embedding_dim)
                
                # Compute ColloSSL loss
                loss, loss_info = self.criterion(
                    anchor_embeddings=anchor_embeddings,
                    sync_embeddings=sync_embeddings,
                    async_embeddings=async_embeddings,
                    anchor_data=anchor_data,
                    sync_data=sync_samples
                )
                
                valid_losses.append(loss.item())
                
                # Accumulate statistics
                positive_device = loss_info.get('positive_device', 'unknown')
                if positive_device not in valid_stats['positive_device_counts']:
                    valid_stats['positive_device_counts'][positive_device] = 0
                valid_stats['positive_device_counts'][positive_device] += 1
                
                # Accumulate MMD distances
                mmd_distances = loss_info.get('mmd_distances', {})
                for device, distance in mmd_distances.items():
                    if device not in valid_stats['avg_mmd_distances']:
                        valid_stats['avg_mmd_distances'][device] = []
                    valid_stats['avg_mmd_distances'][device].append(distance)
                
                valid_stats['avg_positive_probs'].append(loss_info.get('avg_positive_prob', 0.0))
                valid_stats['total_negative_devices'] += loss_info.get('num_negative_devices', 0)
                valid_stats['total_negative_samples'] += loss_info.get('total_negative_samples', 0)
        
        valid_loss = np.average(valid_losses)
        
        # Compute average statistics
        for device in valid_stats['avg_mmd_distances']:
            valid_stats['avg_mmd_distances'][device] = np.mean(valid_stats['avg_mmd_distances'][device])
        valid_stats['avg_positive_prob'] = np.mean(valid_stats['avg_positive_probs'])
        valid_stats['avg_negative_devices'] = valid_stats['total_negative_devices'] / batch_count if batch_count > 0 else 0
        valid_stats['avg_negative_samples'] = valid_stats['total_negative_samples'] / batch_count if batch_count > 0 else 0
        
        self.logger.info(f"Completed validation with {batch_count} batches")
        
        return valid_loss, valid_stats
    
    def train(self, train_loader: DataLoader, valid_loader: DataLoader) -> ColloSSLEncoder:
        """
        Complete training process for ColloSSL
        
        Args:
            train_loader: Training data loader
            valid_loader: Validation data loader
            
        Returns:
            Trained ColloSSL encoder model
        """
        self.logger.info(f"Starting ColloSSL encoder training, saving to: {self.save_path}")
        
        # Convert to ColloSSL data loaders using config batch size
        device_channel_mapping = self.args.device_channel_mapping
        neg_sample_size = getattr(self.args, 'neg_sample_size', 1)
        
        collossl_train_loader = create_collossl_dataloader(
            train_loader, device_channel_mapping, 
            anchor_device=self.args.anchor_device,
            batch_size=self.args.batch_size,
            neg_sample_size=neg_sample_size
        )
        
        collossl_valid_loader = create_collossl_dataloader(
            valid_loader, device_channel_mapping,
            anchor_device=self.args.anchor_device,
            batch_size=self.args.batch_size,
            neg_sample_size=neg_sample_size
        )
        
        self.logger.info(f"Using batch size: {self.args.batch_size}")
        self.logger.info(f"Using negative sample size: {neg_sample_size}")
        
        for epoch in range(self.epochs):
            # Training phase
            train_loss, epoch_time, train_stats = self.train_epoch(collossl_train_loader)
            
            # Log training progress with detailed stats
            log_message = f"Epoch: {epoch+1}, train_loss: {train_loss:.7f}, time: {epoch_time:.2f}s"
            log_message += f", avg_pos_prob: {train_stats['avg_positive_prob']:.4f}"
            log_message += f", pos_devices: {train_stats['positive_device_counts']}"
            log_message += f", avg_neg_samples: {train_stats['avg_negative_samples']:.1f}"
            self.logger.info(log_message)
            
            # Validation phase
            valid_loss, valid_stats = self.validate(collossl_valid_loader)
            
            # Log validation results with detailed stats
            log_message = f"Validation: Epoch: {epoch+1}, Train Loss: {train_loss:.7f}, Valid Loss: {valid_loss:.7f}"
            log_message += f", Valid Pos Prob: {valid_stats['avg_positive_prob']:.4f}"
            log_message += f", Valid MMD: {valid_stats['avg_mmd_distances']}"
            log_message += f", Valid Neg Samples: {valid_stats['avg_negative_samples']:.1f}"
            self.logger.info(log_message)
            
            # Early stopping check
            self.early_stopping(valid_loss, self.model, self.save_path, None)
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping triggered")
                break
        
        # Training complete
        self.logger.info("ColloSSL encoder training completed")
        return self.model


def load_pretrained_collossl_encoder(encoder: ColloSSLEncoder, path: str) -> ColloSSLEncoder:
    """
    Load pretrained ColloSSL encoder weights from checkpoint file
    
    Args:
        encoder: ColloSSL encoder model instance
        path: Path to checkpoint file
        
    Returns:
        ColloSSL encoder with loaded weights
        
    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist
        RuntimeError: If there's an error loading the state dict
    """
    logger = Logger("collossl_encoder_loader")
    logger.info(f"Loading pretrained ColloSSL encoder from: {path}")
    
    if not os.path.exists(path):
        logger.error(f"Checkpoint file not found: {path}")
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    
    try:
        checkpoint = torch.load(path, map_location=encoder.device, weights_only=False)
        encoder.load_state_dict(checkpoint['model_state_dict'])
        val_loss = checkpoint.get('val_loss', 'N/A')
        logger.info(f"Successfully loaded ColloSSL model with validation loss: {val_loss}")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
    
    return encoder 