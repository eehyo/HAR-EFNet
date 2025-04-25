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

# Initialize global logger
Logger.initialize(log_dir='logs')

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
        batch_count = 0
        # batch_x.shape: (128, 1, 168, 9)
        # batch_x.shape: (64, 1, 168, 9)
        for batch_x, _ in train_loader:  # label is not needed for ECDF features
            batch_count += 1
            self.logger.debug(f"Processing batch #{batch_count} in train epoch")
            
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
        self.logger.info(f"Completed epoch with {batch_count} batches")
        
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
        batch_count = 0
        
        with torch.no_grad():
            for batch_x, _ in valid_loader:  
                batch_count += 1
                self.logger.debug(f"Processing batch #{batch_count} in validation")
                
                batch_x = batch_x.float().to(self.device)
                
                batch_ecdf = compute_batch_ecdf_features(batch_x.cpu().numpy())
                batch_ecdf = torch.tensor(batch_ecdf).float().to(self.device)
                
                predicted_ecdf = self.model(batch_x)
                
                loss = self.criterion(predicted_ecdf, batch_ecdf)
                
                valid_loss.append(loss.item())
        
        valid_loss = np.average(valid_loss)
        self.logger.info(f"Completed validation with {batch_count} batches")
        
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
        
        # 실행 ID를 포함한 파일 이름으로 저장
        run_id = Logger.get_run_id()
        model_path = os.path.join(save_path, f'best_model_{run_id}.pth')
        
        # save model state and metadata
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'run_id': run_id
        }
        
        if metric is not None:
            checkpoint['metric'] = metric
            
        torch.save(checkpoint, model_path)
        self.val_loss_min = val_loss

def create_encoder(args):
    """
    Create encoder model based on configuration
    
    Args:
        args: Configuration parameters
        
    Returns:
        Created encoder model
    """
    # Initialize logger for encoder creation
    logger = Logger("encoder_creator")
    
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
        # conv_channels = encoder_args.get('conv_channels', ['unknown'])
        # dropout_rate = encoder_args.get('dropout_rate', 'unknown')
        # logger.info(f"Created CNN encoder with channels {conv_channels} and dropout {dropout_rate}")
    elif args.encoder_type == 'lstm':
        encoder_args.update(encoder_config['lstm'])
        encoder = LSTMEncoder(encoder_args)
    #     hidden_size = encoder_args.get('hidden_size', 'unknown')
    #     num_layers = encoder_args.get('num_layers', 'unknown')
    #     bidirectional = encoder_args.get('bidirectional', 'unknown')
    #     logger.info(f"Created LSTM encoder with hidden size {hidden_size}, layers {num_layers}, bidirectional={bidirectional}")
    else:
        logger.error(f"Unsupported encoder type: {args.encoder_type}")
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
    logger.info(f"Successfully loaded model with validation loss: {checkpoint.get('val_loss', 'N/A')}")

    
    return encoder 