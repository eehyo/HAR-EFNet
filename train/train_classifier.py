import os
import time
import torch
import numpy as np
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score
from typing import Tuple, List, Dict, Any, Optional

from classifiers.deepconvlstm_classifier import DeepConvLSTMClassifier
from classifiers.deepconvlstm_attn_classifier import DeepConvLSTMAttnClassifier
from classifiers.sa_har_classifier import SAHARClassifier
from utils.training_utils import EarlyStopping, adjust_learning_rate, set_seed, visualize_confusion_matrix
from utils.logger import Logger

# Make sure logger is initialized
if Logger._run_id is None:
    Logger.initialize(log_dir='logs')

class ClassifierTrainer:
    """
    Classifier trainer class
    """
    def __init__(self, args: Any, model: nn.Module, save_path: str):
        """
        Initialize trainer
        
        Args:
            args: Configuration parameters
            model: Model to train
            save_path: Path to save model
        """
        self.model = model
        self.device = args.device
        self.model.to(self.device)
        
        # Initialize logger
        self.logger = Logger(f"classifier_{args.encoder_type}_{args.classifier_type}")
        self.logger.info(f"Using device: {self.device}")
        
        # Check if encoder freezing is enabled and apply the correct freezing mode
        if args.freeze_encoder:
            self.logger.info("Freezing encoder parameters during classifier training")
            if args.encoder_type == 'deepconvlstm_attn':
                # For attention models, freeze only CNN and LSTM parts
                self.logger.info("For DeepConvLSTM-Attn: Freezing only CNN and LSTM layers, keeping attention layers trainable")
                self.model.freeze_encoder(freeze_mode='cnn_lstm_only')
            else:
                # For other encoder types, freeze all parameters
                self.logger.info("Freezing all encoder parameters")
                self.model.freeze_encoder(freeze_mode='all')
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer setup
        if args.optimizer == "Adam":
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                       lr=args.learning_rate)
        else:
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                       lr=args.learning_rate)
        
        # Save path and logging setup
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # Training settings
        self.epochs = args.train_epochs
        
        # Early stopping and learning rate adjustment
        self.early_stopping = EarlyStopping(patience=args.early_stop_patience, verbose=True,
                                          logger_name=f"es_classifier_{args.encoder_type}_{args.classifier_type}")
        # self.learning_rate_adapter = adjust_learning_rate(args, verbose=True,
        #                                               logger_name=f"lr_classifier_{args.encoder_type}_{args.classifier_type}")
    
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
        
        for batch_x, batch_y in train_loader:
            batch_count += 1
            self.logger.debug(f"Processing batch #{batch_count} in training")
            
            # Process input data
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.long().to(self.device)
            
            # Model forward pass
            # (128, 168, 9) -> (128, 12)
            outputs = self.model(batch_x)
            
            # Calculate loss
            loss = self.criterion(outputs, batch_y)
            
            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss.append(loss.item())
        
        epoch_time = time.time() - epoch_time
        train_loss = np.average(train_loss)
        self.logger.info(f"Completed epoch with {batch_count} batches")
        
        return train_loss, epoch_time
    
    def validate(self, valid_loader: DataLoader) -> Tuple[float, float, float, float, float]:
        """
        Validate model
        
        Args:
            valid_loader: Validation data loader
            
        Returns:
            Validation loss, accuracy, F1 scores (weighted, macro, micro)
        """
        self.model.eval()
        valid_loss = []
        predictions = []
        true_labels = []
        batch_count = 0
        
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_count += 1
                self.logger.debug(f"Processing batch #{batch_count} in validation")
                
                # Process input data
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)
                
                # Model forward pass
                outputs = self.model(batch_x)
                
                # Calculate loss
                loss = self.criterion(outputs, batch_y)
                valid_loss.append(loss.item())
                
                # Save predictions and labels
                pred = outputs.argmax(dim=1).cpu().numpy()
                true = batch_y.cpu().numpy()
                
                predictions.extend(pred)
                true_labels.extend(true)
        
        # Calculate average loss
        valid_loss = np.average(valid_loss)
        
        # Calculate performance metrics
        acc = accuracy_score(true_labels, predictions)
        f_w = f1_score(true_labels, predictions, average='weighted')
        f_macro = f1_score(true_labels, predictions, average='macro')
        f_micro = f1_score(true_labels, predictions, average='micro')
        self.logger.info(f"Completed validation with {batch_count} batches")
        
        return valid_loss, acc, f_w, f_macro, f_micro
    
    def train(self, train_loader: DataLoader, valid_loader: DataLoader) -> nn.Module:
        """
        Complete training process
        
        Args:
            train_loader: Training data loader
            valid_loader: Validation data loader
            
        Returns:
            Trained model
        """
        self.logger.info(f"Starting classifier training, saving to: {self.save_path}")
        
        for epoch in range(self.epochs):
            # Training phase
            train_loss, epoch_time = self.train_epoch(train_loader)
            
            # Log training progress
            log_message = f"Epoch: {epoch+1}, train_loss: {train_loss:.7f}, time: {epoch_time:.2f}s"
            self.logger.info(log_message)
            
            # Validation phase
            valid_loss, acc, f_w, f_macro, f_micro = self.validate(valid_loader)
            
            # Log validation results
            log_message = f"Validation: Epoch: {epoch+1}, " \
                        f"Train Loss: {train_loss:.7f}, " \
                        f"Valid Loss: {valid_loss:.7f}, " \
                        f"Valid Accuracy: {acc:.7f}, " \
                        f"Valid F1 Weighted: {f_w:.7f}, " \
                        f"Valid F1 Macro: {f_macro:.7f}, " \
                        f"Valid F1 Micro: {f_micro:.7f}"
            self.logger.info(log_message)
            
            # Early stopping check
            self.early_stopping(valid_loss, self.model, self.save_path, f_macro)
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping triggered")
                break
            
            # Learning rate adjustment
            # self.learning_rate_adapter(self.optimizer, valid_loss)
        
        # Training complete
        self.logger.info("Classifier training completed")
        return self.model

def create_classifier(args: Any, encoder: nn.Module) -> nn.Module:
    """
    Create classifier model based on configuration
    
    Args:
        args: Configuration parameters
        encoder: Pretrained encoder
        
    Returns:
        Created classifier model
    """
    # Initialize logger
    logger = Logger("classifier_creator")
    
    # Load model configuration - use relative path for flexibility
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'model.yaml')
    with open(config_path, mode='r') as config_file:
        model_config = yaml.load(config_file, Loader=yaml.FullLoader)
    
    classifier_config = model_config['efnet_classifier']
    
    if  args.classifier_type == 'deepconvlstm_classifier':
        classifier_args = classifier_config['deepconvlstm_classifier']
        model_class = DeepConvLSTMClassifier
        logger.info(f"Using DeepConvLSTM specific classifier configuration")
    elif args.classifier_type == 'deepconvlstm_attn_classifier':
        classifier_args = classifier_config['deepconvlstm_attn_classifier']
        model_class = DeepConvLSTMAttnClassifier
        logger.info(f"Using DeepConvLSTM-Attn specific classifier configuration")
    elif args.classifier_type == 'sa_har_classifier':
        classifier_args = classifier_config['sa_har_classifier']
        model_class = SAHARClassifier
        logger.info(f"Using SA-HAR specific classifier configuration")
    else:
        if args.encoder_type == 'deepconvlstm':
            classifier_args = classifier_config['deepconvlstm_classifier']
            model_class = DeepConvLSTMClassifier
            logger.info(f"Auto-selecting DeepConvLSTM specific classifier for {args.encoder_type} encoder")
        elif args.encoder_type == 'deepconvlstm_attn':
            classifier_args = classifier_config['deepconvlstm_attn_classifier']
            model_class = DeepConvLSTMAttnClassifier
            logger.info(f"Auto-selecting DeepConvLSTM-Attn specific classifier for {args.encoder_type} encoder")
        elif args.encoder_type == 'sa_har':
            classifier_args = classifier_config['sa_har_classifier']
            model_class = SAHARClassifier
            logger.info(f"Auto-selecting SA-HAR specific classifier for {args.encoder_type} encoder")
        else:
            raise ValueError(f"Unsupported encoder type: {args.encoder_type}")
    # Create selected classifier model
    model = model_class(
        encoder=encoder,
        num_classes=args.num_classes,
        config=classifier_args
    )
    
    logger.info(f"Created {args.classifier_type} with {args.num_classes} classes")
    return model

def evaluate_classifier(args: Any, model: nn.Module, test_loader: DataLoader, 
                      save_path: Optional[str] = None) -> Tuple[float, float, float, float]:
    """
    Evaluate classifier on test data
    
    Args:
        args: Configuration parameters
        model: Classifier model to evaluate
        test_loader: Test data loader
        save_path: Path to save results (optional)
        
    Returns:
        Tuple of (accuracy, F1 weighted score, F1 macro score, F1 micro score)
    """
    logger = Logger(f"eval_classifier_{args.encoder_type}_{args.classifier_type}")
    logger.info("Testing classifier...")
    logger.info(f"Test Subject: {args.test_subject} (Fold {args.fold_idx})")
    
    device = args.device
    model.to(device)
    model.eval()
    
    predictions = []
    true_labels = []
    batch_count = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_count += 1
            logger.debug(f"Processing test batch #{batch_count}")
            
            # Process input data
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.long().to(device)
            
            # Model forward pass and predictions
            outputs = model(batch_x) # (128, 12)
            pred = outputs.argmax(dim=1).cpu().numpy() # (128,)
            true = batch_y.cpu().numpy() # (128,)
            
            predictions.extend(pred)
            true_labels.extend(true)
    
    logger.info(f"Completed testing with {batch_count} batches")
    
    # Calculate metrics
    acc = accuracy_score(true_labels, predictions)
    f_w = f1_score(true_labels, predictions, average='weighted')
    f_macro = f1_score(true_labels, predictions, average='macro')
    f_micro = f1_score(true_labels, predictions, average='micro')
    
    # Log results
    results_str = f"Test Results: Accuracy: {acc:.4f}, F1 Weighted: {f_w:.4f}, F1 Macro: {f_macro:.4f}, F1 Micro: {f_micro:.4f}"
    
    logger.info(results_str)
    
    # Save test results and confusion matrix
    if save_path:
        # Save test results
        results_file = os.path.join(save_path, "test_results.txt")
        with open(results_file, 'w') as f:
            f.write(results_str)
        logger.info(f"Test results saved to: {results_file}")
        
        cm_files = visualize_confusion_matrix(
            true_labels=true_labels, 
            predictions=predictions, 
            save_path=save_path
        )
        
        logger.info(f"Results and confusion matrix visualization saved to {save_path}")
    
    return acc, f_w, f_macro, f_micro 