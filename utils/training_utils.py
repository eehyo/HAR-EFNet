import numpy as np
import random
import torch
import os
import scipy as sp
from scipy.fftpack import fft, fftfreq, ifft

from .logger import Logger

if not hasattr(Logger, '_run_id') or Logger._run_id is None:
    Logger.initialize(log_dir='logs')

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
  
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.metric_max = -np.inf 
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
        # For classifier training, use f1_macro (metric) if provided
        if metric is not None:
            score = metric 
        else:
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
            if metric is not None:
                self.logger.info(f'F1 Macro Score increased ({self.metric_max:.6f} --> {metric:.6f}). Saving model...')
            else:
                self.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        
        # path check
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # save model state
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
            self.metric_max = metric  
            
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


def save_results_summary(results, args, timestamp):
    """Save a summary of experiment results"""
    # Create results directory
    results_path = os.path.join(args.results_save_path, f"{args.encoder_type}_{timestamp}")
    os.makedirs(results_path, exist_ok=True)
    
    # Calculate overall statistics
    mean_acc = np.mean(results['accuracy'])
    std_acc = np.std(results['accuracy'])
    mean_f_w = np.mean(results['f1_weighted'])
    std_f_w = np.std(results['f1_weighted'])
    mean_f_macro = np.mean(results['f1_macro'])
    std_f_macro = np.std(results['f1_macro'])
    mean_f_micro = np.mean(results['f1_micro'])
    std_f_micro = np.std(results['f1_micro'])
    
    # List of subjects that were tested
    tested_subjects = ", ".join([str(s) for s in results['subject_id']])
    
    logger = Logger(f"results_{args.encoder_type}")
    
    # Log overall performance
    logger.info("\n===== Performance Summary =====")
    logger.info(f"Tested Subjects: {tested_subjects}")
    logger.info(f"Accuracy: mean={mean_acc:.7f}, std={std_acc:.7f}")
    logger.info(f"F1 Weighted: mean={mean_f_w:.7f}, std={std_f_w:.7f}")
    logger.info(f"F1 Macro: mean={mean_f_macro:.7f}, std={std_f_macro:.7f}")
    logger.info(f"F1 Micro: mean={mean_f_micro:.7f}, std={std_f_micro:.7f}")
    
    # Save results to file
    with open(os.path.join(results_path, "loocv_results.txt"), "w") as f:
        f.write(f"Model: {args.encoder_type}_{timestamp}\n\n")
        f.write(f"Experiment Settings:\n")
        f.write(f"- Encoder Type: {args.encoder_type}\n")
        f.write(f"- Tested Subjects: {tested_subjects}\n")
        f.write(f"- Freeze Encoder: {args.freeze_encoder}\n")
        f.write(f"- Classifier Learning Rate: {args.classifier_lr}\n")
        f.write(f"- Classifier Epochs: {args.classifier_epochs}\n\n")
        
        # Subject-specific results
        for i, subj_id in enumerate(results['subject_id']):
            f.write(f"Subject {subj_id} Results:\n")
            f.write(f"Accuracy: {results['accuracy'][i]:.7f}\n")
            f.write(f"F1 Weighted: {results['f1_weighted'][i]:.7f}\n")
            f.write(f"F1 Macro: {results['f1_macro'][i]:.7f}\n")
            f.write(f"F1 Micro: {results['f1_micro'][i]:.7f}\n")
            f.write("-" * 50 + "\n\n")
        
        # Overall summary
        f.write("===== Performance Summary =====\n")
        f.write(f"Accuracy: mean={mean_acc:.7f}, std={std_acc:.7f}\n")
        f.write(f"F1 Weighted: mean={mean_f_w:.7f}, std={std_f_w:.7f}\n")
        f.write(f"F1 Macro: mean={mean_f_macro:.7f}, std={std_f_macro:.7f}\n")
        f.write(f"F1 Micro: mean={mean_f_micro:.7f}, std={std_f_micro:.7f}\n")

def visualize_confusion_matrix(true_labels, predictions, save_path, filename_prefix="confusion_matrix"):

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import os
    
    logger = Logger("confusion_matrix_visualizer")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    cm = confusion_matrix(true_labels, predictions)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    cm_file = os.path.join(save_path, f"{filename_prefix}.txt")
    with open(cm_file, 'w') as f:
        f.write("Raw Counts:\n")
        np.savetxt(f, cm, fmt='%d')
        f.write("\nPercentages (%):\n")
        np.savetxt(f, cm_percentage, fmt='%.2f')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues') 
    plt.title('Confusion Matrix (%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_plot_file = os.path.join(save_path, f"{filename_prefix}_percent.png")
    plt.savefig(cm_plot_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 10))
    annot_labels = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            row.append(f"{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)")
        annot_labels.append(row)
        
    annot_array = np.array(annot_labels)
    sns.heatmap(cm_percentage, annot=annot_array, fmt='', cmap='Blues')
    plt.title('Confusion Matrix (count & %)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    combined_cm_plot_file = os.path.join(save_path, f"{filename_prefix}_combined.png")
    plt.savefig(combined_cm_plot_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    logger.info(f"Confusion matrix saved to: {cm_file}, {cm_plot_file}, and {combined_cm_plot_file}")
    
    return [cm_file, cm_plot_file, combined_cm_plot_file]
