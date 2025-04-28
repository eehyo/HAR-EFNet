import os
import datetime
import numpy as np
import torch

from configs.config import get_args
from dataloaders.data_loader import PAMAP2, get_data
from utils.training_utils import set_seed, save_results_summary
from utils.logger import Logger

from encoders import CNNEncoder, LSTMEncoder, BaseEncoder
from classifiers import MLPClassifierModel, BaseClassifierModel
from train_encoder import create_encoder, load_pretrained_encoder, EncoderTrainer
from train_classifier import create_classifier, ClassifierTrainer, evaluate_classifier

if __name__ == '__main__': 
    args = get_args()

    # Timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.timestamp = timestamp
    
    logger = Logger(f"har_{args.encoder_type}_{args.classifier_type}")

    # Random Seed
    set_seed(args.seed)
    logger.info(f"Random Seed: {args.seed}")
    logger.info(f"Using encoder type: {args.encoder_type}, classifier type: {args.classifier_type}")

    # Load dataset
    dataset = PAMAP2(args)
    logger.info(f"Dataset: {args.data_name} loaded") 

    # for test performance aggregation (if args.test==True)
    results = {
        'subject_id': [],
        'accuracy': [],
        'f1_weighted': [],
        'f1_macro': [],
        'f1_micro': []
    }
    
    # Filter to specific subject if requested
    if args.specific_subject is not None:
        fold_range = [args.specific_subject - 1] 
        logger.info(f"Restricting training/evaluation: Testing only Subject {args.specific_subject}")
    else:
        fold_range = range(len(dataset.LOCV_keys))
        logger.info(f"Running all {len(dataset.LOCV_keys)} subjects")

    # Cross-validation for each test subject
    logger.info(f"Starting {len(fold_range)}-fold")

    # range(0, 8)
    for fold_idx in fold_range:
        # Set dataset state to the specified fold index
        dataset.index_of_cv = fold_idx
        # train, valid, test key 재설정
        dataset.update_train_val_test_keys()
        
        current_test_subject = dataset.test_keys[0]  # Extract the current test subject ID
        
        logger.info(f"Fold {fold_idx+1}/{len(dataset.LOCV_keys)}: Testing on Subject {current_test_subject}, Training on Subjects {dataset.train_keys}")

        # Create data loaders - use classifier batch size if training classifier
        batch_size = args.classifier_batch_size if args.train_classifier else args.batch_size
        train_loader = get_data(dataset, batch_size, flag="train")
        valid_loader = get_data(dataset, batch_size, flag="valid") 
        test_loader = get_data(dataset, batch_size, flag="test")
        
        # Set save paths for current fold
        # current_test_subject:1~8
        fold_dir = f"fold_{fold_idx+1}_subj_{current_test_subject}"
        
        # Get run_id from Logger (format: YYYYMMDD_HHMMSS)
        run_id = Logger.get_run_id()
        
        # Directory timestamp format should match the folder structure
        encoder_save_path = os.path.join(args.encoder_save_path, f"{args.encoder_type}_{timestamp}", fold_dir)
        classifier_save_path = os.path.join(args.classifier_save_path, f"{args.encoder_type}_{args.classifier_type}_{timestamp}", fold_dir)
        
        # Create directories and save fold information
        os.makedirs(encoder_save_path, exist_ok=True)
        os.makedirs(classifier_save_path, exist_ok=True)
        
        # Save fold details
        with open(os.path.join(classifier_save_path, "fold_info.txt"), "w") as f:
            f.write(f"Fold: {fold_idx+1}/{len(dataset.LOCV_keys)}\n")
            f.write(f"Test Subject: {current_test_subject}\n")
            f.write(f"Train Subjects: {dataset.train_keys}\n")
            f.write(f"Data Split: train={len(train_loader.dataset)}, valid={len(valid_loader.dataset)}, test={len(test_loader.dataset)} samples\n")
            f.write(f"Encoder Type: {args.encoder_type}\n")
            f.write(f"Classifier Type: {args.classifier_type}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Run ID: {run_id}\n")

         # ============Step 1: Encoder Training============
        if args.train_encoder:
            logger.info(f"Training encoder for fold {fold_idx+1}, test subject {current_test_subject}")
            
            encoder = create_encoder(args)
            encoder_trainer = EncoderTrainer(args, encoder, encoder_save_path)
            encoder = encoder_trainer.train(train_loader, valid_loader)
            
            logger.info(f"Encoder training completed for fold {fold_idx+1}")
        
        # ============Step 2: Classifier Training============
        if args.train_classifier:
            logger.info(f"Training classifier for fold {fold_idx+1}, test subject {current_test_subject}")
            
            args.learning_rate = args.classifier_lr
            args.train_epochs = args.classifier_epochs
            
            # 파라미터는 랜덤 초기화된 상태
            encoder = create_encoder(args)
            if args.load_encoder:
                encoder_checkpoint_path = args.encoder_path
            else:
                encoder_model_name = f"best_model_{run_id}.pth"
                encoder_checkpoint_path = os.path.join(encoder_save_path, encoder_model_name)
            
            logger.info(f"Loading encoder from: {encoder_checkpoint_path}")
            encoder = load_pretrained_encoder(encoder, encoder_checkpoint_path)
            
            # Configure whether to freeze encoder parameters
            if args.freeze_encoder:
                logger.info("Freezing encoder parameters during classifier training")
                for param in encoder.parameters():
                    param.requires_grad = False
            
            # Create and train classifier
            classifier = create_classifier(args, encoder)
            classifier_trainer = ClassifierTrainer(args, classifier, classifier_save_path)
            classifier = classifier_trainer.train(train_loader, valid_loader)
            
            logger.info(f"Classifier training completed for fold {fold_idx+1}")
        
        # ============Step 3: Testing============
        if args.test:
            logger.info(f"Testing on subject {current_test_subject} (fold {fold_idx+1})")
            
            # Load models
            encoder = create_encoder(args)
            if args.load_encoder:
                encoder_checkpoint_path = args.encoder_path
            else:
                encoder_model_name = f"best_model_{run_id}.pth"
                encoder_checkpoint_path = os.path.join(encoder_save_path, encoder_model_name)
            
            logger.info(f"Loading encoder from: {encoder_checkpoint_path}")
            encoder = load_pretrained_encoder(encoder, encoder_checkpoint_path)
            
            classifier = create_classifier(args, encoder)
            
            # Load classifier checkpoint
            if args.load_classifier and args.classifier_path:
                classifier_checkpoint_path = args.classifier_path
            else:
                classifier_model_name = f"best_model_{run_id}.pth"
                classifier_checkpoint_path = os.path.join(classifier_save_path, classifier_model_name)
            
            logger.info(f"Loading classifier from: {classifier_checkpoint_path}")
            checkpoint = torch.load(classifier_checkpoint_path, map_location=args.device, weights_only=False)
            classifier.load_state_dict(checkpoint['model_state_dict'])
            
            # Set current test subject and fold index for result tracking
            args.test_subject = current_test_subject
            args.fold_idx = fold_idx + 1
            
            # Evaluate on test set (left-out subject)
            acc, f_w, f_macro, f_micro = evaluate_classifier(args, classifier, test_loader, classifier_save_path)
            
            results['subject_id'].append(current_test_subject)
            results['accuracy'].append(acc)
            results['f1_weighted'].append(f_w)
            results['f1_macro'].append(f_macro)
            results['f1_micro'].append(f_micro)
    
    # Summarize all fold results if testing was performed
    if args.test and len(results['subject_id']) > 0:
        save_results_summary(results, args, timestamp)
    
    logger.info("All processes completed successfully.") 