import os
import datetime
import numpy as np
import torch

from configs.config import get_args
from dataloaders.data_loader import PAMAP2, get_data
from utils.training_utils import set_seed
from utils.logger import Logger

from encoders import CNNEncoder, LSTMEncoder
from classifiers import ClassifierModel
from train_encoder import create_encoder, load_pretrained_encoder, EncoderTrainer
from train_classifier import create_classifier, ClassifierTrainer, evaluate_classifier

if __name__ == '__main__': 
    # args in configs/config.py
    args = get_args()

    # Timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.timestamp = timestamp
    
    # Initialize logger
    logger = Logger(f"har_ssl_{args.encoder_type}")

    # Random Seed
    set_seed(args.seed)
    logger.info(f"Random Seed: {args.seed}")

    # Load dataset
    dataset = PAMAP2(args)
    logger.info(f"Dataset: {args.data_name} loaded") 

    # for test performance aggregation (if args.test==True)
    acc_list = []
    f_w_list = []
    f_macro_list = []
    f_micro_list = []

    # Cross-validation for each test subject
    for test_sub in range(1, 9):
        # Update test subject
        dataset.update_train_val_test_keys()
        logger.info(f"Using subject {dataset.index_of_cv} as test subject")

        # Create data loaders
        train_loader = get_data(dataset, args.batch_size, flag="train")
        valid_loader = get_data(dataset, args.batch_size, flag="valid")
        test_loader = get_data(dataset, args.batch_size, flag="test")

        # Set save paths
        encoder_save_path = os.path.join(args.encoder_save_path, f"{args.encoder_type}_{timestamp}", str(dataset.index_of_cv))
        classifier_save_path = os.path.join(args.classifier_save_path, f"{args.encoder_type}_{timestamp}", str(dataset.index_of_cv))

        # ============Step 1: Encoder Training============
        if args.train_encoder:
            logger.info("Starting encoder training...")
            
            # Create encoder model
            encoder = create_encoder(args)
            
            # Initialize encoder trainer
            encoder_trainer = EncoderTrainer(args, encoder, encoder_save_path)
            
            # Train encoder
            encoder = encoder_trainer.train(train_loader, valid_loader)
            
            logger.info("Encoder training completed!")
        
        # ============Step 2: Classifier Training============
        if args.train_classifier:
            logger.info("Starting classifier training...")
            
            # Load encoder
            encoder = create_encoder(args)
            if args.load_encoder:
                # Use specified encoder path if available
                encoder_path = args.encoder_path
            else:
                # Otherwise use the encoder just trained
                encoder_path = os.path.join(encoder_save_path, "best_model.pth")
            
            # Load pretrained encoder
            encoder = load_pretrained_encoder(encoder, encoder_path)
            
            # Create classifier model
            classifier = create_classifier(args, encoder)
            
            # Initialize classifier trainer
            classifier_trainer = ClassifierTrainer(args, classifier, classifier_save_path)
            
            # Train classifier
            classifier = classifier_trainer.train(train_loader, valid_loader)
            
            logger.info("Classifier training completed!")
        
        # ============Step 3: Testing============
        if args.test:
            logger.info(f"Starting testing (test subject: {dataset.index_of_cv})")
            
            # Load final model
            encoder = create_encoder(args)
            encoder = load_pretrained_encoder(encoder, os.path.join(encoder_save_path, "best_model.pth"))
            classifier = create_classifier(args, encoder)
            
            # Load classifier checkpoint
            checkpoint = torch.load(os.path.join(classifier_save_path, "best_model.pth"), map_location=args.device)
            classifier.load_state_dict(checkpoint['model_state_dict'])
            
            # Perform testing
            acc, f_w, f_macro, f_micro = evaluate_classifier(args, classifier, test_loader, classifier_save_path)
            
            # Save performance metrics
            acc_list.append(acc)
            f_w_list.append(f_w)
            f_macro_list.append(f_macro)
            f_micro_list.append(f_micro)
            
            # Report overall performance after testing the last subject
            # Indicates completion of cross-validation across all subjects
            if test_sub == 8:
                results_path = os.path.join(args.classifier_save_path, f"{args.encoder_type}_{timestamp}")
                if not os.path.exists(results_path):
                    os.makedirs(results_path)
                
                # Log overall results
                logger.info("\n===== Overall Performance Summary =====")
                logger.info(f"Model: {args.encoder_type}_{timestamp}")
                logger.info(f"Accuracy: mean={np.mean(acc_list):.7f}, std={np.std(acc_list):.7f}")
                logger.info(f"F1 Weighted: mean={np.mean(f_w_list):.7f}, std={np.std(f_w_list):.7f}")
                logger.info(f"F1 Macro: mean={np.mean(f_macro_list):.7f}, std={np.std(f_macro_list):.7f}")
                logger.info(f"F1 Micro: mean={np.mean(f_micro_list):.7f}, std={np.std(f_micro_list):.7f}")
                
                # # Save results to file
                # with open(os.path.join(results_path, "overall_results.txt"), "a") as f:
                #     f.write(f"Model: {args.encoder_type}_{timestamp}\n")
                #     f.write(f"Accuracy: mean={np.mean(acc_list):.7f}, std={np.std(acc_list):.7f}\n")
                #     f.write(f"F1 Weighted: mean={np.mean(f_w_list):.7f}, std={np.std(f_w_list):.7f}\n")
                #     f.write(f"F1 Macro: mean={np.mean(f_macro_list):.7f}, std={np.std(f_macro_list):.7f}\n")
                #     f.write(f"F1 Micro: mean={np.mean(f_micro_list):.7f}, std={np.std(f_micro_list):.7f}\n")
    
    logger.info("All processes completed successfully.") 