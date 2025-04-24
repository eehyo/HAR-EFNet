import argparse
import os
import torch
import yaml

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='HAR encoder and classifier training')
    
    # Dataset
    parser.add_argument('-d', '--data_name', default='pamap2', type=str, help='Name of the Dataset')
    
    # Model
    parser.add_argument('-e', '--encoder_type', default='cnn', type=str, help='Encoder Type (cnn, lstm)')
    
    # Training mode settings
    parser.add_argument('--train_encoder', default=True, type=str2bool, help='Train Encoder')
    parser.add_argument('--train_classifier', default=False, type=str2bool, help='Train Classifier')
    parser.add_argument('--test', default=False, type=str2bool, help='perform testing')
    
    # load encoder weights
    parser.add_argument('--load_encoder', default=False, type=str2bool, help='Load pre-trained encoder')
    parser.add_argument('--encoder_path', default=None, type=str, help='Path to pre-trained encoder')
    
    args = parser.parse_args()
    
    # data config
    config_file = open('HAR_SSL/configs/data.yaml', mode='r')
    data_config = yaml.load(config_file, Loader=yaml.FullLoader)
    data_config = data_config[args.data_name]
    
    # path settings
    args.data_path = os.path.join("datasets", data_config['filename'])
    args.save_path = "saved"
    args.encoder_save_path = os.path.join(args.save_path, "encoders")
    args.classifier_save_path = os.path.join(args.save_path, "classifiers")
    
    args.use_gpu = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_gpu else "cpu")
    args.gpu = 6
    args.use_multi_gpu = False
    
    args.optimizer = "Adam"
    args.criterion = "MSELoss" if args.train_encoder else "CrossEntropy"
    args.exp_mode = "LOCV"
    args.datanorm_type = "standardization"
    
    # training settings
    args.train_epochs = 150
    args.learning_rate = 0.0005
    args.learning_rate_patience = 7
    args.learning_rate_factor = 0.1
    args.early_stop_patience = 15
    args.batch_size = 128
    args.shuffle = True
    args.drop_last = False
    args.train_vali_quote = 0.90
    
    # Time series input settings
    window_seconds = data_config["window_seconds"]
    args.window_size = int(window_seconds * data_config["sampling_freq"])
    args.input_length = args.window_size
    args.input_channels = data_config["num_channels"]
    args.sampling_freq = data_config["sampling_freq"]
    args.num_classes = data_config["num_classes"]
    
    # ECDF feature dimension
    args.n_ecdf_points = 25
    args.output_size = 234  # 26차원(25개 포인트+평균) × 9채널(3축 × 3부위)
    
    # Random seed and other settings
    args.sensor_select = ["acc"]
    args.seed = 42
    args.filtering = True
    args.freq1 = 0.001
    args.freq2 = 25.0
    
    return args 