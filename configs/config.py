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
    parser.add_argument('-e', '--encoder_type', default='deepconvlstm_attn', type=str, 
                         help='Encoder Type (deepconvlstm, deepconvlstm_attn, sa_har)')
    parser.add_argument('-c', '--classifier_type', default='deepconvlstm_attn_classifier', type=str, 
                         help='Classifier Type (deepconvlstm_classifier, deepconvlstm_attn_classifier, sa_har_classifier). If not specified, will auto-select based on encoder type.')
    
    # Training mode settings
    parser.add_argument('--train_encoder', default=True, type=str2bool, help='Train Encoder')
    parser.add_argument('--train_classifier', default=True, type=str2bool, help='Train Classifier')
    parser.add_argument('--test', default=True, type=str2bool, help='perform testing')
    
    # load encoder weights
    parser.add_argument('--load_encoder', default=False, type=str2bool, help='Load pre-trained encoder')
    parser.add_argument('--encoder_path', default=None, type=str, help='Path to pre-trained encoder')
    
    # load classifier weights
    parser.add_argument('--load_classifier', default=False, type=str2bool, help='Load pre-trained classifier')
    parser.add_argument('--classifier_path', default=None, type=str, help='Path to pre-trained classifier')
 
    # LOOCV settings
    parser.add_argument('--specific_subject', default=None, type=int, help='Test only specific subject (1-8), None to test all subjects')
    
    # MTL pretraining settings
    parser.add_argument('--mtl_mode', default=False, type=str2bool, help='Use MTL SSL pretraining')
    parser.add_argument('--task_weights_noise', type=float, default=1.0, help='Weight for noise task')
    parser.add_argument('--task_weights_scaling', type=float, default=1.0, help='Weight for scaling task')
    parser.add_argument('--task_weights_time_warp', type=float, default=1.0, help='Weight for time warping task')
    parser.add_argument('--task_weights_rotation', type=float, default=1.0, help='Weight for rotation task')
    parser.add_argument('--task_weights_time_segment_permutation', type=float, default=1.0, help='Weight for time segment permutation task')
    parser.add_argument('--task_weights_negate', type=float, default=1.0, help='Weight for negation task')
    parser.add_argument('--task_weights_horizontal_flip', type=float, default=1.0, help='Weight for horizontal flip task')
    parser.add_argument('--task_weights_channel_shuffle', type=float, default=1.0, help='Weight for channel shuffle task')

    # SimCLR pretraining settings
    parser.add_argument('--simclr_mode', default=False, type=str2bool, help='Use SimCLR SSL pretraining')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature parameter for NT-Xent loss')
    parser.add_argument('--projection_dim', type=int, default=50, help='Projection head output dimension')
    parser.add_argument('--projection_hidden_dim1', type=int, default=256, help='Projection head first hidden dimension')
    parser.add_argument('--projection_hidden_dim2', type=int, default=128, help='Projection head second hidden dimension')

    
    # SimCLR transformation functions
    parser.add_argument('--transform_funcs', nargs='+', 
                        default=['channel_shuffle', 'time_segment_permutation'],
                        help='List of transformation function names to use for SimCLR')
    
    args = parser.parse_args()
    
    # MTL task weights
    args.task_weights = {
        'noise': args.task_weights_noise,
        'scaling': args.task_weights_scaling,
        'time_warp': args.task_weights_time_warp,
        'rotation': args.task_weights_rotation,
        'time_segment_permutation': args.task_weights_time_segment_permutation,
        'negate': args.task_weights_negate,
        'horizontal_flip': args.task_weights_horizontal_flip,
        'channel_shuffle': args.task_weights_channel_shuffle,
    }
        
    # data config
    config_file = open('configs/data.yaml', mode='r')
    data_config = yaml.load(config_file, Loader=yaml.FullLoader)
    data_config = data_config[args.data_name]
    
    # path settings
    args.data_path = os.path.join("datasets", data_config['filename'])
    args.save_path = "saved"
    args.encoder_save_path = os.path.join(args.save_path, "encoders")
    args.classifier_save_path = os.path.join(args.save_path, "classifiers")
    args.results_save_path = os.path.join(args.save_path, "results")
    
    # GPU settings
    args.use_gpu = True if torch.cuda.is_available() else False
    args.use_multi_gpu = True
    args.gpu = 5
    args.devices = "0,1,2,3,4,5,6,7"
    
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
        args.device = torch.device(f'cuda:{args.gpu}')
        print(f'Device: GPU, cuda:{args.gpu}')
    else:
        args.device = torch.device('cpu')
        print('Device: CPU')
    
    args.optimizer = "Adam"
    args.criterion = "MSELoss" if args.train_encoder else "CrossEntropy"
    args.exp_mode = "LOCV"
    args.datanorm_type = "standardization"
    
    # training settings
    # mtl: maximum of 30 epochs with a learning rate of 0.0003
    args.train_epochs = 300
    args.learning_rate = 0.0005 # 0.0003 
    args.weight_decay = 0.0001  # mtl, simclr weight decay
    args.learning_rate_patience = 7
    args.learning_rate_factor = 0.1
    args.early_stop_patience = 20
    args.batch_size = 256
    args.shuffle = True
    args.drop_last = False
    args.train_vali_quote = 0.90

    args.classifier_lr = 0.0001
    args.classifier_epochs = 300
    args.classifier_batch_size = 256
    args.freeze_encoder = True  # Freeze
    
    # Time series input settings
    window_seconds = data_config["window_seconds"]
    args.window_size = int(window_seconds * data_config["sampling_freq"])
    args.input_length = args.window_size
    args.input_channels = data_config["num_channels"]
    args.sampling_freq = data_config["sampling_freq"]
    args.num_classes = data_config["num_classes"]
    
    # ECDF feature dimension
    args.n_ecdf_points = 25
    args.output_size = (3, 78) 
    
    # Random seed and other settings
    args.sensor_select = ["acc"]
    args.seed = 10
    args.filtering = True
    args.freq1 = 0.001
    args.freq2 = 25.0

    ## saved pickle file paths - data preprocessing
    # 1) preprocessed data x and y for all subject
    # 2) window indices 
    args.pkl_save_path = os.path.join("datasets", args.data_name, f"window_size_{args.window_size}")
    
    return args 