import os
import logging
from datetime import datetime
from pathlib import Path

class Logger:
    _run_id = None
    _log_dir_path = None
    _loggers = {}
    _train_file_handler = None
    _debug_file_handler = None
    _is_initialized = False
    
    @classmethod
    def initialize(cls, log_dir='logs'):
        """
        Initialize logger at class level. Sets up file handlers for all instances to use.
        
        Args:
            log_dir: Directory where log files will be saved
        """
        # Skip if already initialized
        if cls._is_initialized:
            return
            
        # Create log directory structure
        base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        log_dir_path = os.path.join(base_dir, log_dir)
        Path(log_dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create training logs directory
        train_log_dir = os.path.join(log_dir_path, 'training')
        Path(train_log_dir).mkdir(parents=True, exist_ok=True)
        
        # Create debug logs directory
        debug_log_dir = os.path.join(log_dir_path, 'debug')
        Path(debug_log_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate run ID (timestamp)
        cls._run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        cls._log_dir_path = log_dir_path
        
        # Create common formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create training file handler
        train_file_path = os.path.join(train_log_dir, f'train_{cls._run_id}.log')
        train_file_handler = logging.FileHandler(train_file_path)
        train_file_handler.setLevel(logging.INFO)
        train_file_handler.setFormatter(formatter)
        cls._train_file_handler = train_file_handler
        
        # Create debug file handler
        debug_file_path = os.path.join(debug_log_dir, f'debug_{cls._run_id}.log')
        debug_file_handler = logging.FileHandler(debug_file_path)
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(formatter)
        cls._debug_file_handler = debug_file_handler
        
        # Clear and reconfigure root logger
        root_logger = logging.getLogger("root")
        if root_logger.handlers:
            root_logger.handlers.clear()
            
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(train_file_handler)
        root_logger.addHandler(debug_file_handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        root_logger.info(f"===== New Session Started: {cls._run_id} =====")
        
        # Mark as initialized
        cls._is_initialized = True
    
    def __init__(self, name, log_level=logging.INFO):
        """
        Initialize a new logger instance
        
        Args:
            name: Name of the logger
            log_level: Minimum log level to display in console (default: INFO)
        """
        self.name = name
        
        # Initialize logger system if not already done
        if not Logger._is_initialized:
            Logger.initialize()
        
        # Reuse existing logger if it exists
        if name in Logger._loggers:
            self.logger = Logger._loggers[name]
            return
            
        # Create new logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Add handlers
        self.logger.addHandler(Logger._train_file_handler)
        self.logger.addHandler(Logger._debug_file_handler)
        
        # Add console handler with specified log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Ensure we don't propagate to root logger to avoid duplicate logs
        self.logger.propagate = False
        
        # Cache logger
        Logger._loggers[name] = self.logger
    
    def get_logger_path(self, log_type='train'):
        """
        Get the path to a specific log file
        
        Args:
            log_type: Type of log (train or debug)
            
        Returns:
            Path to the log file
        """
        if log_type == 'train':
            return os.path.join(Logger._log_dir_path, 'training', f'train_{Logger._run_id}.log')
        elif log_type == 'debug':
            return os.path.join(Logger._log_dir_path, 'debug', f'debug_{Logger._run_id}.log')
        else:
            return None
        
    def info(self, msg):
        """Log an info message"""
        self.logger.info(msg)
        
    def debug(self, msg):
        """Log a debug message"""
        self.logger.debug(msg)
        
    def warning(self, msg):
        """Log a warning message"""
        self.logger.warning(msg)
        
    def error(self, msg):
        """Log an error message"""
        self.logger.error(msg)
        
    @classmethod
    def get_run_id(cls):
        """Return the current run ID"""
        if not cls._is_initialized:
            cls.initialize()
        return cls._run_id 