import os
import logging
from datetime import datetime
from pathlib import Path

class Logger:
    def __init__(self, name, log_dir='detailed_logs'):
        self.name = name
        
        # Create logs directory if it doesn't exist
        log_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', log_dir)
        Path(log_dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            os.path.join(log_dir_path, f'{name}_{current_time}.log')
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def info(self, msg):
        self.logger.info(msg)
        
    def debug(self, msg):
        self.logger.debug(msg)
        
    def warning(self, msg):
        self.logger.warning(msg)
        
    def error(self, msg):
        self.logger.error(msg) 