"""
Logger Module
This module provides logging functionality for the project
"""

import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name):
    """
    Get existing logger or create new one
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)

class LoggerContext:
    """Context manager for temporary logging"""
    
    def __init__(self, name, log_file, level=logging.INFO):
        """
        Initialize logger context
        
        Args:
            name: Logger name
            log_file: Path to log file
            level: Logging level
        """
        self.logger = setup_logger(name, log_file, level)
    
    def __enter__(self):
        """Enter context"""
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context"""
        # Close and remove handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

def log_execution_time(func):
    """
    Decorator to log function execution time
    
    Args:
        func: Function to decorate
        
    Returns:
        function: Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = datetime.now()
        
        logger.info(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"Completed {func.__name__} in {duration:.2f} seconds")
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"Error in {func.__name__} after {duration:.2f} seconds: {str(e)}")
            raise
    
    return wrapper

def log_error(logger, error, context=None):
    """
    Log error with context
    
    Args:
        logger: Logger instance
        error: Error exception
        context: Additional context information
    """
    error_msg = f"Error: {str(error)}"
    
    if context:
        error_msg += f" | Context: {context}"
    
    logger.error(error_msg, exc_info=True)

def create_run_logger(run_name=None):
    """
    Create logger for a specific run
    
    Args:
        run_name: Name of the run
        
    Returns:
        logging.Logger: Logger for the run
    """
    if run_name is None:
        run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    log_file = f'logs/run_{run_name}.log'
    logger = setup_logger(f'run_{run_name}', log_file)
    
    return logger

if __name__ == '__main__':
    # Test logger
    logger = setup_logger('test_logger', 'logs/test.log')
    
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test decorator
    @log_execution_time
    def test_function():
        import time
        time.sleep(1)
        return "Done"
    
    result = test_function()
    print(result)
