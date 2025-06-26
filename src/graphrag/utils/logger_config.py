import logging
import os
import sys
from datetime import datetime
from typing import Optional


class GraphRAGLogger:
    """Logger configuration cho GraphRAG System"""
    
    def __init__(self, 
                 name: str = "GraphRAG",
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
        """
        Khởi tạo logger
        
        Args:
            name: Tên logger
            log_level: Level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Đường dẫn file log (optional)
            log_format: Format của log message
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_file = log_file
        self.log_format = log_format
        
        # Tạo logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Tạo formatter
        formatter = logging.Formatter(self.log_format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (nếu có)
        if log_file:
            # Tạo thư mục log nếu chưa tồn tại
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def exception(self, message: str):
        """Log exception message với traceback"""
        self.logger.exception(message)


def setup_logger(name: str = "GraphRAG",
                log_level: str = "INFO",
                log_dir: str = "./logs",
                log_file: Optional[str] = None) -> GraphRAGLogger:
    """
    Setup logger cho hệ thống
    
    Args:
        name: Tên logger
        log_level: Level logging
        log_dir: Thư mục chứa log files
        log_file: Tên file log (optional, nếu None sẽ tự động tạo)
    
    Returns:
        GraphRAGLogger: Logger instance
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    return GraphRAGLogger(name, log_level, log_file)


# Global logger instance
_global_logger = None


def get_logger() -> GraphRAGLogger:
    """Lấy global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logger()
    return _global_logger


def set_logger(logger: GraphRAGLogger):
    """Set global logger instance"""
    global _global_logger
    _global_logger = logger 