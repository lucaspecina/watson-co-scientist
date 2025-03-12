"""
Logging setup and utilities for the Co-Scientist system.
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
import platform

# Check if we're running on Windows
IS_WINDOWS = platform.system() == "Windows"

# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"
    
    @staticmethod
    def enabled():
        """Check if colors should be enabled"""
        if IS_WINDOWS:
            # Enable ANSI color support on Windows
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                return True
            except:
                return False
        return True

# Custom formatter that colorizes output
class ColorizedFormatter(logging.Formatter):
    """A formatter that adds colors to logs based on level"""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.BLUE,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BG_RED + Colors.WHITE
    }
    
    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and Colors.enabled()
    
    def format(self, record):
        # Save original values
        levelname = record.levelname
        message = record.msg
        
        if self.use_colors:
            # Apply colors based on log level
            color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
            
            # Format the levelname with color
            record.levelname = f"{color}{levelname}{Colors.RESET}"
            
            # Add color to important messages for visibility
            if record.levelno >= logging.WARNING:
                record.msg = f"{color}{message}{Colors.RESET}"
        
        result = super().format(record)
        
        # Restore original values
        record.levelname = levelname
        record.msg = message
        
        return result

# Create a formatter for pretty-printing user-facing output
class UserOutputFormatter:
    """Utility class for formatting user-facing output with colors and structure"""
    
    @staticmethod
    def section_header(text, width=60):
        """Format a section header with a box"""
        if Colors.enabled():
            return f"\n{Colors.BG_BLUE}{Colors.WHITE}{Colors.BOLD} {text.center(width-2)} {Colors.RESET}\n"
        else:
            return f"\n{'=' * width}\n{text.center(width)}\n{'=' * width}\n"
    
    @staticmethod
    def subsection_header(text, width=60):
        """Format a subsection header"""
        if Colors.enabled():
            return f"\n{Colors.CYAN}{Colors.BOLD}{text}{Colors.RESET}\n{'-' * len(text)}\n"
        else:
            return f"\n{text}\n{'-' * len(text)}\n"
    
    @staticmethod
    def highlight(text):
        """Highlight important text"""
        if Colors.enabled():
            return f"{Colors.YELLOW}{Colors.BOLD}{text}{Colors.RESET}"
        else:
            return f"** {text} **"
    
    @staticmethod
    def success(text):
        """Format success messages"""
        if Colors.enabled():
            return f"{Colors.GREEN}{Colors.BOLD}✓ {text}{Colors.RESET}"
        else:
            return f"[SUCCESS] {text}"
    
    @staticmethod
    def error(text):
        """Format error messages"""
        if Colors.enabled():
            return f"{Colors.RED}{Colors.BOLD}✗ {text}{Colors.RESET}"
        else:
            return f"[ERROR] {text}"
    
    @staticmethod
    def info(text):
        """Format info messages"""
        if Colors.enabled():
            return f"{Colors.BLUE}ℹ {text}{Colors.RESET}"
        else:
            return f"[INFO] {text}"

def setup_logger(name="co_scientist", level=logging.INFO):
    """
    Set up and configure a logger.
    
    Args:
        name (str): The name of the logger.
        level (int): The logging level.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    use_colors = Colors.enabled()
    
    # Detailed formatter for file logs
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # More concise, colorized formatter for console
    console_formatter = ColorizedFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        use_colors=use_colors
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create file handler
    log_file = f"logs/{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10485760, backupCount=5
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Add a user-output utility to the module for easy import
user_output = UserOutputFormatter()