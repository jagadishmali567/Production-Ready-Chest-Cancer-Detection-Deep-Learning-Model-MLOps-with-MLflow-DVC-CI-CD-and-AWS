import os
import sys
import logging
from logging.handlers import RotatingFileHandler

# Define logging format string
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Ensure log directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Define log file path
log_filepath = os.path.join(log_dir, "running_logs.log")

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        RotatingFileHandler(log_filepath, maxBytes=10**6, backupCount=5),  # Rotate logs when file size exceeds 1MB
        logging.StreamHandler(sys.stdout)
    ]
)

# Get logger instance
logger = logging.getLogger("cnnClassifierLogger")

# Example usage of the logger
logger.info("Logging setup is complete.")
