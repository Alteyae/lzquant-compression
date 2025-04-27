"""
Hybrid Image Compression API Entry Point

This file serves as the main entry point for the application,
importing and running the FastAPI application defined in the app package.

Run with uvicorn:
    uvicorn main:app --reload
"""
import os
import logging
import sys
from app import app

# Configure logging based on environment variables
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Configure root logger
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    stream=sys.stdout
)

# Set up logger
logger = logging.getLogger(__name__)

# Check that required dependencies are installed
try:
    import PIL
    import lz4
    import zstandard
    import psutil
    import numpy
    import sklearn
    logger.info("All required dependencies are available")
except ImportError as e:
    logger.critical(f"Missing required dependency: {str(e)}")
    logger.critical("Please install all dependencies: pip install -r requirements.txt")
    sys.exit(1)

# Check for pngquant
try:
    import subprocess
    subprocess.run(["pngquant", "--version"], check=True, capture_output=True)
    logger.info("pngquant is installed and working")
except (subprocess.SubprocessError, FileNotFoundError):
    logger.warning("pngquant is not installed or not working. PLZ compression will not function correctly.")

# Entry point for running with uvicorn
if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment variables
    port = int(os.environ.get("PORT", 8000))
    workers = int(os.environ.get("WORKERS", 1))
    debug = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")
    
    logger.info(f"Starting Hybrid Compression API on port {port} with {workers} workers")
    
    # Using multiprocessing workers for better performance
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=port,
        workers=workers,
        reload=debug
    )