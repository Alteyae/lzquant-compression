"""
Hybrid Image Compression API Entry Point

Run with uvicorn:
    uvicorn main:app --reload
"""
import os
import logging
import sys
from app import app

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

try:
    import PIL
    import lz4
    import zstandard
    import psutil
    import numpy
    logger.info("All required dependencies are available")
except ImportError as e:
    logger.critical(f"Missing required dependency: {str(e)}")
    sys.exit(1)

try:
    import subprocess
    subprocess.run(["pngquant", "--version"], check=True, capture_output=True)
    logger.info("pngquant is installed and working")
except (subprocess.SubprocessError, FileNotFoundError):
    logger.warning("pngquant not found. PLZ compression will not function correctly.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    workers = int(os.environ.get("WORKERS", 1))
    debug = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=workers, reload=debug)
