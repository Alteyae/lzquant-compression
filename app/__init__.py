"""
Hybrid Image Compression API Application

This package implements a FastAPI application for hybrid image compression,
providing multiple compression algorithms:
- PLZ format (PNGQuant + LZ4)
- Zstandard compression

Features include:
- Single and batch image compression
- Quality metrics (PSNR, SSIM)
- Performance metrics
- Metadata support
"""
import tempfile
import os

# Create a temporary directory for file storage
TEMP_DIR = tempfile.mkdtemp()

# Export the app instance
from app.api import app

__all__ = ['app', 'TEMP_DIR']