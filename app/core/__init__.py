"""
Core compression implementations for the hybrid compression project.

This package contains the implementations of different compression algorithms:
- PLZ: A custom format combining PNGQuant and LZ4
- ZST: Zstandard compression for images
"""
from app.core.plz import (
    PLZ_MAGIC, 
    PNGQUANT_DEFAULT_QUALITY,
    create_plz_file, 
    extract_plz_file, 
    read_plz_metadata,
    compress_image_to_plz
)

from app.core.zst import (
    DEFAULT_COMPRESSION_LEVEL,
    compress_image, 
    decompress_image, 
    decompress_zstd,
    compress_image_with_zstd
)

__all__ = [
    # PLZ format
    'PLZ_MAGIC',
    'PNGQUANT_DEFAULT_QUALITY',
    'create_plz_file',
    'extract_plz_file',
    'read_plz_metadata',
    'compress_image_to_plz',
    
    # ZST format
    'DEFAULT_COMPRESSION_LEVEL',
    'compress_image',
    'decompress_image',
    'decompress_zstd',
    'compress_image_with_zstd'
]