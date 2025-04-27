"""
Simplified Zstandard (zstd) compression implementation for images.

This module provides basic functions for compressing and decompressing images 
using the Zstandard library directly.
"""
import os
import json
import time
import math
import logging
import zstandard as zstd
from typing import Dict, Any, Tuple, Union, Optional

# Set up logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_COMPRESSION_LEVEL = 10

def compress_image(
    input_data: Union[bytes, str],
    compression_level: int = DEFAULT_COMPRESSION_LEVEL
) -> bytes:
    """
    Compress image data using Zstandard.
    
    Args:
        input_data: Raw image data as bytes or path to image file
        compression_level: Zstandard compression level (1-22)
        
    Returns:
        Compressed data as bytes
    """
    # Input handling
    if isinstance(input_data, str):
        # It's a file path
        with open(input_data, 'rb') as f:
            data = f.read()
    else:
        # It's raw data
        data = input_data
    
    # Create compressor with specified level
    cctx = zstd.ZstdCompressor(level=compression_level)
    
    # Compress the data
    compressed_data = cctx.compress(data)
    
    # Verify data is different after compression (for debugging)
    from app.utils.metrics import logger as metrics_logger
    if data == compressed_data:
        metrics_logger.warning("WARNING: Compressed data is identical to original data. This suggests no actual compression occurred.")
    
    return compressed_data


def decompress_image(compressed_data: bytes) -> bytes:
    """
    Decompress Zstandard compressed data.
    
    Args:
        compressed_data: Data compressed with Zstandard
        
    Returns:
        Decompressed data as bytes
    """
    # Create decompressor
    dctx = zstd.ZstdDecompressor()
    
    # Decompress the data
    decompressed_data = dctx.decompress(compressed_data)
    
    return decompressed_data


def compress_image_with_zstd(
    input_data: Union[bytes, str],
    output_path: str,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Compress an image using Zstandard compression and write to file.
    
    Args:
        input_data: Raw image data as bytes or path to image file
        output_path: Path to save the compressed output
        compression_level: Zstandard compression level (1-22)
        include_metadata: Not used in this simplified version
        
    Returns:
        Dictionary with compression metrics
    """
    # Get file ID from output path
    file_id = os.path.basename(output_path).split('_')[0]
    
    # Input handling
    if isinstance(input_data, str):
        # It's a file path
        with open(input_data, 'rb') as f:
            original_data = f.read()
    else:
        # It's raw data
        original_data = input_data
    
    original_size = len(original_data)
    
    # Time the compression
    start_time = time.time()
    compressed_data = compress_image(original_data, compression_level)
    compression_time = time.time() - start_time
    
    # Write compressed data to output
    with open(output_path, "wb") as f:
        f.write(compressed_data)
    
    # Calculate metrics
    compressed_size = len(compressed_data)
    compression_ratio = round(original_size / compressed_size, 2) if compressed_size > 0 else 0
    space_savings = round((1 - (compressed_size / original_size)) * 100, 2) if original_size > 0 else 0
    
    # Get CPU and memory usage
    try:
        # Import here to avoid circular imports
        from app.utils.metrics import get_cpu_mem
        cpu_mem = get_cpu_mem()
        cpu_usage = cpu_mem["cpu_usage"]
        memory_usage = cpu_mem["memory_usage"]
    except Exception:
        cpu_usage = 0
        memory_usage = 0
    
    # Calculate image quality metrics (PSNR & SSIM)
    psnr = None
    ssim = None
    try:
        from io import BytesIO
        from PIL import Image
        from app.utils.metrics import calculate_image_metrics
        
        # Open original image
        try:
            original_image = Image.open(BytesIO(original_data))
            
            # Decompress to test quality
            decompressed_data = decompress_image(compressed_data)
            decompressed_image = Image.open(BytesIO(decompressed_data))
            
            # Make sure both images are in RGB mode for consistent comparison
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
            if decompressed_image.mode != 'RGB':
                decompressed_image = decompressed_image.convert('RGB')
                
            # Calculate quality metrics
            psnr, ssim = calculate_image_metrics(original_image, decompressed_image)
            
            if psnr is None:
                logger.warning("PSNR calculation returned None even though images were loaded successfully")
        except Exception as e:
            logger.warning(f"Error processing images for metrics: {str(e)}")
    except Exception as e:
        logger.warning(f"Could not calculate image quality metrics: {str(e)}")
    
    # Sanitize values for JSON compliance (handling inf, nan)
    def sanitize_float(value):
        if value is None:
            return None
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    
    # Return compression metrics with sanitized values
    return {
        "file_id": file_id,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": sanitize_float(compression_ratio),
        "space_savings_percent": sanitize_float(space_savings),
        "compression_time": sanitize_float(round(compression_time, 4)),
        "compression_level": compression_level,
        "cpu_usage": sanitize_float(cpu_usage),
        "memory_usage": sanitize_float(memory_usage),
        "psnr": sanitize_float(psnr),
        "ssim": sanitize_float(ssim)
    }


def decompress_zstd(compressed_data: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """
    Decompress Zstandard compressed data.
    
    Args:
        compressed_data: Compressed data
        
    Returns:
        Tuple of (decompressed_data, empty_metadata_dict)
    """
    try:
        decompressed_data = decompress_image(compressed_data)
        return decompressed_data, {}
    except Exception as e:
        logger.error(f"Decompression failed: {str(e)}")
        raise ValueError(f"Decompression failed: {str(e)}")