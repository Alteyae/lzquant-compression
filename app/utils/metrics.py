"""
Utilities for measuring compression performance and image quality.
"""
import time
import logging
import numpy as np
import psutil
from PIL import Image
from typing import Tuple, Optional, Dict, Any, Union
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Set up logging
logger = logging.getLogger(__name__)


def get_cpu_mem() -> Dict[str, float]:
    """
    Get current CPU and memory usage.
    
    Returns:
        Dictionary with CPU and memory usage percentages
    """
    return {
        "cpu_usage": psutil.cpu_percent(interval=None),
        "memory_usage": psutil.virtual_memory().percent
    }


def calculate_image_metrics(
    original_img: Union[np.ndarray, Image.Image], 
    decompressed_img: Union[np.ndarray, Image.Image]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate PSNR and SSIM for image quality comparison.
    
    Args:
        original_img: Original image (PIL Image or numpy array)
        decompressed_img: Decompressed image (PIL Image or numpy array)
        
    Returns:
        Tuple of (PSNR, SSIM) values, rounded to 2 and 4 decimal places respectively
        Returns (None, None) if calculation fails
    """
    # Convert PIL images to numpy arrays if needed
    if not isinstance(original_img, np.ndarray):
        try:
            original_img = np.array(original_img.convert("RGB"))
        except Exception as e:
            logger.error(f"Failed to convert original image to array: {e}")
            return None, None
            
    if not isinstance(decompressed_img, np.ndarray):
        try:
            decompressed_img = np.array(decompressed_img.convert("RGB"))
        except Exception as e:
            logger.error(f"Failed to convert decompressed image to array: {e}")
            return None, None
    
    try:
        # Make sure images are same size
        if original_img.shape != decompressed_img.shape:
            logger.info(f"Image shapes don't match: original {original_img.shape} vs decompressed {decompressed_img.shape}")
            # Resize decompressed to match original's dimensions
            try:
                decompressed_pil = Image.fromarray(decompressed_img)
                original_pil = Image.fromarray(original_img)
                decompressed_pil = decompressed_pil.resize((original_pil.width, original_pil.height))
                decompressed_img = np.array(decompressed_pil)
                logger.info(f"Resized decompressed image to {decompressed_img.shape}")
            except Exception as e:
                logger.error(f"Failed to resize images to match: {e}")
                return None, None
        
        # Check for data type and valid range
        if np.issubdtype(original_img.dtype, np.integer) and np.issubdtype(decompressed_img.dtype, np.integer):
            # Handle integer images - make sure both arrays have same dtype for comparison
            max_val = 255
            if original_img.dtype != decompressed_img.dtype:
                logger.info(f"Converting image dtypes to match: {original_img.dtype} and {decompressed_img.dtype}")
                decompressed_img = decompressed_img.astype(original_img.dtype)
        else:
            # Handle float images
            max_val = 1.0
            if original_img.max() > 1.0 or decompressed_img.max() > 1.0:
                logger.info("Normalizing float images to [0, 1] range")
                original_img = original_img / 255.0 if original_img.max() > 1.0 else original_img
                decompressed_img = decompressed_img / 255.0 if decompressed_img.max() > 1.0 else decompressed_img
        
        # Compare the images to check if they're different
        image_diff = np.abs(original_img.astype(np.float32) - decompressed_img.astype(np.float32))
        mse = np.mean(np.square(image_diff))
        
        logger.info(f"MSE between original and decompressed: {mse}")
        
        # Handle case when images are very similar but not identical
        if mse < 1e-10:  # Extremely close to identical
            logger.info("Images are nearly identical, MSE is extremely small")
            psnr = 100.0  # Set an arbitrarily high PSNR value
        elif mse == 0:  # Truly identical
            logger.info("Images are exactly identical, MSE is zero")
            psnr = float('inf')  # Mathematically correct value for identical images
        else:
            # Normal case - images have some difference
            logger.info(f"Calculating PSNR with MSE={mse}")
            try:
                psnr = peak_signal_noise_ratio(original_img, decompressed_img, data_range=max_val)
                logger.info(f"Calculated PSNR: {psnr}")
            except Exception as e:
                logger.error(f"Error in PSNR calculation: {e}")
                # Manual PSNR calculation if scikit-image fails
                psnr = 10 * np.log10((max_val ** 2) / max(mse, 1e-10))
        
        ssim = structural_similarity(original_img, decompressed_img, data_range=max_val, channel_axis=2)
        
        return round(psnr, 2), round(ssim, 4)
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return None, None


def measure_compression_performance(
    original_size: int, 
    compressed_size: int, 
    compression_time: float
) -> Dict[str, float]:
    """
    Calculate compression performance metrics.
    
    Args:
        original_size: Size of the original file in bytes
        compressed_size: Size of the compressed file in bytes
        compression_time: Time taken for compression in seconds
        
    Returns:
        Dictionary with compression ratio, space savings percentage, and compression speed
    """
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    space_savings = (1 - (compressed_size / original_size)) * 100 if original_size > 0 else 0
    compression_speed = original_size / (compression_time * 1024 * 1024) if compression_time > 0 else 0  # MB/s
    
    return {
        "compression_ratio": round(compression_ratio, 2),
        "space_savings_percent": round(space_savings, 2),
        "compression_speed_mbps": round(compression_speed, 2)
    }


class PerformanceTimer:
    """
    Context manager for measuring execution time.
    
    Example:
        with PerformanceTimer() as timer:
            # Code to measure
        execution_time = timer.execution_time
    """
    
    def __init__(self):
        self.start_time = None
        self.execution_time = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.execution_time = time.time() - self.start_time
        return False  # Don't suppress exceptions