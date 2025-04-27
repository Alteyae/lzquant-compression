"""
Implementation of the PLZ compression format.

PLZ is a custom image compression format combining PNGQuant with LZ4 compression.
File format structure:
- 4 bytes: Magic number "PLZ\0"
- 4 bytes: Header size (uint32)
- N bytes: JSON header with metadata
- Rest: LZ4 compressed PNG data
"""
import os
import json
import struct
import logging
import lz4.frame
import subprocess
from PIL import Image
from typing import Dict, Any, Optional, Tuple, BinaryIO
from io import BytesIO

from app.utils.metrics import get_cpu_mem, calculate_image_metrics, PerformanceTimer
from app.utils.file_handling import get_temp_filepath, schedule_cleanup

# Set up logging
logger = logging.getLogger(__name__)

# PLZ format constants
PLZ_MAGIC = b'PLZ\0'
PNGQUANT_DEFAULT_QUALITY = 80


def create_plz_file(pngquant_path: str, plz_path: str, metadata: Optional[Dict[str, Any]] = None) -> int:
    """
    Create a PLZ file from a pngquant-compressed PNG.

    Args:
        pngquant_path: Path to the PNGQuant compressed PNG file
        plz_path: Output path for the PLZ file
        metadata: Optional metadata to include in the PLZ header

    Returns:
        Size of the created PLZ file in bytes
    """
    if metadata is None:
        metadata = {}
    
    # Compress the PNG with LZ4
    with open(pngquant_path, 'rb') as f:
        png_data = f.read()
    
    compressed_data = lz4.frame.compress(
        png_data, 
        compression_level=lz4.frame.COMPRESSIONLEVEL_MAX
    )
    
    # Create the PLZ header
    header_json = json.dumps(metadata).encode('utf-8')
    header_size = len(header_json)
    
    # Write the PLZ file
    with open(plz_path, 'wb') as f:
        # Magic number
        f.write(PLZ_MAGIC)
        # Header size
        f.write(struct.pack('<I', header_size))
        # Header data
        f.write(header_json)
        # Compressed data
        f.write(compressed_data)
    
    return os.path.getsize(plz_path)


def extract_plz_file(plz_path: str, output_path: str) -> Dict[str, Any]:
    """
    Extract a PNG from a PLZ file.

    Args:
        plz_path: Path to the PLZ file
        output_path: Output path for the extracted PNG

    Returns:
        Metadata from the PLZ file
    
    Raises:
        ValueError: If the file is not a valid PLZ file
    """
    with open(plz_path, 'rb') as f:
        # Read magic number
        magic = f.read(4)
        if magic != PLZ_MAGIC:
            raise ValueError("Not a valid PLZ file (wrong magic number)")
        
        # Read header size
        header_size = struct.unpack('<I', f.read(4))[0]
        
        # Read header data
        header_json = f.read(header_size)
        metadata = json.loads(header_json.decode('utf-8'))
        
        # Read compressed data
        compressed_data = f.read()
        
        # Decompress data
        decompressed_data = lz4.frame.decompress(compressed_data)
        
        # Write decompressed PNG
        with open(output_path, 'wb') as out_f:
            out_f.write(decompressed_data)
    
    return metadata


def read_plz_metadata(plz_path: str) -> Tuple[Dict[str, Any], bool]:
    """
    Read metadata from a PLZ file without extracting the full file.
    
    Args:
        plz_path: Path to the PLZ file
        
    Returns:
        Tuple of (metadata, is_valid) where is_valid indicates if the file is a valid PLZ file
    """
    try:
        with open(plz_path, 'rb') as f:
            # Read magic number
            magic = f.read(4)
            if magic != PLZ_MAGIC:
                return {"error": "Not a valid PLZ file"}, False
            
            # Read header size
            header_size = struct.unpack('<I', f.read(4))[0]
            
            # Read header data
            header_json = f.read(header_size)
            metadata = json.loads(header_json.decode('utf-8'))
            
            return metadata, True
    except Exception as e:
        logger.error(f"Failed to read PLZ metadata: {str(e)}")
        return {"error": f"Failed to read PLZ metadata: {str(e)}"}, False


def compress_image_to_plz(
    input_data: bytes, 
    output_path: str, 
    quality: int = PNGQUANT_DEFAULT_QUALITY,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compress an image to PLZ format.
    
    Args:
        input_data: Raw image data as bytes
        output_path: Path where to save the PLZ file
        quality: PNGQuant quality setting (0-100)
        metadata: Optional metadata to include
        
    Returns:
        Dictionary with compression metrics
    
    Raises:
        ValueError: If compression fails
    """
    # Generate temporary file paths
    file_id = os.path.basename(output_path).split('_')[0]
    input_path = get_temp_filepath(file_id, "_input.png")
    pngquant_path = get_temp_filepath(file_id, "_pngquant.png")
    
    # Start performance monitoring
    timer = PerformanceTimer()
    results = {}
    
    try:
        # Save input data to file
        with open(input_path, "wb") as f:
            f.write(input_data)
        
        # Check if it's a valid image and convert to PNG if needed
        img = Image.open(BytesIO(input_data))
        img.save(input_path, format="PNG")
        
        # Store original image for quality comparison
        original_image = img.convert("RGB")
        original_size = len(input_data)
        
        # Compress with pngquant
        with timer:
            try:
                subprocess.run(
                    ["pngquant", "--force", "--output", pngquant_path, 
                     "--quality", f"{max(0, quality-10)}-{quality}", input_path], 
                    check=True, capture_output=True
                )
            except subprocess.CalledProcessError as e:
                raise ValueError(f"pngquant compression failed: {e.stderr.decode('utf-8')}")
            except FileNotFoundError:
                raise ValueError("pngquant not installed. Please install pngquant for hybrid compression.")
        
        compression_time = timer.execution_time
        pngquant_size = os.path.getsize(pngquant_path)
        
        # Create PLZ file
        if metadata is None:
            metadata = {}
        
        # Add compression metadata
        metadata.update({
            "compression_quality": quality,
            "created_timestamp": timer.start_time,
            "version": "1.0"
        })
        
        final_size = create_plz_file(pngquant_path, output_path, metadata)
        
        # Decompress for quality metrics
        decompressed_path = get_temp_filepath(file_id, "_decompressed.png")
        
        with timer:
            extract_plz_file(output_path, decompressed_path)
        decompression_time = timer.execution_time
        
        # Image quality metrics
        try:
            decompressed_img = Image.open(decompressed_path).convert("RGB")
            psnr, ssim = calculate_image_metrics(original_image, decompressed_img)
        except Exception as e:
            logger.error(f"Error calculating image metrics: {str(e)}")
            psnr, ssim = None, None
        
        # Calculate compression ratios
        pngquant_ratio = ((original_size - pngquant_size) / original_size) * 100 if original_size > 0 else 0
        lz4_ratio = ((pngquant_size - final_size) / pngquant_size) * 100 if pngquant_size > 0 else 0
        total_ratio = ((original_size - final_size) / original_size) * 100 if original_size > 0 else 0
        
        # System resource usage
        cpu_mem = get_cpu_mem()
        
        # Populate results
        results = {
            "original_size": original_size,
            "pngquant_size": pngquant_size,
            "final_size": final_size,
            "compressed_size": final_size,  # Add compressed_size field explicitly
            "pngquant_compression_ratio": pngquant_ratio,
            "lz4_compression_ratio": lz4_ratio,
            "total_compression_ratio": total_ratio,
            "compression_ratio": original_size / final_size if final_size > 0 else 0,
            "space_savings_percent": total_ratio,
            "compression_time": round(compression_time, 4),
            "decompression_time": round(decompression_time, 4),
            "cpu_usage": cpu_mem["cpu_usage"],
            "memory_usage": cpu_mem["memory_usage"],
            "psnr": psnr,
            "ssim": ssim
        }
        
        # Schedule cleanup of temporary files
        for path in [input_path, pngquant_path, decompressed_path]:
            if os.path.exists(path):
                schedule_cleanup(path)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in PLZ compression: {str(e)}")
        # Clean up any temporary files
        for path in [input_path, pngquant_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        raise