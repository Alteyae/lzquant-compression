"""
PLZ/LZQuant Comprehensive Implementation

This module provides a full implementation of the PLZ compression format,
which combines PNGQuant color quantization with LZ4 compression to achieve
optimal image compression with quality control.

File format structure:
- 4 bytes: Magic number "PLZ\0"
- 4 bytes: Header size (uint32)
- N bytes: JSON header with metadata
- Rest: LZ4 compressed PNG data

Author: Claude
Version: 1.0
"""

import os
import io
import json
import struct
import time
import logging
import subprocess
import tempfile
import numpy as np
import lz4.frame
from PIL import Image
from typing import Dict, Any, Optional, Tuple, BinaryIO, List, Union
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import psutil

# Constants
PLZ_MAGIC = b'PLZ\0'
PNGQUANT_DEFAULT_QUALITY = 80
TEMP_FILE_PREFIX = "plz_temp_"
DEFAULT_TEMP_DIR = tempfile.gettempdir()
LZ4_COMPRESSION_LEVEL = lz4.frame.COMPRESSIONLEVEL_MAX
LZ4_BLOCK_SIZE = lz4.frame.BLOCKSIZE_MAX64KB
VERSION = "1.0"

# Logging configuration
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class CompressionOptions:
    """Configuration options for PLZ compression."""
    
    def __init__(
        self, 
        quality: int = PNGQUANT_DEFAULT_QUALITY,
        temp_dir: Optional[str] = None,
        dithering: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        collect_metrics: bool = True,
        cleanup_temp_files: bool = True
    ):
        """
        Initialize compression options.
        
        Args:
            quality: PNGQuant quality setting (0-100, higher is better quality)
            temp_dir: Directory for temporary files
            dithering: Whether to use dithering in PNGQuant
            metadata: Additional metadata to include in the PLZ header
            collect_metrics: Whether to collect detailed performance metrics
            cleanup_temp_files: Whether to automatically clean up temporary files
        """
        self.quality = max(0, min(100, quality))  # Clamp between 0-100
        self.temp_dir = temp_dir if temp_dir else DEFAULT_TEMP_DIR
        self.dithering = dithering
        self.metadata = metadata if metadata else {}
        self.collect_metrics = collect_metrics
        self.cleanup_temp_files = cleanup_temp_files


class CompressionMetrics:
    """Detailed metrics about a compression operation."""
    
    def __init__(self):
        # File sizes
        self.original_size: int = 0
        self.quantized_size: int = 0
        self.final_size: int = 0
        
        # Compression ratios
        self.pngquant_compression_ratio: float = 0.0
        self.lz4_compression_ratio: float = 0.0
        self.total_compression_ratio: float = 0.0
        self.compression_ratio: float = 0.0
        
        # Timing
        self.start_time: float = 0.0
        self.quantization_time: float = 0.0
        self.lz4_compression_time: float = 0.0
        self.total_compression_time: float = 0.0
        self.decompression_time: float = 0.0
        
        # Quality metrics
        self.psnr: Optional[float] = None
        self.ssim: Optional[float] = None
        
        # System resources
        self.peak_memory_usage: float = 0.0
        self.cpu_usage: float = 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary."""
        return {
            # Size metrics
            "original_size": self.original_size,
            "quantized_size": self.quantized_size,
            "final_size": self.final_size,
            "compressed_size": self.final_size,  # Alias for compatibility
            
            # Compression ratios
            "pngquant_compression_ratio": round(self.pngquant_compression_ratio, 2),
            "lz4_compression_ratio": round(self.lz4_compression_ratio, 2),
            "total_compression_ratio": round(self.total_compression_ratio, 2),
            "compression_ratio": round(self.compression_ratio, 2),
            "space_savings_percent": round(self.total_compression_ratio, 2),
            
            # Timing
            "quantization_time": round(self.quantization_time, 4),
            "lz4_compression_time": round(self.lz4_compression_time, 4),
            "total_compression_time": round(self.total_compression_time, 4),
            "decompression_time": round(self.decompression_time, 4),
            
            # Quality metrics
            "psnr": round(self.psnr, 2) if self.psnr is not None else None,
            "ssim": round(self.ssim, 4) if self.ssim is not None else None,
            
            # System resources
            "peak_memory_usage_mb": round(self.peak_memory_usage, 2),
            "cpu_usage_percent": round(self.cpu_usage, 2)
        }


class PLZCompressor:
    """Main class for PLZ format compression operations."""
    
    def __init__(self, options: Optional[CompressionOptions] = None):
        """
        Initialize the PLZ compressor.
        
        Args:
            options: Compression options, or None to use defaults
        """
        self.options = options if options else CompressionOptions()
        self._verify_dependencies()
    
    def _verify_dependencies(self) -> None:
        """Verify that all required dependencies are available."""
        try:
            result = subprocess.run(
                ["pngquant", "--version"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            if result.returncode != 0:
                logger.warning("pngquant not found or not working correctly")
        except FileNotFoundError:
            logger.warning("pngquant not installed. PLZ compression requires pngquant.")
    
    def _get_temp_filepath(self, prefix: str, suffix: str) -> str:
        """
        Generate a temporary file path.
        
        Args:
            prefix: Prefix for the temporary file
            suffix: Suffix for the temporary file
            
        Returns:
            Path to a temporary file
        """
        return os.path.join(
            self.options.temp_dir, 
            f"{TEMP_FILE_PREFIX}{prefix}_{int(time.time() * 1000)}{suffix}"
        )
    
    def _cleanup_temp_files(self, files: List[str]) -> None:
        """
        Clean up temporary files.
        
        Args:
            files: List of file paths to clean up
        """
        if not self.options.cleanup_temp_files:
            return
            
        for file_path in files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {file_path}: {e}")
    
    def _get_system_resources(self) -> Dict[str, float]:
        """
        Get current system resource usage.
        
        Returns:
            Dictionary with CPU and memory usage
        """
        process = psutil.Process(os.getpid())
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": process.memory_info().rss / (1024 * 1024)  # MB
        }
    
    def _calculate_image_metrics(
        self, 
        original_image: Image.Image, 
        compressed_image: Image.Image
    ) -> Tuple[float, float]:
        """
        Calculate image quality metrics.
        
        Args:
            original_image: Original image
            compressed_image: Compressed image
            
        Returns:
            Tuple of (PSNR, SSIM)
        """
        # Convert images to same size and mode if needed
        if original_image.size != compressed_image.size:
            compressed_image = compressed_image.resize(original_image.size)
        
        # Convert to RGB for comparison
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        if compressed_image.mode != 'RGB':
            compressed_image = compressed_image.convert('RGB')
        
        # Convert to numpy arrays
        original_array = np.array(original_image)
        compressed_array = np.array(compressed_image)
        
        # Calculate PSNR (Peak Signal-to-Noise Ratio)
        psnr = peak_signal_noise_ratio(original_array, compressed_array)
        
        # Calculate SSIM (Structural Similarity Index)
        ssim_result = structural_similarity(
            original_array, 
            compressed_array, 
            channel_axis=2,  # RGB has 3 channels
            data_range=255
        )
        
        return psnr, ssim_result
    
    def _apply_pngquant(
        self, 
        input_path: str, 
        output_path: str, 
        metrics: CompressionMetrics
    ) -> bool:
        """
        Apply PNGQuant compression to an image.
        
        Args:
            input_path: Path to input PNG file
            output_path: Path to output quantized PNG file
            metrics: Metrics object to update
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        # Prepare PNGQuant command
        min_quality = max(0, self.options.quality - 10)
        max_quality = self.options.quality
        quality_param = f"{min_quality}-{max_quality}"
        
        cmd = ["pngquant", "--force", "--output", output_path]
        
        # Add quality parameter
        cmd.extend(["--quality", quality_param])
        
        # Add dithering parameter
        if not self.options.dithering:
            cmd.append("--nofs")  # No Floyd-Steinberg dithering
            
        # Add input file
        cmd.append(input_path)
        
        # Run PNGQuant
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True
            )
            
            metrics.quantization_time = time.time() - start_time
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"PNGQuant compression failed: {e.stderr.decode('utf-8')}")
            return False
            
        except FileNotFoundError:
            logger.error("PNGQuant not installed or not in PATH")
            return False
    
    def _create_plz_file(
        self, 
        pngquant_path: str, 
        output_path: str, 
        metadata: Dict[str, Any], 
        metrics: CompressionMetrics
    ) -> bool:
        """
        Create a PLZ file from a quantized PNG.
        
        Args:
            pngquant_path: Path to the quantized PNG file
            output_path: Path to the output PLZ file
            metadata: Metadata to include in the PLZ header
            metrics: Metrics object to update
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            # Read the quantized PNG
            with open(pngquant_path, 'rb') as f:
                png_data = f.read()
            
            # Apply LZ4 compression
            lz4_start_time = time.time()
            compressed_data = lz4.frame.compress(
                png_data,
                compression_level=LZ4_COMPRESSION_LEVEL,
                block_size=LZ4_BLOCK_SIZE
            )
            metrics.lz4_compression_time = time.time() - lz4_start_time
            
            # Create the PLZ header
            header_json = json.dumps(metadata).encode('utf-8')
            header_size = len(header_json)
            
            # Write the PLZ file
            with open(output_path, 'wb') as f:
                # Magic number
                f.write(PLZ_MAGIC)
                # Header size
                f.write(struct.pack('<I', header_size))
                # Header data
                f.write(header_json)
                # Compressed data
                f.write(compressed_data)
            
            metrics.final_size = os.path.getsize(output_path)
            metrics.lz4_compression_ratio = ((metrics.quantized_size - metrics.final_size) / metrics.quantized_size) * 100
            metrics.total_compression_time = time.time() - start_time
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating PLZ file: {str(e)}")
            return False
    
    def compress(
        self, 
        input_data: bytes, 
        output_path: str, 
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compress an image to PLZ format.
        
        Args:
            input_data: Raw image data as bytes
            output_path: Path to save the PLZ file
            custom_metadata: Additional metadata to include
            
        Returns:
            Dictionary with compression metrics
            
        Raises:
            ValueError: If compression fails
        """
        metrics = CompressionMetrics()
        metrics.start_time = time.time()
        metrics.original_size = len(input_data)
        
        # Start monitoring system resources
        initial_resources = self._get_system_resources()
        
        # Generate temporary file paths
        temp_id = f"{os.path.basename(output_path).split('.')[0]}_{int(time.time() * 1000)}"
        input_path = self._get_temp_filepath(temp_id, "_input.png")
        pngquant_path = self._get_temp_filepath(temp_id, "_quantized.png")
        decompressed_path = self._get_temp_filepath(temp_id, "_decompressed.png")
        
        temp_files = [input_path, pngquant_path, decompressed_path]
        
        try:
            # Save input data as PNG
            try:
                img = Image.open(io.BytesIO(input_data))
                img.save(input_path, format="PNG")
                
                # Store original image for quality comparison
                original_image = img.convert("RGB")
                
            except Exception as e:
                raise ValueError(f"Invalid image data: {str(e)}")
            
            # Apply PNGQuant compression
            if not self._apply_pngquant(input_path, pngquant_path, metrics):
                raise ValueError("PNGQuant compression failed")
            
            # Get quantized size
            metrics.quantized_size = os.path.getsize(pngquant_path)
            metrics.pngquant_compression_ratio = (
                (metrics.original_size - metrics.quantized_size) / metrics.original_size
            ) * 100
            
            # Prepare metadata
            metadata = {
                "compression_quality": self.options.quality,
                "created_timestamp": metrics.start_time,
                "version": VERSION
            }
            
            # Add custom metadata
            if custom_metadata:
                metadata.update(custom_metadata)
            
            # Add options metadata
            metadata["dithering"] = self.options.dithering
            
            # Create PLZ file
            if not self._create_plz_file(pngquant_path, output_path, metadata, metrics):
                raise ValueError("Failed to create PLZ file")
            
            # Calculate total compression metrics
            metrics.total_compression_ratio = (
                (metrics.original_size - metrics.final_size) / metrics.original_size
            ) * 100
            metrics.compression_ratio = metrics.original_size / metrics.final_size if metrics.final_size > 0 else 0
            
            # If metrics collection is enabled, perform decompression test and quality analysis
            if self.options.collect_metrics:
                # Decompress for quality assessment
                decomp_start_time = time.time()
                self.decompress(output_path, decompressed_path)
                metrics.decompression_time = time.time() - decomp_start_time
                
                # Calculate quality metrics
                try:
                    decompressed_img = Image.open(decompressed_path).convert("RGB")
                    metrics.psnr, metrics.ssim = self._calculate_image_metrics(
                        original_image, decompressed_img
                    )
                except Exception as e:
                    logger.warning(f"Error calculating image quality metrics: {str(e)}")
            
            # Final resource measurements
            final_resources = self._get_system_resources()
            metrics.cpu_usage = final_resources["cpu_usage"] - initial_resources["cpu_usage"]
            metrics.peak_memory_usage = final_resources["memory_usage"]
            
            return metrics.to_dict()
            
        except Exception as e:
            logger.error(f"Error in PLZ compression: {str(e)}")
            raise
            
        finally:
            # Clean up temporary files
            self._cleanup_temp_files(temp_files)
    
    def decompress(self, plz_path: str, output_path: str) -> Dict[str, Any]:
        """
        Decompress a PLZ file to a PNG image.
        
        Args:
            plz_path: Path to the PLZ file
            output_path: Path to save the decompressed PNG
            
        Returns:
            Dictionary with metadata and decompression information
            
        Raises:
            ValueError: If decompression fails or file is invalid
        """
        start_time = time.time()
        
        try:
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
            
            decompression_time = time.time() - start_time
            
            # Add decompression information to metadata
            result = metadata.copy()
            result.update({
                "decompression_time": round(decompression_time, 4),
                "decompressed_size": os.path.getsize(output_path),
                "original_plz_size": os.path.getsize(plz_path)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error decompressing PLZ file: {str(e)}")
            raise ValueError(f"Failed to decompress PLZ file: {str(e)}")
    
    def read_metadata(self, plz_path: str) -> Dict[str, Any]:
        """
        Read metadata from a PLZ file without extracting it.
        
        Args:
            plz_path: Path to the PLZ file
            
        Returns:
            Dictionary with file metadata
            
        Raises:
            ValueError: If the file is not a valid PLZ file
        """
        try:
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
                
                # Add file size information
                metadata["file_size"] = os.path.getsize(plz_path)
                
                return metadata
                
        except Exception as e:
            logger.error(f"Error reading PLZ metadata: {str(e)}")
            raise ValueError(f"Failed to read PLZ metadata: {str(e)}")
    
    def compress_file(
        self, 
        input_path: str, 
        output_path: str, 
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compress an image file to PLZ format.
        
        Args:
            input_path: Path to input image file
            output_path: Path to save the PLZ file
            custom_metadata: Additional metadata to include
            
        Returns:
            Dictionary with compression metrics
            
        Raises:
            ValueError: If compression fails
        """
        try:
            with open(input_path, 'rb') as f:
                input_data = f.read()
            
            # Add original filename to metadata
            if custom_metadata is None:
                custom_metadata = {}
            custom_metadata["original_filename"] = os.path.basename(input_path)
            
            return self.compress(input_data, output_path, custom_metadata)
            
        except Exception as e:
            logger.error(f"Error compressing file {input_path}: {str(e)}")
            raise
    
    def batch_compress(
        self, 
        input_paths: List[str], 
        output_dir: str, 
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Compress multiple image files to PLZ format.
        
        Args:
            input_paths: List of paths to input image files
            output_dir: Directory to save the PLZ files
            custom_metadata: Additional metadata to include in all files
            
        Returns:
            List of dictionaries with compression metrics for each file
            
        Raises:
            ValueError: If compression fails for any file
        """
        results = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        for input_path in input_paths:
            try:
                filename = os.path.basename(input_path)
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.plz")
                
                # Add file-specific metadata
                file_metadata = custom_metadata.copy() if custom_metadata else {}
                file_metadata["original_filename"] = filename
                
                result = self.compress_file(input_path, output_path, file_metadata)
                result["input_path"] = input_path
                result["output_path"] = output_path
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {input_path}: {str(e)}")
                results.append({
                    "input_path": input_path,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def compress_base64(
        self, 
        base64_data: str, 
        output_path: str, 
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compress a base64-encoded image to PLZ format.
        
        Args:
            base64_data: Base64-encoded image data
            output_path: Path to save the PLZ file
            custom_metadata: Additional metadata to include
            
        Returns:
            Dictionary with compression metrics
            
        Raises:
            ValueError: If compression fails
        """
        import base64
        
        try:
            # Decode base64 data
            image_data = base64.b64decode(base64_data)
            
            # Add encoding info to metadata
            if custom_metadata is None:
                custom_metadata = {}
            custom_metadata["source_encoding"] = "base64"
            
            return self.compress(image_data, output_path, custom_metadata)
            
        except Exception as e:
            logger.error(f"Error compressing base64 data: {str(e)}")
            raise
    
    def decompress_to_base64(self, plz_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Decompress a PLZ file and return the image as base64-encoded data.
        
        Args:
            plz_path: Path to the PLZ file
            
        Returns:
            Tuple of (base64-encoded image data, metadata)
            
        Raises:
            ValueError: If decompression fails
        """
        import base64
        temp_path = self._get_temp_filepath("decomp", ".png")
        
        try:
            # Decompress to temporary file
            metadata = self.decompress(plz_path, temp_path)
            
            # Read decompressed file and convert to base64
            with open(temp_path, 'rb') as f:
                image_data = f.read()
            
            base64_data = base64.b64encode(image_data).decode('utf-8')
            
            return base64_data, metadata
            
        except Exception as e:
            logger.error(f"Error decompressing to base64: {str(e)}")
            raise
            
        finally:
            # Clean up temporary file
            self._cleanup_temp_files([temp_path])


# Convenient functions for direct use

def compress_image(
    input_data: Union[bytes, str, Image.Image],
    output_path: str,
    quality: int = PNGQUANT_DEFAULT_QUALITY,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compress an image to PLZ format.
    
    Args:
        input_data: Image data as bytes, file path, or PIL Image
        output_path: Path to save the PLZ file
        quality: PNGQuant quality setting (0-100, higher is better quality)
        metadata: Additional metadata to include in the PLZ header
        
    Returns:
        Dictionary with compression metrics
        
    Raises:
        ValueError: If compression fails
    """
    options = CompressionOptions(quality=quality)
    compressor = PLZCompressor(options)
    
    # Handle different input types
    if isinstance(input_data, str):
        # Input is a file path
        return compressor.compress_file(input_data, output_path, metadata)
    elif isinstance(input_data, Image.Image):
        # Input is a PIL Image
        buffer = io.BytesIO()
        input_data.save(buffer, format="PNG")
        return compressor.compress(buffer.getvalue(), output_path, metadata)
    else:
        # Input is bytes
        return compressor.compress(input_data, output_path, metadata)

def decompress_image(plz_path: str, output_path: str) -> Dict[str, Any]:
    """
    Decompress a PLZ file to a PNG image.
    
    Args:
        plz_path: Path to the PLZ file
        output_path: Path to save the decompressed PNG
        
    Returns:
        Dictionary with metadata and decompression information
        
    Raises:
        ValueError: If decompression fails or file is invalid
    """
    compressor = PLZCompressor()
    return compressor.decompress(plz_path, output_path)

def read_plz_metadata(plz_path: str) -> Dict[str, Any]:
    """
    Read metadata from a PLZ file without extracting it.
    
    Args:
        plz_path: Path to the PLZ file
        
    Returns:
        Dictionary with file metadata
        
    Raises:
        ValueError: If the file is not a valid PLZ file
    """
    compressor = PLZCompressor()
    return compressor.read_metadata(plz_path)

def batch_compress_images(
    input_paths: List[str],
    output_dir: str,
    quality: int = PNGQUANT_DEFAULT_QUALITY,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Compress multiple image files to PLZ format.
    
    Args:
        input_paths: List of paths to input image files
        output_dir: Directory to save the PLZ files
        quality: PNGQuant quality setting (0-100, higher is better quality)
        metadata: Additional metadata to include in all files
        
    Returns:
        List of dictionaries with compression metrics for each file
        
    Raises:
        ValueError: If compression fails for any file
    """
    options = CompressionOptions(quality=quality)
    compressor = PLZCompressor(options)
    return compressor.batch_compress(input_paths, output_dir, metadata)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="PLZ/LZQuant image compression tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Compress command
    compress_parser = subparsers.add_parser("compress", help="Compress an image to PLZ format")
    compress_parser.add_argument("input", help="Input image file path")
    compress_parser.add_argument("output", help="Output PLZ file path")
    compress_parser.add_argument(
        "--quality", type=int, default=PNGQUANT_DEFAULT_QUALITY,
        help=f"PNGQuant quality setting (0-100, default: {PNGQUANT_DEFAULT_QUALITY})"
    )
    compress_parser.add_argument(
        "--no-dither", action="store_true",
        help="Disable dithering in PNGQuant"
    )
    
    # Decompress command
    decompress_parser = subparsers.add_parser("decompress", help="Decompress a PLZ file")
    decompress_parser.add_argument("input", help="Input PLZ file path")
    decompress_parser.add_argument("output", help="Output image file path")
    
    # Metadata command
    metadata_parser = subparsers.add_parser("metadata", help="View metadata of a PLZ file")
    metadata_parser.add_argument("input", help="Input PLZ file path")
    
    # Batch compress command
    batch_parser = subparsers.add_parser("batch", help="Compress multiple images")
    batch_parser.add_argument("input_dir", help="Input directory with images")
    batch_parser.add_argument("output_dir", help="Output directory for PLZ files")
    batch_parser.add_argument(
        "--quality", type=int, default=PNGQUANT_DEFAULT_QUALITY,
        help=f"PNGQuant quality setting (0-100, default: {PNGQUANT_DEFAULT_QUALITY})"
    )
    batch_parser.add_argument(
        "--pattern", default="*.png",
        help="File pattern to match (default: *.png)"
    )
    
    args = parser.parse_args()
    
    if args.command == "compress":
        options = CompressionOptions(
            quality=args.quality,
            dithering=not args.no_dither
        )
        compressor = PLZCompressor(options)
        result = compressor.compress_file(args.input, args.output)
        print(json.dumps(result, indent=2))
        
    elif args.command == "decompress":
        compressor = PLZCompressor()
        result = compressor.decompress(args.input, args.output)
        print(json.dumps(result, indent=2))
        
    elif args.command == "metadata":
        compressor = PLZCompressor()
        metadata = compressor.read_metadata(args.input)
        print(json.dumps(metadata, indent=2))
        
    elif args.command == "batch":
        import glob
        
        # Find input files
        pattern = os.path.join(args.input_dir, args.pattern)
        input_files = glob.glob(pattern)
        
        if not input_files:
            print(f"No files found matching pattern: {pattern}")
            exit(1)
        
        print(f"Found {len(input_files)} files matching pattern: {pattern}")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Process files
        options = CompressionOptions(quality=args.quality)
        compressor = PLZCompressor(options)
        results = compressor.batch_compress(input_files, args.output_dir)
        
        # Print summary
        success_count = sum(1 for r in results if "error" not in r)
        print(f"Processed {len(results)} files, {success_count} successful")
        
        # Calculate average compression ratio
        if success_count > 0:
            avg_ratio = sum(r.get("total_compression_ratio", 0) for r in results if "error" not in r) / success_count
            print(f"Average compression ratio: {avg_ratio:.2f}%")
    
    else:
        parser.print_help()