"""
Models for Zstandard compression operations.

Models for image compression and decompression using Zstandard,
including support for batch operations.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class ZstCompressionRequest(BaseModel):
    """Request model for Zstandard compression options"""
    compression_level: int = Field(
        10, ge=1, le=22, 
        description="Zstandard compression level (1-22)"
    )
    include_metadata: bool = Field(
        True, 
        description="Whether to include metadata in compressed file"
    )

class ZstCompressionResponse(BaseModel):
    """Response model for Zstandard compression results"""
    file_id: str = Field(..., description="Unique ID for the compressed file")
    original_size: int = Field(..., description="Size of original file in bytes")
    compressed_size: int = Field(..., description="Size of compressed file in bytes")
    compression_ratio: float = Field(..., description="Compression ratio (original/compressed)")
    space_savings_percent: float = Field(..., description="Percentage of space saved")
    compression_time: float = Field(..., description="Time taken for compression in seconds")
    compression_level: int = Field(..., description="Zstandard compression level used (1-22)")
    cpu_usage: float = Field(..., description="CPU usage during compression")
    memory_usage: float = Field(..., description="Memory usage during compression")
    psnr: Optional[float] = Field(None, description="Peak Signal-to-Noise Ratio (if available)")
    ssim: Optional[float] = Field(None, description="Structural Similarity Index (if available)")

class ZstDecompressionResponse(BaseModel):
    """Response model for Zstandard decompression results"""
    file_id: str = Field(..., description="Unique ID for the file")
    compressed_size: int = Field(..., description="Size of compressed file in bytes")
    decompressed_size: int = Field(..., description="Size of decompressed file in bytes")
    decompression_time: float = Field(..., description="Time taken for decompression in seconds")
    cpu_usage: float = Field(..., description="CPU usage during decompression")
    memory_usage: float = Field(..., description="Memory usage during decompression")
    download_url: str = Field(..., description="URL to download the decompressed file")
    metadata: Dict[str, Any] = Field({}, description="Metadata extracted from the compressed file")


class ZstBatchCompressRequest(BaseModel):
    """Request model for batch compression"""
    compression_level: int = Field(
        10, ge=1, le=22, 
        description="Zstandard compression level (1-22)"
    )
    include_metadata: bool = Field(
        True, 
        description="Whether to include metadata in compressed files"
    )


class ZstBatchCompressResult(BaseModel):
    """Result model for a single file in batch compression"""
    file_id: str = Field(..., description="Unique ID for the compressed file")
    original_filename: str = Field(..., description="Original filename")
    original_size: int = Field(..., description="Size of original file in bytes")
    compressed_size: int = Field(..., description="Size of compressed file in bytes")
    compression_ratio: float = Field(..., description="Compression ratio (original/compressed)")
    space_savings_percent: float = Field(..., description="Percentage of space saved")
    compression_time: float = Field(..., description="Time taken for compression in seconds")
    status: str = Field(..., description="Status of the compression (success/failed)")
    error: Optional[str] = Field(None, description="Error message if compression failed")


class ZstBatchCompressResponse(BaseModel):
    """Response model for batch compression results"""
    results: List[ZstBatchCompressResult] = Field(..., description="Compression results for each file")
    successful_count: int = Field(..., description="Number of successfully compressed files")
    failed_count: int = Field(..., description="Number of failed files")
    total_original_size: int = Field(..., description="Total size of all original files in bytes")
    total_compressed_size: int = Field(..., description="Total size of all compressed files in bytes")
    overall_compression_ratio: float = Field(..., description="Overall compression ratio")
    total_time: float = Field(..., description="Total time taken for all compressions in seconds")
    download_url: Optional[str] = Field(None, description="URL to download all compressed files as ZIP")


class ZstBatchDecompressRequest(BaseModel):
    """Request model for batch decompression"""
    file_ids: List[str] = Field(..., description="List of file IDs to decompress")