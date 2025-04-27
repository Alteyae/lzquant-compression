"""
Data models for compression API responses and requests.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CompressionResponse(BaseModel):
    """Response model for basic compression results"""
    original_size: int
    pngquant_size: int
    final_size: int
    pngquant_compression_ratio: float
    lz4_compression_ratio: float
    total_compression_ratio: float
    file_id: str
    compression_time: float
    decompression_time: float
    cpu_usage: float
    memory_usage: float
    psnr: float
    ssim: float


class DecompressionResponse(BaseModel):
    """Response model for basic decompression results"""
    file_id: str
    compressed_size: int
    decompressed_size: int
    decompression_time: float
    cpu_usage: float
    memory_usage: float


class DecompressRequest(BaseModel):
    """Request model for batch decompression operations"""
    file_ids: List[str]


class Base64CompressRequest(BaseModel):
    """Request model for Base64 image compression"""
    base64_images: List[str]


# Zstandard Compression Models

class ZstCompressionRequest(BaseModel):
    """Options for Zstandard compression"""
    compression_level: int = Field(10, ge=1, le=22, description="Zstandard compression level (1-22)")
    include_metadata: bool = Field(True, description="Whether to include metadata in compressed file")


class ZstCompressionResponse(BaseModel):
    """Response model for Zstandard compression results"""
    file_id: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    space_savings_percent: float
    compression_time: float
    compression_level: int
    cpu_usage: float
    memory_usage: float
    psnr: Optional[float] = None
    ssim: Optional[float] = None


class ZstDecompressionResponse(BaseModel):
    """Response model for Zstandard decompression results"""
    file_id: str
    compressed_size: int
    decompressed_size: int
    decompression_time: float
    cpu_usage: float
    memory_usage: float
    psnr: Optional[float] = None
    ssim: Optional[float] = None


# Remove the batch-related models since we're simplifying the API