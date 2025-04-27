"""
Base models for hybrid compression API.
These models define common fields for reuse across different 
compression formats.
"""
from pydantic import BaseModel, Field
from typing import Optional


class BaseMetrics(BaseModel):
    """Base class for performance and resource metrics"""
    cpu_usage: float = Field(..., description="CPU usage during operation (%)")
    memory_usage: float = Field(..., description="Memory usage during operation (%)")


class BaseCompressionResponse(BaseMetrics):
    """Base class for compression operation responses"""
    file_id: str = Field(..., description="Unique identifier for the file")
    original_size: int = Field(..., description="Size of the original file in bytes")
    compressed_size: int = Field(..., description="Size of the compressed file in bytes")
    compression_ratio: float = Field(
        ..., description="Compression ratio (original_size / compressed_size)"
    )
    compression_time: float = Field(
        ..., description="Time taken for compression operation in seconds"
    )
    

class BaseDecompressionResponse(BaseMetrics):
    """Base class for decompression operation responses"""
    file_id: str = Field(..., description="Unique identifier for the file")
    compressed_size: int = Field(..., description="Size of the compressed file in bytes")
    decompressed_size: int = Field(..., description="Size of the decompressed file in bytes")
    decompression_time: float = Field(
        ..., description="Time taken for decompression operation in seconds"
    )


class BaseQualityMetrics(BaseModel):
    """Base class for image quality metrics"""
    psnr: Optional[float] = Field(
        None, description="Peak Signal-to-Noise Ratio between original and decompressed images"
    )
    ssim: Optional[float] = Field(
        None, 
        description="Structural Similarity Index between original and decompressed images"
    )


class BaseCompressionRequest(BaseModel):
    """Base class for compression operation requests"""
    pass


class BaseDecompressionRequest(BaseModel):
    """Base class for decompression operation requests"""
    pass