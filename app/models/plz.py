"""
Models for PLZ format compression operations.

PLZ combines PNGQuant compression with LZ4 for highly efficient
image compression with minimal quality loss.
"""
from typing import List, Optional
from pydantic import BaseModel, Field

from app.models.base import (
    BaseCompressionResponse, 
    BaseDecompressionResponse,
    BaseQualityMetrics
)


class PLZCompressionResponse(BaseCompressionResponse, BaseQualityMetrics):
    """Response model for PLZ compression results"""
    pngquant_size: int = Field(
        ..., description="Size of the intermediate PNGQuant-compressed file in bytes"
    )
    pngquant_compression_ratio: float = Field(
        ..., 
        description="Compression ratio achieved by PNGQuant step (%)"
    )
    lz4_compression_ratio: float = Field(
        ...,
        description="Compression ratio achieved by LZ4 step (%)"
    )
    space_savings_percent: float = Field(
        ...,
        description="Percentage of space saved through compression"
    )


class PLZDecompressionResponse(BaseDecompressionResponse, BaseQualityMetrics):
    """Response model for PLZ decompression results"""
    pass


class PLZBatchCompressRequest(BaseModel):
    """Request model for batch Base64 image compression with PLZ"""
    base64_images: List[str] = Field(
        ..., description="List of Base64-encoded images to compress"
    )
    quality: int = Field(
        80, ge=0, le=100, 
        description="PNGQuant compression quality (0-100)"
    )


class PLZBatchDecompressRequest(BaseModel):
    """Request model for batch PLZ decompression operations"""
    file_ids: List[str] = Field(
        ..., description="List of file IDs to decompress"
    )