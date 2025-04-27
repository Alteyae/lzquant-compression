"""
Data models for hybrid compression API.

This module provides Pydantic models for request/response validation and
documentation across different compression formats.
"""
from app.models.base import (
    BaseMetrics,
    BaseCompressionResponse,
    BaseDecompressionResponse,
    BaseQualityMetrics,
    BaseCompressionRequest,
    BaseDecompressionRequest
)

from app.models.plz import (
    PLZCompressionResponse,
    PLZDecompressionResponse,
    PLZBatchCompressRequest,
    PLZBatchDecompressRequest
)

from app.models.zst import (
    ZstCompressionRequest,
    ZstCompressionResponse,
    ZstDecompressionResponse
)

__all__ = [
    # Base models
    'BaseMetrics',
    'BaseCompressionResponse',
    'BaseDecompressionResponse',
    'BaseQualityMetrics',
    'BaseCompressionRequest',
    'BaseDecompressionRequest',
    
    # PLZ models
    'PLZCompressionResponse',
    'PLZDecompressionResponse',
    'PLZBatchCompressRequest',
    'PLZBatchDecompressRequest',
    
    # ZST models
    'ZstCompressionRequest',
    'ZstCompressionResponse',
    'ZstDecompressionResponse'
]