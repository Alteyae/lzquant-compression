"""
Zstandard (zstd) compression for images.
"""
import math
import os
import time
import logging
from io import BytesIO
from typing import Any, Dict, Optional, Tuple, Union

import zstandard as zstd
from PIL import Image

from app.utils.metrics import get_cpu_mem, calculate_image_metrics

logger = logging.getLogger(__name__)

DEFAULT_COMPRESSION_LEVEL = 10


def compress_image(data: bytes, compression_level: int = DEFAULT_COMPRESSION_LEVEL) -> bytes:
    """Compress raw bytes with Zstandard."""
    return zstd.ZstdCompressor(level=compression_level).compress(data)


def decompress_image(compressed_data: bytes) -> bytes:
    """Decompress Zstandard-compressed bytes."""
    return zstd.ZstdDecompressor().decompress(compressed_data)


def compress_image_with_zstd(
    input_data: Union[bytes, str],
    output_path: str,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    include_metadata: bool = True,  # reserved — kept for API compatibility
) -> Dict[str, Any]:
    """Compress an image and write the result to *output_path*.

    Returns a metrics dict compatible with ZstCompressionResponse.
    """
    if isinstance(input_data, str):
        with open(input_data, "rb") as f:
            original_data = f.read()
    else:
        original_data = input_data

    original_size = len(original_data)
    file_id = os.path.basename(output_path).split("_")[0]

    t0 = time.monotonic()
    compressed_data = compress_image(original_data, compression_level)
    compression_time = time.monotonic() - t0

    with open(output_path, "wb") as f:
        f.write(compressed_data)

    compressed_size = len(compressed_data)
    compression_ratio = round(original_size / compressed_size, 2) if compressed_size else 0
    space_savings = round((1 - compressed_size / original_size) * 100, 2) if original_size else 0

    cpu_mem = get_cpu_mem()

    # Quality metrics (PSNR / SSIM)
    psnr = ssim = None
    try:
        original_img = Image.open(BytesIO(original_data)).convert("RGB")
        decompressed_img = Image.open(BytesIO(decompress_image(compressed_data))).convert("RGB")
        psnr, ssim = calculate_image_metrics(original_img, decompressed_img)
    except Exception as exc:
        logger.warning("Could not calculate image quality metrics: %s", exc)

    def _clean(v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        return None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v

    return {
        "file_id": file_id,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": _clean(compression_ratio),
        "space_savings_percent": _clean(space_savings),
        "compression_time": _clean(round(compression_time, 4)),
        "compression_level": compression_level,
        "cpu_usage": _clean(cpu_mem["cpu_usage"]),
        "memory_usage": _clean(cpu_mem["memory_usage"]),
        "psnr": _clean(psnr),
        "ssim": _clean(ssim),
    }


def decompress_zstd(compressed_data: bytes) -> Tuple[bytes, Dict]:
    """Decompress and return (data, empty_metadata).  Kept for API compatibility."""
    return decompress_image(compressed_data), {}
