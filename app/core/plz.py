"""
PLZ compression format implementation.

Combines PNGQuant color quantization with LZ4 compression.

File format:
  4 bytes  — magic "PLZ\\0"
  4 bytes  — header size (uint32 LE)
  N bytes  — JSON metadata
  rest     — LZ4-compressed PNG data
"""
import os
import json
import struct
import time
import logging
import subprocess
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import lz4.frame
from PIL import Image

from app.utils.metrics import get_cpu_mem, calculate_image_metrics
from app.utils.file_handling import get_temp_filepath, schedule_cleanup

logger = logging.getLogger(__name__)

PLZ_MAGIC = b"PLZ\0"
PNGQUANT_DEFAULT_QUALITY = 80


# ── Low-level format helpers ──────────────────────────────────────────────────

def create_plz_file(
    pngquant_path: str,
    plz_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """Write a PLZ file from a pngquant-output PNG.  Returns final file size."""
    metadata = metadata or {}
    with open(pngquant_path, "rb") as f:
        png_data = f.read()

    compressed_data = lz4.frame.compress(
        png_data, compression_level=lz4.frame.COMPRESSIONLEVEL_MAX
    )
    header_json = json.dumps(metadata).encode()
    with open(plz_path, "wb") as f:
        f.write(PLZ_MAGIC)
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)
        f.write(compressed_data)

    return os.path.getsize(plz_path)


def extract_plz_file(plz_path: str, output_path: str) -> Dict[str, Any]:
    """Decompress a PLZ file to a PNG.  Returns embedded metadata.

    Raises:
        ValueError: if the magic number is wrong.
    """
    with open(plz_path, "rb") as f:
        magic = f.read(4)
        if magic != PLZ_MAGIC:
            raise ValueError("Not a valid PLZ file (wrong magic number)")
        header_size = struct.unpack("<I", f.read(4))[0]
        metadata = json.loads(f.read(header_size).decode())
        decompressed = lz4.frame.decompress(f.read())

    with open(output_path, "wb") as f:
        f.write(decompressed)

    return metadata


def read_plz_metadata(plz_path: str) -> Tuple[Dict[str, Any], bool]:
    """Read metadata from a PLZ file without decompressing the image data.

    Returns:
        (metadata, is_valid)
    """
    try:
        with open(plz_path, "rb") as f:
            if f.read(4) != PLZ_MAGIC:
                return {"error": "Not a valid PLZ file"}, False
            header_size = struct.unpack("<I", f.read(4))[0]
            metadata = json.loads(f.read(header_size).decode())
        return metadata, True
    except Exception as exc:
        logger.error("Failed to read PLZ metadata: %s", exc)
        return {"error": str(exc)}, False


# ── High-level compression pipeline ──────────────────────────────────────────

def compress_image_to_plz(
    input_data: bytes,
    output_path: str,
    quality: int = PNGQUANT_DEFAULT_QUALITY,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compress raw image bytes → PLZ file.

    Pipeline: PNG conversion → pngquant quantisation → LZ4 → PLZ.

    Returns:
        Dict with compression sizes, ratios, timing, and quality metrics.

    Raises:
        ValueError: if pngquant is missing or fails.
    """
    file_id = os.path.splitext(os.path.basename(output_path))[0]
    input_path = get_temp_filepath(file_id, "_input.png")
    pngquant_path = get_temp_filepath(file_id, "_pngquant.png")
    decompressed_path = get_temp_filepath(file_id, "_decompressed.png")

    try:
        # ── 1. Save as PNG ────────────────────────────────────────────────────
        img = Image.open(BytesIO(input_data))
        img.save(input_path, format="PNG")
        original_image = img.convert("RGB")
        original_size = len(input_data)

        # ── 2. pngquant ───────────────────────────────────────────────────────
        t0 = time.monotonic()
        try:
            subprocess.run(
                [
                    "pngquant", "--force", "--output", pngquant_path,
                    "--quality", f"{max(0, quality - 10)}-{quality}",
                    input_path,
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"pngquant failed: {exc.stderr.decode()}") from exc
        except FileNotFoundError as exc:
            raise ValueError(
                "pngquant not installed — required for PLZ compression."
            ) from exc
        pngquant_time = time.monotonic() - t0
        pngquant_size = os.path.getsize(pngquant_path)

        # ── 3. Pack into PLZ ──────────────────────────────────────────────────
        file_metadata = {
            "compression_quality": quality,
            "created_timestamp": time.time(),
            "version": "1.0",
            **(metadata or {}),
        }
        t1 = time.monotonic()
        final_size = create_plz_file(pngquant_path, output_path, file_metadata)
        lz4_time = time.monotonic() - t1

        # ── 4. Decompress to measure quality ─────────────────────────────────
        t2 = time.monotonic()
        extract_plz_file(output_path, decompressed_path)
        decompression_time = time.monotonic() - t2

        psnr = ssim = None
        try:
            decompressed_img = Image.open(decompressed_path).convert("RGB")
            psnr, ssim = calculate_image_metrics(original_image, decompressed_img)
        except Exception as exc:
            logger.warning("Quality metric calculation failed: %s", exc)

        # ── 5. Ratios ─────────────────────────────────────────────────────────
        pngquant_ratio = (1 - pngquant_size / original_size) * 100 if original_size else 0
        lz4_ratio = (1 - final_size / pngquant_size) * 100 if pngquant_size else 0
        total_ratio = (1 - final_size / original_size) * 100 if original_size else 0

        cpu_mem = get_cpu_mem()
        return {
            "original_size": original_size,
            "pngquant_size": pngquant_size,
            "final_size": final_size,
            "compressed_size": final_size,
            "pngquant_compression_ratio": round(pngquant_ratio, 2),
            "lz4_compression_ratio": round(lz4_ratio, 2),
            "total_compression_ratio": round(total_ratio, 2),
            "compression_ratio": round(original_size / final_size, 2) if final_size else 0,
            "space_savings_percent": round(total_ratio, 2),
            "compression_time": round(pngquant_time + lz4_time, 4),
            "decompression_time": round(decompression_time, 4),
            "cpu_usage": cpu_mem["cpu_usage"],
            "memory_usage": cpu_mem["memory_usage"],
            "psnr": psnr,
            "ssim": ssim,
        }

    except Exception:
        raise
    finally:
        for path in (input_path, pngquant_path, decompressed_path):
            if os.path.exists(path):
                schedule_cleanup(path, delay=0)
