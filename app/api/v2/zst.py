"""
API v2 — Zstandard compression endpoints.
"""
import os
import time
import logging
import asyncio
import zipfile
from io import BytesIO
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

from app import TEMP_DIR
from app.models.zst import (
    ZstCompressionResponse,
    ZstDecompressionResponse,
    ZstBatchCompressResult,
    ZstBatchCompressResponse,
    ZstBatchDecompressRequest,
)
from app.core.zst import (
    DEFAULT_COMPRESSION_LEVEL,
    compress_image_with_zstd,
    decompress_image,
)
from app.utils.file_handling import get_temp_filepath, schedule_cleanup
from app.utils.metrics import get_cpu_mem

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2", tags=["Zstandard Compression v2"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compressed_path(file_id: str) -> str:
    return get_temp_filepath(file_id, "_compressed.zst")


def _find_compressed(file_id: str) -> Optional[str]:
    """Locate the .zst file for *file_id*, trying a few known patterns."""
    candidates = [
        _compressed_path(file_id),
        os.path.join(TEMP_DIR, f"{file_id}_compressed.zst"),
        os.path.join(TEMP_DIR, f"{file_id}.zst"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # Fallback: scan TEMP_DIR
    for name in os.listdir(TEMP_DIR):
        if file_id in name and name.endswith(".zst"):
            return os.path.join(TEMP_DIR, name)
    return None


def _image_ext(data: bytes) -> tuple[str, str]:
    """Return (extension, mime_type) by probing with Pillow."""
    try:
        fmt = Image.open(BytesIO(data)).format or ""
        fmt = fmt.lower()
        mapping = {
            "png": ("image/png", ".png"),
            "jpeg": ("image/jpeg", ".jpg"),
            "webp": ("image/webp", ".webp"),
            "gif": ("image/gif", ".gif"),
        }
        if fmt in mapping:
            return mapping[fmt]
    except Exception:
        pass
    return "application/octet-stream", ".bin"


# ── Single-file compress / decompress ────────────────────────────────────────

@router.post("/compress/zst", response_model=ZstCompressionResponse)
async def compress_image_zst(
    file: UploadFile = File(...),
    compression_level: int = Form(DEFAULT_COMPRESSION_LEVEL),
    include_metadata: bool = Form(True),
):
    """Compress an uploaded image with Zstandard."""
    if not 1 <= compression_level <= 22:
        raise HTTPException(status_code=400, detail="Compression level must be 1–22")

    file_id = get_temp_filepath().rsplit(os.sep, 1)[-1]
    compressed_path = _compressed_path(file_id)
    original_data = await file.read()

    logger.info("Compressing %s (%d B) at level %d", file.filename, len(original_data), compression_level)
    try:
        result = compress_image_with_zstd(original_data, compressed_path, compression_level, include_metadata)
        logger.info("Compressed to %d B (ratio %.2f)", result["compressed_size"], result["compression_ratio"])
        return ZstCompressionResponse(**result)
    except Exception as exc:
        logger.error("Compression failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Compression failed: {exc}")


@router.get("/decompress/zst/{file_id}", response_model=ZstDecompressionResponse)
async def decompress_image_zst(file_id: str):
    """Decompress a previously compressed Zstandard file."""
    compressed_path = _find_compressed(file_id)
    if not compressed_path:
        raise HTTPException(status_code=404, detail=f"Compressed file not found for ID: {file_id}")

    try:
        t0 = time.monotonic()
        with open(compressed_path, "rb") as f:
            compressed_data = f.read()
        decompressed_data = decompress_image(compressed_data)
        decompression_time = time.monotonic() - t0
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Decompression failed: {exc}")

    mime, ext = _image_ext(decompressed_data)
    output_path = get_temp_filepath(file_id, f"_decompressed{ext}")
    with open(output_path, "wb") as f:
        f.write(decompressed_data)

    schedule_cleanup(output_path, 3600)
    cpu_mem = get_cpu_mem()

    return ZstDecompressionResponse(
        file_id=file_id,
        compressed_size=len(compressed_data),
        decompressed_size=len(decompressed_data),
        decompression_time=round(decompression_time, 4),
        cpu_usage=cpu_mem["cpu_usage"],
        memory_usage=cpu_mem["memory_usage"],
        download_url=f"/api/v2/download/{file_id}",
        metadata={},
    )


@router.get("/download/{file_id}", response_class=FileResponse)
async def download_decompressed_file(file_id: str):
    """Download a decompressed file (decompresses on-demand if needed)."""
    # Check for an already-decompressed file
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bin"):
        path = get_temp_filepath(file_id, f"_decompressed{ext}")
        if os.path.exists(path):
            mime = {
                ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp", ".gif": "image/gif",
            }.get(ext, "application/octet-stream")
            return FileResponse(path, media_type=mime, filename=f"{file_id}{ext}")

    # Fall back to decompressing now
    compressed_path = _find_compressed(file_id)
    if not compressed_path:
        raise HTTPException(status_code=404, detail="Decompressed file not found")

    try:
        with open(compressed_path, "rb") as f:
            compressed_data = f.read()
        decompressed_data = decompress_image(compressed_data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Decompression failed: {exc}")

    mime, ext = _image_ext(decompressed_data)
    output_path = get_temp_filepath(file_id, f"_decompressed{ext}")
    with open(output_path, "wb") as f:
        f.write(decompressed_data)
    schedule_cleanup(output_path, 3600)

    return FileResponse(output_path, media_type=mime, filename=f"{file_id}{ext}")


# ── Batch operations ──────────────────────────────────────────────────────────

@router.post("/compress/zst/batch", response_model=ZstBatchCompressResponse, tags=["Batch Operations"])
async def batch_compress_zst(
    files: List[UploadFile] = File(...),
    compression_level: int = Form(DEFAULT_COMPRESSION_LEVEL),
    include_metadata: bool = Form(True),
):
    """Compress multiple files; returns a ZIP of all .zst outputs."""
    if not 1 <= compression_level <= 22:
        raise HTTPException(status_code=400, detail="Compression level must be 1–22")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results: List[ZstBatchCompressResult] = []
    successful = failed = 0
    total_original = total_compressed = total_time = 0

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for upload in files:
            file_id = get_temp_filepath().rsplit(os.sep, 1)[-1]
            compressed_path = _compressed_path(file_id)
            try:
                data = await upload.read()
                t0 = time.monotonic()
                r = compress_image_with_zstd(data, compressed_path, compression_level, include_metadata)
                elapsed = time.monotonic() - t0
                total_time += elapsed

                with open(compressed_path, "rb") as f:
                    zf.writestr(f"{os.path.splitext(upload.filename)[0]}.zst", f.read())

                total_original += r["original_size"]
                total_compressed += r["compressed_size"]
                successful += 1
                results.append(ZstBatchCompressResult(
                    file_id=file_id,
                    original_filename=upload.filename,
                    original_size=r["original_size"],
                    compressed_size=r["compressed_size"],
                    compression_ratio=r["compression_ratio"],
                    space_savings_percent=r["space_savings_percent"],
                    compression_time=r["compression_time"],
                    status="success",
                    error=None,
                ))
            except Exception as exc:
                logger.error("Failed to compress %s: %s", upload.filename, exc)
                failed += 1
                results.append(ZstBatchCompressResult(
                    file_id="error",
                    original_filename=upload.filename,
                    original_size=0, compressed_size=0,
                    compression_ratio=0, space_savings_percent=0, compression_time=0,
                    status="failed", error=str(exc),
                ))

    zip_id = get_temp_filepath().rsplit(os.sep, 1)[-1]
    zip_path = get_temp_filepath(zip_id, "_compressed_files.zip")
    with open(zip_path, "wb") as f:
        f.write(zip_buffer.getvalue())
    schedule_cleanup(zip_path, 3600)

    overall_ratio = round(total_original / total_compressed, 2) if total_compressed else 0
    return ZstBatchCompressResponse(
        results=results,
        successful_count=successful,
        failed_count=failed,
        total_original_size=total_original,
        total_compressed_size=total_compressed,
        overall_compression_ratio=overall_ratio,
        total_time=round(total_time, 4),
        download_url=f"/api/v2/download/zip/{zip_id}",
    )


@router.post("/decompress/zst/batch", tags=["Batch Operations"])
async def batch_decompress_zst(request: ZstBatchDecompressRequest):
    """Decompress multiple files; returns a ZIP of all decompressed outputs."""
    if not request.file_ids:
        raise HTTPException(status_code=400, detail="No file IDs provided")

    zip_buffer = BytesIO()
    successful: List[str] = []
    failed: list = []

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_id in request.file_ids:
            compressed_path = _find_compressed(file_id)
            if not compressed_path:
                failed.append({"file_id": file_id, "error": "File not found"})
                continue
            try:
                with open(compressed_path, "rb") as f:
                    compressed_data = f.read()
                decompressed_data = decompress_image(compressed_data)
                _, ext = _image_ext(decompressed_data)
                zf.writestr(f"{file_id}{ext}", decompressed_data)
                successful.append(file_id)
            except Exception as exc:
                logger.error("Failed to decompress %s: %s", file_id, exc)
                failed.append({"file_id": file_id, "error": str(exc)})

    if not successful:
        raise HTTPException(status_code=400, detail=f"No files decompressed. Errors: {failed}")

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": "attachment; filename=decompressed_files.zip",
            "X-Successful-Files": str(len(successful)),
            "X-Failed-Files": str(len(failed)),
        },
    )


@router.get("/download/zip/{zip_id}")
async def download_zip_file(zip_id: str):
    """Download a pre-built ZIP archive by its ID."""
    zip_path = get_temp_filepath(zip_id, "_compressed_files.zip")
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="ZIP file not found")
    return FileResponse(zip_path, media_type="application/zip", filename="compressed_files.zip")
