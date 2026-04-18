"""
API v1 — PLZ format compression endpoints.
"""
import os
import base64
import asyncio
import time
import zipfile
from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from app import TEMP_DIR
from app.models.plz import (
    PLZCompressionResponse,
    PLZDecompressionResponse,
    PLZBatchCompressRequest,
    PLZBatchDecompressRequest,
)
from app.core.plz import (
    PNGQUANT_DEFAULT_QUALITY,
    extract_plz_file,
    read_plz_metadata,
    compress_image_to_plz,
)
from app.utils.file_handling import get_temp_filepath
from app.utils.metrics import get_cpu_mem

router = APIRouter(prefix="/v1", tags=["PLZ Compression v1"])


def _plz_path(file_id: str) -> str:
    return os.path.join(TEMP_DIR, f"{file_id}_final.plz")


def _decomp_path(file_id: str) -> str:
    return os.path.join(TEMP_DIR, f"{file_id}_decompressed.png")


def _require_plz(file_id: str) -> str:
    path = _plz_path(file_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Compressed file not found")
    return path


# ── Single-file compression ───────────────────────────────────────────────────

@router.post("/compress/image", response_model=PLZCompressionResponse)
async def compress_image(
    file: UploadFile = File(...),
    quality: int = Query(
        PNGQUANT_DEFAULT_QUALITY, ge=0, le=100,
        description="PNGQuant quality (0–100)",
    ),
):
    """Compress a PNG/JPEG image with PNGQuant + LZ4 (PLZ format)."""
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only PNG and JPEG files are supported")

    file_id = get_temp_filepath().rsplit(os.sep, 1)[-1]  # bare UUID
    final_path = _plz_path(file_id)
    content = await file.read()

    try:
        r = compress_image_to_plz(
            content, final_path, quality,
            metadata={"original_filename": file.filename},
        )
        return PLZCompressionResponse(
            file_id=file_id,
            original_size=r["original_size"],
            pngquant_size=r["pngquant_size"],
            final_size=r["final_size"],
            compressed_size=r["final_size"],
            pngquant_compression_ratio=r["pngquant_compression_ratio"],
            lz4_compression_ratio=r["lz4_compression_ratio"],
            total_compression_ratio=r["total_compression_ratio"],
            compression_ratio=r["compression_ratio"],
            space_savings_percent=r["space_savings_percent"],
            compression_time=r["compression_time"],
            decompression_time=r["decompression_time"],
            cpu_usage=r["cpu_usage"],
            memory_usage=r["memory_usage"],
            psnr=r["psnr"],
            ssim=r["ssim"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Compression failed: {exc}")


# ── Decompression ─────────────────────────────────────────────────────────────

@router.get("/decompress/{file_id}", response_class=FileResponse)
async def decompress_image(file_id: str):
    """Decompress a previously compressed PLZ file and return the PNG."""
    plz_path = _require_plz(file_id)
    output_path = _decomp_path(file_id)

    try:
        metadata = extract_plz_file(plz_path, output_path)
        original_name = metadata.get("original_filename", "")
        if original_name.lower().endswith((".jpg", ".jpeg", ".png")):
            filename = f"decompressed_{os.path.splitext(original_name)[0]}.png"
        else:
            filename = f"decompressed_{file_id}.png"
        return FileResponse(output_path, media_type="image/png", filename=filename)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Decompression failed: {exc}")


@router.get("/decompress/{file_id}/info", response_model=PLZDecompressionResponse)
async def decompress_image_info(file_id: str):
    """Decompress and return stats (sizes, timing, resource usage)."""
    plz_path = _require_plz(file_id)
    output_path = _decomp_path(file_id)

    try:
        t0 = time.monotonic()
        extract_plz_file(plz_path, output_path)
        decompression_time = time.monotonic() - t0

        cpu_mem = get_cpu_mem()
        return PLZDecompressionResponse(
            file_id=file_id,
            compressed_size=os.path.getsize(plz_path),
            decompressed_size=os.path.getsize(output_path),
            decompression_time=decompression_time,
            cpu_usage=cpu_mem["cpu_usage"],
            memory_usage=cpu_mem["memory_usage"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Decompression failed: {exc}")


# ── File info & download ──────────────────────────────────────────────────────

@router.get("/info/{file_id}")
async def get_compression_info(file_id: str):
    """List sizes and metadata for all intermediate files for this job."""
    candidates = {
        "original": os.path.join(TEMP_DIR, f"{file_id}_input.png"),
        "pngquant": os.path.join(TEMP_DIR, f"{file_id}_pngquant.png"),
        "plz": _plz_path(file_id),
        "decompressed": _decomp_path(file_id),
    }
    result = {}
    for key, path in candidates.items():
        if os.path.exists(path):
            entry: dict = {"exists": True, "size": os.path.getsize(path)}
            if key == "plz":
                meta, _ = read_plz_metadata(path)
                entry["metadata"] = meta
            result[key] = entry
        else:
            result[key] = {"exists": False}
    return JSONResponse(content=result)


@router.get("/download/{file_id}.plz")
async def download_plz_file(file_id: str):
    """Download the compressed PLZ file."""
    plz_path = _require_plz(file_id)

    try:
        meta, _ = read_plz_metadata(plz_path)
        base = os.path.splitext(meta.get("original_filename", file_id))[0]
        return FileResponse(
            plz_path,
            media_type="application/octet-stream",
            filename=f"{base}.plz",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Download failed: {exc}")


# ── Batch operations ──────────────────────────────────────────────────────────

@router.post("/compress/batch/base64")
async def batch_compress_base64(
    request: PLZBatchCompressRequest,
    quality: int = Query(
        PNGQUANT_DEFAULT_QUALITY, ge=0, le=100,
        description="PNGQuant quality (0–100)",
    ),
):
    """Compress a list of base64-encoded images to PLZ."""
    results = []
    for b64 in request.base64_images:
        file_id = get_temp_filepath().rsplit(os.sep, 1)[-1]
        final_path = _plz_path(file_id)
        try:
            image_bytes = base64.b64decode(b64.split(",")[-1])
            r = compress_image_to_plz(
                image_bytes, final_path, quality,
                metadata={"source": "base64", "quality": quality},
            )
            results.append({"file_id": file_id, **r})
        except Exception as exc:
            results.append({"file_id": file_id, "status": "failed", "error": str(exc)})
    return results


@router.post("/decompress/batch")
async def batch_decompress_plz(request: PLZBatchDecompressRequest):
    """Decompress multiple PLZ files; returns info + base64 previews."""
    results = []
    for file_id in request.file_ids:
        plz_path = _plz_path(file_id)
        output_path = _decomp_path(file_id)

        if not os.path.exists(plz_path):
            results.append({"file_id": file_id, "status": "failed", "error": "File not found"})
            continue

        try:
            t0 = time.monotonic()
            metadata = extract_plz_file(plz_path, output_path)
            decompression_time = time.monotonic() - t0

            cpu_mem = get_cpu_mem()
            with open(output_path, "rb") as img_file:
                preview = base64.b64encode(img_file.read()).decode()

            results.append({
                "file_id": file_id,
                "status": "success",
                "compressed_size": os.path.getsize(plz_path),
                "decompressed_size": os.path.getsize(output_path),
                "decompression_time": round(decompression_time, 4),
                "cpu_usage": cpu_mem["cpu_usage"],
                "memory_usage": cpu_mem["memory_usage"],
                "download_url": f"/v1/decompress/{file_id}",
                "metadata": metadata,
                "image_preview": f"data:image/png;base64,{preview}",
            })
        except Exception as exc:
            results.append({"file_id": file_id, "status": "failed", "error": str(exc)})

    return results


@router.post("/decompress/batch/download")
async def batch_download_decompressed(request: PLZBatchDecompressRequest):
    """Download multiple decompressed PNGs as a single ZIP archive."""
    zip_id = get_temp_filepath().rsplit(os.sep, 1)[-1]
    zip_path = os.path.join(TEMP_DIR, f"{zip_id}_decompressed_files.zip")
    files_added: List[str] = []

    try:
        with zipfile.ZipFile(zip_path, "w") as zf:
            for file_id in request.file_ids:
                plz_path = _plz_path(file_id)
                output_path = _decomp_path(file_id)

                if not os.path.exists(output_path):
                    if not os.path.exists(plz_path):
                        continue
                    try:
                        extract_plz_file(plz_path, output_path)
                    except Exception:
                        continue

                # Derive a human-friendly filename
                meta, _ = read_plz_metadata(plz_path) if os.path.exists(plz_path) else ({}, False)
                original = meta.get("original_filename", "")
                filename = (
                    f"{os.path.splitext(original)[0]}.png"
                    if original
                    else f"{file_id}.png"
                )
                zf.write(output_path, arcname=filename)
                files_added.append(file_id)

        if not files_added:
            raise HTTPException(status_code=404, detail="No valid files found to download")

        # Schedule ZIP cleanup after it's served
        async def _delete_zip():
            await asyncio.sleep(60)
            if os.path.exists(zip_path):
                os.remove(zip_path)

        asyncio.create_task(_delete_zip())
        return FileResponse(zip_path, media_type="application/zip", filename="decompressed_images.zip")

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create ZIP: {exc}")
