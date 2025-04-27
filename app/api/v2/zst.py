"""
Zstandard compression API endpoints.

Provides compression and decompression operations for images with both
single-file and batch processing capabilities.
"""
import os
import sys
import time
import logging
import asyncio
import zipfile
from io import BytesIO
from typing import List, Dict, Any

import zstandard as zstd
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Response, Body
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app import TEMP_DIR
from app.models.zst import (
    ZstCompressionRequest,
    ZstCompressionResponse,
    ZstDecompressionResponse,
    ZstBatchCompressRequest,
    ZstBatchCompressResult,
    ZstBatchCompressResponse,
    ZstBatchDecompressRequest
)
from app.core.zst import (
    DEFAULT_COMPRESSION_LEVEL,
    compress_image_with_zstd,
    decompress_zstd,
    decompress_image
)
from app.utils.file_handling import get_temp_filepath, schedule_cleanup
from app.utils.metrics import get_cpu_mem

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/v2", tags=["Zstandard Compression v2"])


@router.post("/compress/zst", response_model=ZstCompressionResponse)
async def compress_image_zst(
    file: UploadFile = File(...),
    compression_level: int = Form(DEFAULT_COMPRESSION_LEVEL),
    include_metadata: bool = Form(True)
):
    """
    Compress an uploaded image using Zstandard compression.
    
    - **file**: The image file to compress
    - **compression_level**: Zstandard compression level (1-22)
    - **include_metadata**: Whether to include metadata in the compressed file
    
    Returns:
        Compression statistics and file ID for the compressed image
    """
    # Validate compression level
    if compression_level < 1 or compression_level > 22:
        raise HTTPException(status_code=400, detail="Compression level must be between 1 and 22")
    
    # Generate file ID and paths
    file_id = os.path.basename(get_temp_filepath())
    compressed_path = get_temp_filepath(file_id, "_compressed.zst")
    
    # Read file data
    original_data = await file.read()
    logger.info(f"Compressing image {file.filename} ({len(original_data)} bytes) with level {compression_level}")
    
    try:
        # Compress the image
        result = compress_image_with_zstd(
            original_data, 
            compressed_path,
            compression_level=compression_level,
            include_metadata=include_metadata
        )
        
        logger.info(f"Successfully compressed to {result['compressed_size']} bytes " + 
                   f"(ratio: {result['compression_ratio']})")
        
        # Create response
        return ZstCompressionResponse(**result)
    except Exception as e:
        logger.error(f"Compression failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Compression failed: {str(e)}")


@router.get("/decompress/zst/{file_id}", response_model=ZstDecompressionResponse)
async def decompress_image_zst(file_id: str):
    """
    Decompress a previously compressed Zstandard file by its ID.
    
    - **file_id**: The ID of the compressed file
    
    Returns:
        Decompression statistics and download link for the decompressed file
    """
    logger.info(f"Decompression request for file_id: {file_id}")
    
    # Try different possible file patterns
    possible_paths = [
        get_temp_filepath(file_id, "_compressed.zst"),
        os.path.join(TEMP_DIR, f"{file_id}_compressed.zst"),
        os.path.join(TEMP_DIR, f"{file_id}.zst")
    ]
    
    # Find the compressed file
    compressed_path = None
    for path in possible_paths:
        if os.path.exists(path):
            compressed_path = path
            logger.info(f"Found compressed file at: {path}")
            break
    
    if not compressed_path:
        # List files in TEMP_DIR that might match
        matching_files = [f for f in os.listdir(TEMP_DIR) if file_id in f and f.endswith(".zst")]
        if matching_files:
            compressed_path = os.path.join(TEMP_DIR, matching_files[0])
            logger.info(f"Found compressed file: {compressed_path}")
        else:
            logger.error(f"No compressed file found for ID: {file_id}")
            raise HTTPException(
                status_code=404, 
                detail=f"Compressed file not found for ID: {file_id}"
            )
    
    # Start timing
    start_time = time.time()
    
    try:
        # Read compressed file
        with open(compressed_path, "rb") as f:
            compressed_data = f.read()
        
        logger.info(f"Read {len(compressed_data)} bytes from {compressed_path}")
        
        # Decompress the data using direct zstd library
        decompressed_data = decompress_image(compressed_data)
        logger.info(f"Successfully decompressed to {len(decompressed_data)} bytes")
        
        # Try to detect the image type
        try:
            import imghdr
            img_type = imghdr.what(None, h=decompressed_data[:32])
            if img_type:
                ext = f".{img_type}"
            else:
                ext = ".bin"
        except Exception:
            ext = ".bin"
        
        # Create output path
        output_path = get_temp_filepath(file_id, f"_decompressed{ext}")
        
        # Write decompressed data
        with open(output_path, "wb") as out_f:
            out_f.write(decompressed_data)
        logger.info(f"Saved decompressed data to {output_path}")
            
    except Exception as e:
        logger.exception(f"Decompression failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Decompression failed: {str(e)}")
    
    # Calculate metrics
    decompression_time = time.time() - start_time
    decompressed_size = len(decompressed_data)
    compressed_size = len(compressed_data)
    
    # Get system resource usage
    cpu_mem = get_cpu_mem()
    
    # Schedule cleanup of the decompressed file
    schedule_cleanup(output_path, 3600)  # Keep for 1 hour
    
    # Create response
    response = ZstDecompressionResponse(
        file_id=file_id,
        compressed_size=compressed_size,
        decompressed_size=decompressed_size,
        decompression_time=round(decompression_time, 4),
        cpu_usage=cpu_mem["cpu_usage"],
        memory_usage=cpu_mem["memory_usage"],
        download_url=f"/api/v2/download/{file_id}",
        metadata={}  # No metadata in simplified version
    )
    
    return response


@router.get("/download/{file_id}", response_class=FileResponse)
async def download_decompressed_file(file_id: str):
    """
    Download a decompressed file by its ID.
    
    - **file_id**: The ID of the decompressed file
    
    Returns:
        The decompressed file for download
    """
    logger.info(f"Download request for file_id: {file_id}")
    
    # Look for existing decompressed file with various possible extensions
    possible_extensions = [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bin"]
    
    for ext in possible_extensions:
        path = get_temp_filepath(file_id, f"_decompressed{ext}")
        if os.path.exists(path):
            logger.info(f"Found decompressed file at: {path}")
            
            # Determine content type
            content_type = "application/octet-stream"
            if ext == ".png":
                content_type = "image/png"
            elif ext in [".jpg", ".jpeg"]:
                content_type = "image/jpeg"
            elif ext == ".webp":
                content_type = "image/webp"
            elif ext == ".gif":
                content_type = "image/gif"
            
            # Determine filename
            filename = f"{file_id}{ext}"
            
            logger.info(f"Serving file {path} as {filename} ({content_type})")
            return FileResponse(
                path,
                media_type=content_type,
                filename=filename
            )
    
    # If no decompressed file exists, find the compressed file
    compressed_path = None
    for path in [
        get_temp_filepath(file_id, "_compressed.zst"),
        os.path.join(TEMP_DIR, f"{file_id}_compressed.zst"),
        os.path.join(TEMP_DIR, f"{file_id}.zst")
    ]:
        if os.path.exists(path):
            compressed_path = path
            break
    
    if not compressed_path:
        # Check for any files that match the file_id
        matching_files = [
            os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) 
            if file_id in f and f.endswith(".zst")
        ]
        if matching_files:
            compressed_path = matching_files[0]
    
    if compressed_path:
        logger.info(f"Found compressed file at {compressed_path}")
        try:
            # Read and decompress
            with open(compressed_path, "rb") as f:
                compressed_data = f.read()
            
            # Simple decompress using zstd library directly
            decompressed_data = decompress_image(compressed_data)
            
            # Try to detect the image type
            try:
                import imghdr
                img_type = imghdr.what(None, h=decompressed_data[:32])
                if img_type:
                    ext = f".{img_type}"
                    if img_type == "png":
                        content_type = "image/png"
                    elif img_type in ["jpg", "jpeg"]:
                        content_type = "image/jpeg"
                    elif img_type == "webp":
                        content_type = "image/webp"
                    elif img_type == "gif":
                        content_type = "image/gif"
                    else:
                        content_type = f"image/{img_type}"
                else:
                    ext = ".bin"
                    content_type = "application/octet-stream"
            except Exception:
                ext = ".bin"
                content_type = "application/octet-stream"
            
            # Create a temp file
            decompressed_path = get_temp_filepath(file_id, f"_decompressed{ext}")
            with open(decompressed_path, "wb") as f:
                f.write(decompressed_data)
            
            # Schedule cleanup
            schedule_cleanup(decompressed_path, 3600)  # 1 hour
            
            logger.info(f"Successfully decompressed to {decompressed_path}")
            
            # Return the file
            return FileResponse(
                decompressed_path,
                media_type=content_type,
                filename=f"{file_id}{ext}"
            )
                
        except Exception as e:
            logger.error(f"Decompression failed: {str(e)}")
    
    # If all fails, return 404
    logger.error(f"No decompressed file found for ID: {file_id}")
    raise HTTPException(status_code=404, detail="Decompressed file not found")


@router.post("/compress/zst/batch", response_model=ZstBatchCompressResponse, tags=["Batch Operations"])
async def batch_compress_zst(
    files: List[UploadFile] = File(...),
    compression_level: int = Form(DEFAULT_COMPRESSION_LEVEL),
    include_metadata: bool = Form(True)
):
    """
    Compress multiple files using Zstandard compression.
    
    - **files**: List of files to compress
    - **compression_level**: Zstandard compression level (1-22)
    - **include_metadata**: Whether to include metadata in compressed files
    
    Returns:
        Compression statistics for all files and a download link for a ZIP archive
    """
    # Validate compression level
    if compression_level < 1 or compression_level > 22:
        raise HTTPException(status_code=400, detail="Compression level must be between 1 and 22")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Prepare results
    results = []
    successful_count = 0
    failed_count = 0
    total_original_size = 0
    total_compressed_size = 0
    total_time = 0
    
    # Create a ZIP file for all compressed files
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Process each file
        for file in files:
            try:
                # Generate file ID and path
                file_id = os.path.basename(get_temp_filepath())
                compressed_path = get_temp_filepath(file_id, "_compressed.zst")
                
                # Read file data
                original_data = await file.read()
                logger.info(f"Compressing file {file.filename} ({len(original_data)} bytes) with level {compression_level}")
                
                # Start time for this file
                file_start_time = time.time()
                
                # Compress the file
                result = compress_image_with_zstd(
                    original_data,
                    compressed_path,
                    compression_level=compression_level,
                    include_metadata=include_metadata
                )
                
                # Calculate time taken for this file
                file_time = time.time() - file_start_time
                total_time += file_time
                
                # Add compressed file to ZIP
                with open(compressed_path, 'rb') as f:
                    compressed_data = f.read()
                    
                # Add to ZIP with original filename but .zst extension
                original_name = file.filename
                filename_root, _ = os.path.splitext(original_name)
                zip_file.writestr(f"{filename_root}.zst", compressed_data)
                
                # Update totals
                total_original_size += result['original_size']
                total_compressed_size += result['compressed_size']
                successful_count += 1
                
                # Add to results
                results.append(ZstBatchCompressResult(
                    file_id=file_id,
                    original_filename=file.filename,
                    original_size=result['original_size'],
                    compressed_size=result['compressed_size'],
                    compression_ratio=result['compression_ratio'],
                    space_savings_percent=result['space_savings_percent'],
                    compression_time=result['compression_time'],
                    status="success",
                    error=None
                ))
                
            except Exception as e:
                logger.error(f"Failed to compress {file.filename}: {str(e)}")
                failed_count += 1
                results.append(ZstBatchCompressResult(
                    file_id="error",
                    original_filename=file.filename,
                    original_size=0,
                    compressed_size=0,
                    compression_ratio=0,
                    space_savings_percent=0,
                    compression_time=0,
                    status="failed",
                    error=str(e)
                ))
    
    # Calculate overall metrics
    overall_compression_ratio = 0
    if total_compressed_size > 0 and total_original_size > 0:
        overall_compression_ratio = round(total_original_size / total_compressed_size, 2)
    
    # Create a temp file for the ZIP
    zip_id = os.path.basename(get_temp_filepath())
    zip_path = get_temp_filepath(zip_id, "_compressed_files.zip")
    with open(zip_path, 'wb') as f:
        f.write(zip_buffer.getvalue())
    
    # Schedule cleanup
    schedule_cleanup(zip_path, 3600)  # 1 hour
    
    # Return response
    return ZstBatchCompressResponse(
        results=results,
        successful_count=successful_count,
        failed_count=failed_count,
        total_original_size=total_original_size,
        total_compressed_size=total_compressed_size,
        overall_compression_ratio=overall_compression_ratio,
        total_time=round(total_time, 4),
        download_url=f"/api/v2/download/zip/{zip_id}"
    )


@router.post("/decompress/zst/batch", tags=["Batch Operations"])
async def batch_decompress_zst(request: ZstBatchDecompressRequest):
    """
    Decompress multiple files and provide a ZIP archive containing all decompressed files.
    
    - **file_ids**: List of file IDs to decompress
    
    Returns:
        A ZIP archive containing all the decompressed files
    """
    if not request.file_ids:
        raise HTTPException(status_code=400, detail="No file IDs provided")
    
    # Create ZIP buffer
    zip_buffer = BytesIO()
    
    # Track successful and failed files
    successful = []
    failed = []
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_id in request.file_ids:
            logger.info(f"Processing file ID: {file_id}")
            
            # Try to find the compressed file
            compressed_path = None
            for path in [
                get_temp_filepath(file_id, "_compressed.zst"),
                os.path.join(TEMP_DIR, f"{file_id}_compressed.zst"),
                os.path.join(TEMP_DIR, f"{file_id}.zst")
            ]:
                if os.path.exists(path):
                    compressed_path = path
                    break
            
            if not compressed_path:
                # Check for any file matching this ID
                matching_files = [f for f in os.listdir(TEMP_DIR) if file_id in f and f.endswith(".zst")]
                if matching_files:
                    compressed_path = os.path.join(TEMP_DIR, matching_files[0])
            
            if not compressed_path:
                logger.error(f"No compressed file found for ID: {file_id}")
                failed.append({"file_id": file_id, "error": "File not found"})
                continue
            
            try:
                # Read and decompress
                with open(compressed_path, 'rb') as f:
                    compressed_data = f.read()
                
                # Decompress
                decompressed_data = decompress_image(compressed_data)
                
                # Try to detect file type
                try:
                    import imghdr
                    img_type = imghdr.what(None, h=decompressed_data[:32])
                    if img_type:
                        ext = f".{img_type}"
                    else:
                        ext = ".bin"
                except:
                    ext = ".bin"
                
                # Add to ZIP
                zip_file.writestr(f"{file_id}{ext}", decompressed_data)
                successful.append(file_id)
                
            except Exception as e:
                logger.error(f"Failed to decompress {file_id}: {str(e)}")
                failed.append({"file_id": file_id, "error": str(e)})
    
    if not successful:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to decompress any files. Errors: {failed}"
        )
    
    # Reset buffer position
    zip_buffer.seek(0)
    
    # Return the ZIP file
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": "attachment; filename=decompressed_files.zip",
            "X-Successful-Files": str(len(successful)),
            "X-Failed-Files": str(len(failed))
        }
    )


@router.get("/download/zip/{zip_id}")
async def download_zip_file(zip_id: str):
    """
    Download a ZIP archive by its ID.
    
    - **zip_id**: The ID of the ZIP file
    
    Returns:
        The ZIP archive file
    """
    # Look for the ZIP file
    zip_path = get_temp_filepath(zip_id, "_compressed_files.zip")
    
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="ZIP file not found")
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="compressed_files.zip"
    )