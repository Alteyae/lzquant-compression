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
import base64
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

# Dictionary to store compressed files for later retrieval
COMPRESSED_FILES = {}
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


@router.post("/compress/batch/base64")
async def batch_compress_base64(
    request: ZstBatchCompressRequest
):
    """
    Compress multiple Base64 images into Zstandard format.
    Accepts JSON input with base64-encoded images and returns detailed compression results.
    
    Example JSON body:
    {
      "base64_images": [
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
        "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."
      ],
      "compression_level": 10, 
      "include_metadata": true
    }
    
    Returns:
        List of compression results for each image with file IDs for later retrieval
    """
    results = []
    
    # Check if base64_images is provided
    if not request.base64_images:
        raise HTTPException(
            status_code=400, 
            detail="base64_images field is required and must contain at least one image"
        )
    
    for i, base64_str in enumerate(request.base64_images):
        # Generate file ID and paths
        file_id = os.path.basename(get_temp_filepath())
        zst_path = get_temp_filepath(file_id, "_compressed.zst")
        
        try:
            # Decode base64 data
            base64_data = base64_str.split(",")[-1]  # Remove 'data:image/png;base64,' if present
            try:
                image_data = base64.b64decode(base64_data)
            except Exception as e:
                logger.error(f"Error decoding base64 image {i}: {str(e)}")
                results.append({
                    "file_id": file_id,
                    "status": "failed",
                    "error": f"Invalid base64 data: {str(e)}"
                })
                continue
            
            # Compress the image
            metadata = {"source": "base64", "compression_level": request.compression_level, "index": i}
            compression_result = compress_image_with_zstd(
                input_data=image_data,
                output_path=zst_path,
                compression_level=request.compression_level,
                include_metadata=True
            )
            
            # Store the compressed file path
            COMPRESSED_FILES[file_id] = {
                "path": zst_path,
                "original_filename": f"image_{i}.png"
            }
            
            # Create response with simplified structure
            # Add basic fields first
            result = {
                "file_id": file_id,
                "status": "success"
            }
            
            # Add all compression metrics
            for key, value in compression_result.items():
                result[key] = value
                
            results.append(result)
            
        except Exception as e:
            logger.exception(f"Error in batch Zstandard compression for image {i}: {str(e)}")
            results.append({
                "file_id": file_id,
                "status": "failed",
                "error": f"Compression failed: {str(e)}"
            })
    
    return results


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


@router.post("/decompress/batch")
async def batch_decompress_zst(request: ZstBatchDecompressRequest):
    """
    Decompress multiple files and return information about each decompression.
    Returns base64-encoded images for preview.
    
    - **file_ids**: List of file IDs to decompress
    
    Returns:
        Detailed decompression results for each file including base64-encoded images
    """
    if not request.file_ids:
        raise HTTPException(status_code=400, detail="No file IDs provided")
    
    # Prepare results
    results = []
    
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
            results.append({
                "file_id": file_id,
                "status": "failed",
                "error": "File not found"
            })
            continue
        
        try:
            # Start timing
            start_time = time.time()
            
            # Read and decompress
            with open(compressed_path, 'rb') as f:
                compressed_data = f.read()
            
            # Decompress
            decompressed_data = decompress_image(compressed_data)
            decompression_time = time.time() - start_time
            
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
            
            # Create a temporary file for the decompressed data
            decompressed_path = get_temp_filepath(file_id, f"_decompressed{ext}")
            with open(decompressed_path, "wb") as f:
                f.write(decompressed_data)
            
            # Schedule cleanup of temp file
            schedule_cleanup(decompressed_path, 3600)  # 1 hour
            
            # Convert to base64
            import base64
            base64_image = base64.b64encode(decompressed_data).decode('utf-8')
            
            # Get system resource usage
            cpu_mem = get_cpu_mem()
            
            # Generate download URL
            download_url = f"/api/v2/download/{file_id}"
            
            # Add to results
            results.append({
                "file_id": file_id,
                "status": "success",
                "compressed_size": len(compressed_data),
                "decompressed_size": len(decompressed_data),
                "decompression_time": round(decompression_time, 4),
                "cpu_usage": cpu_mem["cpu_usage"],
                "memory_usage": cpu_mem["memory_usage"],
                "download_url": download_url,
                "metadata": {},  # No metadata in simplified version
                "image_preview": f"data:image/{img_type if img_type else 'png'};base64,{base64_image}"
            })
            
        except Exception as e:
            logger.error(f"Failed to decompress {file_id}: {str(e)}")
            results.append({
                "file_id": file_id,
                "status": "failed",
                "error": str(e)
            })
    
    return results


@router.post("/decompress/batch/download")
async def batch_download_decompressed(request: ZstBatchDecompressRequest):
    """
    Download multiple decompressed files as a ZIP archive.
    Decompresses Zstandard files and packages them into a single ZIP file.
    
    - **file_ids**: List of file IDs to decompress
    
    Returns:
        A ZIP archive containing all the decompressed files
    """
    if not request.file_ids:
        raise HTTPException(status_code=400, detail="No file IDs provided")
    
    # Create a temporary zip file
    zip_file_id = os.path.basename(get_temp_filepath())
    zip_path = os.path.join(TEMP_DIR, f"{zip_file_id}_decompressed_files.zip")
    
    # Track which files were successfully added to the ZIP
    files_added = []
    
    try:
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            for file_id in request.file_ids:
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
                    continue
                
                # Find existing decompressed file, or decompress if it doesn't exist
                decompressed_path = None
                
                # Look for existing decompressed file
                for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bin"]:
                    path = get_temp_filepath(file_id, f"_decompressed{ext}")
                    if os.path.exists(path):
                        decompressed_path = path
                        break
                
                # If no decompressed file exists, decompress now
                if not decompressed_path:
                    try:
                        # Decompress the file
                        with open(compressed_path, 'rb') as f:
                            compressed_data = f.read()
                        
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
                        
                        # Create temporary file
                        decompressed_path = get_temp_filepath(file_id, f"_decompressed{ext}")
                        with open(decompressed_path, "wb") as f:
                            f.write(decompressed_data)
                        
                        # Schedule cleanup
                        schedule_cleanup(decompressed_path, 3600)  # 1 hour
                        
                    except Exception as e:
                        logger.error(f"Failed to decompress {file_id}: {str(e)}")
                        continue
                
                if decompressed_path and os.path.exists(decompressed_path):
                    # Get filename from metadata if possible
                    filename = f"{file_id}{os.path.splitext(decompressed_path)[1]}"
                    
                    # Try to extract original filename from metadata
                    if file_id in COMPRESSED_FILES and "original_filename" in COMPRESSED_FILES[file_id]:
                        original_name = COMPRESSED_FILES[file_id]["original_filename"]
                        base_name = os.path.splitext(original_name)[0]
                        ext = os.path.splitext(decompressed_path)[1]
                        filename = f"{base_name}{ext}"
                    
                    # Add file to ZIP
                    zip_file.write(decompressed_path, arcname=filename)
                    files_added.append(file_id)
        
        if not files_added:
            raise HTTPException(status_code=404, detail="No valid files found to download")
        
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="decompressed_images.zip"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create ZIP archive: {str(e)}")
    finally:
        # Clean up the ZIP file after it's been sent
        if os.path.exists(zip_path):
            # Schedule deletion (can't delete immediately as it's being served)
            async def delete_file():
                await asyncio.sleep(60)  # Wait 60 seconds before deleting
                if os.path.exists(zip_path):
                    os.remove(zip_path)
            
            asyncio.create_task(delete_file())


@router.post("/decompress/batch/base64")
async def batch_decompress_base64(
    request: ZstBatchDecompressRequest
):
    """
    Decompress multiple files and return their base64-encoded content.
    
    - **file_ids**: List of file IDs to decompress
    
    Returns:
        Detailed decompression results for each file with base64-encoded images
    """
    if not request.file_ids:
        raise HTTPException(status_code=400, detail="No file IDs provided")
    
    # Prepare results
    results = []
    
    for file_id in request.file_ids:
        logger.info(f"Processing file ID for base64 decompression: {file_id}")
        
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
            results.append({
                "file_id": file_id,
                "error": "File not found",
                "success": False
            })
            continue
        
        try:
            # Start timing
            start_time = time.time()
            
            # Read and decompress
            with open(compressed_path, 'rb') as f:
                compressed_data = f.read()
            
            # Decompress
            decompressed_data = decompress_image(compressed_data)
            decompression_time = time.time() - start_time
            
            # Convert to base64
            import base64
            base64_image = base64.b64encode(decompressed_data).decode('utf-8')
            
            # Get system resource usage
            cpu_mem = get_cpu_mem()
            
            # Try to detect file type
            try:
                import imghdr
                img_type = imghdr.what(None, h=decompressed_data[:32])
                if img_type:
                    ext = f".{img_type}"
                    mime_type = f"image/{img_type}"
                else:
                    ext = ".bin"
                    mime_type = "application/octet-stream"
            except:
                ext = ".bin"
                mime_type = "application/octet-stream"
            
            # Add to results
            results.append({
                "file_id": file_id,
                "original_filename": f"{file_id}{ext}",
                "mime_type": mime_type,
                "decompressed_size": len(decompressed_data),
                "compressed_size": len(compressed_data),
                "decompression_time": round(decompression_time, 4),
                "cpu_usage": cpu_mem["cpu_usage"],
                "memory_usage": cpu_mem["memory_usage"],
                "base64_image": base64_image,
                "success": True
            })
            
        except Exception as e:
            logger.error(f"Failed to decompress {file_id} to base64: {str(e)}")
            results.append({
                "file_id": file_id,
                "error": str(e),
                "success": False
            })
    
    return results


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