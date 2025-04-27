"""
API v1 - PLZ format compression endpoints.
"""
import os
import base64
import asyncio
import io
from typing import List, Dict, Any
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from app import TEMP_DIR
from app.models.plz import (
    PLZCompressionResponse,
    PLZDecompressionResponse,
    PLZBatchCompressRequest,
    PLZBatchDecompressRequest
)
from app.core.plz import (
    PNGQUANT_DEFAULT_QUALITY,
    extract_plz_file,
    compress_image_to_plz
)
from app.utils.file_handling import get_temp_filepath
from app.utils.metrics import get_cpu_mem

router = APIRouter(prefix="/v1", tags=["PLZ Compression v1"])


@router.post("/compress/image", response_model=PLZCompressionResponse)
async def compress_image(
    file: UploadFile = File(...), 
    quality: int = Query(PNGQUANT_DEFAULT_QUALITY, ge=0, le=100, description="PNG compression quality (0-99)")
):
    """
    Compress a PNG image using PNGQuant and LZ4 in sequence.
    
    Returns compression statistics and metrics comparing the original and decompressed images.
    Only PNG and JPEG files are supported - other formats will be converted to PNG.
    """
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only PNG and JPEG files are supported")
    
    # Generate a unique ID and paths
    file_id = str(os.path.basename(file.filename).split('.')[0]) + "_" + os.path.basename(get_temp_filepath())
    final_path = get_temp_filepath(file_id, "_final.plz")
    
    # Read file content
    content = await file.read()
    
    # Add filename to metadata
    metadata = {
        "original_filename": file.filename,
    }
    
    try:
        # Compress the image
        compression_result = compress_image_to_plz(content, final_path, quality, metadata)
        
        # Create response
        return PLZCompressionResponse(
            file_id=file_id,
            original_size=compression_result["original_size"],
            pngquant_size=compression_result["pngquant_size"],
            final_size=compression_result["final_size"],
            compressed_size=compression_result["final_size"],  # Use final_size as compressed_size
            pngquant_compression_ratio=compression_result["pngquant_compression_ratio"],
            lz4_compression_ratio=compression_result["lz4_compression_ratio"],
            total_compression_ratio=compression_result["total_compression_ratio"],
            compression_ratio=compression_result["compression_ratio"],
            space_savings_percent=compression_result["space_savings_percent"],
            compression_time=compression_result["compression_time"],
            decompression_time=compression_result["decompression_time"],
            cpu_usage=compression_result["cpu_usage"],
            memory_usage=compression_result["memory_usage"],
            psnr=compression_result["psnr"],
            ssim=compression_result["ssim"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compression failed: {str(e)}")


@router.get("/decompress/{file_id}", response_class=FileResponse)
async def decompress_image(file_id: str):
    """
    Decompress a previously compressed image and return it as a file.
    """
    compressed_path = os.path.join(TEMP_DIR, f"{file_id}_final.plz")
    output_path = os.path.join(TEMP_DIR, f"{file_id}_decompressed.png")
    
    if not os.path.exists(compressed_path):
        raise HTTPException(status_code=404, detail="Compressed file not found")
    
    try:
        metadata = extract_plz_file(compressed_path, output_path)
        
        # Use original filename if available in metadata
        filename = f"decompressed_{file_id}.png"
        if metadata and "original_filename" in metadata:
            original_name = metadata["original_filename"]
            if original_name.lower().endswith((".jpg", ".jpeg", ".png")):
                filename = f"decompressed_{os.path.splitext(original_name)[0]}.png"
        
        return FileResponse(
            output_path, 
            media_type="image/png", 
            filename=filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decompression failed: {str(e)}")


@router.get("/decompress/{file_id}/info", response_model=PLZDecompressionResponse)
async def decompress_image_info(file_id: str):
    """
    Decompress a previously compressed image and return detailed information about the decompression.
    """
    compressed_path = os.path.join(TEMP_DIR, f"{file_id}_final.plz")
    output_path = os.path.join(TEMP_DIR, f"{file_id}_decompressed.png")
    
    if not os.path.exists(compressed_path):
        raise HTTPException(status_code=404, detail="Compressed file not found")
    
    try:
        # Start timing
        import time
        start_time = time.time()
        
        # Decompress
        metadata = extract_plz_file(compressed_path, output_path)
        decompression_time = time.time() - start_time
        
        # File sizes
        compressed_size = os.path.getsize(compressed_path)
        decompressed_size = os.path.getsize(output_path)
        
        # System resource usage
        cpu_mem = get_cpu_mem()
        
        return PLZDecompressionResponse(
            file_id=file_id,
            compressed_size=compressed_size,
            decompressed_size=decompressed_size,
            decompression_time=decompression_time,
            cpu_usage=cpu_mem["cpu_usage"],
            memory_usage=cpu_mem["memory_usage"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decompression failed: {str(e)}")


@router.get("/info/{file_id}")
async def get_compression_info(file_id: str):
    """
    Get information about a previously compressed file including paths to original, 
    pngquant-compressed, and PLZ-compressed versions.
    """
    paths = {
        "original": os.path.join(TEMP_DIR, f"{file_id}_input.png"),
        "pngquant": os.path.join(TEMP_DIR, f"{file_id}_pngquant.png"),
        "plz": os.path.join(TEMP_DIR, f"{file_id}_final.plz"),
        "decompressed": os.path.join(TEMP_DIR, f"{file_id}_decompressed.png")
    }
    
    results = {}
    for key, path in paths.items():
        if os.path.exists(path):
            results[key] = {
                "exists": True,
                "size": os.path.getsize(path)
            }
            
            # Extract metadata from PLZ file if it exists
            if key == "plz" and os.path.exists(path):
                try:
                    import json
                    import struct
                    with open(path, 'rb') as f:
                        f.read(4)  # Skip magic number
                        header_size = struct.unpack('<I', f.read(4))[0]
                        header_json = f.read(header_size)
                        metadata = json.loads(header_json.decode('utf-8'))
                        results[key]["metadata"] = metadata
                except Exception:
                    results[key]["metadata"] = {"error": "Failed to extract metadata"}
        else:
            results[key] = {
                "exists": False
            }
    
    return JSONResponse(content=results)


@router.get("/download/{file_id}.plz")
async def download_plz_file(file_id: str):
    """
    Download the compressed PLZ file.
    """
    plz_path = os.path.join(TEMP_DIR, f"{file_id}_final.plz")
    
    if not os.path.exists(plz_path):
        raise HTTPException(status_code=404, detail="Compressed PLZ file not found")
    
    try:
        # Extract metadata for original filename
        import json
        import struct
        with open(plz_path, 'rb') as f:
            f.read(4)  # Skip magic number
            header_size = struct.unpack('<I', f.read(4))[0]
            header_json = f.read(header_size)
            metadata = json.loads(header_json.decode('utf-8'))
        
        # Use original filename if available
        filename = f"{file_id}.plz"
        if "original_filename" in metadata:
            base_name = os.path.splitext(metadata["original_filename"])[0]
            filename = f"{base_name}.plz"
        
        return FileResponse(
            plz_path,
            media_type="application/octet-stream",
            filename=filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download PLZ file: {str(e)}")


@router.post("/compress/batch/base64")
async def batch_compress_base64(
    request: PLZBatchCompressRequest,
    quality: int = Query(PNGQUANT_DEFAULT_QUALITY, ge=0, le=100, description="PNG compression quality (0-99)")
):
    """
    Compress multiple Base64 images into PLZ format.
    Accepts JSON input with "base64_images": ["data:image/png;base64,...", "data:image/png;base64,..."]
    """
    results = []

    for base64_str in request.base64_images:
        file_id = os.path.basename(get_temp_filepath())
        final_path = get_temp_filepath(file_id, "_final.plz")

        try:
            # Decode base64 data
            base64_data = base64_str.split(",")[-1]  # Remove 'data:image/png;base64,' if present
            image_bytes = base64.b64decode(base64_data)
            
            # Compress the image
            metadata = {"source": "base64", "quality": quality}
            compression_result = compress_image_to_plz(image_bytes, final_path, quality, metadata)
            
            # Create a properly formatted result with all required fields
            results.append({
                "file_id": file_id,
                "original_size": compression_result["original_size"],
                "pngquant_size": compression_result["pngquant_size"],
                "final_size": compression_result["final_size"],
                "compressed_size": compression_result["final_size"],  # Use final_size as compressed_size
                "pngquant_compression_ratio": compression_result["pngquant_compression_ratio"],
                "lz4_compression_ratio": compression_result["lz4_compression_ratio"],
                "total_compression_ratio": compression_result["total_compression_ratio"],
                "compression_ratio": compression_result["compression_ratio"],
                "space_savings_percent": compression_result["space_savings_percent"],
                "compression_time": compression_result["compression_time"],
                "decompression_time": compression_result["decompression_time"],
                "cpu_usage": compression_result["cpu_usage"],
                "memory_usage": compression_result["memory_usage"],
                "psnr": compression_result["psnr"],
                "ssim": compression_result["ssim"]
            })
            
        except Exception as e:
            results.append({
                "file_id": file_id,
                "status": "failed",
                "error": str(e)
            })

    return results


@router.post("/decompress/batch")
async def batch_decompress_plz(request: PLZBatchDecompressRequest):
    """
    Decompress multiple PLZ files and return information about each decompression.
    The decompressed files can be downloaded individually.
    """
    results = []

    for file_id in request.file_ids:
        plz_path = os.path.join(TEMP_DIR, f"{file_id}_final.plz")
        output_path = os.path.join(TEMP_DIR, f"{file_id}_decompressed.png")

        if not os.path.exists(plz_path):
            results.append({
                "file_id": file_id,
                "status": "failed",
                "error": "File not found"
            })
            continue

        try:
            import time
            start_time = time.time()
            metadata = extract_plz_file(plz_path, output_path)
            decompression_time = time.time() - start_time
            
            compressed_size = os.path.getsize(plz_path)
            decompressed_size = os.path.getsize(output_path)
            
            # System resource usage
            cpu_mem = get_cpu_mem()
            
            # Generate download URL
            download_url = f"/v1/decompress/{file_id}"
            
            # Convert decompressed image to Base64
            with open(output_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            results.append({
                "file_id": file_id,
                "status": "success",
                "compressed_size": compressed_size,
                "decompressed_size": decompressed_size,
                "decompression_time": decompression_time,
                "cpu_usage": cpu_mem["cpu_usage"],
                "memory_usage": cpu_mem["memory_usage"],
                "download_url": download_url,
                "metadata": metadata,
                "image_preview": f"data:image/png;base64,{image_base64}"  
            })
        except Exception as e:
            results.append({
                "file_id": file_id,
                "status": "failed",
                "error": str(e)
            })

    return results


@router.post("/decompress/batch/download")
async def batch_download_decompressed(request: PLZBatchDecompressRequest):
    """
    Download multiple decompressed files as a ZIP archive.
    """
    # Create a temporary zip file
    import zipfile
    zip_file_id = os.path.basename(get_temp_filepath())
    zip_path = os.path.join(TEMP_DIR, f"{zip_file_id}_decompressed_files.zip")
    
    # Track which files were successfully added to the ZIP
    files_added = []
    
    try:
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            for file_id in request.file_ids:
                decompressed_path = os.path.join(TEMP_DIR, f"{file_id}_decompressed.png")
                plz_path = os.path.join(TEMP_DIR, f"{file_id}_final.plz")
                
                if not os.path.exists(decompressed_path):
                    # If the file doesn't exist but the PLZ file does, try to decompress it
                    if os.path.exists(plz_path):
                        try:
                            metadata = extract_plz_file(plz_path, decompressed_path)
                        except Exception:
                            continue
                    else:
                        continue
                
                # Get filename from metadata if possible
                filename = f"{file_id}.png"
                try:
                    import json
                    import struct
                    if os.path.exists(plz_path):
                        with open(plz_path, 'rb') as f:
                            f.read(4)  # Skip magic number
                            header_size = struct.unpack('<I', f.read(4))[0]
                            header_json = f.read(header_size)
                            metadata = json.loads(header_json.decode('utf-8'))
                            if "original_filename" in metadata:
                                original_name = metadata["original_filename"]
                                filename = f"{os.path.splitext(original_name)[0]}.png"
                except Exception:
                    pass
                
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