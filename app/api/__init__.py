"""
API module for the hybrid compression application.
"""
import logging
import shutil
import asyncio
import time
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app import TEMP_DIR
from app.api.v1 import router as v1_router
from app.api.v2 import router as v2_router

# Set up logging
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Hybrid Image Compression API",
    description="""
    API for compressing images using multiple algorithms:
    - PLZ format (PNGQuant + LZ4)
    - Zstandard compression
    
    Provides single and batch compression, quality metrics, and performance statistics.
    """,
    version="2.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(v1_router, prefix="/api")
app.include_router(v2_router, prefix="/api")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unexpected errors."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred", "error": str(exc)}
    )


# Health check endpoints
@app.get("/health")
async def health_check():
    """Check if the API is running."""
    return {"status": "healthy", "version": "2.0.0"}

@app.get("/health/detailed")
async def detailed_health_check():
    """
    Provides detailed health information including system metrics and component status.
    """
    import psutil
    import platform
    import tempfile
    import shutil
    import zstandard as zstd
    import lz4.frame
    
    # System info
    system_info = {
        "cpu_usage": psutil.cpu_percent(interval=0.1),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "python_version": platform.python_version(),
        "platform": platform.platform()
    }
    
    # Check compression libraries
    compression_status = {}
    
    # Check zstd
    try:
        test_data = b"test data for compression"
        cctx = zstd.ZstdCompressor()
        compressed = cctx.compress(test_data)
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(compressed)
        compression_status["zstd"] = {
            "status": "ok" if decompressed == test_data else "error",
            "compression_ratio": len(test_data) / len(compressed)
        }
    except Exception as e:
        compression_status["zstd"] = {"status": "error", "message": str(e)}
    
    # Check lz4
    try:
        test_data = b"test data for compression"
        compressed = lz4.frame.compress(test_data)
        decompressed = lz4.frame.decompress(compressed)
        compression_status["lz4"] = {
            "status": "ok" if decompressed == test_data else "error",
            "compression_ratio": len(test_data) / len(compressed)
        }
    except Exception as e:
        compression_status["lz4"] = {"status": "error", "message": str(e)}
    
    # Check pngquant
    try:
        import subprocess
        result = subprocess.run(["pngquant", "--version"], 
                              capture_output=True, text=True, check=True)
        compression_status["pngquant"] = {"status": "ok", "version": result.stdout.strip()}
    except Exception as e:
        compression_status["pngquant"] = {"status": "error", "message": str(e)}
    
    # Check temp directory
    temp_status = {}
    try:
        from app import TEMP_DIR
        
        # Check if temp dir exists
        temp_status["exists"] = os.path.exists(TEMP_DIR)
        
        # Check if writable
        if temp_status["exists"]:
            test_file = os.path.join(TEMP_DIR, "test_write.tmp")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                temp_status["writable"] = True
                os.remove(test_file)
            except Exception as e:
                temp_status["writable"] = False
                temp_status["write_error"] = str(e)
                
            # Check free space
            try:
                temp_status["free_space_mb"] = shutil.disk_usage(TEMP_DIR).free / (1024 * 1024)
            except Exception as e:
                temp_status["space_error"] = str(e)
    except Exception as e:
        temp_status["error"] = str(e)
        
    return {
        "status": "healthy",
        "version": "2.0.0",
        "system": system_info,
        "compression": compression_status,
        "temp_directory": temp_status,
        "timestamp": time.time()
    }


# Cleanup event handler
@app.on_event("shutdown")
async def cleanup():
    """Clean up temporary files when the application shuts down."""
    logger.info(f"Cleaning up temporary directory: {TEMP_DIR}")
    try:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    except Exception as e:
        logger.error(f"Error cleaning up temporary directory: {str(e)}")