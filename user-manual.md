# Hybrid Image Compression System - User Manual

## Table of Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation and Setup](#installation-and-setup)
- [Quick Start Guide](#quick-start-guide)
- [API Reference](#api-reference)
- [Compression Algorithms](#compression-algorithms)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Overview

The Hybrid Image Compression System is a FastAPI-based service that provides advanced image compression capabilities using two distinct compression algorithms:

1. **PLZ Format** - A custom format combining PNGQuant color quantization with LZ4 compression
2. **Zstandard (Zst)** - High-performance compression using the Zstandard algorithm

The system offers both single image and batch processing capabilities with detailed compression metrics and quality assessment.

### Key Features
- Multi-algorithm compression (PLZ and Zstandard)
- Single and batch image processing
- Base64 input/output support
- Quality metrics (PSNR, SSIM)
- Performance monitoring
- RESTful API with versioning
- Docker containerization
- Health monitoring endpoints

## System Requirements

### Software Requirements
- **Python**: 3.11 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 2GB RAM (4GB+ recommended for batch processing)
- **Storage**: Sufficient space for temporary files (varies by workload)

### External Dependencies
- **pngquant**: Required for PLZ compression
  - Ubuntu/Debian: `sudo apt-get install pngquant`
  - macOS (Homebrew): `brew install pngquant`
  - Windows: Download from [pngquant.org](https://pngquant.org/)

### Docker Requirements (Optional)
- Docker Engine 20.x or higher
- Docker Compose 2.x or higher

## Installation and Setup

### Method 1: Local Python Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd hybrid-compression
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install pngquant**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install pngquant
   
   # macOS with Homebrew
   brew install pngquant
   
   # Verify installation
   pngquant --version
   ```

5. **Run the application**:
   ```bash
   python main.py
   ```

### Method 2: Docker Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd hybrid-compression
   ```

2. **Build and run with Docker Compose**:
   ```bash
   # Development mode
   docker-compose up --build
   
   # Production mode (if available)
   docker-compose -f docker-compose.prod.yml up --build -d
   ```

3. **Access the application**:
   - API: http://localhost:8000
   - Health check: http://localhost:8000/health
   - API documentation: http://localhost:8000/docs

## Quick Start Guide

### 1. Start the Server
```bash
python main.py
```
The server will start on http://localhost:8000 by default.

### 2. Test the Health Endpoint
```bash
curl http://localhost:8000/health
```

### 3. Compress Your First Image

#### Using curl (PLZ format):
```bash
curl -X POST "http://localhost:8000/api/v1/compress/image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-image.png" \
  -F "quality=80"
```

#### Using curl (Zstandard format):
```bash
curl -X POST "http://localhost:8000/api/v2/compress/zst" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-image.png" \
  -F "compression_level=10"
```

### 4. Download Compressed File
Use the `file_id` from the compression response:
```bash
# Download PLZ file
curl -O "http://localhost:8000/api/v1/download/{file_id}.plz"

# Download decompressed image
curl -O "http://localhost:8000/api/v2/download/{file_id}"
```

## API Reference

### Base URLs
- **API v1 (PLZ)**: `http://localhost:8000/api/v1/`
- **API v2 (Zstandard)**: `http://localhost:8000/api/v2/`

### Authentication
No authentication required for current version.

### Common Response Format
All compression endpoints return detailed metrics:
```json
{
  "file_id": "unique_identifier",
  "original_size": 1024000,
  "compressed_size": 256000,
  "compression_ratio": 4.0,
  "space_savings_percent": 75.0,
  "compression_time": 1.234,
  "psnr": 45.67,
  "ssim": 0.95,
  "cpu_usage": 15.2,
  "memory_usage": 512.0
}
```

### API v1 - PLZ Format Endpoints

#### Single Image Compression
```http
POST /api/v1/compress/image
Content-Type: multipart/form-data

Parameters:
- file: Image file (PNG, JPEG)
- quality: Compression quality (0-100, default: 80)
```

#### Batch Compression (Base64)
```http
POST /api/v1/compress/batch/base64
Content-Type: application/json

Body:
{
  "base64_images": [
    "data:image/png;base64,iVBORw0KGgo...",
    "data:image/jpeg;base64,/9j/4AAQSkZJ..."
  ]
}
```

#### Decompression
```http
GET /api/v1/decompress/{file_id}
```

#### Batch Decompression
```http
POST /api/v1/decompress/batch
Content-Type: application/json

Body:
{
  "file_ids": ["file1", "file2", "file3"]
}
```

#### Download Operations
```http
GET /api/v1/download/{file_id}.plz          # Download PLZ file
POST /api/v1/decompress/batch/download       # Download ZIP of decompressed files
```

### API v2 - Zstandard Endpoints

#### Single Image Compression
```http
POST /api/v2/compress/zst
Content-Type: multipart/form-data

Parameters:
- file: Image file
- compression_level: Zstd level (1-22, default: 10)
- include_metadata: Include metadata (boolean, default: true)
```

#### Batch Compression (Base64)
```http
POST /api/v2/compress/batch/base64
Content-Type: application/json

Body:
{
  "base64_images": ["base64_string1", "base64_string2"],
  "compression_level": 10,
  "include_metadata": true
}
```

#### Decompression
```http
GET /api/v2/decompress/zst/{file_id}
```

#### Download Operations
```http
GET /api/v2/download/{file_id}              # Download decompressed file
POST /api/v2/decompress/batch/download      # Download ZIP of decompressed files
```

### Health and Monitoring

#### Basic Health Check
```http
GET /health
```
Returns: `{"status": "healthy"}`

#### Detailed Health Check
```http
GET /health/detailed
```
Returns comprehensive system status including:
- System resources (CPU, memory, disk)
- Component availability (pngquant, libraries)
- Temporary directory status
- Recent performance metrics

## Compression Algorithms

### PLZ Format (LZQuant Algorithm)

PLZ combines two compression stages:

1. **Color Quantization** using PNGQuant:
   - Reduces color palette while maintaining visual quality
   - Uses advanced dithering algorithms
   - Configurable quality parameter (0-100)

2. **LZ4 Compression** for size reduction:
   - Fast dictionary-based compression
   - Optimized for speed and decent compression ratios
   - Maintains quick decompression

**File Format Structure**:
```
[4 bytes] Magic number: "PLZ\0"
[4 bytes] Header size (uint32, little-endian)
[N bytes] JSON metadata header
[remaining] LZ4 compressed PNG data
```

**Best Use Cases**:
- Web images requiring fast decompression
- When moderate compression with high speed is needed
- Images with many colors that benefit from quantization

### Zstandard (Zst) Format

Zstandard provides:
- Superior compression ratios compared to traditional algorithms
- Fast compression and decompression
- Configurable compression levels (1-22)
- Built-in integrity checking

**Best Use Cases**:
- Archival storage where compression ratio is critical
- Batch processing of large image datasets
- When maximum compression is more important than speed

### Algorithm Comparison

| Aspect | PLZ (LZQuant) | Zstandard |
|--------|---------------|-----------|
| Compression Ratio | Good | Excellent |
| Compression Speed | Fast | Moderate |
| Decompression Speed | Very Fast | Fast |
| Image Quality | Configurable loss | Lossless |
| Best For | Web/real-time | Archival/batch |

## Usage Examples

### Example 1: Single Image Compression with Quality Assessment

```python
import requests
import json

# Compress image with PLZ
with open('image.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/compress/image',
        files={'file': f},
        data={'quality': 80}
    )

result = response.json()
print(f"Compression ratio: {result['compression_ratio']}x")
print(f"Space savings: {result['space_savings_percent']}%")
print(f"PSNR: {result['psnr']} dB")
print(f"SSIM: {result['ssim']}")

# Download compressed file
file_id = result['file_id']
compressed_response = requests.get(f'http://localhost:8000/api/v1/download/{file_id}.plz')
with open(f'{file_id}.plz', 'wb') as f:
    f.write(compressed_response.content)
```

### Example 2: Batch Processing with Base64

```python
import base64
import requests

# Prepare base64 images
def image_to_base64(file_path):
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

images = [
    f"data:image/png;base64,{image_to_base64('image1.png')}",
    f"data:image/png;base64,{image_to_base64('image2.png')}",
    f"data:image/png;base64,{image_to_base64('image3.png')}"
]

# Batch compress with Zstandard
response = requests.post(
    'http://localhost:8000/api/v2/compress/batch/base64',
    json={
        'base64_images': images,
        'compression_level': 15
    }
)

results = response.json()
for result in results:
    if result.get('status') == 'success':
        print(f"File {result['file_id']}: {result['compression_ratio']}x compression")
    else:
        print(f"Failed: {result.get('error')}")
```

### Example 3: Batch Decompression and Download

```python
import requests

# Get file IDs from previous compression
file_ids = ['file1_id', 'file2_id', 'file3_id']

# Batch decompress and download as ZIP
response = requests.post(
    'http://localhost:8000/api/v2/decompress/batch/download',
    json={'file_ids': file_ids}
)

# Save the ZIP file
with open('decompressed_images.zip', 'wb') as f:
    f.write(response.content)

print("Batch decompression complete. Files saved to decompressed_images.zip")
```

### Example 4: Performance Monitoring

```python
import requests
import time

def monitor_compression_performance():
    # Get system health
    health = requests.get('http://localhost:8000/health/detailed').json()
    print(f"System Status: {health['status']}")
    print(f"CPU Usage: {health.get('cpu_usage', 'N/A')}%")
    print(f"Memory Usage: {health.get('memory_usage', 'N/A')} MB")
    
    # Compress test image and measure performance
    start_time = time.time()
    
    with open('test_image.png', 'rb') as f:
        response = requests.post(
            'http://localhost:8000/api/v1/compress/image',
            files={'file': f}
        )
    
    end_time = time.time()
    result = response.json()
    
    print(f"\nCompression Performance:")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Compression time: {result['compression_time']:.2f} seconds")
    print(f"CPU usage during compression: {result['cpu_usage']}%")
    print(f"Memory usage: {result['memory_usage']} MB")

monitor_compression_performance()
```

## Configuration

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `PORT` | Server port | 8000 | `PORT=3000` |
| `WORKERS` | Uvicorn worker processes | 1 | `WORKERS=4` |
| `LOG_LEVEL` | Logging verbosity | INFO | `LOG_LEVEL=DEBUG` |
| `DEBUG` | Enable debug mode | false | `DEBUG=true` |
| `TEMP_DIR` | Temporary files directory | auto-detected | `TEMP_DIR=/tmp/hybrid-compression` |

### Setting Configuration

#### Using Environment Variables:
```bash
export PORT=3000
export WORKERS=4
export LOG_LEVEL=DEBUG
python main.py
```

#### Using Docker:
```bash
# docker-compose.yml
environment:
  - PORT=3000
  - WORKERS=4
  - LOG_LEVEL=DEBUG
```

#### Inline with Command:
```bash
PORT=3000 WORKERS=4 LOG_LEVEL=DEBUG python main.py
```

### Compression Parameters

#### PLZ Format:
- **Quality Range**: 0-100
  - 0-30: Maximum compression, lower quality
  - 30-60: Balanced compression and quality
  - 60-90: High quality, moderate compression
  - 90-100: Minimal compression, highest quality

#### Zstandard:
- **Compression Levels**: 1-22
  - 1-3: Fast compression, lower ratios
  - 4-9: Balanced speed and compression
  - 10-15: Better compression, slower
  - 16-22: Maximum compression, slowest

## Performance Tuning

### CPU Optimization

1. **Worker Configuration**:
   ```bash
   # Set workers to CPU cores
   WORKERS=$(nproc) python main.py
   
   # For mixed workloads, use 2x cores
   WORKERS=$(($(nproc) * 2)) python main.py
   ```

2. **Compression Level Tuning**:
   ```python
   # For real-time applications
   {
     "compression_level": 3,  # Fast Zstd
     "quality": 70           # Moderate PLZ quality
   }
   
   # For archival/batch processing
   {
     "compression_level": 19, # High Zstd compression
     "quality": 60           # Balanced PLZ quality
   }
   ```

### Memory Optimization

1. **Batch Size Limits**:
   ```python
   # Process large batches in chunks
   def process_large_batch(images, chunk_size=20):
       for i in range(0, len(images), chunk_size):
           chunk = images[i:i+chunk_size]
           # Process chunk
   ```

2. **Temporary File Management**:
   - Files are automatically cleaned up after processing
   - Customize cleanup timing if needed
   - Monitor disk space for large workloads

### Network Optimization

1. **Request Batching**:
   ```python
   # Batch multiple images in single request
   response = requests.post(
       'http://localhost:8000/api/v1/compress/batch/base64',
       json={'base64_images': batch_of_images}
   )
   ```

2. **Streaming Large Files**:
   ```python
   # For very large files, consider streaming
   with open('large_image.png', 'rb') as f:
       response = requests.post(
           'http://localhost:8000/api/v1/compress/image',
           files={'file': f},
           stream=True
       )
   ```

## Troubleshooting

### Common Issues and Solutions

#### 1. "pngquant not found" Error

**Problem**: PLZ compression fails with pngquant not found.

**Solution**:
```bash
# Install pngquant
sudo apt-get install pngquant  # Ubuntu/Debian
brew install pngquant          # macOS

# Verify installation
pngquant --version

# Check PATH
which pngquant
```

#### 2. Memory Issues with Large Batches

**Problem**: Server crashes or becomes unresponsive with large batch processing.

**Solutions**:
- Reduce batch size (< 50 images per request)
- Increase system memory
- Reduce worker count
- Use streaming for very large files

```bash
# Reduce workers for memory-constrained systems
WORKERS=1 python main.py
```

#### 3. Slow Performance

**Problem**: Compression takes too long.

**Solutions**:
1. **Optimize compression parameters**:
   ```python
   # Faster PLZ compression
   {"quality": 50}
   
   # Faster Zstandard compression
   {"compression_level": 3}
   ```

2. **Increase worker processes**:
   ```bash
   WORKERS=4 python main.py
   ```

3. **Use appropriate algorithm**:
   - PLZ for speed-critical applications
   - Zstandard for maximum compression

#### 4. File Not Found Errors

**Problem**: Cannot find compressed files for decompression.

**Causes & Solutions**:
- **Temporary file cleanup**: Files may have been cleaned up
  ```python
  # Check file immediately after compression
  # Files are cleaned up after inactivity
  ```
- **Incorrect file_id**: Verify the file_id from compression response
- **Server restart**: Temporary files are lost on restart

#### 5. API Connection Issues

**Problem**: Cannot connect to API endpoints.

**Diagnosis**:
```bash
# Check if server is running
curl http://localhost:8000/health

# Check server logs
python main.py  # Look for startup messages

# Test with verbose output
curl -v http://localhost:8000/health
```

#### 6. Docker Issues

**Problem**: Docker container won't start or crashes.

**Solutions**:
```bash
# Check Docker logs
docker-compose logs

# Rebuild containers
docker-compose down
docker-compose up --build

# Check resource allocation
docker stats
```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
DEBUG=true LOG_LEVEL=DEBUG python main.py
```

This provides:
- Detailed API request/response logging
- Compression algorithm step-by-step output
- System resource monitoring
- Temporary file management details

### Health Check Diagnostics

Use the detailed health endpoint for system diagnostics:

```bash
curl http://localhost:8000/health/detailed
```

Returns:
- System resource usage
- Component availability
- Recent operation statistics
- Temporary directory status

### Performance Monitoring

Monitor system performance during operation:

```python
import psutil
import time

def monitor_system():
    while True:
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        print(f"CPU: {cpu}%, Memory: {memory.percent}%")
        time.sleep(5)
```

## Advanced Usage

### Custom Integration

#### Integrating with Web Applications

```javascript
// JavaScript example for web integration
async function compressImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('quality', '80');
    
    const response = await fetch('/api/v1/compress/image', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}

// Handle file upload
document.getElementById('imageInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        const result = await compressImage(file);
        console.log(`Compressed: ${result.compression_ratio}x smaller`);
    }
});
```

#### Python Library Integration

```python
class CompressionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def compress_image(self, image_path, algorithm='plz', **kwargs):
        if algorithm == 'plz':
            return self._compress_plz(image_path, **kwargs)
        elif algorithm == 'zst':
            return self._compress_zst(image_path, **kwargs)
            
    def _compress_plz(self, image_path, quality=80):
        with open(image_path, 'rb') as f:
            response = requests.post(
                f'{self.base_url}/api/v1/compress/image',
                files={'file': f},
                data={'quality': quality}
            )
        return response.json()
    
    def _compress_zst(self, image_path, compression_level=10):
        with open(image_path, 'rb') as f:
            response = requests.post(
                f'{self.base_url}/api/v2/compress/zst',
                files={'file': f},
                data={'compression_level': compression_level}
            )
        return response.json()

# Usage
client = CompressionClient()
result = client.compress_image('image.png', algorithm='plz', quality=85)
```

### Batch Processing Workflows

#### Processing Large Datasets

```python
import os
import concurrent.futures
from pathlib import Path

class BatchProcessor:
    def __init__(self, api_url="http://localhost:8000", max_workers=4):
        self.api_url = api_url
        self.max_workers = max_workers
    
    def process_directory(self, input_dir, output_dir, algorithm='zst'):
        """Process all images in a directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get all image files
        image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
        
        # Process in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for image_file in image_files:
                future = executor.submit(self._process_single_image, image_file, output_path, algorithm)
                futures.append((future, image_file.name))
            
            # Collect results
            results = []
            for future, filename in futures:
                try:
                    result = future.result(timeout=60)
                    result['filename'] = filename
                    results.append(result)
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")
        
        return results
    
    def _process_single_image(self, image_path, output_dir, algorithm):
        """Process a single image"""
        with open(image_path, 'rb') as f:
            if algorithm == 'plz':
                response = requests.post(
                    f'{self.api_url}/api/v1/compress/image',
                    files={'file': f}
                )
            else:  # zst
                response = requests.post(
                    f'{self.api_url}/api/v2/compress/zst',
                    files={'file': f}
                )
        
        result = response.json()
        
        # Download compressed file
        file_id = result['file_id']
        if algorithm == 'plz':
            download_url = f'{self.api_url}/api/v1/download/{file_id}.plz'
            output_file = output_dir / f"{image_path.stem}.plz"
        else:
            download_url = f'{self.api_url}/api/v2/download/{file_id}'
            output_file = output_dir / f"{image_path.stem}.zst"
        
        download_response = requests.get(download_url)
        with open(output_file, 'wb') as f:
            f.write(download_response.content)
        
        return result

# Usage
processor = BatchProcessor()
results = processor.process_directory('/path/to/images', '/path/to/output', algorithm='zst')

# Print summary
total_original = sum(r['original_size'] for r in results)
total_compressed = sum(r['compressed_size'] for r in results)
overall_ratio = total_original / total_compressed
print(f"Processed {len(results)} images")
print(f"Overall compression ratio: {overall_ratio:.2f}x")
print(f"Total space saved: {(total_original - total_compressed) / 1024 / 1024:.1f} MB")
```

### API Client Libraries

#### Python Client with Error Handling

```python
import requests
import time
import logging
from typing import Optional, Dict, List

class HybridCompressionClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
    
    def health_check(self) -> Dict:
        """Check API health"""
        try:
            response = self.session.get(f'{self.base_url}/health', timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Health check failed: {e}")
            raise
    
    def compress_plz(self, image_path: str, quality: int = 80) -> Dict:
        """Compress image using PLZ format"""
        try:
            with open(image_path, 'rb') as f:
                response = self.session.post(
                    f'{self.base_url}/api/v1/compress/image',
                    files={'file': f},
                    data={'quality': quality},
                    timeout=self.timeout
                )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"PLZ compression failed: {e}")
            raise
    
    def compress_zst(self, image_path: str, compression_level: int = 10) -> Dict:
        """Compress image using Zstandard"""
        try:
            with open(image_path, 'rb') as f:
                response = self.session.post(
                    f'{self.base_url}/api/v2/compress/zst',
                    files={'file': f},
                    data={'compression_level': compression_level},
                    timeout=self.timeout
                )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Zstandard compression failed: {e}")
            raise
    
    def batch_compress_base64(self, base64_images: List[str], 
                             algorithm: str = 'zst', **kwargs) -> List[Dict]:
        """Batch compress base64 images"""
        if algorithm == 'plz':
            url = f'{self.base_url}/api/v1/compress/batch/base64'
            data = {'base64_images': base64_images}
        elif algorithm == 'zst':
            url = f'{self.base_url}/api/v2/compress/batch/base64'
            data = {
                'base64_images': base64_images,
                'compression_level': kwargs.get('compression_level', 10),
                'include_metadata': kwargs.get('include_metadata', True)
            }
        else:
            raise ValueError("Algorithm must be 'plz' or 'zst'")
        
        try:
            response = self.session.post(url, json=data, timeout=self.timeout * 2)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Batch compression failed: {e}")
            raise
    
    def download_file(self, file_id: str, algorithm: str = 'zst', output_path: str = None):
        """Download compressed or decompressed file"""
        if algorithm == 'plz':
            url = f'{self.base_url}/api/v1/download/{file_id}.plz'
            default_filename = f'{file_id}.plz'
        elif algorithm == 'zst':
            url = f'{self.base_url}/api/v2/download/{file_id}'
            default_filename = f'{file_id}.bin'
        else:
            raise ValueError("Algorithm must be 'plz' or 'zst'")
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            output_path = output_path or default_filename
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return output_path
        except requests.RequestException as e:
            self.logger.error(f"Download failed: {e}")
            raise

# Usage with error handling
client = HybridCompressionClient("http://localhost:8000")

try:
    # Check health
    health = client.health_check()
    print(f"API Status: {health['status']}")
    
    # Compress image
    result = client.compress_zst('image.png', compression_level=15)
    print(f"Compression: {result['compression_ratio']}x")
    
    # Download result
    output_file = client.download_file(result['file_id'], 'zst', 'compressed_image.zst')
    print(f"Downloaded: {output_file}")
    
except Exception as e:
    print(f"Operation failed: {e}")
```

This comprehensive user manual covers all aspects of the Hybrid Image Compression System, from basic setup to advanced integration scenarios. Users can reference specific sections based on their needs and experience level.