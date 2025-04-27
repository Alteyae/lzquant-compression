# Hybrid Image Compression API

A FastAPI-based service for compressing images using hybrid techniques:
- **PLZ Format**: PNGQuant + LZ4 compression
- **Zstandard**: High-performance compression algorithm

## Features

- Single and batch image compression
- Multiple compression algorithms
- Image quality metrics (PSNR, SSIM)
- Performance monitoring
- Base64 input/output support
- API versioning (v1/v2)
- Detailed debugging information

## Requirements

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- pngquant (for PLZ compression)

## Quick Start

### Development

```bash
# Install dependencies
pip install -r requirements.txt

# Install pngquant
# On Ubuntu/Debian: sudo apt-get install pngquant
# On macOS with Homebrew: brew install pngquant

# Run the application
python main.py
```

### Docker Deployment

```bash
# Development build
docker-compose up --build

# Production build
docker-compose -f docker-compose.prod.yml up --build -d
```

## API Endpoints

The API is organized into versioned endpoints:

### V1 - PLZ Format (PNGQuant + LZ4)

- **POST /api/v1/compress/image**: Compress a single image
- **POST /api/v1/compress/batch/base64**: Compress multiple base64-encoded images
- **GET /api/v1/decompress/{file_id}**: Decompress a previously compressed image
- **POST /api/v1/decompress/batch**: Decompress multiple images with detailed stats
- **POST /api/v1/decompress/batch/download**: Download multiple decompressed images as ZIP

### V2 - Zstandard Compression

- **POST /api/v2/compress/zst**: Compress a single image with Zstandard
- **POST /api/v2/compress/zst/batch/base64**: Compress multiple base64-encoded images
- **GET /api/v2/decompress/zst/{file_id}**: Decompress a Zstandard-compressed image
- **POST /api/v2/decompress/batch**: Decompress multiple images with detailed stats
- **POST /api/v2/decompress/batch/download**: Download multiple decompressed images as ZIP
- **POST /api/v2/decompress/zst/batch/zip**: Alternative batch download method

### Utilities

- **GET /health**: Basic health check
- **GET /health/detailed**: Comprehensive system and component status

## Configuration

Configuration is managed through environment variables:

| Variable      | Description                                | Default |
|---------------|--------------------------------------------|---------|
| PORT          | Server port                                | 8000    |
| WORKERS       | Number of Uvicorn workers                  | 1       |
| LOG_LEVEL     | Logging level (INFO, DEBUG, etc.)          | INFO    |
| DEBUG         | Enable debug mode                          | false   |
| TEMP_DIR      | Directory for temporary files              | auto    |

## Performance Tuning

For optimal performance:

1. Increase worker count according to CPU cores:
   ```
   WORKERS=4 python main.py
   ```

2. Use the production Docker setup for better resource management:
   ```
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. For large batches, consider using smaller batch sizes (20-50 images) to avoid memory issues.

## Troubleshooting

- **Slow Docker Builds**: Use BuildKit with `DOCKER_BUILDKIT=1 docker-compose build`
- **Compression Failures**: Check that pngquant is installed and working
- **Memory Issues**: Reduce worker count or batch size

For more detailed diagnostics, check the `/health/detailed` endpoint.# plz-compression-site

For more information about the lzquant algorithm, please check out here: https://github.com/Alteyae/lzquant-compression/blob/main/LZQUANT_ALGORITHM_PROCESS.md
