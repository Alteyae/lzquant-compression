"""
Utility functions for the hybrid compression application.
"""
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from app.utils.metrics import (
    get_cpu_mem,
    calculate_image_metrics,
    measure_compression_performance,
    PerformanceTimer
)

from app.utils.file_handling import (
    get_temp_filepath,
    cleanup_file_later,
    schedule_cleanup,
    temp_file_context,
    process_in_batches
)

__all__ = [
    # Metrics utilities
    'get_cpu_mem',
    'calculate_image_metrics',
    'measure_compression_performance',
    'PerformanceTimer',
    
    # File handling utilities
    'get_temp_filepath',
    'cleanup_file_later',
    'schedule_cleanup',
    'temp_file_context',
    'process_in_batches'
]