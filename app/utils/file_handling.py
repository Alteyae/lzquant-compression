"""
Utilities for file handling and temporary file management.
"""
import os
import asyncio
import logging
import contextlib
import uuid
from typing import Optional, Callable, Any, AsyncGenerator
from app import TEMP_DIR

# Set up logging
logger = logging.getLogger(__name__)


def get_temp_filepath(file_id: Optional[str] = None, suffix: str = "") -> str:
    """
    Generate a path for a temporary file.
    
    Args:
        file_id: Optional file ID to use (generates a new UUID if not provided)
        suffix: Optional file suffix/extension
        
    Returns:
        Absolute path to a temporary file
    """
    if file_id is None:
        file_id = str(uuid.uuid4())
        
    return os.path.join(TEMP_DIR, f"{file_id}{suffix}")


async def cleanup_file_later(file_path: str, delay: int = 60) -> None:
    """
    Schedule a file for deletion after a delay.
    
    Args:
        file_path: Path to the file to delete
        delay: Delay in seconds before deletion (default: 60)
    """
    logger.debug(f"Scheduling cleanup of {file_path} in {delay} seconds")
    await asyncio.sleep(delay)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to clean up temporary file {file_path}: {e}")


def schedule_cleanup(file_path: str, delay: int = 60) -> None:
    """
    Schedule a file for deletion after a delay (non-blocking).
    
    Args:
        file_path: Path to the file to delete
        delay: Delay in seconds before deletion (default: 60)
    """
    # Create and schedule the cleanup task
    asyncio.create_task(cleanup_file_later(file_path, delay))


@contextlib.contextmanager
def temp_file_context(suffix: str = "", cleanup: bool = True) -> str:
    """
    Context manager that creates a temporary file and optionally cleans it up.
    
    Args:
        suffix: File extension to use
        cleanup: Whether to delete the file after the context exits
        
    Yields:
        Path to the temporary file
    """
    temp_path = get_temp_filepath(suffix=suffix)
    try:
        yield temp_path
    finally:
        if cleanup and os.path.exists(temp_path):
            os.remove(temp_path)


async def process_in_batches(
    items: list,
    batch_size: int,
    processor: Callable[[Any], Any]
) -> AsyncGenerator[Any, None]:
    """
    Process items in batches asynchronously.
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        processor: Function to process each item
        
    Yields:
        Results from each batch
    """
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        # Process batch
        results = []
        for item in batch:
            try:
                result = await processor(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item {item}: {e}")
                results.append({"error": str(e)})
        
        yield results