"""
Asynchronous processing utilities for CircuitKit.
Provides async I/O, batch processing, and concurrent operations.
"""

import asyncio

try:
    import aiofiles
except ImportError:
    aiofiles = None
import concurrent.futures
import json
import pickle
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from circuitkit.utils.logging import get_logger

logger = get_logger(__name__)


class AsyncIO:
    """Asynchronous I/O operations."""

    @staticmethod
    async def read_file_async(file_path: Union[str, Path]) -> str:
        """
        Asynchronously read a text file.

        Args:
            file_path: Path to the file

        Returns:
            File contents as string
        """
        if aiofiles is None:
            # Fallback to synchronous reading
            with open(file_path, "r") as f:
                return f.read()

        async with aiofiles.open(file_path, "r") as f:
            content = await f.read()
        return content

    @staticmethod
    async def write_file_async(file_path: Union[str, Path], content: str) -> None:
        """
        Asynchronously write content to a file.

        Args:
            file_path: Path to the file
            content: Content to write
        """
        async with aiofiles.open(file_path, "w") as f:
            await f.write(content)

    @staticmethod
    async def read_json_async(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Asynchronously read a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Parsed JSON data
        """
        content = await AsyncIO.read_file_async(file_path)
        return json.loads(content)

    @staticmethod
    async def write_json_async(file_path: Union[str, Path], data: Dict[str, Any]) -> None:
        """
        Asynchronously write data to a JSON file.

        Args:
            file_path: Path to the JSON file
            data: Data to write
        """
        content = json.dumps(data, indent=2)
        await AsyncIO.write_file_async(file_path, content)

    @staticmethod
    async def read_pickle_async(file_path: Union[str, Path]) -> Any:
        """
        Asynchronously read a pickle file.

        Args:
            file_path: Path to the pickle file

        Returns:
            Unpickled data
        """
        async with aiofiles.open(file_path, "rb") as f:
            content = await f.read()
        return pickle.loads(content)

    @staticmethod
    async def write_pickle_async(file_path: Union[str, Path], data: Any) -> None:
        """
        Asynchronously write data to a pickle file.

        Args:
            file_path: Path to the pickle file
            data: Data to write
        """
        content = pickle.dumps(data)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)


class AsyncBatchProcessor:
    """Asynchronous batch processing utilities."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def process_batch_async(
        self, items: List[Any], process_func: Callable, batch_size: int = 10
    ) -> List[Any]:
        """
        Process items in batches asynchronously.

        Args:
            items: List of items to process
            process_func: Function to process each item
            batch_size: Size of each batch

        Returns:
            List of processed results
        """
        results = []

        # Split items into batches
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

        # Process batches concurrently
        tasks = []
        for batch in batches:
            task = asyncio.create_task(self._process_batch(batch, process_func))
            tasks.append(task)

        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks)

        # Flatten results
        for batch_result in batch_results:
            results.extend(batch_result)

        return results

    async def _process_batch(self, batch: List[Any], process_func: Callable) -> List[Any]:
        """Process a single batch."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, process_func, batch)

    def close(self):
        """Close the executor."""
        self.executor.shutdown(wait=True)


class AsyncDataLoader:
    """Asynchronous data loading utilities."""

    def __init__(self, max_queue_size: int = 100):
        self.max_queue_size = max_queue_size
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.loader_thread = None

    def start_async_loading(self, data_source: Callable) -> None:
        """
        Start asynchronous data loading.

        Args:
            data_source: Function that yields data items
        """

        def loader():
            try:
                for item in data_source():
                    if self.stop_event.is_set():
                        break
                    self.queue.put(item, block=True, timeout=1.0)
            except queue.Full:
                logger.warning("Data loader queue is full, dropping items")
            except Exception as e:
                logger.error(f"Error in data loader: {e}")

        self.loader_thread = threading.Thread(target=loader, daemon=True)
        self.loader_thread.start()

    def get_next_item(self, timeout: float = 1.0) -> Optional[Any]:
        """
        Get the next item from the async loader.

        Args:
            timeout: Timeout in seconds

        Returns:
            Next item or None if timeout
        """
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop_loading(self) -> None:
        """Stop the async data loading."""
        self.stop_event.set()
        if self.loader_thread:
            self.loader_thread.join(timeout=5.0)


class AsyncModelProcessor:
    """Asynchronous model processing utilities."""

    def __init__(self, model, max_workers: int = 2):
        self.model = model
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def process_async(self, inputs: List[Any]) -> List[Any]:
        """
        Process inputs asynchronously using the model.

        Args:
            inputs: List of inputs to process

        Returns:
            List of model outputs
        """
        loop = asyncio.get_event_loop()

        # Process inputs in parallel
        tasks = []
        for input_item in inputs:
            task = loop.run_in_executor(self.executor, self._process_single, input_item)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    def _process_single(self, input_item: Any) -> Any:
        """Process a single input item."""
        try:
            # Handle different input types
            if isinstance(input_item, dict):
                # Extract text from dictionary
                text = input_item.get("text", str(input_item))
                return self.model(text)
            elif isinstance(input_item, str):
                return self.model(input_item)
            else:
                # Convert to string and process
                return self.model(str(input_item))
        except Exception as e:
            logger.error(f"Error processing item: {e}")
            return None

    def close(self):
        """Close the executor."""
        self.executor.shutdown(wait=True)


class AsyncCache:
    """Asynchronous caching system."""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.access_times: Dict[str, float] = {}
        self.lock = threading.Lock()

    async def get_async(self, key: str) -> Optional[Any]:
        """
        Asynchronously get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None

    async def set_async(self, key: str, value: Any) -> None:
        """
        Asynchronously set item in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Remove oldest items if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]

            self.cache[key] = value
            self.access_times[key] = time.time()

    async def clear_async(self) -> None:
        """Asynchronously clear the cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()


# Global instances
async_io = AsyncIO()
async_batch_processor = AsyncBatchProcessor()
async_data_loader = AsyncDataLoader()
async_cache = AsyncCache()


async def process_files_async(
    file_paths: List[Union[str, Path]], process_func: Callable
) -> List[Any]:
    """
    Process multiple files asynchronously.

    Args:
        file_paths: List of file paths
        process_func: Function to process each file

    Returns:
        List of processed results
    """
    tasks = []
    for file_path in file_paths:
        task = asyncio.create_task(process_func(file_path))
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results


async def save_results_async(results: List[Any], output_paths: List[Union[str, Path]]) -> None:
    """
    Save multiple results asynchronously.

    Args:
        results: List of results to save
        output_paths: List of output paths
    """
    if len(results) != len(output_paths):
        raise ValueError("Number of results must match number of output paths")

    tasks = []
    for result, output_path in zip(results, output_paths):
        if isinstance(result, str):
            task = asyncio.create_task(async_io.write_file_async(output_path, result))
        elif isinstance(result, dict):
            task = asyncio.create_task(async_io.write_json_async(output_path, result))
        else:
            task = asyncio.create_task(async_io.write_pickle_async(output_path, result))
        tasks.append(task)

    await asyncio.gather(*tasks)
