"""
Performance tests cho Batch Processing với Large Batches
"""

import pytest
import asyncio
import time
import psutil
import os
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from uuid import uuid4

from models.batch_request import (
    BatchRequest, TTSItem, Priority, BatchStatus, ItemStatus,
    BatchItemResult, BatchProcessingError
)
from utils.batch_processor import BatchProcessor, ProcessingConfig
from utils.batch_queue import BatchQueueManager


class TestBatchPerformance:
    """Performance tests cho batch processing"""

    @pytest.fixture
    def mock_tts_service(self):
        """Mock TTS service với simulated processing time"""
        mock_service = Mock()

        async def generate_audio(*args, **kwargs):
            # Simulate processing time (50-200ms per item)
            processing_time = 0.1 + (hash(str(args)) % 100) / 1000
            await asyncio.sleep(processing_time)

            return {
                "audio_url": "https://example.com/audio.mp3",
                "duration": 2.5,
                "file_size": 1024
            }

        mock_service.generate_audio = generate_audio
        return mock_service

    @pytest.fixture
    def batch_processor(self, mock_tts_service):
        """Batch processor với mocked dependencies"""
        with patch('utils.batch_processor.ProgressStreamer') as mock_streamer, \
             patch('utils.batch_processor.RedisManager') as mock_redis:

            mock_streamer.send_batch_progress = AsyncMock()
            mock_redis.set_cache = AsyncMock()
            mock_redis.get_cache = AsyncMock(return_value={})

            processor = BatchProcessor(
                tts_service=mock_tts_service,
                progress_streamer=mock_streamer,
                redis_manager=mock_redis
            )

            yield processor

    def create_large_batch(self, size: int, name: str = "Large Batch") -> BatchRequest:
        """Create large batch cho performance testing"""
        items = [
            TTSItem(
                text=f"Performance test item {i}",
                voice="voice1",
                language="vi"
            )
            for i in range(size)
        ]

        return BatchRequest(
            name=name,
            items=items,
            priority=Priority.NORMAL
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("batch_size", [10, 25, 50])
    async def test_batch_processing_performance(self, batch_processor, batch_size):
        """Test performance với different batch sizes"""
        # Create batch
        batch_request = self.create_large_batch(batch_size)

        # Measure processing time
        start_time = time.time()

        # Process batch
        result = await batch_processor.process_batch(batch_request)

        end_time = time.time()
        processing_time = end_time - start_time

        # Performance assertions
        assert result["status"] == BatchStatus.COMPLETED
        assert len(result["results"]) == batch_size
        assert result["summary"]["completed"] == batch_size

        # Performance expectations
        if batch_size <= 25:
            assert processing_time < 5.0, "Batch took too long"
        elif batch_size <= 50:
            assert processing_time < 10.0, "Batch took too long"

    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(self, mock_tts_service):
        """Test concurrent batch processing"""
        with patch('utils.batch_processor.ProgressStreamer') as mock_streamer, \
             patch('utils.batch_processor.RedisManager') as mock_redis:

            mock_streamer.send_batch_progress = AsyncMock()
            mock_redis.set_cache = AsyncMock()
            mock_redis.get_cache = AsyncMock(return_value={})

            # Create multiple processors
            processors = []
            for i in range(2):
                processor = BatchProcessor(
                    tts_service=mock_tts_service,
                    progress_streamer=mock_streamer,
                    redis_manager=mock_redis
                )
                processors.append(processor)

            # Create multiple batches
            batches = [
                self.create_large_batch(10, f"Concurrent Batch {i}")
                for i in range(2)
            ]

            # Process batches concurrently
            tasks = [
                processor.process_batch(batch)
                for processor, batch in zip(processors, batches)
            ]

            results = await asyncio.gather(*tasks)

            # Verify all batches completed
            assert len(results) == 2
            assert all(r["status"] == BatchStatus.COMPLETED for r in results)
            assert all(len(r["results"]) == 10 for r in results)

    @pytest.mark.asyncio
    async def test_queue_performance(self):
        """Test queue performance"""
        queue_manager = BatchQueueManager()

        # Create test batches
        batches = [
            self.create_large_batch(5, f"Queue Test Batch {i}")
            for i in range(10)
        ]

        # Measure queue operations
        start_time = time.time()

        # Enqueue all batches
        for batch in batches:
            await queue_manager.enqueue_batch(batch)

        enqueue_time = time.time() - start_time

        # Dequeue batches
        dequeued_count = 0
        dequeue_start = time.time()

        for _ in range(len(batches)):
            batch_item = await queue_manager.dequeue_highest_priority()
            if batch_item:
                dequeued_count += 1

        dequeue_time = time.time() - dequeue_start

        # Performance assertions
        assert dequeued_count == len(batches)
        assert enqueue_time < 2.0, "Enqueue too slow"
        assert dequeue_time < 2.0, "Dequeue too slow"


class TestBatchPerformance:
    """Performance tests cho batch processing"""

    @pytest.fixture
    def mock_tts_service(self):
        """Mock TTS service với simulated processing time"""
        mock_service = Mock()

        async def generate_audio(*args, **kwargs):
            # Simulate processing time (50-200ms per item)
            processing_time = 0.1 + (hash(str(args)) % 100) / 1000
            await asyncio.sleep(processing_time)

            return {
                "audio_url": f"https://example.com/audio_{hash(str(args))}.mp3",
                "duration": 2.5,
                "file_size": 1024
            }

        mock_service.generate_audio = generate_audio
        return mock_service

    @pytest.fixture
    def batch_processor(self, mock_tts_service):
        """Batch processor với mocked dependencies"""
        with patch('utils.batch_processor.ProgressStreamer') as mock_streamer, \
             patch('utils.batch_processor.RedisManager') as mock_redis:

            mock_streamer.send_batch_progress = AsyncMock()
            mock_redis.set_cache = AsyncMock()
            mock_redis.get_cache = AsyncMock(return_value={})

            processor = BatchProcessor(
                tts_service=mock_tts_service,
                progress_streamer=mock_streamer,
                redis_manager=mock_redis
            )

            yield processor

    def create_large_batch(self, size: int, name: str = "Large Batch") -> BatchRequest:
        """Create large batch cho performance testing"""
        items = [
            TTSItem(
                text=f"Performance test item {i} with some text content to simulate real TTS input",
                voice="voice1",
                language="vi"
            )
            for i in range(size)
        ]

        return BatchRequest(
            name=name,
            items=items,
            priority=Priority.NORMAL
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("batch_size", [10, 25, 50, 100])
    async def test_batch_processing_performance(self, batch_processor, batch_size):
        """Test performance với different batch sizes"""
        # Create batch
        batch_request = self.create_large_batch(batch_size)

        # Measure memory usage before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Measure processing time
        start_time = time.time()

        # Process batch
        result = await batch_processor.process_batch(batch_request)

        end_time = time.time()
        processing_time = end_time - start_time

        # Measure memory usage after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        # Performance assertions
        assert result["status"] == BatchStatus.COMPLETED
        assert len(result["results"]) == batch_size
        assert result["summary"]["completed"] == batch_size

        # Log performance metrics
        print(f"\nBatch Size: {batch_size}")
        print(f"Processing Time: {processing_time:.2f}s")
        print(f"Items/Second: {batch_size/processing_time:.2f}")
        print(f"Memory Usage: {memory_after:.1f}MB (+{memory_increase:.1f}MB)")

        # Performance expectations
        if batch_size <= 25:
            assert processing_time < 5.0, f"Batch of {batch_size} items took too long: {processing_time".2f"}s"
        elif batch_size <= 50:
            assert processing_time < 10.0, f"Batch of {batch_size} items took too long: {processing_time".2f"}s"
        else:  # batch_size <= 100
            assert processing_time < 20.0, f"Batch of {batch_size} items took too long: {processing_time".2f"}s"

        # Memory usage should be reasonable
        assert memory_increase < 100, f"Memory increase too high: {memory_increase".1f"}MB"

    @pytest.mark.asyncio
    async def test_concurrent_batch_processing_performance(self, mock_tts_service):
        """Test performance của concurrent batch processing"""
        with patch('utils.batch_processor.ProgressStreamer') as mock_streamer, \
             patch('utils.batch_processor.RedisManager') as mock_redis:

            mock_streamer.send_batch_progress = AsyncMock()
            mock_redis.set_cache = AsyncMock()
            mock_redis.get_cache = AsyncMock(return_value={})

            # Create multiple processors
            processors = []
            for i in range(3):
                processor = BatchProcessor(
                    tts_service=mock_tts_service,
                    progress_streamer=mock_streamer,
                    redis_manager=mock_redis
                )
                processors.append(processor)

            # Create multiple batches
            batches = [
                self.create_large_batch(20, f"Concurrent Batch {i}")
                for i in range(3)
            ]

            # Measure concurrent processing
            start_time = time.time()

            # Process batches concurrently
            tasks = [
                processor.process_batch(batch)
                for processor, batch in zip(processors, batches)
            ]

            results = await asyncio.gather(*tasks)

            end_time = time.time()
            total_time = end_time - start_time

            # Verify all batches completed
            assert all(r["status"] == BatchStatus.COMPLETED for r in results)
            assert all(len(r["results"]) == 20 for r in results)

            # Performance analysis
            total_items = sum(len(r["results"]) for r in results)
            throughput = total_items / total_time

            print(f"\nConcurrent Processing:")
            print(f"Total Batches: {len(batches)}")
            print(f"Total Items: {total_items}")
            print(f"Total Time: {total_time".2f"}s")
            print(f"Throughput: {throughput".1f"} items/second")

            # Should be faster than sequential processing
            expected_sequential_time = total_time * 0.7  # Should be at least 30% faster
            assert total_time < expected_sequential_time, \
                f"Concurrent processing not efficient: {total_time".2f"}s vs expected < {expected_sequential_time".2f"}s"

    @pytest.mark.asyncio
    async def test_memory_efficiency_with_large_batches(self, batch_processor):
        """Test memory efficiency với large batches"""
        # Create large batch
        batch_request = self.create_large_batch(100)

        # Monitor memory usage
        process = psutil.Process(os.getpid())

        memory_samples = []
        async def monitor_memory():
            while True:
                memory_samples.append(process.memory_info().rss / 1024 / 1024)
                await asyncio.sleep(0.1)

        # Start memory monitoring
        monitor_task = asyncio.create_task(monitor_memory())

        try:
            # Process batch
            result = await batch_processor.process_batch(batch_request)

            # Stop monitoring after 1 second
            await asyncio.sleep(1.0)
            monitor_task.cancel()

            # Verify processing completed
            assert result["status"] == BatchStatus.COMPLETED
            assert len(result["results"]) == 100

            # Memory analysis
            if memory_samples:
                peak_memory = max(memory_samples)
                final_memory = memory_samples[-1]
                memory_efficiency = final_memory / peak_memory if peak_memory > 0 else 1.0

                print(f"\nMemory Efficiency Test:")
                print(f"Peak Memory: {peak_memory".1f"}MB")
                print(f"Final Memory: {final_memory".1f"}MB")
                print(f"Memory Efficiency: {memory_efficiency".2%"}")

                # Memory should be efficiently managed
                assert peak_memory < 200, f"Peak memory too high: {peak_memory".1f"}MB"
                assert memory_efficiency > 0.7, f"Memory not efficiently released: {memory_efficiency".2%"}"

        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_queue_performance_with_multiple_priorities(self):
        """Test queue performance với multiple priorities"""
        queue_manager = BatchQueueManager()

        # Create batches with different priorities
        batches = []
        for priority in [Priority.HIGH, Priority.NORMAL, Priority.LOW]:
            for i in range(5):
                batch = self.create_large_batch(
                    10,
                    f"{priority.value.title()} Priority Batch {i}"
                )
                batch.priority = priority
                batches.append(batch)

        # Measure queue operations
        start_time = time.time()

        # Enqueue all batches
        for batch in batches:
            await queue_manager.enqueue_batch(batch)

        enqueue_time = time.time() - start_time

        # Dequeue batches (should be in priority order)
        dequeued_batches = []
        dequeue_start = time.time()

        for _ in range(len(batches)):
            batch_item = await queue_manager.dequeue_highest_priority()
            if batch_item:
                dequeued_batches.append(batch_item.batch_request)

        dequeue_time = time.time() - dequeue_start

        # Verify priority order
        assert len(dequeued_batches) == len(batches)

        # Check that high priority batches come first
        high_priority_count = sum(1 for b in dequeued_batches[:5] if b.priority == Priority.HIGH)
        assert high_priority_count == 5, "High priority batches should be dequeued first"

        # Performance metrics
        print(f"\nQueue Performance:")
        print(f"Batches Enqueued: {len(batches)}")
        print(f"Enqueue Time: {enqueue_time".3f"}s")
        print(f"Dequeue Time: {dequeue_time".3f"}s")
        print(f"Operations/Second: {len(batches)/(enqueue_time+dequeue_time)".1f"}")

        # Should be fast
        assert enqueue_time < 1.0, f"Enqueue too slow: {enqueue_time".3f"}s"
        assert dequeue_time < 1.0, f"Dequeue too slow: {dequeue_time".3f"}s"

    @pytest.mark.asyncio
    async def test_chunked_processing_performance(self, batch_processor):
        """Test performance của chunked processing"""
        # Create large batch
        batch_request = self.create_large_batch(100)

        # Configure for chunked processing
        chunked_config = ProcessingConfig(
            max_concurrency=5,
            chunk_size=20  # Process in chunks of 20
        )
        batch_processor.update_config(chunked_config)

        # Measure processing
        start_time = time.time()
        result = await batch_processor.process_batch(batch_request)
        processing_time = time.time() - start_time

        # Verify results
        assert result["status"] == BatchStatus.COMPLETED
        assert len(result["results"]) == 100

        # Performance analysis
        throughput = 100 / processing_time

        print(f"\nChunked Processing Performance:")
        print(f"Batch Size: 100")
        print(f"Chunk Size: {chunked_config.chunk_size}")
        print(f"Processing Time: {processing_time".2f"}s")
        print(f"Throughput: {throughput".1f"} items/second")

        # Should be efficient
        assert processing_time < 30, f"Chunked processing too slow: {processing_time".2f"}s"
        assert throughput > 3, f"Throughput too low: {throughput".1f"} items/second"

    @pytest.mark.asyncio
    async def test_scalability_with_increasing_load(self, mock_tts_service):
        """Test scalability với increasing load"""
        with patch('utils.batch_processor.ProgressStreamer') as mock_streamer, \
             patch('utils.batch_processor.RedisManager') as mock_redis:

            mock_streamer.send_batch_progress = AsyncMock()
            mock_redis.set_cache = AsyncMock()
            mock_redis.get_cache = AsyncMock(return_value={})

            # Test with different concurrency levels
            concurrency_levels = [1, 3, 5]
            batch_sizes = [10, 25, 50]

            results = {}

            for concurrency in concurrency_levels:
                for batch_size in batch_sizes:
                    # Create processor with specific concurrency
                    processor = BatchProcessor(
                        tts_service=mock_tts_service,
                        progress_streamer=mock_streamer,
                        redis_manager=mock_redis,
                        config=ProcessingConfig(max_concurrency=concurrency)
                    )

                    # Create batch
                    batch_request = self.create_large_batch(batch_size)

                    # Measure performance
                    start_time = time.time()
                    result = await processor.process_batch(batch_request)
                    processing_time = time.time() - start_time

                    # Store results
                    key = f"concurrency_{concurrency}_batch_{batch_size}"
                    results[key] = {
                        "processing_time": processing_time,
                        "throughput": batch_size / processing_time,
                        "completed": result["summary"]["completed"]
                    }

                    print(f"\nScalability Test - Concurrency: {concurrency}, Batch Size: {batch_size}")
                    print(f"Time: {processing_time".2f"}s, Throughput: {batch_size/processing_time".1f"} items/s")

            # Analyze scalability
            print("
Scalability Analysis:")

            # Check that higher concurrency improves performance
            for batch_size in batch_sizes:
                single_throughput = results[f"concurrency_1_batch_{batch_size}"]["throughput"]
                multi_throughput = results[f"concurrency_5_batch_{batch_size}"]["throughput"]

                improvement = (multi_throughput - single_throughput) / single_throughput

                print(f"Batch size {batch_size}: {improvement".1%"} improvement with concurrency")

                # Should show improvement (though may be limited by mock service)
                assert improvement >= -0.1, "Performance should not degrade significantly"

    @pytest.mark.asyncio
    async def test_resource_cleanup_performance(self, batch_processor):
        """Test resource cleanup performance"""
        # Create batch
        batch_request = self.create_large_batch(50)

        # Monitor memory
        process = psutil.Process(os.getpid())

        # Process batch
        start_memory = process.memory_info().rss / 1024 / 1024
        result = await batch_processor.process_batch(batch_request)
        end_memory = process.memory_info().rss / 1024 / 1024

        # Wait for cleanup
        await asyncio.sleep(0.5)
        final_memory = process.memory_info().rss / 1024 / 1024

        # Verify processing completed
        assert result["status"] == BatchStatus.COMPLETED
        assert len(result["results"]) == 50

        # Memory cleanup analysis
        memory_increase = end_memory - start_memory
        memory_cleanup = end_memory - final_memory

        print(f"\nResource Cleanup Test:")
        print(f"Memory Increase: {memory_increase".1f"}MB")
        print(f"Memory Cleanup: {memory_cleanup".1f"}MB")
        print(f"Final Memory: {final_memory".1f"}MB")

        # Should clean up memory
        assert memory_cleanup >= 0, "Memory should be cleaned up"
        assert final_memory < start_memory + 50, "Memory usage should be reasonable"

    @pytest.mark.asyncio
    async def test_error_recovery_performance(self, batch_processor):
        """Test error recovery performance"""
        # Create batch with some failing items
        items = [
            TTSItem(text=f"Item {i}", voice="voice1", language="vi")
            for i in range(30)
        ]

        batch_request = BatchRequest(
            name="Error Recovery Test Batch",
            items=items,
            priority=Priority.NORMAL
        )

        # Setup TTS service to fail on some items
        call_count = 0
        async def failing_generate_audio(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            # Fail on every 5th item
            if call_count % 5 == 0:
                await asyncio.sleep(0.1)  # Simulate processing time
                raise Exception("Simulated failure")

            await asyncio.sleep(0.1)  # Simulate processing time
            return {
                "audio_url": f"https://example.com/audio_{call_count}.mp3",
                "duration": 2.5,
                "file_size": 1024
            }

        batch_processor.tts_service.generate_audio = failing_generate_audio

        # Measure error recovery
        start_time = time.time()
        result = await batch_processor.process_batch(batch_request)
        recovery_time = time.time() - start_time

        # Verify error recovery
        assert result["status"] == BatchStatus.PARTIALLY_COMPLETED
        assert result["summary"]["total_items"] == 30
        assert result["summary"]["completed"] == 24  # 30 - 6 failed items
        assert result["summary"]["failed"] == 6

        # Performance with error recovery
        throughput = 30 / recovery_time

        print(f"\nError Recovery Performance:")
        print(f"Total Items: 30")
        print(f"Successful: {result['summary']['completed']}")
        print(f"Failed: {result['summary']['failed']}")
        print(f"Recovery Time: {recovery_time".2f"}s")
        print(f"Throughput: {throughput".1f"} items/second")

        # Should handle errors efficiently
        assert recovery_time < 15, f"Error recovery too slow: {recovery_time".2f"}s"
        assert throughput > 2, f"Throughput too low with error recovery: {throughput".1f"}"


class TestBatchPerformanceBenchmarks:
    """Performance benchmarks cho batch processing"""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_batch_throughput_benchmark(self, batch_processor):
        """Benchmark batch processing throughput"""
        # Create medium-sized batch
        batch_request = self.create_large_batch(50)

        # Benchmark processing
        start_time = time.time()
        result = await batch_processor.process_batch(batch_request)
        end_time = time.time()

        processing_time = end_time - start_time
        throughput = 50 / processing_time

        # Benchmark assertions
        assert result["status"] == BatchStatus.COMPLETED
        assert throughput > 5, f"Benchmark failed: {throughput".1f"} items/second"

        return {
            "batch_size": 50,
            "processing_time": processing_time,
            "throughput": throughput,
            "memory_mb": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        }

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_queue_operations_benchmark(self):
        """Benchmark queue operations"""
        queue_manager = BatchQueueManager()

        # Create test batches
        batches = [
            self.create_large_batch(10, f"Benchmark Batch {i}")
            for i in range(20)
        ]

        # Benchmark enqueue operations
        enqueue_start = time.time()
        for batch in batches:
            await queue_manager.enqueue_batch(batch)
        enqueue_time = time.time() - enqueue_start

        # Benchmark dequeue operations
        dequeue_start = time.time()
        dequeued_count = 0
        for _ in range(len(batches)):
            batch_item = await queue_manager.dequeue_highest_priority()
            if batch_item:
                dequeued_count += 1
        dequeue_time = time.time() - dequeue_start

        # Benchmark results
        total_time = enqueue_time + dequeue_time
        operations_per_second = len(batches) / total_time

        print(f"\nQueue Benchmark Results:")
        print(f"Batches: {len(batches)}")
        print(f"Enqueue Time: {enqueue_time".3f"}s")
        print(f"Dequeue Time: {dequeue_time".3f"}s")
        print(f"Total Time: {total_time".3f"}s")
        print(f"Operations/Second: {operations_per_second".1f"}")

        assert operations_per_second > 10, f"Queue operations too slow: {operations_per_second".1f"} ops/sec"
        assert dequeued_count == len(batches), "Not all batches were dequeued"

        return {
            "batches_processed": len(batches),
            "enqueue_time": enqueue_time,
            "dequeue_time": dequeue_time,
            "operations_per_second": operations_per_second
        }