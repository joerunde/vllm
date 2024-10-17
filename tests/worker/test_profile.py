import torch

from vllm.engine.arg_utils import EngineArgs
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.worker import Worker


def test_gpu_memory_profiling():
    # Tests the gpu profiling that happens in order to determine the number of
    # KV cache blocks that we can allocate on the GPU.
    # This test mocks the maximum available gpu memory so that it can run on
    # any gpu setup.

    # Set up engine args to build a worker.
    engine_args = EngineArgs(model="facebook/opt-125m",
                             dtype="half",
                             load_format="dummy")
    engine_config = engine_args.create_engine_config()
    engine_config.cache_config.num_gpu_blocks = 1000
    engine_config.cache_config.num_cpu_blocks = 1000

    # Create the worker.
    distributed_init_method = get_distributed_init_method(
        get_ip(), get_open_port())
    worker = Worker(
        model_config=engine_config.model_config,
        parallel_config=engine_config.parallel_config,
        scheduler_config=engine_config.scheduler_config,
        device_config=engine_config.device_config,
        cache_config=engine_config.cache_config,
        load_config=engine_config.load_config,
        local_rank=0,
        rank=0,
        distributed_init_method=distributed_init_method,
        is_driver_worker=True,
    )

    # Load the model so we can profile it
    worker.init_device()
    worker.load_model()

    # Set 10GiB as the total gpu ram to be device-agnostic
    def mock_mem_info():
        current_usage = torch.cuda.memory_stats(
        )["allocated_bytes.all.current"]
        mock_total_bytes = 10 * 1024**3
        free = mock_total_bytes - current_usage

        return (free, mock_total_bytes)

    from unittest.mock import patch
    with patch("torch.cuda.mem_get_info", side_effect=mock_mem_info):
        gpu_blocks, _ = worker.determine_num_available_blocks()

    # Peak vram usage by torch should be 0.7077 GiB
    # Non-torch allocations should be 0.0079 GiB
    # 9.0 GiB should be the utilization target
    # 8.2843 GiB should be available for the KV cache
    block_size = CacheEngine.get_cache_block_size(
        engine_config.cache_config, engine_config.model_config,
        engine_config.parallel_config)

    assert gpu_blocks == (8.2843 * 1024**3) // block_size
