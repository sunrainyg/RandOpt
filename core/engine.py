"""vLLM engine setup and management."""
import os
import time
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM


class RandOptNcclLLM(LLM):
    """Custom LLM class with NCCL backend configuration."""
    
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)


def launch_engines(num_engines: int, model_name: str, precision: str = "bfloat16", batch_size: int = 25, tensor_parallel_size: int = 1, enable_prefix_caching: bool = False, gpu_memory_utilization: float = 0.75, multimodal: bool = False):
    """Launch vLLM engines on Ray with batched initialization.
    
    Args:
        num_engines: Number of engines to launch
        model_name: Path to model or HuggingFace model name
        precision: Model precision (bfloat16 or float16)
        batch_size: Number of engines to initialize per batch (reduces NFS contention)
        tensor_parallel_size: Number of GPUs per engine for tensor parallelism (for large models)
        multimodal: Whether to enable multimodal support (for VL models with images)
    
    Returns:
        Tuple of (engines list, placement groups list)
    """
    # Check available GPUs before creating placement groups
    required_gpus = num_engines * tensor_parallel_size
    cluster_resources = ray.cluster_resources()
    available_gpus = int(cluster_resources.get("GPU", 0))
    
    print(f"Cluster resources: {available_gpus} GPUs available, {required_gpus} GPUs required")
    
    if available_gpus < required_gpus:
        # Reduce num_engines to fit available GPUs instead of failing
        max_engines = available_gpus // tensor_parallel_size
        print(f"WARNING: Insufficient GPUs in Ray cluster! "
              f"Required: {required_gpus} GPUs ({num_engines} engines × TP={tensor_parallel_size}), "
              f"Available: {available_gpus} GPUs. "
              f"Reducing num_engines from {num_engines} to {max_engines} to continue.")
        num_engines = max_engines
        if num_engines == 0:
            raise RuntimeError(
                f"No GPUs available to launch any engines (need at least {tensor_parallel_size} GPUs for TP={tensor_parallel_size})."
            )
    
    print(f"Creating {num_engines} placement groups (TP={tensor_parallel_size})...")
    pg_start = time.time()
    # Each bundle can only have 1 GPU (vLLM requirement), so create TP bundles per engine
    pg_bundles = [{"GPU": 1, "CPU": 0} for _ in range(tensor_parallel_size)]
    pgs = [placement_group(pg_bundles, lifetime="detached") for _ in range(num_engines)]
    
    # Wait for placement groups with timeout
    try:
        ray.get([pg.ready() for pg in pgs], timeout=120)  # 2 minute timeout
    except ray.exceptions.GetTimeoutError:
        # Clean up any created placement groups
        from ray.util.placement_group import remove_placement_group
        for pg in pgs:
            try:
                remove_placement_group(pg)
            except:
                pass
        raise RuntimeError(
            f"Timeout waiting for placement groups after 120s. "
            f"Required: {required_gpus} GPUs, Available: {available_gpus} GPUs. "
            f"Some GPUs may be in use or nodes may have failed to join."
        )
    
    print(f"Placement groups ready in {time.time() - pg_start:.1f}s")

    strategies = [
        PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0
        )
        for pg in pgs
    ]

    engines = []
    num_batches = (num_engines + batch_size - 1) // batch_size
    total_start = time.time()
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_engines)
        batch_strategies = strategies[start_idx:end_idx]
        
        print(f"Launching batch {batch_idx + 1}/{num_batches} (engines {start_idx}-{end_idx - 1})...")
        batch_start = time.time()
        
        # Build engine kwargs
        engine_kwargs = dict(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="ray",
            worker_extension_cls="utils.worker_extn.WorkerExtension",
            dtype=precision,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=True,
            gpu_memory_utilization=gpu_memory_utilization,
            disable_log_stats=True,
        )
        if multimodal:
            engine_kwargs["limit_mm_per_prompt"] = {"image": 1}
        
        batch_engines = [
            ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(RandOptNcclLLM).remote(
                **engine_kwargs
            )
            for strategy in batch_strategies
        ]
        
        # Wait for batch to initialize
        ray.get([e.collective_rpc.remote("store_base_weights", args=()) for e in batch_engines])
        print(f"  Batch {batch_idx + 1} initialized in {time.time() - batch_start:.1f}s")
        engines.extend(batch_engines)
    
    print(f"All {num_engines} engines launched in {time.time() - total_start:.1f}s")
    return engines, pgs


def cleanup_engines(engines: list, pgs: list):
    """Clean up vLLM engines and placement groups."""
    from ray.util.placement_group import remove_placement_group
    
    for llm in engines:
        try:
            ray.kill(llm)
        except:
            pass
    for pg in pgs:
        try:
            remove_placement_group(pg)
        except:
            pass
    ray.shutdown()

