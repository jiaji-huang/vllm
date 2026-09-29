# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark the TP-sharded hierarchical and dense LM-head paths.

Launch with ``torchrun`` so the benchmark uses the production tensor-parallel
process group and collectives. Timings are the maximum latency across ranks.
"""

import json
import os
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.gemma4_mtp import (
    Gemma4MTPMaskedEmbedder,
    _decode_local_top_token,
    _grouped_gemm,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _initialize_distributed() -> tuple[int, int, torch.device, VllmConfig]:
    required = {"RANK", "WORLD_SIZE", "LOCAL_RANK"}
    if not required <= os.environ.keys():
        raise RuntimeError(
            "Launch this benchmark with torchrun, for example: "
            "torchrun --standalone --nproc-per-node=8 "
            "benchmarks/kernels/benchmark_cluster_parallel_lm_head.py"
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.accelerator.set_device_index(device)
    config = VllmConfig(device_config=DeviceConfig(device="cuda"))
    init_distributed_environment()
    with set_current_vllm_config(config):
        initialize_model_parallel(tensor_model_parallel_size=world_size)
    return rank, world_size, device, config


def _build_heads(
    args,
    rank: int,
    world_size: int,
    device: torch.device,
    config: VllmConfig,
) -> tuple[Gemma4MTPMaskedEmbedder, ParallelLMHead, LogitsProcessor]:
    dtype = torch.bfloat16
    with set_current_vllm_config(config):
        hierarchical = Gemma4MTPMaskedEmbedder(
            hidden_size=args.hidden_size,
            vocab_size=args.vocab_size,
            num_centroids=args.num_clusters,
            centroid_intermediate_top_k=args.top_k,
            prefix="benchmark.masked_embedding",
            centroid_shard_rank=rank,
            centroid_shard_size=world_size,
        ).to(device=device, dtype=dtype)
        dense = ParallelLMHead(
            args.vocab_size,
            args.hidden_size,
            params_dtype=dtype,
            prefix="benchmark.lm_head",
        ).to(device=device)
        logits_processor = LogitsProcessor(args.vocab_size)

    hierarchical.use_moe_backend = True
    shared_generator = torch.Generator(device=device).manual_seed(args.seed)
    weight_generator = torch.Generator(device=device).manual_seed(args.seed + rank + 1)
    with torch.no_grad():
        hierarchical.centroids.weight.normal_(generator=shared_generator)
        hierarchical.token_ordering.copy_(
            torch.randperm(
                args.vocab_size,
                generator=shared_generator,
                device=device,
            )
        )
        dense.weight.normal_(generator=weight_generator)

    hierarchical.build_sharded_centroid_weight(
        dense.weight,
        vocab_start=dense.shard_indices.org_vocab_start_index,
        vocab_end=dense.shard_indices.org_vocab_end_index,
    )
    return hierarchical, dense, logits_processor


def _global_pair_winner(local_pair: torch.Tensor, tp_size: int) -> torch.Tensor:
    gathered = tensor_model_parallel_all_gather(local_pair, dim=-1).view(
        local_pair.shape[0], tp_size, 2
    )
    winning_rank = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    return (
        gathered[:, :, 1]
        .gather(dim=-1, index=winning_rank)
        .squeeze(-1)
        .to(torch.int64)
    )


def _hierarchical_gemm(
    head: Gemma4MTPMaskedEmbedder,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    assert head.centroid_weight is not None
    return _grouped_gemm(
        hidden_states,
        head.centroid_weight,
        topk_ids,
        head.centroid_intermediate_top_k,
        head.local_num_centroids,
        expert_map=head.centroid_expert_map,
    )


def _hierarchical_local_pair(
    head: Gemma4MTPMaskedEmbedder,
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    local_max, token_ids = _decode_local_top_token(
        logits,
        topk_ids,
        head.token_ordering,
        local_centroid_start=head.local_centroid_start,
        local_num_centroids=head.local_num_centroids,
        vocab_size_per_centroid=head.vocab_size_per_centroid,
    )
    return torch.stack((local_max.float(), token_ids.float()), dim=-1)


def _dense_logits(
    processor: LogitsProcessor,
    head: ParallelLMHead,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    return processor._apply_head(head, hidden_states, None)


def _dense_local_pair(head: ParallelLMHead, logits: torch.Tensor) -> torch.Tensor:
    num_pad = head.shard_indices.num_org_vocab_padding
    if num_pad > 0:
        logits = logits.clone()
        logits[..., -num_pad:] = -float("inf")
    local_max, local_ids = logits.max(dim=-1)
    global_ids = local_ids + head.shard_indices.org_vocab_start_index
    return torch.stack((local_max.float(), global_ids.float()), dim=-1)


def _max_rank_latency_us(
    fn: Callable[[int], Any],
    *,
    pool_size: int,
    warmup_replays: int,
    iterations: int,
    samples: int,
    device: torch.device,
) -> float:
    """Measure fixed replay loops and return median max-rank GPU latency."""
    for replay in range(warmup_replays):
        fn(replay % pool_size)
    torch.cuda.synchronize(device)
    dist.barrier(group=get_tp_group().device_group)

    sample_us: list[float] = []
    for sample in range(samples):
        dist.barrier(group=get_tp_group().device_group)
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        offset = sample * iterations
        for replay in range(iterations):
            fn((offset + replay) % pool_size)
        end.record()
        end.synchronize()
        latency = torch.tensor(
            start.elapsed_time(end) * 1000.0 / iterations,
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(
            latency,
            op=dist.ReduceOp.MAX,
            group=get_tp_group().device_group,
        )
        sample_us.append(latency.item())
    return statistics.median(sample_us)


def _validate_paths(
    hierarchical: Gemma4MTPMaskedEmbedder,
    dense: ParallelLMHead,
    processor: LogitsProcessor,
    hidden_states: torch.Tensor,
    tp_size: int,
) -> None:
    topk_ids = hierarchical._route(hidden_states).to(torch.int32)
    hierarchical_logits = _hierarchical_gemm(
        hierarchical, hidden_states, topk_ids
    )
    hierarchical_pair = _hierarchical_local_pair(
        hierarchical, hierarchical_logits, topk_ids
    )
    manual_hierarchical = _global_pair_winner(hierarchical_pair, tp_size)
    production_hierarchical = hierarchical.get_top_tokens(
        hidden_states, dense.weight
    )
    torch.testing.assert_close(manual_hierarchical, production_hierarchical)

    dense_logits = _dense_logits(processor, dense, hidden_states)
    dense_pair = _dense_local_pair(dense, dense_logits)
    manual_dense = _global_pair_winner(dense_pair, tp_size)
    production_dense = processor.get_top_tokens(dense, hidden_states)
    torch.testing.assert_close(manual_dense, production_dense)


def _benchmark_rows(
    args,
    num_rows: int,
    device: torch.device,
    tp_size: int,
    hierarchical: Gemma4MTPMaskedEmbedder,
    dense: ParallelLMHead,
    processor: LogitsProcessor,
) -> dict[str, float | int]:
    generators = [
        torch.Generator(device=device).manual_seed(args.seed + 1000 + index)
        for index in range(args.input_pool_size)
    ]
    hidden_pool = [
        torch.randn(
            num_rows,
            args.hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        for generator in generators
    ]
    with torch.no_grad():
        topk_pool = [
            hierarchical._route(hidden_states).to(torch.int32)
            for hidden_states in hidden_pool
        ]
        hierarchical_logits_pool = [
            _hierarchical_gemm(hierarchical, hidden_states, topk_ids)
            for hidden_states, topk_ids in zip(hidden_pool, topk_pool, strict=True)
        ]
        hierarchical_pair_pool = [
            _hierarchical_local_pair(hierarchical, logits, topk_ids)
            for logits, topk_ids in zip(
                hierarchical_logits_pool, topk_pool, strict=True
            )
        ]
        dense_logits_pool = [
            _dense_logits(processor, dense, hidden_states)
            for hidden_states in hidden_pool
        ]
        dense_pair_pool = [
            _dense_local_pair(dense, logits) for logits in dense_logits_pool
        ]
        _validate_paths(
            hierarchical, dense, processor, hidden_pool[0], tp_size
        )

        common = {
            "pool_size": args.input_pool_size,
            "warmup_replays": args.warmup_replays,
            "iterations": args.iterations,
            "samples": args.samples,
            "device": device,
        }

        def measure(fn: Callable[[int], Any]) -> float:
            return _max_rank_latency_us(fn, **common)

        route_us = measure(lambda index: hierarchical._route(hidden_pool[index]))
        hierarchical_gemm_us = measure(
            lambda index: _hierarchical_gemm(
                hierarchical, hidden_pool[index], topk_pool[index]
            )
        )
        hierarchical_local_us = measure(
            lambda index: _hierarchical_local_pair(
                hierarchical,
                hierarchical_logits_pool[index],
                topk_pool[index],
            )
        )
        hierarchical_collective_us = measure(
            lambda index: _global_pair_winner(
                hierarchical_pair_pool[index], tp_size
            )
        )
        hierarchical_e2e_us = measure(
            lambda index: hierarchical.get_top_tokens(
                hidden_pool[index], dense.weight
            )
        )
        dense_gemm_us = measure(
            lambda index: _dense_logits(processor, dense, hidden_pool[index])
        )
        dense_local_us = measure(
            lambda index: _dense_local_pair(dense, dense_logits_pool[index])
        )
        dense_collective_us = measure(
            lambda index: _global_pair_winner(dense_pair_pool[index], tp_size)
        )
        dense_e2e_us = measure(
            lambda index: processor.get_top_tokens(dense, hidden_pool[index])
        )

    hierarchical_component_us = (
        route_us
        + hierarchical_gemm_us
        + hierarchical_local_us
        + hierarchical_collective_us
    )
    dense_component_us = dense_gemm_us + dense_local_us + dense_collective_us
    return {
        "rows": num_rows,
        "hier_route_us": route_us,
        "hier_gemm_us": hierarchical_gemm_us,
        "hier_local_us": hierarchical_local_us,
        "hier_collective_us": hierarchical_collective_us,
        "hier_component_sum_us": hierarchical_component_us,
        "hier_e2e_us": hierarchical_e2e_us,
        "hier_residual_us": hierarchical_e2e_us - hierarchical_component_us,
        "dense_gemm_us": dense_gemm_us,
        "dense_local_us": dense_local_us,
        "dense_collective_us": dense_collective_us,
        "dense_component_sum_us": dense_component_us,
        "dense_e2e_us": dense_e2e_us,
        "dense_residual_us": dense_e2e_us - dense_component_us,
        "dense_over_hierarchical": dense_e2e_us / hierarchical_e2e_us,
    }


def _print_results(results: list[dict[str, float | int]]) -> None:
    columns = (
        ("rows", "rows"),
        ("route", "hier_route_us"),
        ("h_gemm", "hier_gemm_us"),
        ("h_local", "hier_local_us"),
        ("h_comm", "hier_collective_us"),
        ("h_sum", "hier_component_sum_us"),
        ("h_e2e", "hier_e2e_us"),
        ("h_resid", "hier_residual_us"),
        ("d_gemm", "dense_gemm_us"),
        ("d_local", "dense_local_us"),
        ("d_comm", "dense_collective_us"),
        ("d_sum", "dense_component_sum_us"),
        ("d_e2e", "dense_e2e_us"),
        ("d_resid", "dense_residual_us"),
        ("speedup", "dense_over_hierarchical"),
    )
    print("\nAll latency columns are microseconds; speedup = dense_e2e / hier_e2e.")
    print(" ".join(f"{title:>10}" for title, _ in columns))
    for result in results:
        values = []
        for title, key in columns:
            value = result[key]
            if key == "rows":
                values.append(f"{int(value):>10}")
            elif key == "dense_over_hierarchical":
                values.append(f"{float(value):>9.3f}x")
            else:
                values.append(f"{float(value):>10.2f}")
        print(" ".join(values))


def main(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if args.vocab_size % args.num_clusters != 0:
        raise ValueError("--vocab-size must be divisible by --num-clusters")
    if not 1 <= args.top_k <= args.num_clusters:
        raise ValueError("--top-k must be between 1 and --num-clusters")
    if args.input_pool_size <= 0:
        raise ValueError("--input-pool-size must be positive")
    if args.warmup_replays < 0 or args.iterations <= 0 or args.samples <= 0:
        raise ValueError("warmup must be nonnegative; iterations/samples positive")

    rank, world_size, device, config = _initialize_distributed()
    if args.num_clusters % world_size != 0:
        raise ValueError("--num-clusters must be divisible by TP world size")
    torch.manual_seed(args.seed + rank)
    hierarchical, dense, processor = _build_heads(
        args, rank, world_size, device, config
    )
    if rank == 0:
        selected = args.top_k * (args.vocab_size // args.num_clusters)
        print(
            f"TP={world_size} vocab={args.vocab_size} hidden={args.hidden_size} "
            f"clusters={args.num_clusters} top_k={args.top_k} "
            f"selected_tokens={selected} pool={args.input_pool_size} "
            f"iterations={args.iterations} samples={args.samples}"
        )

    results = [
        _benchmark_rows(
            args,
            num_rows,
            device,
            world_size,
            hierarchical,
            dense,
            processor,
        )
        for num_rows in args.num_rows
    ]
    if rank == 0:
        _print_results(results)
        if args.output_json:
            output = Path(args.output_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(results, indent=2) + "\n")
            print(f"\nWrote {output}")
    dist.barrier(group=get_tp_group().device_group)
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=2688)
    parser.add_argument("--vocab-size", type=int, default=131072)
    parser.add_argument("--num-clusters", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument(
        "--num-rows",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128, 512, 2048],
    )
    parser.add_argument("--input-pool-size", type=int, default=20)
    parser.add_argument("--warmup-replays", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=str)
    main(parser.parse_args())
