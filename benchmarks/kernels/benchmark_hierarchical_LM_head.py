# SPDX-License-Identifier: Apache-2.0
"""Benchmark the hierarchical MoE LM head against a dense LM head on one GPU.

The hierarchical path includes cluster routing, top-k selection, grouped GEMM
over the selected clusters, and sparse argmax/token decoding. Inputs cycle
through a preallocated random pool inside each captured CUDA graph so repeated
measurements do not continually route to the same cache-hot clusters.
"""

import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.gemma4_mtp import Gemma4MTPMaskedEmbedder
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_open_port


def _initialize_single_gpu_model_parallel(
    device: torch.device,
    benchmark_config: VllmConfig,
) -> None:
    """Set up the world-size-one TP group required by production layers."""
    torch.accelerator.set_device_index(device)
    with set_current_vllm_config(benchmark_config):
        if not torch.distributed.is_initialized():
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=device.index or 0,
                distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            )
        if not model_parallel_is_initialized():
            initialize_model_parallel(tensor_model_parallel_size=1)


def _build_heads(
    args,
    lm_head_weight: torch.Tensor,
    centroid_signatures: torch.Tensor,
    token_ordering: torch.Tensor,
    benchmark_config: VllmConfig,
) -> tuple[Gemma4MTPMaskedEmbedder, ParallelLMHead, LogitsProcessor]:
    """Construct the production hierarchical and dense argmax paths."""
    with set_current_vllm_config(benchmark_config):
        hierarchical_head = Gemma4MTPMaskedEmbedder(
            hidden_size=args.hidden_size,
            vocab_size=args.vocab_size,
            num_centroids=args.num_centroids,
            centroid_intermediate_top_k=args.top_k,
            prefix="benchmark.masked_embedding",
        ).to(device=lm_head_weight.device, dtype=lm_head_weight.dtype)
        dense_head = ParallelLMHead(
            args.vocab_size,
            args.hidden_size,
            params_dtype=lm_head_weight.dtype,
            prefix="benchmark.lm_head",
        ).to(device=lm_head_weight.device)
        logits_processor = LogitsProcessor(args.vocab_size)

    hierarchical_head.use_moe_backend = True
    with torch.no_grad():
        hierarchical_head.centroids.weight.copy_(centroid_signatures)
        hierarchical_head.token_ordering.copy_(token_ordering)
        dense_head.weight_loader(dense_head.weight, lm_head_weight)
    hierarchical_head.build_centroid_weight(dense_head.weight)
    return hierarchical_head, dense_head, logits_processor


def _hierarchical_top_tokens(
    hierarchical_head: Gemma4MTPMaskedEmbedder,
    lm_head_weight: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    return hierarchical_head.get_top_tokens(hidden_states, lm_head_weight)


def _dense_top_tokens(
    logits_processor: LogitsProcessor,
    dense_head: ParallelLMHead,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    return logits_processor.get_top_tokens(dense_head, hidden_states)


def _bench_input_pool(
    fn,
    input_pool: list[torch.Tensor],
    *static_args,
    rep_ms: int,
) -> float:
    """Capture a graph whose unrolled calls rotate through random inputs."""
    input_index = 0

    def invoke():
        nonlocal input_index
        hidden_states = input_pool[input_index % len(input_pool)]
        input_index += 1
        return fn(*static_args, hidden_states)

    median_ms, *_ = triton.testing.do_bench_cudagraph(
        invoke,
        rep=rep_ms,
        quantiles=[0.5, 0.2, 0.8],
    )
    return median_ms


def main(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    if args.input_pool_size < 2:
        raise ValueError("--input-pool-size must be at least 2")
    if args.vocab_size % args.num_centroids != 0:
        raise ValueError("vocab_size must be divisible by num_centroids")
    if not 1 <= args.top_k <= args.num_centroids:
        raise ValueError("top_k must be between 1 and num_centroids")

    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    benchmark_config = VllmConfig(device_config=DeviceConfig(device="cuda"))
    _initialize_single_gpu_model_parallel(device, benchmark_config)
    torch.manual_seed(args.seed)

    vocab_per_centroid = args.vocab_size // args.num_centroids
    num_selected = args.top_k * vocab_per_centroid

    lm_head_weight = torch.randn(
        args.vocab_size,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    centroid_signatures = torch.randn(
        args.num_centroids,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    token_ordering = torch.randperm(args.vocab_size, device=device)
    hierarchical_head, dense_head, logits_processor = _build_heads(
        args,
        lm_head_weight,
        centroid_signatures,
        token_ordering,
        benchmark_config,
    )
    del lm_head_weight, centroid_signatures, token_ordering

    print(
        f"vocab={args.vocab_size} hidden={args.hidden_size} "
        f"clusters={args.num_centroids} top_k={args.top_k} "
        f"selected_tokens={num_selected} "
        f"({100.0 * num_selected / args.vocab_size:.2f}% of vocab) "
        f"input_pool={args.input_pool_size}"
    )
    print(f"{'T':>6} {'hierarchical_ms':>16} {'dense_ms':>12} {'speedup':>10}")

    for num_tokens in args.batch_sizes:
        input_pool = [
            torch.randn(
                num_tokens,
                args.hidden_size,
                device=device,
                dtype=dtype,
            )
            for _ in range(args.input_pool_size)
        ]

        hierarchical_ms = _bench_input_pool(
            _hierarchical_top_tokens,
            input_pool,
            hierarchical_head,
            dense_head.weight,
            rep_ms=args.rep_ms,
        )
        dense_ms = _bench_input_pool(
            _dense_top_tokens,
            input_pool,
            logits_processor,
            dense_head,
            rep_ms=args.rep_ms,
        )
        speedup = dense_ms / hierarchical_ms
        print(
            f"{num_tokens:>6} {hierarchical_ms:>16.4f} "
            f"{dense_ms:>12.4f} {speedup:>9.2f}x"
        )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=2688)
    parser.add_argument("--vocab-size", type=int, default=131072)
    parser.add_argument("--num-centroids", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument("--input-pool-size", type=int, default=20)
    parser.add_argument(
        "--rep-ms",
        type=int,
        default=20,
        help="Approximate duration in milliseconds of each captured replay.",
    )
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
