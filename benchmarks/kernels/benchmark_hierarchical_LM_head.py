# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare either hierarchical BF16 backend against a dense BF16 head."""

import argparse

import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph

from vllm.model_executor.layers.hierarchical_lm_head import HierarchicalLMHead


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=2688)
    parser.add_argument("--vocab-size", type=int, default=131072)
    parser.add_argument("--num-centroids", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128]
    )
    parser.add_argument(
        "--sidecar", help="Optional trained router; otherwise use random centroids"
    )
    parser.add_argument("--backend", choices=("compiled", "moe"), default="compiled")
    parser.add_argument(
        "--moe-routing",
        choices=("auto", "unsorted", "sorted"),
        default="unsorted",
        help="MoE assignment policy; sorted forces grouping across tokens at any C",
    )
    parser.add_argument(
        "--router-precision",
        choices=("bf16", "w4a4"),
        default="bf16",
        help="Quantize centroid signatures at load time; vocabulary stays BF16",
    )
    parser.add_argument(
        "--report-router",
        action="store_true",
        help="Also time the router and report top-K overlap with BF16 routing",
    )
    args = parser.parse_args()
    torch.manual_seed(0)
    head = HierarchicalLMHead(
        args.hidden_size,
        args.vocab_size,
        args.num_centroids,
        args.top_k,
        backend=args.backend,
        moe_routing=args.moe_routing,
        router_precision=args.router_precision,
    ).to(device="cuda", dtype=torch.bfloat16)
    if args.sidecar:
        head.load_sidecar(args.sidecar)
    weight = torch.randn(
        args.vocab_size, args.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    head.prepare_weights(weight)
    print(f"backend={args.backend}")
    print("vocabulary_precision=bf16")
    print(f"router_precision={head.router_precision}")
    print(f"router_kernel={head.router_kernel_name}")
    if args.backend == "moe":
        print(f"moe_routing={args.moe_routing}")
    header = "batch,dense_us,hierarchical_us,speedup"
    if args.report_router:
        header += ",router_us,router_topk_overlap_pct"
    print(header)
    with torch.inference_mode():
        for batch in args.batch_sizes:
            hidden = torch.randn(
                batch, args.hidden_size, device="cuda", dtype=torch.bfloat16
            )

            def dense():
                return F.linear(hidden, weight).argmax(-1)

            def sparse():
                return head.get_top_tokens(hidden, weight)

            # Compile outside CUDA graph capture, including batch=1 specialization.
            sparse()
            torch.cuda.synchronize()
            dense_ms = do_bench_cudagraph(dense)
            sparse_ms = do_bench_cudagraph(sparse)
            row = (
                f"{batch},{dense_ms * 1000:.3f},{sparse_ms * 1000:.3f},"
                f"{dense_ms / sparse_ms:.3f}"
            )
            if args.report_router:
                router_ms = do_bench_cudagraph(lambda: head.router_logits(hidden))
                reference = head.centroids(hidden).topk(args.top_k, dim=-1).indices
                actual = head.router_logits(hidden).topk(args.top_k, dim=-1).indices
                overlap = (reference[:, :, None] == actual[:, None, :]).any(-1)
                overlap_pct = overlap.float().mean().item() * 100
                row += f",{router_ms * 1000:.3f},{overlap_pct:.2f}"
            print(row)


if __name__ == "__main__":
    main()
