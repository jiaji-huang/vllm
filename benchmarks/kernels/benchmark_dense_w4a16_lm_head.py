# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare dense BF16 and artifact-backed NVFP4 W4A16/W4A4 LM-head latency."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton.testing
from safetensors import safe_open

from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.nemotron_h_mtp import NemotronHDenseNvFp4LMHeadMethod


def build_heads(args):
    with safe_open(args.bf16_weight_path, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(args.bf16_weight_key)
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise ValueError("The BF16 reference must be a two-dimensional BF16 weight")
    vocab, hidden = weight.shape
    with torch.device("cuda"):
        bf16 = ParallelLMHead(
            vocab, hidden, params_dtype=torch.bfloat16, disable_tp=True
        )
        w4a16 = ParallelLMHead(
            vocab, hidden, params_dtype=torch.bfloat16, disable_tp=True
        )
    bf16.weight.zero_()
    bf16.weight[:vocab].copy_(weight)
    del weight
    bf16.quant_method.process_weights_after_loading(bf16)
    w4a16.quant_method = NemotronHDenseNvFp4LMHeadMethod(
        args.artifact,
        use_a16=args.precision == "w4a16",
        activation_global_scale=args.activation_global_scale,
    )
    w4a16.quant_method.process_weights_after_loading(w4a16)
    return bf16, w4a16, LogitsProcessor(vocab), vocab, hidden


def bench_pool(fn, pool, rep_ms):
    # Distinct inputs avoid timing only one fixed hidden-state buffer.
    def run_pool():
        for x in pool:
            fn(x)

    return triton.testing.do_bench_cudagraph(run_pool, rep=rep_ms) / len(pool)


@torch.inference_mode()
def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    backend_choice = args.backend
    if backend_choice is None:
        backend_choice = "cutlass" if args.precision == "w4a4" else "auto"
    if backend_choice == "flashinfer":
        backend_choice = "flashinfer_cutedsl"
    config = VllmConfig(kernel_config=KernelConfig(linear_backend=backend_choice))
    with set_current_vllm_config(config):
        bf16, w4a16, processor, vocab, hidden = build_heads(args)
        backend = type(w4a16.quant_method.kernel).__name__
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(
            f"Vocab={vocab}, hidden={hidden}, precision={args.precision}, "
            f"backend={backend}"
        )
        print("CUDA graphs; excludes artifact loading, repacking, and compilation.")
        print(f"Speedup = BF16 latency / {args.precision.upper()} latency.")
        if args.precision == "w4a4":
            print(
                f"Projection includes activation quantization; "
                f"activation dequantization global scale={args.activation_global_scale}"
            )
        print(
            f"{'Tokens':>7} {'Operation':>16} {'BF16 us':>12} "
            f"{(args.precision.upper() + ' us'):>12} {'Speedup':>9}"
        )
        rows = []
        for batch in args.batch_sizes:
            pool = [
                torch.randn(batch, hidden, device="cuda", dtype=torch.bfloat16)
                for _ in range(args.input_pool_size)
            ]
            for operation in ("projection", "projection+argmax"):
                if operation == "projection":
                    functions = [
                        lambda x, head=head: head.quant_method.apply(head, x)
                        for head in (bf16, w4a16)
                    ]
                else:
                    functions = [
                        lambda x, head=head: processor.get_top_tokens(head, x)
                        for head in (bf16, w4a16)
                    ]
                for fn in functions:
                    for _ in range(3):
                        for x in pool:
                            fn(x)
                torch.cuda.synchronize()
                samples = [[], []]
                for repeat in range(args.repeats):
                    # Alternate order to reduce warmup/clock drift bias.
                    for index in (0, 1) if repeat % 2 == 0 else (1, 0):
                        samples[index].append(
                            bench_pool(functions[index], pool, args.rep_ms)
                        )
                bf16_ms, w4a16_ms = map(statistics.median, samples)
                speedup = bf16_ms / w4a16_ms
                print(
                    f"{batch:>7} {operation:>16} {bf16_ms * 1000:>12.2f} "
                    f"{w4a16_ms * 1000:>12.2f} {speedup:>8.3f}x",
                    flush=True,
                )
                rows.append(
                    {
                        "tokens": batch,
                        "operation": operation,
                        "bf16_ms": bf16_ms,
                        f"{args.precision}_ms": w4a16_ms,
                        "speedup": speedup,
                        "bf16_samples_ms": samples[0],
                        f"{args.precision}_samples_ms": samples[1],
                    }
                )
        if args.output_json:
            path = Path(args.output_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "gpu": torch.cuda.get_device_name(),
                        "backend": backend,
                        "precision": args.precision,
                        "vocab_size": vocab,
                        "hidden_size": hidden,
                        "arguments": vars(args),
                        "results": rows,
                    },
                    indent=2,
                )
                + "\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", required=True, help="NVFP4 weight artifact file or its directory"
    )
    parser.add_argument(
        "--bf16-weight-path",
        required=True,
        help="Original BF16 checkpoint shard containing the LM head",
    )
    parser.add_argument("--bf16-weight-key", default="lm_head.weight")
    parser.add_argument("--precision", choices=("w4a16", "w4a4"), default="w4a16")
    parser.add_argument(
        "--activation-global-scale",
        type=float,
        default=1.0,
        help="W4A4 activation dequantization multiplier; default 1.0 is uncalibrated",
    )
    parser.add_argument(
        "--backend",
        choices=(
            "auto",
            "cutlass",
            "marlin",
            "flashinfer",
            "flashinfer_cutedsl",
            "flashinfer_cutlass",
            "flashinfer_trtllm",
            "flashinfer_cudnn",
            "humming",
        ),
        help="Default: auto for W4A16, cutlass for W4A4",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument("--input-pool-size", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if any(
        value <= 0
        for value in (
            *args.batch_sizes,
            args.input_pool_size,
            args.repeats,
            args.rep_ms,
        )
    ):
        parser.error("Batch sizes, pool size, repeats, and rep-ms must be positive")
    run(args)


if __name__ == "__main__":
    main()
