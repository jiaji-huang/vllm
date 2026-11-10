# Hierarchical LM head backends (experimental)

The `hlm_with_w4` experiment uses the model-independent
`vllm.model_executor.layers.hierarchical_lm_head.HierarchicalLMHead` layer.
Gemma4 MTP and NemotronH MTP import this layer directly. The layer owns centroid
routing and vocabulary ordering; the caller supplies the unsharded floating-point
LM-head weights. Choose `backend="compiled"` (default) or `backend="moe"`
when constructing the layer:

- `compiled` compiles `(lm_head_weights[sel] * hidden[:, None, :]).sum(-1)`
  with `dynamic=True`. Token count is dynamic; model dimensions are marked
  static so Inductor can tile each model separately. The gather stays inside
  compilation to allow fusion.
- `moe` reuses the existing unquantized Triton MoE grouped-GEMM kernel for
  selected clusters. There is no activation, second GEMM, or routing-weight sum.
  It retains a full copy of the head weights arranged by cluster. Call
  `prepare_weights(lm_head_weight)` after loading or reloading weights or token
  ordering, before CUDA graph capture. Gemma and Nemotron do this automatically.
  Reloads refill the buffer in place to preserve captured graph addresses.

Actual lowering and performance depend on PyTorch and hardware. Neither backend
requires changes to CUDA sources. The optional NVFP4 router below works with both BF16 scoring backends.

MoE assignment is controlled by `moe_routing`:

- `auto` calls vLLM's `_prepare_expert_assignment` directly, using
  its heuristic to choose grouped or ungrouped assignment. For this unquantized
  path it avoids grouping when `batch * top_k * 4 <= num_clusters`. When it
  chooses grouping, the upstream CUDA alignment kernel's expert-count limit
  still applies. At C4096/K32, batches above 32 can hit that limit; use an
  explicit `sorted` or `unsorted` policy for those benchmarks.
- `unsorted` (default) forces one block per token/cluster pair at any cluster count.
- `sorted` forces grouping of routed rows across tokens by cluster, then pads
  each group to the GEMM tile size. It uses PyTorch sorting and prefix sums,
  supports C4096, and keeps buffer shapes fixed for CUDA graph capture. The GEMM
  writes results back to their original token/top-K slots.

Sorting adds work. It may save weight loads and GEMM blocks when many tokens
select the same cluster, but performance must include sorting and padding costs.
For uniformly distributed routes, average rows per cluster are `batch * K / C`:
0.25 at batch 32, 4 at batch 512, and 16 at batch 2048 with C4096/K32.
The initial sorted implementation is an experiment, not a tuned routing kernel.

For NemotronH MTP on a single GPU, set these variables before launching vLLM:

```bash
export VLLM_NEMOTRON_MTP_CLUSTERED_LM_HEAD=1
export VLLM_NEMOTRON_MTP_CLUSTERED_LM_HEAD_PATH=/path/to/router/model.pt
export VLLM_NEMOTRON_MTP_LM_HEAD_NUM_CLUSTERS=4096
export VLLM_NEMOTRON_MTP_LM_HEAD_CLUSTER_TOP_K=32
export VLLM_NEMOTRON_MTP_SPARSE_HEAD_BACKEND=compiled  # or moe
# For backend=moe, optionally force cross-token grouping:
# export VLLM_NEMOTRON_MTP_MOE_ROUTING=sorted  # unsorted (default), auto, sorted
```

For Gemma4 MTP with ordered embeddings, set the assistant HF config field
`hierarchical_lm_head_backend` to `compiled` (default) or `moe`, and optionally
`hierarchical_lm_head_moe_routing` to `unsorted` (default), `auto`, or `sorted`.
The Nemotron single-GPU job script honors the backend environment variable above.

The feature is disabled by default. A router sidecar is required when enabled.
The sidecar may contain `centroids.weight` and `token_ordering`, their
`masked_embedding.` variants, or training keys `cluster_signature` and
`token_to_cluster`. Clusters must be balanced and cover the entire vocabulary.
The head reuses the target model's shared BF16 weights. Nemotron tensor
parallelism greater than one is unsupported.

`get_top_tokens` returns selected token IDs without allocating full-vocabulary
logits. `forward` returns masked full-vocabulary logits. Selection is approximate
and may change draft acceptance compared with a dense head. The multiply/reduce
expression can also round differently from a matrix multiplication.

Benchmark compilation separately from steady-state CUDA graph execution:

```bash
python benchmarks/kernels/benchmark_hierarchical_LM_head.py \
  --num-centroids 4096 --top-k 32 --backend moe --moe-routing sorted \
  --batch-sizes 1 8 32 128 512 1024 2048
```

This benchmark includes routing, scoring and argmax for both heads. Random weights
measure execution time, not acceptance quality; use a trained sidecar and a real
speculative decoding workload to measure acceptance and end-to-end throughput.


## W4A4 centroid router with BF16 vocabulary scoring

Set `router_precision="w4a4"` to quantize the loaded centroid signatures once
before inference. The trained sidecar stays unchanged: load its BF16 signatures,
move the layer to its final CUDA device/dtype, then call
`prepare_weights(lm_head_weight)` (or `prepare_router()` for compiled scoring).
Gemma and Nemotron prepare the router automatically after loading.

The router uses vLLM's native NVFP4 linear-kernel selection, including FlashInfer
CuTe DSL on supported hardware. Weights use packed E2M1 values, E4M3 scales per
16 values and one FP32 global scale derived from their maximum magnitude.
No calibration dataset or offline conversion is required. Input activations are
quantized on each projection with global quantization multiplier 1.0.
This requires a native W4A4 NVFP4 kernel; weight-only and emulated fallbacks are
rejected. Native hardware support is generally SM100 or newer.

Only centroid routing is quantized. Selected vocabulary rows always use the
original BF16/FP weights through the compiled or MoE scoring backend. Derived
packed router weights are excluded from source checkpoint state. Reloading
same-layout weights refills their storage to preserve CUDA graph addresses.
BF16 routing remains the default. Quantized routing can change the selected
clusters and acceptance length, so evaluate it on the intended workload.

For Nemotron, set `VLLM_NEMOTRON_MTP_ROUTER_PRECISION=w4a4`. For Gemma, set
`hierarchical_lm_head_router_precision` to `w4a4` in the assistant HF config.
From the parent project, launch with the existing BF16 router checkpoint:

```bash
VLLM_NEMOTRON_MTP_SPARSE_HEAD_BACKEND=compiled \
VLLM_NEMOTRON_MTP_ROUTER_PRECISION=w4a4 \
sbatch job_Nemotron35_MTP_hierarchical_AL_singleGPU.sh \
  C4096_drl5/step_010000.pt 4096 32 1 \
  logs/compare_backends/qualitative/compiled_router_w4a4_1
```

For the head-only benchmark, include router latency and top-K overlap against
BF16 routing (the comparison is outside the head timing):

```bash
python benchmarks/kernels/benchmark_hierarchical_LM_head.py \
  --backend compiled --router-precision w4a4 --report-router \
  --num-centroids 4096 --top-k 32 --batch-sizes 1 2 4 8 16 32
```

The earlier compiled packed-vocabulary W4A4 experiment is removed. Old
`VLLM_NEMOTRON_MTP_NVFP4_LM_HEAD_PATH` settings and comma-separated
router/quantized-vocabulary sidecars are no longer supported by this launcher.
