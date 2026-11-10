# MTP (Multi-Token Prediction)

MTP is a speculative decoding method where the target model includes native
multi-token prediction capability. Unlike draft-model-based methods, you do not
need to provide a separate draft model.

MTP is useful when:

- Your model natively supports MTP.
- You want model-based speculative decoding with minimal extra configuration.

## Gemma 4 Assistant Models

Gemma 4 assistant checkpoints use vLLM's Gemma 4 MTP path. They are not generic
draft models, even though they are passed through the `model` field in
`--speculative-config`.

Use `"method": "mtp"` when serving Gemma 4 with an assistant checkpoint:

```bash
vllm serve google/gemma-4-E2B-it \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --speculative-config '{"method":"mtp","model":"gg-hf-am/gemma-4-E2B-it-assistant","num_speculative_tokens":1}'
```

The E2B, E4B, 12B, 26B-A4B, and 31B Gemma 4 IT assistant checkpoints are supported.
Tower-based variants use `model_type: gemma4_assistant` and the encoder-free
Gemma 4 Unified variant (12B) uses `model_type: gemma4_unified_assistant`.
vLLM maps both to `Gemma4MTPModel` internally and wires the assistant layers
to share KV cache with the target model.

If an older vLLM release logs `SpeculativeConfig(method='draft_model', ...)`
for a Gemma 4 assistant checkpoint, that release is treating the assistant as a
generic draft model and may fail during initialization for multimodal Gemma 4
targets. Upgrade to a version with Gemma 4 MTP support instead.

## Offline Example

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="XiaomiMiMo/MiMo-7B-Base",
    tensor_parallel_size=1,
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## Online Example

```bash
vllm serve XiaomiMiMo/MiMo-7B-Base \
    --tensor-parallel-size 1 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## Notes

- MTP only works for model families that support MTP in vLLM.
- `num_speculative_tokens` controls speculative depth. A small value like `1`
  is a good default to start with.
- If your model does not support MTP, use another method such as EAGLE or draft
  model speculation.
- When the MTP head shares the target's unquantized lm_head (most MTP models),
  `"draft_lm_head_quantization": "fp8"`, `"nvfp4"`, or `"nvfp4_w4a4"` in `--speculative-config`
  gives the drafter a quantized copy of that head (Model Runner V2). The target
  still verifies with its own head, so outputs are unchanged; only the
  acceptance length can move. It helps most on bandwidth-bound GPUs with a
  large vocabulary. The copy costs about one byte (FP8) or 0.56 bytes (NVFP4)
  per lm_head weight, e.g. 0.6 or 0.35 GiB for a 248k x 2560 head; it is
  counted in the model weights, so with `--kv-cache-memory-bytes` leave room
  for it.

### Dense NVFP4 draft head

`draft_lm_head_quantization="nvfp4"` keeps activations in the model dtype
(W4A16). `"nvfp4_w4a4"` quantizes activations dynamically in blocks of 16 and
uses a native FP4 tensor-core kernel (SM100+ hardware). Its activation global
scale is 1.0, matching the earlier dense Nemotron experiment. The two modes
share the same weight quantization; they do not quantize the target head.
Weights are quantized once from the loaded BF16/FP16 target by default, without
calibration data. Kernel selection follows `--linear-backend`.

To reuse an existing NVFP4 artifact at TP=1, also set
`draft_lm_head_quantization_path` to its file or directory. A directory resolves
to `lm_head_w4a16.safetensors`, containing `weight` (packed uint8 E2M1),
`weight_scale` (logical E4M3 scales per 16 weights), and `weight_scale_2`
(one positive FP32 dequantization scale). Vocabulary padding is applied at
load time. Reloading an artifact-backed copy rereads that artifact; update the
artifact as well when changing source weights. Reloads preserve graph storage.
The option is supported wherever a target LM head can be shared by the draft.

For example:

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --tensor-parallel-size 1 \
  --kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a16":"flashinfer_cutedsl","nvfp4_w4a4":"flashinfer_cutedsl"}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":7,"draft_lm_head_quantization":"nvfp4_w4a4","draft_lm_head_quantization_path":"./quantized/nm3.5nano/"}'
```

In the parent project's single-GPU E2E launcher, the earlier artifact-backed
commands are supported again:

```bash
sbatch job_Nemotron35_MTP_hierarchical_AL_singleGPU.sh \
  ./quantized/nm3.5nano/ 1 logs/dense_w4a16_1
sbatch job_Nemotron35_MTP_hierarchical_AL_singleGPU.sh \
  --w4a4 ./quantized/nm3.5nano/ 1 logs/dense_w4a4_1
```

Both launcher modes select FlashInfer CuTe DSL through NVFP4-specific backend
overrides, preserving the target backbone's BF16 kernel selection. For
quantization at load time,
replace the artifact argument with `--w4a16`, or omit it after `--w4a4`.
