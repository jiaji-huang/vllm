# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online-quantized copies of an already-loaded lm_head."""

from itertools import chain
from pathlib import Path
from typing import Literal
from weakref import WeakKeyDictionary

import torch
from torch.nn import Module

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PtpcOnlineLinearMethod,
    OnlineLinearBase,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    FLOAT4_E2M1_MAX,
    ref_nvfp4_quant,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import weight_amax
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
from vllm.model_executor.utils import replace_parameter
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


def _quantize_nvfp4(
    weight: torch.Tensor, chunk_rows: int = 8192
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4-quantize the rows of `weight` without the SM100+ quant kernel.

    Returns packed E2M1 codes (low nibble first), E4M3 scales per 16 values and
    the fp32 dequantization global scale, in the checkpoint layout.
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    global_scale = (weight_amax(weight).float() / (FLOAT4_E2M1_MAX * fp8_max)).clamp(
        min=1e-12
    )
    inv_global_scale = (1.0 / global_scale).reshape(())
    # E2M1 magnitudes 0, 0.5, ..., 6 doubled to integers, mapped to their codes.
    code_of = torch.zeros(13, dtype=torch.uint8, device=weight.device)
    code_of[torch.tensor([0, 1, 2, 3, 4, 6, 8, 12])] = torch.arange(
        8, dtype=torch.uint8
    ).to(weight.device)
    n, k = weight.shape
    packed = torch.empty(n, k // 2, dtype=torch.uint8, device=weight.device)
    scales = torch.empty(n, k // 16, dtype=torch.float8_e4m3fn, device=weight.device)
    for start in range(0, n, chunk_rows):
        rows = slice(start, start + chunk_rows)
        values, block_scales = ref_nvfp4_quant(weight[rows], inv_global_scale, 16)
        codes = code_of[(values.abs() * 2).long()] | ((values < 0).to(torch.uint8) << 3)
        packed[rows] = codes[:, 0::2] | (codes[:, 1::2] << 4)
        scales[rows] = block_scales.to(torch.float8_e4m3fn)
    return packed, scales, global_scale


def _nvfp4_weights(
    layer: Module, artifact_path: str | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if artifact_path is None:
        return _quantize_nvfp4(layer.weight)
    from safetensors.torch import load_file

    if layer.tp_size != 1:
        raise ValueError("An NVFP4 lm_head artifact currently requires TP=1")
    path = Path(artifact_path).expanduser()
    if path.is_dir():
        path /= "lm_head_w4a16.safetensors"
    tensors = load_file(str(path), device="cpu")
    vocab, hidden = layer.org_vocab_size, layer.embedding_dim
    expected = {
        "weight": ((vocab, hidden // 2), torch.uint8),
        "weight_scale": ((vocab, hidden // 16), torch.float8_e4m3fn),
        "weight_scale_2": ((1,), torch.float32),
    }
    for name, (shape, dtype) in expected.items():
        value = tensors.get(name)
        if value is None or tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(
                f"NVFP4 lm_head {name} must have shape {shape} and dtype {dtype}"
            )
    scale = tensors["weight_scale_2"]
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("NVFP4 lm_head global scale must be finite and positive")
    device = layer.weight.device
    values = []
    for name in ("weight", "weight_scale"):
        value = tensors[name]
        padded = torch.zeros(
            layer.num_embeddings_per_partition,
            value.shape[1],
            dtype=value.dtype,
            device=device,
        )
        padded[:vocab].copy_(value)
        values.append(padded)
    logger.info("Loaded NVFP4 lm_head artifact from %s", path)
    return values[0], values[1], scale.to(device)


class _LMHeadNvfp4Method(OnlineLinearBase):
    """NVFP4 draft head with BF16/FP16 activations (W4A16)."""

    use_a16 = True

    def __init__(self, artifact_path: str | None = None):
        super().__init__()
        self.artifact_path = artifact_path

    def process_weights_after_loading(self, layer: Module) -> None:
        packed, scales, global_scale = _nvfp4_weights(layer, self.artifact_path)
        replace_parameter(layer, "weight", packed)
        replace_parameter(layer, "weight_scale", scales)
        replace_parameter(layer, "weight_global_scale", global_scale)
        self.kernel = init_nvfp4_linear_kernel(use_a16=self.use_a16)
        if not self.use_a16:
            # Match the earlier dense W4A4 path: dynamic block scales, global 1.
            if type(self.kernel).__name__ not in {
                "FlashInferCuteDslNvFp4LinearKernel",
                "FlashInferCutlassNvFp4LinearKernel",
                "FlashInferB12xNvFp4LinearKernel",
                "FlashInferTrtllmNvFp4LinearKernel",
                "FlashInferCudnnNvFp4LinearKernel",
                "CutlassNvFp4LinearKernel",
                "B12xNvFp4LinearKernel",
                "TorchNvFp4LinearKernel",
                "FbgemmNvFp4LinearKernel",
            }:
                raise ValueError("Dense W4A4 requires a native NVFP4 GEMM kernel")
            input_scale = torch.ones(1, dtype=torch.float32, device=packed.device)
            replace_parameter(layer, "input_global_scale", input_scale)
            replace_parameter(
                layer, "input_global_scale_inv", input_scale.reciprocal()
            )
            replace_parameter(layer, "alpha", global_scale * input_scale)
        self.kernel.process_weights_after_loading(layer)
        logger.info(
            "Dense draft LM head: %s, kernel=%s, weights=%s",
            "W4A16" if self.use_a16 else "W4A4",
            type(self.kernel).__name__,
            self.artifact_path or "quantized at load time",
        )

    def apply(
        self, layer: Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


class _LMHeadNvfp4W4A4Method(_LMHeadNvfp4Method):
    """Native NVFP4 tensor-core projection with quantized activations."""

    use_a16 = False


_LM_HEAD_METHODS: dict[str, type[OnlineLinearBase]] = {
    "fp8": Fp8PtpcOnlineLinearMethod,
    "nvfp4": _LMHeadNvfp4Method,
    "nvfp4_w4a4": _LMHeadNvfp4W4A4Method,
}


def _quantize_copy(
    lm_head: ParallelLMHead, quantization: str, artifact_path: str | None = None
) -> ParallelLMHead:
    if artifact_path is not None and quantization not in ("nvfp4", "nvfp4_w4a4"):
        raise ValueError("An lm_head artifact requires NVFP4 quantization")
    if (
        not isinstance(
            lm_head.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        )
        or getattr(lm_head, "bias", None) is not None
        or lm_head.weight.dtype not in (torch.float16, torch.bfloat16)
    ):
        raise ValueError(
            "Quantizing an lm_head copy needs an unquantized fp16/bf16 lm_head "
            "without bias."
        )
    # Online methods take their output dtype from the default dtype, which is
    # only the model dtype while the model itself is being built.
    dtype = lm_head.weight.dtype
    with set_default_torch_dtype(dtype):
        method_cls = _LM_HEAD_METHODS[quantization]
        method = (
            method_cls(artifact_path=artifact_path)
            if issubclass(method_cls, _LMHeadNvfp4Method)
            else method_cls()
        )
        head = ParallelLMHead(
            lm_head.num_embeddings,
            lm_head.embedding_dim,
            params_dtype=dtype,
            org_num_embeddings=lm_head.org_vocab_size,
            padding_size=lm_head.padding_size,
            prefix=lm_head.prefix,
            disable_tp=lm_head.disable_tp,
            quant_method=method,
        )
    # Never loaded: a new param sharing the source storage, never a copy of it.
    head.weight = torch.nn.Parameter(lm_head.weight.data, requires_grad=False)
    head.quant_method.process_weights_after_loading(head)
    # The copy is never loaded. The online loader wrapper captures the layer,
    # so keeping it would leave each copy to the cycle collector.
    for param in head.parameters(recurse=False):
        if hasattr(param, "weight_loader"):
            del param.weight_loader
    return head


# Each quantized copy maps to the head it is derived from. Weak keys, and no
# module attribute, so the copy never owns the source as a submodule.
_SOURCES: WeakKeyDictionary[
    Module, tuple[ParallelLMHead, str, str | None]
] = WeakKeyDictionary()


def quantized_lm_head_copy(
    lm_head: ParallelLMHead,
    quantization: Literal["fp8", "nvfp4", "nvfp4_w4a4"],
    artifact_path: str | None = None,
) -> ParallelLMHead:
    """A ParallelLMHead holding this rank's shard of `lm_head`, quantized.

    "fp8" uses per-row weight and per-token activation scales on the platform's
    FP8 linear kernel; "nvfp4" is weight-only on the
    platform's W4A16 kernel. "nvfp4_w4a4" also quantizes activations and uses
    native FP4 tensor cores. NVFP4 weights can come from an optional TP=1
    artifact in the earlier dense-head format. `lm_head` is left
    unchanged and no bf16 copy of its weight is made. The copy is derived state:
    `refresh_quantized_lm_heads` re-derives it after `lm_head` changes.
    """
    head = _quantize_copy(lm_head, quantization, artifact_path)
    # Keep it out of layerwise reload, which would re-run
    # process_weights_after_loading on the already-quantized weight.
    get_layerwise_info(head).reset()
    _SOURCES[head] = (lm_head, quantization, artifact_path)
    logger.info(
        "Quantized a %s copy of the lm_head (%d x %d per rank).",
        quantization,
        head.output_size_per_partition,
        head.input_size_per_partition,
    )
    return head


@torch.no_grad()
def refresh_quantized_lm_heads(model: Module) -> None:
    """Re-derive the quantized lm_head copies in `model` from their source heads.

    Copies in place, so CUDA graphs keep pointing at valid tensors.
    """
    for module in model.modules():
        if module not in _SOURCES:
            continue
        fresh = _quantize_copy(*_SOURCES[module])
        for name, tensor in chain(
            fresh.named_parameters(recurse=False), fresh.named_buffers(recurse=False)
        ):
            getattr(module, name).copy_(tensor)


# (parent, attribute, copy) for each copy swapped out by share_source_lm_heads.
_SWAPPED: dict[Module, list[tuple[Module, str, Module]]] = {}


def share_source_lm_heads(model: Module) -> None:
    """Replace each quantized copy in `model` with its source head.

    `model` then loads weights like a drafter sharing the unquantized head,
    until `restore_quantized_lm_heads` swaps the copies back in.
    """
    swapped = [
        (module, name, child)
        for module in model.modules()
        for name, child in module.named_children()
        if child in _SOURCES
    ]
    for module, name, child in swapped:
        setattr(module, name, _SOURCES[child][0])
    _SWAPPED[model] = swapped


def restore_quantized_lm_heads(model: Module, refresh: bool = True) -> None:
    """Undo `share_source_lm_heads`, then re-derive the copies in `model`.

    Args:
        model: The model passed to `share_source_lm_heads`, if any.
        refresh: Whether to re-derive the copies; False when the source heads
            may be mid-reload.

    """
    for module, name, head in _SWAPPED.pop(model, ()):
        setattr(module, name, head)
    if refresh:
        refresh_quantized_lm_heads(model)
