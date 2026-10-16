# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NemotronH-MTP model with attention layers."""

import math
import typing
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from vllm import envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.config.parallel import ParallelConfig
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import (
    WeightsMapper,
    get_draft_quant_config,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig

from .interfaces import SupportsPP, SupportsQuant
from .nemotron_h import (
    NemotronHAttentionDecoderLayer,
    NemotronHMoEDecoderLayer,
)

logger = init_logger(__name__)


class NemotronHDenseNvFp4LMHeadMethod(QuantizeMethodBase):
    """Dense artifact-backed NVFP4 projection with BF16 or FP4 activations."""

    def __init__(
        self,
        artifact_path: str,
        *,
        use_a16: bool = True,
        activation_global_scale: float = 1.0,
    ) -> None:
        if not math.isfinite(activation_global_scale) or activation_global_scale <= 0:
            raise ValueError("Activation global scale must be finite and positive")
        self.use_a16 = use_a16
        self.precision = "W4A16" if use_a16 else "W4A4"
        self.activation_global_scale = activation_global_scale
        path = Path(artifact_path).expanduser()
        if path.is_dir():
            path = path / "lm_head_w4a16.safetensors"
        if not path.is_file():
            raise ValueError(f"NVFP4 LM-head artifact does not exist: {path}")
        self.artifact_path = path
        self.kernel = init_nvfp4_linear_kernel(use_a16=use_a16)
        if not use_a16:
            from vllm.model_executor.kernels.linear.nvfp4.cutlass import (
                CutlassNvFp4LinearKernel,
            )
            from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
                FlashInferB12xNvFp4LinearKernel,
                FlashInferCudnnNvFp4LinearKernel,
                FlashInferCuteDslNvFp4LinearKernel,
                FlashInferCutlassNvFp4LinearKernel,
                FlashInferTrtllmNvFp4LinearKernel,
            )

            if not isinstance(
                self.kernel,
                (
                    CutlassNvFp4LinearKernel,
                    FlashInferCuteDslNvFp4LinearKernel,
                    FlashInferCutlassNvFp4LinearKernel,
                    FlashInferTrtllmNvFp4LinearKernel,
                    FlashInferCudnnNvFp4LinearKernel,
                    FlashInferB12xNvFp4LinearKernel,
                ),
            ):
                raise ValueError("Dense W4A4 requires a native NVFP4 GEMM backend")

    def create_weights(self, layer: nn.Module, *args, **kwargs) -> None:
        UnquantizedEmbeddingMethod().create_weights(layer, *args, **kwargs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        tensors = load_file(str(self.artifact_path), device="cpu")
        vocab, hidden = layer.org_vocab_size, layer.embedding_dim
        expected = {
            "weight": ((vocab, hidden // 2), torch.uint8),
            "weight_scale": ((vocab, hidden // 16), torch.float8_e4m3fn),
            "weight_scale_2": ((1,), torch.float32),
        }
        for name, (shape, dtype) in expected.items():
            value = tensors.get(name)
            if value is None or tuple(value.shape) != shape or value.dtype != dtype:
                actual = None if value is None else (value.shape, value.dtype)
                raise ValueError(
                    f"NVFP4 LM-head {name} must have shape {shape} and dtype "
                    f"{dtype}, got {actual}"
                )
        global_scale = tensors["weight_scale_2"]
        if not torch.isfinite(global_scale).all() or not (global_scale > 0).all():
            raise ValueError("NVFP4 LM-head global scale must be finite and positive")

        device = layer.weight.device
        padded_vocab = layer.num_embeddings_per_partition
        for name in ("weight", "weight_scale"):
            value = tensors[name]
            padded = torch.zeros(
                (padded_vocab, value.shape[1]), dtype=value.dtype, device=device
            )
            padded[:vocab].copy_(value)
            setattr(layer, name, nn.Parameter(padded, requires_grad=False))
        layer.weight_global_scale = nn.Parameter(
            global_scale.to(device), requires_grad=False
        )
        layer.input_size_per_partition = hidden
        layer.output_size_per_partition = padded_vocab
        if not self.use_a16:
            # Global scales are dequantization multipliers; the quantizer takes
            # their reciprocal. Per-16-element activation scales remain dynamic.
            layer.input_global_scale = nn.Parameter(
                torch.tensor(
                    [self.activation_global_scale], device=device, dtype=torch.float32
                ),
                requires_grad=False,
            )
            layer.input_global_scale_inv = nn.Parameter(
                layer.input_global_scale.reciprocal(),
                requires_grad=False,
            )
            layer.alpha = nn.Parameter(
                layer.input_global_scale * layer.weight_global_scale,
                requires_grad=False,
            )
        self.kernel.process_weights_after_loading(layer)
        logger.info(
            "Loaded dense MTP %s NVFP4 LM head from %s using %s "
            "(BF16 input, vocab=%d, hidden=%d, activation_global_scale=%g).",
            self.precision,
            self.artifact_path,
            type(self.kernel).__name__,
            vocab,
            hidden,
            self.activation_global_scale,
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise ValueError(f"Dense MTP NVFP4 requires BF16 input, got {x.dtype}")
        return self.kernel.apply_weights(layer, x, bias)


class NemotronHDenseW4A16LMHeadMethod(NemotronHDenseNvFp4LMHeadMethod):
    """Compatibility adapter for the existing dense W4A16 path."""


class NemotronHMTPAttentionDecoderLayer(NemotronHAttentionDecoderLayer):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
        has_start_projections: bool = False,
        has_end_norm: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            parallel_config=parallel_config,
            prefix=prefix,
        )
        self.has_start_projections = has_start_projections
        self.has_end_norm = has_end_norm

        if has_start_projections:
            self.enorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
            self.hnorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

            # Fusion layer to combine embeddings with target hidden states
            self.eh_proj = ColumnParallelLinear(
                input_size=config.hidden_size * 2,
                output_size=config.hidden_size,
                bias=False,
                gather_output=True,
                params_dtype=config.dtype
                if hasattr(config, "dtype")
                else torch.bfloat16,
                quant_config=quant_config,
                prefix=f"{prefix}.eh_proj",
            )

        if has_end_norm:
            self.final_layernorm = RMSNorm(
                config.hidden_size,
                eps=getattr(config, "layer_norm_epsilon", 1e-5),
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Start projections (Fusion)
        if self.has_start_projections:
            # Normalize both inputs before fusion
            assert inputs_embeds is not None
            inputs_embeds_normed = self.enorm(inputs_embeds)
            previous_hidden_states_normed = self.hnorm(hidden_states)

            # Fuse via concatenation and linear projection
            fused = torch.cat(
                [inputs_embeds_normed, previous_hidden_states_normed], dim=-1
            )
            hidden_states, _ = self.eh_proj(fused)

        # Call parent forward (Attention)
        # Parent forward expects: hidden_states, residual
        hidden_states, residual = super().forward(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

        # End norm
        if self.has_end_norm:
            if residual is not None:
                hidden_states = hidden_states + residual
                residual = None  # Consumed residual

            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, residual


class NemotronHMTPMoEDecoderLayer(NemotronHMoEDecoderLayer):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
        has_start_projections: bool = False,
        has_end_norm: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            parallel_config=parallel_config,
            prefix=prefix,
        )
        self.has_start_projections = has_start_projections
        self.has_end_norm = has_end_norm

        if has_start_projections:
            self.enorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
            self.hnorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

            # Fusion layer to combine embeddings with target hidden states
            self.eh_proj = ColumnParallelLinear(
                input_size=config.hidden_size * 2,
                output_size=config.hidden_size,
                bias=False,
                gather_output=True,
                params_dtype=config.dtype
                if hasattr(config, "dtype")
                else torch.bfloat16,
                quant_config=quant_config,
                prefix=f"{prefix}.eh_proj",
            )

        if has_end_norm:
            self.final_layernorm = RMSNorm(
                config.hidden_size,
                eps=getattr(config, "layer_norm_epsilon", 1e-5),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Start projections (Fusion)
        if self.has_start_projections:
            # Normalize both inputs before fusion
            assert inputs_embeds is not None
            inputs_embeds_normed = self.enorm(inputs_embeds)
            previous_hidden_states_normed = self.hnorm(hidden_states)

            # Fuse via concatenation and linear projection
            fused = torch.cat(
                [inputs_embeds_normed, previous_hidden_states_normed], dim=-1
            )
            hidden_states, _ = self.eh_proj(fused)

        # Call parent forward (MoE)
        hidden_states, residual = super().forward(
            hidden_states=hidden_states,
            residual=residual,
        )

        # End norm
        if self.has_end_norm:
            if residual is not None:
                hidden_states = hidden_states + residual
                residual = None  # Consumed residual

            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, residual


@support_torch_compile
class NemotronHMultiTokenPredictor(nn.Module):
    """MTP predictor with NemotronH layers."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        draft_model_config = speculative_config.draft_model_config
        assert draft_model_config is not None
        config = draft_model_config.hf_config.get_text_config()
        if quant_config is None:
            quant_config = get_draft_quant_config(vllm_config)

        self.config = config
        self.vocab_size = config.vocab_size
        self.org_vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)
        assert self.num_mtp_layers == 1, (
            "Only one MTP layer is supported for NemotronH-MTP"
        )

        self.pattern_str = config.mtp_hybrid_override_pattern
        self.pattern_len = len(self.pattern_str)
        assert self.pattern_len > 0

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        # Build flat list of layers
        self.layers = torch.nn.ModuleDict()

        # Total number of physical layers = num_steps * pattern_len
        total_layers = self.num_mtp_layers * self.pattern_len
        for i in range(total_layers):
            step_rel_idx = i % self.pattern_len

            char = self.pattern_str[step_rel_idx]

            is_start_of_step = step_rel_idx == 0
            is_end_of_step = step_rel_idx == self.pattern_len - 1

            layer_prefix = f"{prefix}.layers.{i}"

            # TODO smor- remove double layers formation
            common_kwargs = dict(
                config=config,
                layer_idx=self.mtp_start_layer_idx + i,
                model_config=draft_model_config,
                cache_config=vllm_config.cache_config,
                quant_config=quant_config,
                parallel_config=vllm_config.parallel_config,
                prefix=layer_prefix,
                has_start_projections=is_start_of_step,
                has_end_norm=is_end_of_step,
            )

            if char == "*":
                self.layers[str(i)] = NemotronHMTPAttentionDecoderLayer(**common_kwargs)
            elif char == "E":
                self.layers[str(i)] = NemotronHMTPMoEDecoderLayer(**common_kwargs)
            else:
                raise NotImplementedError(
                    f"Pattern char '{char}' in {self.pattern_str} not implemented"
                )

        self.make_empty_intermediate_tensors: Callable[..., IntermediateTensors] = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None, (
            "embed_tokens not initialized - must be shared from target model"
        )
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.get_input_embeddings(input_ids)

        residual = None

        for i in range(self.pattern_len):
            hidden_states, residual = self.layers[str(i)](
                inputs_embeds=inputs_embeds,
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        return hidden_states


class NemotronHMTP(nn.Module, SupportsPP, SupportsQuant):
    """NemotronH MTP model."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"language_model.": ""},
        orig_to_new_substr={"embeddings": "embed_tokens"},
    )

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        draft_model_config = speculative_config.draft_model_config
        assert draft_model_config is not None
        config = draft_model_config.hf_config.get_text_config()
        self.vllm_config = vllm_config
        self.config = config
        # Needed for load_weights mapping
        self.mtp_start_layer_idx = config.num_hidden_layers

        # EPLB config for experts
        self.num_redundant_experts = 0
        if vllm_config.parallel_config and vllm_config.parallel_config.eplb_config:
            self.num_redundant_experts = (
                vllm_config.parallel_config.eplb_config.num_redundant_experts
            )

        # MTP predictor
        self.model = NemotronHMultiTokenPredictor(
            vllm_config=vllm_config,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "mtp"),
        )

        # LM head for generating logits
        w4a16_path = envs.VLLM_NEMOTRON_MTP_W4A16_LM_HEAD_PATH
        activation_bits = envs.VLLM_NEMOTRON_MTP_NVFP4_ACTIVATION_BITS
        if activation_bits not in (4, 16):
            raise ValueError("NVFP4 LM-head activation bits must be 4 or 16")
        self.has_own_lm_head = bool(w4a16_path)
        if w4a16_path:
            if vllm_config.parallel_config.tensor_parallel_size != 1:
                raise ValueError("Dense MTP NVFP4 LM head currently requires TP=1")
            if vllm_config.model_config.dtype != torch.bfloat16:
                raise ValueError("Dense MTP NVFP4 LM head requires a BF16 model")
            if self.config.hidden_size % 16:
                raise ValueError("NVFP4 LM-head hidden size must be divisible by 16")
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=None if w4a16_path else self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if w4a16_path:
            self.lm_head.quant_method = NemotronHDenseNvFp4LMHeadMethod(
                w4a16_path,
                use_a16=activation_bits == 16,
                activation_global_scale=(
                    envs.VLLM_NEMOTRON_MTP_NVFP4_ACTIVATION_GLOBAL_SCALE
                ),
            )
            self.lm_head.is_nvfp4_lm_head = True

        self.logits_processor = LogitsProcessor(self.config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    @staticmethod
    def _find_quant_config(*args, **kwargs) -> QuantizationConfig | None:
        vllm_config = kwargs.get("vllm_config")
        assert isinstance(vllm_config, VllmConfig)
        return get_draft_quant_config(vllm_config)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward - applies attention-based MTP."""
        assert hidden_states is not None
        hidden_states = self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute logits for DRAFT token generation."""
        assert self.lm_head is not None, (
            "lm_head not initialized - must be shared from target model"
        )
        return self.logits_processor(self.lm_head, hidden_states)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project draft hidden states and select tokens using local argmax."""
        assert self.lm_head is not None, "lm_head not initialized"
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load MTP weights with proper name remapping."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        expert_params_mapping = []
        num_experts = getattr(self.config, "n_routed_experts", None)
        if getattr(self.config, "model_type", None) == "nemotron_h_puzzle":
            num_experts = self.config.mtp_n_routed_experts
        if num_experts is not None:
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="up_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="",  # Empty - non-gated MoE
                num_experts=num_experts,
                num_redundant_experts=self.num_redundant_experts,
            )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # MTP weights are nested in "language_model."
            # in Multimodal Nemotron-H checkpoints.
            name = name.removeprefix("language_model.")
            is_lm_head_weight = name.startswith("lm_head.")
            if is_lm_head_weight and getattr(self.lm_head, "is_nvfp4_lm_head", False):
                # The sidecar supplies this head during post-load processing.
                continue
            if is_lm_head_weight:
                self.has_own_lm_head = True
            # Only process MTP and LM head weights -
            # skip all non-MTP and non-LM head weights
            if (
                not name.startswith("mtp.")
                and "embeddings" not in name
                and not is_lm_head_weight
            ):
                continue
            # Skip rotary embeddings (computed, not loaded)
            if "rotary_emb.inv_freq" in name:
                continue

            name = name.replace("mtp.layers.", "model.layers.")

            if "embeddings" in name:
                name = name.replace("embeddings", "embed_tokens")
                if name.startswith("backbone."):
                    name = name.replace("backbone.", "model.")

            if "scale" in name or "zero_point" in name:
                remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                if remapped_name is None:
                    continue
                name = remapped_name

            # Handle stacked parameters (qkv_proj) for attention layers
            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Must be in a mixer (attention layer)
                if ".mixer." not in name:
                    continue

                is_stacked = True
                stacked_name = name.replace(weight_name, param_name)

                if stacked_name.endswith(".bias") and stacked_name not in params_dict:
                    continue

                if stacked_name not in params_dict:
                    # Might be that mapping failed or param doesn't exist
                    continue

                param = params_dict[stacked_name]
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(stacked_name)
                break

            if is_stacked:
                continue

            is_expert_weight = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                # weight_name is like "experts.0.up_proj."
                if weight_name not in name:
                    continue

                is_expert_weight = True

                # Replace the expert-specific weight name with fused parameter name
                # e.g., "experts.0.up_proj." -> "experts.w13_"
                name_mapped = name.replace(weight_name, param_name)

                if name_mapped not in params_dict:
                    continue

                param = params_dict[name_mapped]
                weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                success = weight_loader(
                    param,
                    loaded_weight,
                    name_mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    loaded_params.add(name_mapped)
                break

            if is_expert_weight:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue

            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            assert weight_loader is not None
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        if not self.has_own_lm_head or getattr(self.lm_head, "is_nvfp4_lm_head", False):
            loaded_params.update(
                name for name in params_dict if name.startswith("lm_head.")
            )

        return loaded_params
