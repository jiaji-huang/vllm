# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NemotronH-MTP model with attention layers."""

import typing
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.config.parallel import ParallelConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
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

from .gemma4_mtp import Gemma4MTPMaskedEmbedder
from .interfaces import SupportsPP, SupportsQuant
from .nemotron_h import (
    NemotronHAttentionDecoderLayer,
    NemotronHMoEDecoderLayer,
)

logger = init_logger(__name__)

def _find_tensor(
    state_dict: dict[str, torch.Tensor], keys: tuple[str, ...]
) -> torch.Tensor | None:
    for key in keys:
        value = state_dict.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _resolve_clustered_lm_head_path(path: str) -> Path:
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_dir():
        for filename in ("model.pt", "pytorch_model.bin", "model.safetensors"):
            candidate = checkpoint_path / filename
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "VLLM_NEMOTRON_MTP_CLUSTERED_LM_HEAD_PATH points to a directory, "
            "but none of model.pt, pytorch_model.bin, or model.safetensors "
            "exists in it: "
            f"{checkpoint_path}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "VLLM_NEMOTRON_MTP_CLUSTERED_LM_HEAD_PATH does not exist or is not "
            f"a file: {checkpoint_path}"
        )
    return checkpoint_path


def _build_token_ordering_from_token_to_cluster(
    token_to_cluster: torch.Tensor,
    *,
    num_centroids: int,
    vocab_size: int,
    vocab_size_per_centroid: int,
) -> torch.Tensor:
    token_to_cluster = token_to_cluster.detach().cpu().to(dtype=torch.long).view(-1)
    if token_to_cluster.numel() != vocab_size:
        raise ValueError(
            "clustered LM-head token_to_cluster has wrong length: "
            f"got {token_to_cluster.numel()}, expected {vocab_size}."
        )
    if token_to_cluster.numel() == 0:
        raise ValueError("clustered LM-head token_to_cluster is empty.")
    min_cluster = int(token_to_cluster.min().item())
    max_cluster = int(token_to_cluster.max().item())
    if min_cluster < 0 or max_cluster >= num_centroids:
        raise ValueError(
            "clustered LM-head token_to_cluster values must be in "
            f"[0, {num_centroids}); got min={min_cluster}, max={max_cluster}."
        )
    counts = torch.bincount(token_to_cluster, minlength=num_centroids)
    if not torch.all(counts == vocab_size_per_centroid):
        raise ValueError(
            "clustered LM-head token_to_cluster must define equal-size "
            f"clusters of size {vocab_size_per_centroid}; got "
            f"min={int(counts.min().item())}, max={int(counts.max().item())}."
        )

    token_ordering = torch.empty(vocab_size, dtype=torch.long)
    offsets = torch.zeros(num_centroids, dtype=torch.long)
    for token_id, cluster_id in enumerate(token_to_cluster.tolist()):
        offset = int(offsets[cluster_id].item())
        token_ordering[cluster_id * vocab_size_per_centroid + offset] = token_id
        offsets[cluster_id] += 1
    return token_ordering


def _validate_token_ordering(
    token_ordering: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    token_ordering = token_ordering.detach().cpu().to(dtype=torch.long).view(-1)
    if token_ordering.numel() != vocab_size:
        raise ValueError(
            "clustered LM-head token_ordering has wrong length: "
            f"got {token_ordering.numel()}, expected {vocab_size}."
        )
    if token_ordering.numel() == 0:
        raise ValueError("clustered LM-head token_ordering is empty.")
    min_token = int(token_ordering.min().item())
    max_token = int(token_ordering.max().item())
    if min_token < 0 or max_token >= vocab_size:
        raise ValueError(
            "clustered LM-head token_ordering values must be in "
            f"[0, {vocab_size}); got min={min_token}, max={max_token}."
        )
    if torch.unique(token_ordering).numel() != vocab_size:
        raise ValueError("clustered LM-head token_ordering must be a permutation.")
    return token_ordering


def _load_clustered_lm_head_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file as load_safetensors_file

        state = load_safetensors_file(str(path), device="cpu")
    else:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError(
            "clustered LM-head sidecar must contain a tensor state dict, got "
            f"{type(state)!r}."
        )
    return {
        str(key).removeprefix("module."): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }


def _load_clustered_lm_head_sidecar(
    masked_embedding: Gemma4MTPMaskedEmbedder,
) -> bool:
    sidecar_path = envs.VLLM_NEMOTRON_MTP_CLUSTERED_LM_HEAD_PATH
    if not sidecar_path:
        return False

    checkpoint_path = _resolve_clustered_lm_head_path(sidecar_path)
    state_dict = _load_clustered_lm_head_state(checkpoint_path)

    centroids = _find_tensor(
        state_dict,
        (
            "cluster_signature",
            "masked_embedding.centroids.weight",
            "centroids.weight",
        ),
    )
    if centroids is None:
        raise ValueError(
            "clustered LM-head sidecar is missing centroid weights. Expected "
            "one of: cluster_signature, masked_embedding.centroids.weight, "
            "centroids.weight."
        )
    expected_centroid_shape = tuple(masked_embedding.centroids.weight.shape)
    if tuple(centroids.shape) != expected_centroid_shape:
        raise ValueError(
            "clustered LM-head centroid shape mismatch: "
            f"got {tuple(centroids.shape)}, expected {expected_centroid_shape}."
        )

    token_ordering = _find_tensor(
        state_dict,
        (
            "token_ordering",
            "masked_embedding.token_ordering",
        ),
    )
    if token_ordering is None:
        cluster_token_table = _find_tensor(
            state_dict,
            (
                "cluster_token_table",
                "masked_embedding.cluster_token_table",
            ),
        )
        if cluster_token_table is not None:
            expected_table_shape = (
                masked_embedding.num_centroids,
                masked_embedding.vocab_size_per_centroid,
            )
            if tuple(cluster_token_table.shape) != expected_table_shape:
                raise ValueError(
                    "clustered LM-head cluster_token_table shape mismatch: "
                    f"got {tuple(cluster_token_table.shape)}, "
                    f"expected {expected_table_shape}."
                )
            token_ordering = cluster_token_table.reshape(-1)
        else:
            token_to_cluster = _find_tensor(
                state_dict,
                (
                    "token_to_cluster",
                    "masked_embedding.token_to_cluster",
                ),
            )
            if token_to_cluster is None:
                raise ValueError(
                    "clustered LM-head sidecar is missing token ordering. "
                    "Expected token_ordering, cluster_token_table, or "
                    "token_to_cluster."
                )
            token_ordering = _build_token_ordering_from_token_to_cluster(
                token_to_cluster,
                num_centroids=masked_embedding.num_centroids,
                vocab_size=masked_embedding.vocab_size,
                vocab_size_per_centroid=masked_embedding.vocab_size_per_centroid,
            )

    token_ordering = _validate_token_ordering(
        token_ordering,
        vocab_size=masked_embedding.vocab_size,
    )

    with torch.no_grad():
        masked_embedding.centroids.weight.copy_(
            centroids.to(
                device=masked_embedding.centroids.weight.device,
                dtype=masked_embedding.centroids.weight.dtype,
            )
        )
        masked_embedding.token_ordering.copy_(
            token_ordering.to(device=masked_embedding.token_ordering.device)
        )

    logger.info(
        "Loaded Nemotron-H MTP clustered LM-head sidecar from %s.",
        checkpoint_path,
    )
    return True


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
        self.has_own_lm_head = False
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        if envs.VLLM_NEMOTRON_MTP_CLUSTERED_LM_HEAD:
            num_clusters = envs.VLLM_NEMOTRON_MTP_LM_HEAD_NUM_CLUSTERS
            top_k = envs.VLLM_NEMOTRON_MTP_LM_HEAD_CLUSTER_TOP_K
            if self.config.vocab_size % num_clusters != 0:
                raise ValueError(
                    "Nemotron-H MTP clustered LM head requires vocab_size "
                    "to be divisible by num_clusters, got "
                    f"vocab_size={self.config.vocab_size} and "
                    f"num_clusters={num_clusters}."
                )
            if not 0 < top_k <= num_clusters:
                raise ValueError(
                    "Nemotron-H MTP clustered LM-head top_k must be between "
                    f"1 and num_clusters, got {top_k=} and {num_clusters=}."
                )

            self.masked_embedding = Gemma4MTPMaskedEmbedder(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                num_centroids=num_clusters,
                centroid_intermediate_top_k=top_k,
                prefix=maybe_prefix(prefix, "masked_embedding"),
            ).to(dtype=vllm_config.model_config.dtype)
            backend = envs.VLLM_NEMOTRON_MTP_SPARSE_HEAD_BACKEND
            if backend not in ("gather", "moe"):
                raise ValueError(
                    "VLLM_NEMOTRON_MTP_SPARSE_HEAD_BACKEND must be "
                    f"'gather' or 'moe', got {backend!r}."
                )
            self.masked_embedding.use_moe_backend = backend == "moe"
            self.masked_embedding.centroids.requires_grad_(False)
            self.masked_embedding.token_ordering.copy_(
                torch.arange(
                    self.config.vocab_size,
                    device=self.masked_embedding.token_ordering.device,
                )
            )
            logger.info(
                "Nemotron-H MTP: clustered LM head enabled "
                "(num_clusters=%d, top_k=%d, active_tokens=%d/%d, "
                "backend=%s).",
                num_clusters,
                top_k,
                top_k * (self.config.vocab_size // num_clusters),
                self.config.vocab_size,
                backend,
            )
        else:
            self.masked_embedding = None

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

    def _get_full_lm_head_weight(self) -> torch.Tensor:
        assert self.masked_embedding is not None
        if self._stable_full_lm_head_weight is not None:
            return self._stable_full_lm_head_weight

        lm_head_weight = self.lm_head.weight
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            lm_head_weight = tensor_model_parallel_all_gather(
                lm_head_weight,
                dim=0,
            )
            lm_head_weight = lm_head_weight[
                : self.masked_embedding.vocab_size
            ].contiguous()
        else:
            lm_head_weight = lm_head_weight[: self.masked_embedding.vocab_size]
        self._stable_full_lm_head_weight = lm_head_weight
        return lm_head_weight

    def prepare_clustered_lm_head(self) -> None:
        """Initialize centroid routing and MoE-packed weights from lm_head."""
        masked_embedding = self.masked_embedding
        if masked_embedding is None:
            return

        masked_embedding.centroid_weight = None
        lm_head_weight = self._get_full_lm_head_weight()
        vocab_size_per_centroid = masked_embedding.vocab_size_per_centroid

        if not _load_clustered_lm_head_sidecar(masked_embedding):
            with torch.no_grad():
                for centroid_idx in range(masked_embedding.num_centroids):
                    start = centroid_idx * vocab_size_per_centroid
                    end = start + vocab_size_per_centroid
                    masked_embedding.centroids.weight[centroid_idx].copy_(
                        lm_head_weight[start:end].mean(dim=0)
                    )
        masked_embedding.build_centroid_weight(lm_head_weight)
        if masked_embedding.use_moe_backend:
            self._stable_full_lm_head_weight = None

    def _get_sparse_lm_head_weight_arg(self) -> torch.Tensor:
        assert self.masked_embedding is not None
        if self.masked_embedding.use_moe_backend:
            return self.lm_head.weight
        return self._get_full_lm_head_weight()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute logits for DRAFT token generation."""
        assert self.lm_head is not None, (
            "lm_head not initialized - must be shared from target model"
        )
        if self.masked_embedding is not None:
            return self.masked_embedding(
                hidden_states,
                self._get_sparse_lm_head_weight_arg(),
            )
        return self.logits_processor(self.lm_head, hidden_states)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.masked_embedding is not None:
            return self.masked_embedding.get_top_tokens(
                hidden_states,
                self._get_sparse_lm_head_weight_arg(),
            )
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load MTP weights with proper name remapping."""
        self._stable_full_lm_head_weight = None
        if self.masked_embedding is not None:
            self.masked_embedding.centroid_weight = None

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

        if not self.has_own_lm_head:
            loaded_params.update(
                name for name in params_dict if name.startswith("lm_head.")
            )

        return loaded_params
