# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from nvfp4_utils import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX, dequantize_nvfp4_to_dtype

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
# m, n, k
SHAPES = [(128, 128, 64), (128, 128, 128), (256, 128, 64), (128, 256, 128)]
PAD_SHAPES = [(150, 128, 64), (128, 128, 96)]
SHAPES.extend(PAD_SHAPES)

SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]


def get_ref_results(
    a_fp4,
    b_fp4,
    a_sf,
    b_sf,
    a_global_scale,
    b_global_scale,
    m,
    n,
    dtype,
    block_size,
    device,
):
    _, m_k = a_fp4.shape
    _, n_k = b_fp4.shape
    assert m_k == n_k
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4, a_sf, a_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    b_in_dtype = dequantize_nvfp4_to_dtype(
        b_fp4, b_sf, b_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    return torch.matmul(a_in_dtype, b_in_dtype.t())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_nvfp4_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)
    m, n, packed_k = shape
    k = packed_k * 2
    block_size = 16
    a_dtype = torch.randn((m, k), dtype=dtype, device=device)
    b_dtype = torch.randn((n, k), dtype=dtype, device=device)

    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    b_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)
    # ops.scaled_fp4_quant returns swizzled scales, while weights
    # from checkpoints are in linear scales.
    a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a_dtype, a_global_scale)
    b_fp4, b_scale_interleaved = ops.scaled_fp4_quant(b_dtype, b_global_scale)

    # get_ref_results unswizzles the scales internally.
    expected_out = get_ref_results(
        a_fp4,
        b_fp4,
        a_scale_interleaved,
        b_scale_interleaved,
        a_global_scale,
        b_global_scale,
        m,
        n,
        dtype,
        block_size,
        device,
    )
    out = ops.cutlass_scaled_fp4_mm(
        a_fp4, b_fp4, a_scale_interleaved, b_scale_interleaved, alpha, dtype
    )

    torch.testing.assert_close(out, expected_out.to(dtype=dtype), atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("num_tokens", [1, 4, 32])
@pytest.mark.parametrize("n,k", [(32, 128), (32, 2688), (48, 272)])
@torch.inference_mode()
def test_nvfp4_routed_lm_head_matches_quantized_reference(num_tokens, n, k):
    """Routes reuse token quantization and retain their original output order."""
    from vllm.model_executor.models.gemma4_mtp import _grouped_gemm_nvfp4

    set_random_seed(42)
    num_experts, top_k = 7, 3
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    x[0].zero_()
    w = torch.randn(num_experts * n, k, dtype=x.dtype, device=x.device)
    activation_scale = torch.tensor([2.0], device=x.device)
    weight_quant_scale = torch.tensor([3.0], device=x.device)
    a_fp4, a_scale = ops.scaled_fp4_quant(
        x, activation_scale, is_sf_swizzled_layout=False
    )
    b_fp4, b_scale = ops.scaled_fp4_quant(
        w, weight_quant_scale, is_sf_swizzled_layout=False
    )
    ids = torch.randint(num_experts, (num_tokens, top_k), device=x.device).int()
    alpha = torch.full((num_experts,), 1 / 3, device=x.device)
    a_ref = dequantize_nvfp4_to_dtype(
        a_fp4,
        a_scale,
        activation_scale,
        torch.float32,
        x.device,
        is_sf_linear_layout=True,
    )
    b_ref = dequantize_nvfp4_to_dtype(
        b_fp4,
        b_scale,
        weight_quant_scale,
        torch.float32,
        x.device,
        is_sf_linear_layout=True,
    ).view(num_experts, n, k)
    expected = torch.einsum("tk,tsnk->tsn", a_ref, b_ref[ids.long()])
    actual = _grouped_gemm_nvfp4(
        x,
        b_fp4.view(num_experts, n, k // 2),
        b_scale.view(num_experts, n, k // 16),
        alpha,
        activation_scale,
        ids,
        top_k,
        num_experts,
    )
    assert actual.shape == (num_tokens, top_k, n)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected.to(x.dtype), rtol=1e-2, atol=1e-2)

    if (num_tokens, n, k) == (4, 32, 2688):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _grouped_gemm_nvfp4(
                x,
                b_fp4.view(num_experts, n, k // 2),
                b_scale.view(num_experts, n, k // 16),
                alpha,
                activation_scale,
                ids,
                top_k,
                num_experts,
            )
        x.mul_(2)
        ids.copy_(ids.flip(1))
        graph.replay()
        torch.testing.assert_close(
            captured, (expected.flip(1) * 2).to(x.dtype), rtol=1e-2, atol=1e-2
        )
