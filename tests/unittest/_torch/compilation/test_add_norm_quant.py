import pytest
import torch

from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.custom_ops import flashinfer_rmsnorm
from tensorrt_llm._torch.modules.rms_norm import RMSNorm


@pytest.fixture(autouse=True)
def reset_backend_pattern_passes():
    Backend._custom_pass_instances = None
    yield


def _skip_fp8_fusion_if_unsupported_gpu():
    """FlashInfer FP8 fused add+RMSNorm+quant is unreliable on pre-Ampere GPUs (e.g. Turing NaNs)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    major, _minor = torch.cuda.get_device_capability()
    if major < 8:
        pytest.skip(
            "FP8 fused add+RMSNorm+quant tests require Ampere (sm_80+) or newer "
            f"(got capability {major}.x)")


def rms_norm(x: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-6):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight
    return y


def _has_fused_add_norm_quant(gm):
    """Check if optimized graph contains the fused add+norm+quant op."""
    fused_target = torch.ops.trtllm.flashinfer_fused_add_rmsnorm_quant.default
    return any(node.target == fused_target for node in gm.graph.nodes)


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("enable_inductor", [False, True])
def test_add_norm_quant_fusion(dtype, enable_inductor):
    _skip_fp8_fusion_if_unsupported_gpu()
    backend = Backend(enable_inductor)
    SEQ_LEN = 16
    HIDDEN_SIZE = 1024
    eps = 1e-6
    torch.manual_seed(42)
    x = torch.randn(SEQ_LEN, HIDDEN_SIZE, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    norm_weight = torch.randn((HIDDEN_SIZE, ), dtype=dtype, device="cuda")
    scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    norm = RMSNorm(hidden_size=HIDDEN_SIZE, eps=eps, dtype=dtype).cuda()
    norm.weight.data.copy_(norm_weight)

    @torch.compile(backend=backend)
    def func(x: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor,
             scale: torch.Tensor, eps: float):
        inter_output = x + residual
        normed = flashinfer_rmsnorm(inter_output, norm_weight, eps)
        fp8_out, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            normed, scale)
        return fp8_out, inter_output

    fp8_output, inter_output = func(x.clone(), residual.clone(), norm_weight,
                                    scale, eps)

    # Check that the fusion pass matched
    assert backend.match_count[0] == 1

    # Reference computation (unfused path)
    torch_inter_output = x + residual
    torch_normed = rms_norm(torch_inter_output, norm_weight, eps)
    torch_fp8_out, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
        torch_normed, scale)

    # Check intermediate add output
    if dtype == torch.float16:
        rtol_inter, atol_inter = 0.05, 0.15
    else:  # bfloat16
        rtol_inter, atol_inter = 0.1, 0.2
    torch.testing.assert_close(
        torch_inter_output,
        inter_output,
        rtol=rtol_inter,
        atol=atol_inter,
    )

    # Check FP8 output (converted back to original dtype for comparison)
    torch_fp8_back = torch_fp8_out.to(dtype)
    fp8_back = fp8_output.to(dtype)
    torch.testing.assert_close(
        torch_fp8_back,
        fp8_back,
        rtol=0.2,
        atol=0.5,
    )


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("enable_inductor", [False, True])
def test_add_norm_quant_does_not_fuse_when_residual_is_used_later(
        dtype, enable_inductor):
    _skip_fp8_fusion_if_unsupported_gpu()
    backend = Backend(enable_inductor)
    SEQ_LEN = 16
    HIDDEN_SIZE = 1024
    eps = 1e-6
    torch.manual_seed(42)
    x = torch.randn(SEQ_LEN, HIDDEN_SIZE, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    norm_weight = torch.randn((HIDDEN_SIZE, ), dtype=dtype, device="cuda")
    scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    norm = RMSNorm(hidden_size=HIDDEN_SIZE, eps=eps, dtype=dtype).cuda()
    norm.weight.data.copy_(norm_weight)

    @torch.compile(backend=backend)
    def func(x: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor,
             scale: torch.Tensor, eps: float):
        inter_output = x + residual
        normed = flashinfer_rmsnorm(inter_output, norm_weight, eps)
        fp8_out, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            normed, scale)
        # Use residual later - this should prevent fusion
        later_use = residual * 2
        return fp8_out, inter_output, later_use

    fp8_output, inter_output, later_use = func(x.clone(), residual.clone(),
                                               norm_weight, scale, eps)

    # Fusion should NOT happen because residual is used after the add
    assert backend.match_count[0] == 0

    # Reference computation
    torch_inter_output = x + residual
    torch_normed = rms_norm(torch_inter_output, norm_weight, eps)
    torch_fp8_out, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
        torch_normed, scale)
    torch_later_use = residual * 2

    # Check outputs match reference
    if dtype == torch.float16:
        rtol_inter, atol_inter = 0.05, 0.15
    else:  # bfloat16
        rtol_inter, atol_inter = 0.1, 0.2
    torch.testing.assert_close(
        torch_inter_output,
        inter_output,
        rtol=rtol_inter,
        atol=atol_inter,
    )

    torch_fp8_back = torch_fp8_out.to(dtype)
    fp8_back = fp8_output.to(dtype)
    torch.testing.assert_close(
        torch_fp8_back,
        fp8_back,
        rtol=0.2,
        atol=0.5,
    )

    torch.testing.assert_close(
        torch_later_use,
        later_use,
        rtol=0.01,
        atol=0.01,
    )


if __name__ == '__main__':
    test_add_norm_quant_fusion(torch.bfloat16, True)
    test_add_norm_quant_does_not_fuse_when_residual_is_used_later(
        torch.bfloat16, True)
