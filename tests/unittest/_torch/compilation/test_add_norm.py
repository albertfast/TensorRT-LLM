import pytest
import torch

from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.custom_ops import flashinfer_rmsnorm
from tensorrt_llm._torch.modules.rms_norm import RMSNorm


def _assert_add_norm_fusion_ran(backend: Backend) -> None:
    """Compatible with Backend layouts that register quant fusion before add+norm fusion.

    Editable/source installs use two PatternMatcherPass instances (match_count often
    starts with ``[0, 1, ...]`` when only add+norm applies). Older single-pass layouts
    record the rewrite at ``match_count[0]``.
    """
    mc = backend.match_count
    if len(mc) >= 2 and mc[0] == 0:
        matches = mc[1]
    else:
        matches = mc[0] if mc else 0
    assert matches >= 1, (
        "Expected residual add + RMSNorm fusion; "
        f"match_count={mc!r}, num_passes={len(backend.custom_passes)}"
    )


@pytest.fixture(autouse=True)
def reset_backend_pattern_passes():
    Backend._custom_pass_instances = None
    yield


def rms_norm(x: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-6):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight
    return y


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("enable_inductor", [False, True])
def test_add_norm_fusion(dtype, enable_inductor):
    backend = Backend(enable_inductor)
    SEQ_LEN = 16
    HIDDEN_SIZE = 1024
    eps = 1e-6
    torch.manual_seed(42)
    x = torch.randn(SEQ_LEN, HIDDEN_SIZE, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    norm_weight = torch.randn((HIDDEN_SIZE, ), dtype=dtype, device="cuda")
    norm = RMSNorm(hidden_size=HIDDEN_SIZE, eps=eps, dtype=dtype).cuda()
    norm.weight.data.copy_(norm_weight)

    @torch.compile(backend=backend)
    def func(x: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor,
             eps: float) -> torch.Tensor:
        inter_output = x + residual
        x = flashinfer_rmsnorm(inter_output, norm_weight, eps)
        return x, inter_output

    final_output, inter_output = func(x.clone(), residual.clone(), norm_weight,
                                      eps)

    _assert_add_norm_fusion_ran(backend)

    torch_inter_output = x + residual
    torch_final_output = rms_norm(torch_inter_output, norm_weight, eps)

    torch.testing.assert_close(
        torch_final_output,
        final_output,
        rtol=0.05,
        atol=0.15,
    )
    torch.testing.assert_close(
        torch_inter_output,
        inter_output,
        rtol=0.05,
        atol=0.15,
    )


if __name__ == '__main__':
    test_add_norm_fusion(torch.bfloat16, True)
