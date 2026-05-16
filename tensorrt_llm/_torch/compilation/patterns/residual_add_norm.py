import torch
from torch._inductor.pattern_matcher import (MULTIPLE, CallFunction, KeywordArg,
                                             Match, MultiOutputPattern,
                                             PatternMatcherPass, fwd_only,
                                             register_replacement)
from operator import getitem

aten = torch.ops.aten
from torch._higher_order_ops.auto_functionalize import auto_functionalized


def _has_no_later_users(value_node: torch.fx.Node,
                        mutation_node: torch.fx.Node) -> bool:
    """Check if value_node has no users after mutation_node in graph order."""
    graph_order = {node: index for index, node in enumerate(mutation_node.graph.nodes)}
    mutation_index = graph_order[mutation_node]

    for user in value_node.users:
        if user is mutation_node:
            continue
        user_index = graph_order.get(user)
        if user_index is not None and user_index > mutation_index:
            return False

    return True


def register_add_norm_quant(custom_pass: PatternMatcherPass):
    residual = KeywordArg("residual")
    add_Tensor = CallFunction(aten.add.Tensor,
                              KeywordArg("input"),
                              residual,
                              _users=MULTIPLE)
    flashinfer_norm_default = CallFunction(
        torch.ops.trtllm.flashinfer_rmsnorm.default,
        add_Tensor,
        KeywordArg("norm_weight"),
        KeywordArg("eps"),
        _users=MULTIPLE)
    static_quantize_e4m3_per_tensor_default = CallFunction(
        torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor.default,
        flashinfer_norm_default,
        KeywordArg("scale"),
        _users=MULTIPLE)
    getitem_default = CallFunction(getitem,
                                   static_quantize_e4m3_per_tensor_default,
                                   0,
                                   _users=MULTIPLE)
    add_norm_quant_pattern = MultiOutputPattern([getitem_default, add_Tensor])

    def empty_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.nn.Parameter,
        eps: float,
        scale: torch.Tensor,
    ):
        return

    def target_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.nn.Parameter,
        eps: float,
        scale: torch.Tensor,
    ):
        out = torch.empty_like(input, dtype=torch.float8_e4m3fn)
        at = auto_functionalized(
            torch.ops.trtllm.flashinfer_fused_add_rmsnorm_quant.default,
            out=out,
            input=input,
            residual=residual,
            weight=norm_weight,
            scale=scale,
            eps=eps)
        return at[1], at[2]

    def extra_check(match: Match):
        # Check the original residual has no other users after the add node
        # since we will inplace update it
        add_node = match.ctx.pattern_to_node[add_Tensor]
        if not isinstance(add_node, torch.fx.graph.Node):
            return False

        residual_arg = add_node.args[1]
        if not isinstance(residual_arg, torch.fx.graph.Node):
            return False

        return _has_no_later_users(residual_arg, add_node)

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=add_norm_quant_pattern,
        extra_check=extra_check,
    )


def register_add_norm(custom_pass: PatternMatcherPass):
    residual = KeywordArg("residual")
    add_Tensor = CallFunction(aten.add.Tensor,
                              KeywordArg("input"),
                              residual,
                              _users=MULTIPLE)
    flashinfer_norm_default = CallFunction(
        torch.ops.trtllm.flashinfer_rmsnorm.default,
        add_Tensor,
        KeywordArg("norm_weight"),
        KeywordArg("eps"),
        _users=MULTIPLE)
    add_norm_pattern = MultiOutputPattern([flashinfer_norm_default, add_Tensor])

    def empty_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.nn.Parameter,
        eps: float,
    ):
        return

    def target_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.nn.Parameter,
        eps: float,
    ):
        at = auto_functionalized(
            torch.ops.trtllm.flashinfer_fused_add_rmsnorm.default,
            input=input,
            residual=residual,
            weight=norm_weight,
            eps=eps)
        return at[1], at[2]

    def extra_check(match: Match):
        # In-place fused kernel: both add operands must have no uses after the add.
        # Avoid relying on dict insertion order of .users (fragile across PyTorch versions).
        add_node = match.ctx.pattern_to_node[add_Tensor]
        if not isinstance(add_node, torch.fx.graph.Node):
            return False

        input_arg = add_node.args[0]
        residual_arg = add_node.args[1]
        if not isinstance(input_arg,
                           torch.fx.graph.Node) or not isinstance(
                               residual_arg, torch.fx.graph.Node):
            return False

        return (_has_no_later_users(input_arg, add_node)
                and _has_no_later_users(residual_arg, add_node))

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=add_norm_pattern,
        extra_check=extra_check,
    )
