# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import torch
import torch.distributed

from cut_cross_entropy.utils import softcapping
from cut_cross_entropy.vocab_parallel.utils import (
    VocabParallelOptions,
    vp_reduce_correct_logit,
    vp_reduce_e_grad_hook,
    vp_reduce_lse,
)


class _VocabParallelLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vp_correct_logit: torch.Tensor,
        vp_lse: torch.Tensor,
        pg: torch.distributed.ProcessGroup | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lse = vp_reduce_lse(vp_lse, pg)
        correct_logit = vp_reduce_correct_logit(vp_correct_logit, pg, dtype=lse.dtype)

        ctx.save_for_backward(vp_lse, lse)

        return lse - correct_logit, lse

    @staticmethod
    def backward(
        ctx,
        grad_loss: torch.Tensor | None,
        grad_lse_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        if grad_loss is not None:
            grad_correct_logit = -grad_loss
        else:
            grad_correct_logit = None

        vp_lse, lse = ctx.saved_tensors

        if grad_lse_out is not None and grad_loss is not None:
            grad_lse = grad_loss + grad_lse_out
        elif grad_loss is not None:
            grad_lse = grad_loss
        elif grad_lse_out is not None:
            grad_lse = grad_lse_out
        else:
            grad_lse = None

        if grad_lse is not None:
            grad_lse = (vp_lse - lse).exp() * grad_lse

        return grad_correct_logit, grad_lse, None


def _vp_loss_fn(
    vp_correct_logit: torch.Tensor,
    vp_lse: torch.Tensor,
    pg: torch.distributed.ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _VocabParallelLossFunction.apply(vp_correct_logit, vp_lse, pg)


def _vp_torch_compile_correct_logit_lse(
    e: torch.Tensor,
    vocab_parallel_c: torch.Tensor,
    targets: torch.Tensor,
    start: int,
    stop: int,
    vocab_parallel_bias: torch.Tensor | None = None,
    softcap: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    vp_logits = e @ vocab_parallel_c.T

    if vocab_parallel_bias is not None:
        vp_logits = vp_logits + vocab_parallel_bias

    if softcap is not None:
        vp_logits = softcapping(vp_logits, softcap)

    vp_lse = torch.logsumexp(vp_logits.float(), -1)

    is_target_in_range = (targets < stop) & (targets >= start)
    masked_targets = torch.where(is_target_in_range, targets - start, targets.new_zeros(()))

    arange_indexer = torch.arange(0, len(vp_lse), device=targets.device, dtype=targets.dtype)
    vp_correct_logit = torch.where(
        is_target_in_range, vp_logits[arange_indexer, masked_targets], vp_logits.new_zeros(())
    )

    return vp_correct_logit, vp_lse


@torch.compile(fullgraph=True)
def vocab_parallel_torch_compile_lce_apply(
    vocab_parallel_options: VocabParallelOptions,
    e: torch.Tensor,
    vocab_parallel_c: torch.Tensor,
    targets: torch.Tensor,
    vocab_parallel_bias: torch.Tensor | None,
    softcap: float | None,
    reduction: str,
    return_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    pg = vocab_parallel_options.group

    e = vp_reduce_e_grad_hook(e, vocab_parallel_options)

    vp_correct_logit, vp_lse = _vp_torch_compile_correct_logit_lse(
        e,
        vocab_parallel_c,
        targets,
        vocab_parallel_options.start,
        vocab_parallel_options.stop,
        vocab_parallel_bias=vocab_parallel_bias,
        softcap=softcap,
    )

    loss, lse = _vp_loss_fn(vp_correct_logit, vp_lse, pg)

    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(f"Unknown reduction {reduction!r}")

    if not return_lse:
        lse = None

    return loss, lse
