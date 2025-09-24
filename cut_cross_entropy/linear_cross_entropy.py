# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import platform
import warnings
from typing import TYPE_CHECKING, Literal, overload

import torch
import torch.nn as nn

from cut_cross_entropy.cce_utils import CCEPreset, CCEPresets, LinearCrossEntropyImpl
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.doc import (
    CCE_OPTS_DOC,
    DTENSOR_NOTE,
    IMPL_DOC,
    LINEAR_CROSS_ENTROPY_DOC,
    add_doc_end,
    add_doc_start,
)
from cut_cross_entropy.torch_compile import torch_compile_linear_cross_entropy
from cut_cross_entropy.utils import (
    CCEWarning,
    is_torch_greater_or_equal_2_5,
    is_triton_3_2,
    maybe_type_as,
    to_full_tensor,
)
from cut_cross_entropy.vocab_parallel import VocabParallelOptions

warnings.filterwarnings("once", category=CCEWarning, module="cut_cross_entropy")

PLATFORM_SYSTEM = platform.system()

if TYPE_CHECKING or PLATFORM_SYSTEM != "Darwin":
    from cut_cross_entropy.cce import cce_linear_cross_entropy

    LCE_IMPL_DEFAULT = LinearCrossEntropyImpl.CCE
else:
    cce_linear_cross_entropy = None
    LCE_IMPL_DEFAULT = LinearCrossEntropyImpl.TORCH_COMPILE

if TYPE_CHECKING or is_torch_greater_or_equal_2_5():
    import torch.distributed.tensor


is_d_tensor_error_message = (
    "Received {name} as a torch.distributed.tensor.DTensor. This is not supported. "
)


@overload
def linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool | int = 0,
    return_lse: Literal[False] = False,
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    vocab_parallel_options: VocabParallelOptions | None = None,
) -> torch.Tensor: ...


@overload
def linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool | int = 0,
    return_lse: Literal[True] = True,
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    vocab_parallel_options: VocabParallelOptions | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool | int = 0,
    return_lse: bool = False,
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    vocab_parallel_options: VocabParallelOptions | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...


@add_doc_start(LINEAR_CROSS_ENTROPY_DOC)
@add_doc_start(*(doc_str + " Only valid for the cce implementation." for doc_str in CCE_OPTS_DOC))
@add_doc_start(IMPL_DOC)
@add_doc_end(DTENSOR_NOTE)
def linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool | int = 0,
    return_lse: bool = False,
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    vocab_parallel_options: VocabParallelOptions | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    :param vocab_parallel_options: Used to enable vocab parallelism."""

    if is_torch_greater_or_equal_2_5():
        maybe_tensor_inputs = dict(e=e, targets=targets)
        for k, v in maybe_tensor_inputs.items():
            if isinstance(v, torch.distributed.tensor.DTensor):
                raise ValueError(is_d_tensor_error_message.format(name=k))

        c = maybe_type_as(to_full_tensor(c), e)
        bias = maybe_type_as(to_full_tensor(bias), e)

    if isinstance(impl, LinearCrossEntropyImpl):
        impl = impl.name.lower()

    if isinstance(shift, int) and (shift < 0 or shift >= targets.size(-1)):
        raise ValueError(f"Shift must be in the range [0, {targets.size(-1)}). Got {shift}.")

    if vocab_parallel_options is not None:
        expected_v_dim_size = vocab_parallel_options.stop - vocab_parallel_options.start
        if c.size(0) != expected_v_dim_size:
            raise ValueError(f"Expected c.size(0) to be {expected_v_dim_size}, got {c.size(0)}.")

    if bias is not None and bias.size(0) != c.size(0):
        raise ValueError(
            f"Bias has a different number of elements than c. {bias.size(0)} vs. {c.size(0)}."
        )

    if impl in CCEPresets.names:
        if platform.system() == "Darwin":
            raise RuntimeError(
                "CCE does not support MacOS. Please use torch_compile when running on MacOS instead."
            )

        if is_triton_3_2():
            warnings.warn(
                "There is a known issue with CCE and Triton 3.2 (the version that ships with PyTorch 2.6)"
                " that can result in incorrect gradients. If possible, please verify that you"
                " are not impacted by this bug by trying a newer triton version (i.e. by installing PyTorch>2.6).",
                CCEWarning,
                stacklevel=2,
            )

        cce_opts = CCEPresets.build_for_impl(
            impl,
            CCEPreset(
                filter_eps=filter_eps,
                accum_e_fp32=accum_e_fp32,
                accum_c_fp32=accum_c_fp32,
                filter_e_grad=filter_e_grad,
                filter_c_grad=filter_c_grad,
            ),
        )

        assert cce_linear_cross_entropy is not None
        loss, lse = cce_linear_cross_entropy(
            e,
            c,
            targets,
            bias,
            ignore_index,
            softcap,
            reduction,
            shift,
            **cce_opts,
            vocab_parallel_options=vocab_parallel_options,
            return_lse=return_lse,
        )
    elif impl == "torch_compile":
        loss, lse = torch_compile_linear_cross_entropy(
            e,
            c,
            targets,
            bias,
            ignore_index,
            softcap,
            reduction,
            shift,
            vocab_parallel_options=vocab_parallel_options,
            return_lse=return_lse,
        )
    else:
        raise NotImplementedError(f"{impl} is not implemented.")

    if return_lse:
        assert lse is not None
        return loss, lse
    else:
        return loss


class LinearCrossEntropy(nn.Module):
    def __init__(
        self,
        ignore_index: int = IGNORE_INDEX,
        softcap: float | None = None,
        reduction: str = "mean",
        shift: bool | int = 0,
        filter_eps: float | str | None = "auto",
        accum_e_fp32: bool = False,
        accum_c_fp32: bool = False,
        filter_e_grad: bool = True,
        filter_c_grad: bool = True,
        impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
        return_lse: bool = False,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.softcap = softcap
        self.reduction = reduction
        self.filter_eps = filter_eps
        self.shift = shift

        self.accum_e_fp32 = accum_e_fp32
        self.accum_c_fp32 = accum_c_fp32

        self.filter_e_grad = filter_e_grad
        self.filter_c_grad = filter_c_grad

        self.impl = impl
        self.return_lse = return_lse

    def forward(
        self,
        e: torch.Tensor,
        c: torch.Tensor,
        targets: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return linear_cross_entropy(
            e,
            c,
            targets,
            bias=bias,
            ignore_index=self.ignore_index,
            softcap=self.softcap,
            reduction=self.reduction,
            shift=self.shift,
            filter_eps=self.filter_eps,
            accum_e_fp32=self.accum_e_fp32,
            accum_c_fp32=self.accum_c_fp32,
            filter_e_grad=self.filter_e_grad,
            filter_c_grad=self.filter_c_grad,
            impl=self.impl,
            return_lse=self.return_lse,
        )
