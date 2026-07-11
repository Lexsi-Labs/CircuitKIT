from typing import Any, Optional

import torch as t
from einops import einsum

from ..types import MaskFn, PatchWrapper
from ..utils.tensor_ops import sample_hard_concrete


class PatchWrapperImpl(PatchWrapper):
    """
    PyTorch module that wraps another module, representing a `Node` in the
    computation graph. It intercepts the forward pass to enable patching.

    If the wrapped module is a `DestNode`, the input is modified by interpolating
    between the default and ablated activations, controlled by `patch_mask`.

    If the wrapped module is a `SrcNode`, its output is cached for use in downstream
    patches.

    Args:
        module_name: Name of the wrapped module.
        module: The module to wrap.
        head_dim: The dimension along which to split the heads (for attention layers).
        seq_dim: The sequence dimension of the model's activations.
        is_src: Whether the wrapped module is a `SrcNode`.
        src_idxs: Slice of indices for this module's outputs in the shared source
            activation tensor.
        is_dest: Whether the wrapped module is a `DestNode`.
        patch_mask: The mask that interpolates between default and ablated activations.
        in_srcs: Slice of indices for this module's inputs in the shared source
            activation tensor.
    """

    def __init__(
        self,
        module_name: str,
        module: t.nn.Module,
        head_dim: Optional[int] = None,
        seq_dim: Optional[int] = None,
        is_src: bool = False,
        src_idxs: Optional[slice] = None,
        is_dest: bool = False,
        patch_mask: Optional[t.Tensor] = None,
        in_srcs: Optional[slice] = None,
    ):
        super().__init__()
        self.module_name: str = module_name
        self.module: t.nn.Module = module
        self.head_dim: Optional[int] = head_dim
        self.seq_dim: Optional[int] = seq_dim
        self.curr_src_outs: Optional[t.Tensor] = None
        self.in_srcs: Optional[slice] = in_srcs

        self.is_src = is_src
        if self.is_src:
            assert src_idxs is not None
            self.src_idxs: slice = src_idxs

        self.is_dest = is_dest
        if self.is_dest:
            assert patch_mask is not None
            self.patch_mask: t.nn.Parameter = t.nn.Parameter(patch_mask)
            self.patch_src_outs: Optional[t.Tensor] = None
            self.mask_fn: MaskFn = None
            self.dropout_layer: t.nn.Module = t.nn.Dropout(p=0.0)
        self.patch_mode = False

        assert head_dim is None or seq_dim is None or head_dim > seq_dim
        dims = range(1, max(head_dim if head_dim else 2, seq_dim if seq_dim else 2))
        self.dims = " ".join(["seq" if i == seq_dim else f"d{i}" for i in dims])

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        arg_0: t.Tensor = args[0].clone()

        if self.patch_mode and self.is_dest:
            assert self.patch_src_outs is not None and self.curr_src_outs is not None
            diff = self.patch_src_outs[self.in_srcs] - self.curr_src_outs[self.in_srcs]

            mask = self.patch_mask
            if self.mask_fn == "hard_concrete":
                mask = sample_hard_concrete(self.patch_mask, arg_0.size(0))
            elif self.mask_fn == "sigmoid":
                mask = t.sigmoid(self.patch_mask)

            mask = self.dropout_layer(mask)

            head_str = "" if self.head_dim is None else "dest"
            seq_str = "" if self.seq_dim is None else "seq"
            ein_pre = f"{seq_str} {head_str} src, src batch {self.dims} ..."
            ein_post = f"batch {self.dims} {head_str} ..."
            arg_0 += einsum(mask, diff, f"{ein_pre} -> {ein_post}")

        new_args = (arg_0,) + args[1:]
        out = self.module(*new_args, **kwargs)

        if self.patch_mode and self.is_src:
            assert self.curr_src_outs is not None
            if self.head_dim is None:
                src_out = out
            else:
                squeeze_dim = self.head_dim if self.head_dim < 0 else self.head_dim + 1
                src_out = t.stack(out.split(1, dim=self.head_dim)).squeeze(squeeze_dim)
            self.curr_src_outs[self.src_idxs] = src_out

        return out
