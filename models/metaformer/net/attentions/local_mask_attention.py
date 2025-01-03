# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from xformers.components.attention import (
    Attention,
    AttentionConfig,
    AttentionMask,
    maybe_sparsify,
    register_attention,
    sparsify,
)
from xformers.components.attention.attention_patterns import (
    causal_1d_pattern,
    local_1d_pattern,
)
from xformers.components.attention.core import scaled_dot_product_attention
import xformers.ops as xops


@dataclass
class LocalMaskAttentionConfig(AttentionConfig):
    causal: Optional[bool] = None
    window_size: Optional[int] = None
    force_sparsity: Optional[bool] = None


@register_attention("local_mask", LocalMaskAttentionConfig)
class LocalMaskAttention(Attention):
    def __init__(
        self,
        dropout: float = 0.0,
        causal: bool = False,
        window_size: int = 5,
        force_sparsity: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.attn_drop = dropout
        self.causal = causal
        self.force_sparsity = force_sparsity

        if not self.causal:
            assert (
                window_size % 2 == 1
            ), "The window size is assumed to be odd (counts self-attention + 2 wings)"

        self.window_size = window_size
        self.attention_mask: Optional[torch.Tensor] = None
        self.requires_same_k_q_dimensions = True

        self.supports_attention_mask = True
        self.supports_key_padding_mask = False

    def _get_local_mask(self, shape: torch.Size) -> torch.Tensor:
        window_size = self.window_size * 2 + 1 if self.causal else self.window_size
        mask = local_1d_pattern(shape[2], window_size)

        if self.causal:
            mask &= causal_1d_pattern(shape[1])

        return mask

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[Union[torch.Tensor, AttentionMask]] = None,
        *args,
        **kwargs,
    ):
        if self.attention_mask is None or self.attention_mask.shape[1] != q.shape[1]:
            self.attention_mask = self._get_local_mask(q.shape).to(q.device)

        if att_mask is None:
            mask = self.attention_mask
        else:
            if isinstance(att_mask, AttentionMask):
                att_mask = att_mask.to_bool()
            heads = v.shape[1]
            am = (
                att_mask.unsqueeze(1)
                .expand(att_mask.shape[0], heads, att_mask.shape[-1])
                .reshape(-1, att_mask.shape[1])
            )
            mask = self.attention_mask.unsqueeze(0) & am.unsqueeze(
                1
            ) 
            mask = mask | (
                torch.eye(mask.shape[1], device=mask.device).unsqueeze(0) == 1
            )

            mask = (~mask).type(q.dtype).masked_fill(~mask, torch.finfo(q.dtype).min)
            mask = mask.reshape(q.shape[0], heads, mask.shape[-2], mask.shape[-1])
        # if self.training:
            return custom_attention(
                q, k, v, attn_bias=mask, dropout_p=self.attn_drop
            )
        # else:
        #     return xops.memory_efficient_attention(q, k, v, attn_bias=mask, p=0.0)

import torch
import torch.nn.functional as F

def custom_attention(q, k, v, attn_bias=None, dropout_p=0.0):
    """
    Custom scaled dot-product attention implementation.
    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim)
        v: Value tensor of shape (batch_size, num_heads, seq_len_k, head_dim)
        attn_bias: Optional attention bias or mask of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        dropout_p: Dropout probability to apply on attention weights.
    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len_q, head_dim)
    """
    # Compute scaled dot-product attention scores
    d_k = q.size(-1)  # Head dimension
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=q.dtype, device=q.device))
    
    # Add attention bias if provided
    if attn_bias is not None:
        scores += attn_bias

    # Apply softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Apply dropout if required
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    # Compute the attention output
    output = torch.matmul(attn_weights, v)
    return output
