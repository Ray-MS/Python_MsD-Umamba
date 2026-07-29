"""Residual DAMamba block built on four-directional DifferentialSS2D.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from models.layers.ss2d import DifferentialSS2D


__all__ = ["DAMamba",]


class DAMamba(nn.Module):
    """Pre-normalized residual DifferentialSS2D block."""

    def __init__(
        self,
        dim: int,
        *,
        depth_index: int,
        state_dim: int = 16,
        ssm_ratio: float = 2.0,
        conv_kernel_size: int = 3,
        scan_chunk_size: int = 64,
        drop_path: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        if not 0.0 <= drop_path < 1.0:
            raise ValueError("drop_path must be in [0, 1)")
        self.norm = nn.LayerNorm(dim)
        # Kept for compatibility with earlier configurations. The official
        # CUDA selective scan controls its own execution strategy.
        self.scan_chunk_size = scan_chunk_size
        self.mixer = DifferentialSS2D(
            d_model=dim,
            depth_index=depth_index,
            d_state=state_dim,
            ssm_ratio=ssm_ratio,
            d_conv=conv_kernel_size,
            **kwargs,
        )
        self.drop_path = drop_path

    @property
    def ssm(self) -> DifferentialSS2D:
        """Compatibility alias used by existing diagnostics."""
        return self.mixer

    def _stochastic_depth(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_path == 0.0:
            return x
        keep_probability = 1.0 - self.drop_path
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(
            shape,
            dtype=x.dtype,
            device=x.device,
        ).bernoulli_(keep_probability)
        return x * mask / keep_probability

    def forward(
        self,
        tokens: Tensor,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        if spatial_shape is None:
            raise ValueError("DAMamba requires the 2D patch-grid shape")
        batch, length, channels = tokens.shape
        if spatial_shape[0] * spatial_shape[1] != length:
            raise ValueError(
                f"Token length {length} does not match {spatial_shape}"
            )

        normalized = self.norm(tokens).reshape(
            batch,
            spatial_shape[0],
            spatial_shape[1],
            channels,
        )
        mixed = self.mixer(normalized).reshape(batch, length, channels)
        return tokens + self._stochastic_depth(mixed)
