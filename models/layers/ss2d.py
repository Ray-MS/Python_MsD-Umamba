"""Two-dimensional selective scans for MsD-UMamba.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


__all__ = ["DifferentialSS2D",]


def _valid_group_count(channels: int, requested_groups: int) -> int:
    groups = min(channels, requested_groups)
    while channels % groups:
        groups -= 1
    return groups


class DifferentialSS2D(nn.Module):
    """Four-directional Differential Attention Mamba scan.

    Args:
        d_model: Input and output channel dimension.
        depth_index: Global DAMamba block index used for lambda initialization.
        d_state: Latent selective-SSM state dimension.
        ssm_ratio: Expansion ratio inside the selective scan.
        dt_rank: Rank of the input-dependent step-size projection.
        d_conv: Kernel size of the depthwise spatial projections.
        dropout: Output dropout probability.
        bias: Whether the input and output linear projections use bias.
        lambda_groups: Requested GroupNorm group count.
    """

    direction_count = 4

    def __init__(
        self,
        d_model: int = 96,
        *,
        depth_index: int = 0,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        dt_rank: int | str = "auto",
        d_conv: int = 3,
        conv_bias: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        lambda_groups: int = 8,
    ) -> None:
        super().__init__()
        if d_conv < 1 or d_conv % 2 != 1:
            raise ValueError("d_conv must be a positive odd number")

        self.d_model = d_model
        self.d_state = d_state
        self.d_hidden = int(ssm_ratio * d_model)
        self.dt_rank = (
            math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        )
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth_index)
        self.last_scan_backend = "not-run"

        self.in_proj = nn.Linear(d_model, 2 * self.d_hidden, bias=bias)
        self.ssm_conv = nn.Conv2d(
            self.d_hidden,
            self.d_hidden,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.d_hidden,
            bias=conv_bias,
        )
        self.gate_conv = nn.Conv2d(
            self.d_hidden,
            self.d_hidden,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.d_hidden,
            bias=conv_bias,
        )

        # Per-direction projections produce one shared delta and two distinct
        # (B, C) pairs: [dt, B1, C1, B2, C2].
        projection_size = self.dt_rank + 4 * d_state
        x_projections = [
            nn.Linear(self.d_hidden, projection_size, bias=False)
            for _ in range(self.direction_count)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([layer.weight for layer in x_projections])
        )

        dt_projections = [
            self._dt_init(
                self.dt_rank,
                self.d_hidden,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
            )
            for _ in range(self.direction_count)
        ]
        self.dt_proj_weight = nn.Parameter(
            torch.stack([layer.weight for layer in dt_projections])
        )
        self.dt_proj_bias = nn.Parameter(
            torch.stack([layer.bias for layer in dt_projections])
        )

        self.a_logs = self._a_log_init(
            d_state,
            self.d_hidden,
            copies=self.direction_count,
        )
        self.skip = self._skip_init(
            self.d_hidden,
            copies=self.direction_count,
        )

        self.lambda_projection = nn.Linear(self.d_hidden, self.d_hidden)
        self.diff_norm = nn.GroupNorm(
            _valid_group_count(self.d_hidden, lambda_groups),
            self.d_hidden,
        )
        self.out_norm = nn.LayerNorm(self.d_hidden)
        self.out_proj = nn.Linear(self.d_hidden, d_model, bias=bias)
        self.dropout = (
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    @staticmethod
    def _dt_init(
        dt_rank: int,
        d_hidden: int,
        dt_scale: float,
        dt_init: str,
        dt_min: float,
        dt_max: float,
        dt_init_floor: float,
    ) -> nn.Linear:
        projection = nn.Linear(dt_rank, d_hidden, bias=True)
        std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(projection.weight, std)
        elif dt_init == "random":
            nn.init.uniform_(projection.weight, -std, std)
        else:
            raise ValueError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(
            torch.rand(d_hidden)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp_min(dt_init_floor)
        inverse_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            projection.bias.copy_(inverse_softplus)
        return projection

    @staticmethod
    def _a_log_init(
        d_state: int,
        d_hidden: int,
        *,
        copies: int,
    ) -> nn.Parameter:
        state = torch.arange(1, d_state + 1, dtype=torch.float32)
        state = state.unsqueeze(0).repeat(d_hidden, 1)
        parameter = nn.Parameter(
            state.log().unsqueeze(0).repeat(copies, 1, 1).flatten(0, 1)
        )
        parameter._no_weight_decay = True
        return parameter

    @staticmethod
    def _skip_init(d_hidden: int, *, copies: int) -> nn.Parameter:
        parameter = nn.Parameter(torch.ones(copies * d_hidden))
        parameter._no_weight_decay = True
        return parameter

    @staticmethod
    def _directional_sequences(x: Tensor) -> Tensor:
        """Return row/column forward/backward sequences [B, 4, D, L]."""
        batch, channels, height, width = x.shape
        row_major = x.flatten(2)
        column_major = x.transpose(2, 3).contiguous().flatten(2)
        forward = torch.stack([row_major, column_major], dim=1)
        backward = torch.flip(forward, dims=[-1])
        sequences = torch.cat([forward, backward], dim=1)
        return sequences.reshape(batch, 4, channels, height * width)

    @staticmethod
    def _restore_directions(
        directional_output: Tensor,
        spatial_shape: Tuple[int, int],
    ) -> Tensor:
        """Restore four directional outputs and sum them as [B, D, L]."""
        height, width = spatial_shape
        batch, _, channels, length = directional_output.shape

        row_forward = directional_output[:, 0]
        column_forward = (
            directional_output[:, 1]
            .reshape(batch, channels, width, height)
            .transpose(2, 3)
            .contiguous()
            .reshape(batch, channels, length)
        )
        row_backward = torch.flip(
            directional_output[:, 2],
            dims=[-1],
        )
        column_backward = torch.flip(
            directional_output[:, 3],
            dims=[-1],
        )
        column_backward = (
            column_backward.reshape(batch, channels, width, height)
            .transpose(2, 3)
            .contiguous()
            .reshape(batch, channels, length)
        )
        return (
            row_forward
            + column_forward
            + row_backward
            + column_backward
        )

    def _differential_scan(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Run two official scans sharing A, delta, and D."""
        batch, _, _, length = x.shape
        projected = torch.einsum(
            "b k d l, k c d -> b k c l",
            x,
            self.x_proj_weight,
        )
        delta_rank, b1, c1, b2, c2 = torch.split(
            projected,
            [
                self.dt_rank,
                self.d_state,
                self.d_state,
                self.d_state,
                self.d_state,
            ],
            dim=2,
        )
        delta = torch.einsum(
            "b k r l, k d r -> b k d l",
            delta_rank,
            self.dt_proj_weight,
        )

        u = x.reshape(batch, -1, length).contiguous()
        delta = delta.reshape(batch, -1, length).contiguous()
        a = -torch.exp(self.a_logs.float())
        skip = self.skip.float()
        delta_bias = self.dt_proj_bias.float().reshape(-1)

        def scan(b: Tensor, c: Tensor) -> Tensor:
            return selective_scan_fn(
                u,
                delta,
                a,
                b.contiguous(),
                c.contiguous(),
                skip,
                z=None,
                delta_bias=delta_bias,
                delta_softplus=True,
                return_last_state=False,
            ).reshape(
                batch,
                self.direction_count,
                self.d_hidden,
                length,
            )

        first = scan(b1, c1)
        second = scan(b2, c2)
        self.last_scan_backend = "mamba_ssm.selective_scan_fn"
        return first, second

    def forward(self, x: Tensor) -> Tensor:
        """Process a channels-last feature map ``[B, H, W, C]``."""
        if x.ndim != 4:
            raise ValueError("DifferentialSS2D expects [B, H, W, C]")
        batch, height, width, _ = x.shape

        ssm_input, gate = self.in_proj(x).chunk(2, dim=-1)
        ssm_input = ssm_input.permute(0, 3, 1, 2).contiguous()
        gate = gate.permute(0, 3, 1, 2).contiguous()
        ssm_input = F.silu(self.ssm_conv(ssm_input))
        gate = F.silu(self.gate_conv(gate))

        sequences = self._directional_sequences(ssm_input)
        first_directions, second_directions = self._differential_scan(
            sequences
        )
        first = self._restore_directions(
            first_directions,
            (height, width),
        )
        second = self._restore_directions(
            second_directions,
            (height, width),
        )

        lambda_value = F.softplus(
            self.lambda_projection(second.mean(dim=-1))
        )
        lambda_value = (lambda_value + self.lambda_init).unsqueeze(-1)
        differential = first - lambda_value * second
        differential = self.diff_norm(differential)
        differential = (1.0 - self.lambda_init) * differential

        differential = differential.reshape(
            batch,
            self.d_hidden,
            height,
            width,
        )
        fused = differential * gate
        fused = fused.permute(0, 2, 3, 1).contiguous()
        fused = self.out_norm(fused)
        return self.dropout(self.out_proj(fused))


# Backward-compatible name for callers that imported SS2D from this project.
SS2D = DifferentialSS2D
