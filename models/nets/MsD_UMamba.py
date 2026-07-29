"""MsD-UMamba implementation corresponding to the manuscript.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from timm.models import register_model

from models.blocks import DAMamba


__all__ = ["MsD_UMamba",]


def _valid_group_count(channels: int, requested_groups: int) -> int:
    """Return the largest requested group count that divides ``channels``."""
    groups = min(channels, requested_groups)
    while channels % groups:
        groups -= 1
    return groups


class ConvNormAct(nn.Sequential):
    """A compact convolution, normalization, and activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(
                _valid_group_count(out_channels, 8),
                out_channels,
            ),
            nn.SiLU(inplace=True),
        )


class PatchEmbed(nn.Module):
    """Split a feature map into non-overlapping patches and embed each patch."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """Return tokens ``[B, L, D]`` and their two-dimensional grid shape."""
        height, width = feature_map.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                "Stem feature size must be divisible by every patch size, "
                f"but received {(height, width)} for patch size "
                f"{self.patch_size}."
            )
        embedded = self.projection(feature_map)
        grid_shape = embedded.shape[-2:]
        tokens = embedded.flatten(2).transpose(1, 2)
        return self.norm(tokens), grid_shape


class DAMambaStage(nn.Module):
    """A stack of DAMamba blocks operating at one representation level."""

    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        first_depth_index: int,
        state_dim: int,
        scan_chunk_size: int,
        drop_path_rates: Sequence[float],
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("Every DAMamba stage must contain at least one block")
        self.blocks = nn.ModuleList(
            DAMamba(
                dim,
                depth_index=first_depth_index + block_index,
                state_dim=state_dim,
                scan_chunk_size=scan_chunk_size,
                drop_path=drop_path_rates[block_index],
            )
            for block_index in range(depth)
        )

    def forward(
        self,
        tokens: Tensor,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        for block in self.blocks:
            tokens = block(tokens, spatial_shape)
        return tokens


class ScaleBranch(nn.Module):
    """One patch-scale branch containing three DAMamba stages."""

    def __init__(
        self,
        in_channels: int,
        stage_dims: Sequence[int],
        stage_depths: Sequence[int],
        patch_size: int,
        *,
        state_dim: int,
        scan_chunk_size: int,
        drop_path_rates: Sequence[float],
    ) -> None:
        super().__init__()
        if len(stage_dims) != 3 or len(stage_depths) != 3:
            raise ValueError("ScaleBranch requires exactly three stages")

        self.patch_embed = PatchEmbed(
            in_channels,
            stage_dims[0],
            patch_size,
        )
        self.transitions = nn.ModuleList(
            nn.Linear(stage_dims[index], stage_dims[index + 1])
            for index in range(2)
        )

        stages = []
        depth_offset = 0
        rate_offset = 0
        for dim, depth in zip(stage_dims, stage_depths):
            stage_rates = drop_path_rates[rate_offset:rate_offset + depth]
            stages.append(
                DAMambaStage(
                    dim,
                    depth,
                    first_depth_index=depth_offset,
                    state_dim=state_dim,
                    scan_chunk_size=scan_chunk_size,
                    drop_path_rates=stage_rates,
                )
            )
            depth_offset += depth
            rate_offset += depth
        self.stages = nn.ModuleList(stages)
        self.output_norms = nn.ModuleList(
            nn.LayerNorm(dim) for dim in stage_dims
        )

    def forward(self, feature_map: Tensor) -> List[Tensor]:
        tokens, grid_shape = self.patch_embed(feature_map)
        outputs = []
        for stage_index, (stage, norm) in enumerate(
            zip(self.stages, self.output_norms)
        ):
            if stage_index:
                tokens = self.transitions[stage_index - 1](tokens)
            tokens = stage(tokens, grid_shape)
            normalized = norm(tokens)
            batch, _, channels = normalized.shape
            output = normalized.transpose(1, 2).reshape(
                batch,
                channels,
                *grid_shape,
            )
            outputs.append(output)
        return outputs


class MsGF(nn.Module):
    """Multi-scale Gated Fusion with channel-wise scale normalization."""

    def __init__(
        self,
        branch_channels: Sequence[int],
        out_channels: int,
        *,
        initial_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, out_channels, kernel_size=1)
            for channels in branch_channels
        )
        self.gating_convs = nn.ModuleList(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
            )
            for _ in branch_channels
        )
        # Softplus keeps the learned temperature strictly positive.
        inverse_softplus = math.log(math.expm1(initial_temperature))
        self.temperature_parameter = nn.Parameter(
            torch.tensor(inverse_softplus, dtype=torch.float32)
        )

    @property
    def temperature(self) -> Tensor:
        return F.softplus(self.temperature_parameter).clamp_min(1e-4)

    def forward(
        self,
        branch_features: Sequence[Tensor],
        target_size: Tuple[int, int],
    ) -> Tensor:
        if len(branch_features) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} branch features, "
                f"received {len(branch_features)}."
            )

        aligned_features = []
        gating_logits = []
        for feature, projection, gating_conv in zip(
            branch_features,
            self.projections,
            self.gating_convs,
        ):
            projected = projection(feature)
            if projected.shape[-2:] != target_size:
                projected = F.interpolate(
                    projected,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            aligned_features.append(projected)
            gate = torch.amax(gating_conv(projected), dim=(-2, -1))
            gating_logits.append(torch.tanh(gate))

        # [B, M, C]: softmax is explicitly applied across patch scales.
        logits = torch.stack(gating_logits, dim=1)
        weights = torch.softmax(logits / self.temperature, dim=1)
        stacked_features = torch.stack(aligned_features, dim=1)
        return torch.sum(weights[..., None, None] * stacked_features, dim=1)


class MultiScaleEncoder(nn.Module):
    """Parallel multi-scale encoder producing manuscript features F1--F5."""

    def __init__(
        self,
        in_channels: int,
        encoder_dims: Sequence[int],
        stage_depths: Sequence[int],
        *,
        patch_sizes: Sequence[int],
        state_dim: int,
        scan_chunk_size: int,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        if len(encoder_dims) != 5:
            raise ValueError("encoder_dims must describe F1 through F5")
        if len(stage_depths) < 3:
            raise ValueError("stage_depths must provide at least three values")
        if len(patch_sizes) < 2:
            raise ValueError("At least two patch scales are required")

        self.patch_sizes = tuple(patch_sizes)
        stage_dims = tuple(encoder_dims[1:4])
        stage_depths = tuple(stage_depths[:3])
        total_blocks = sum(stage_depths)
        branch_drop_rates = torch.linspace(
            0.0,
            drop_path_rate,
            total_blocks,
        ).tolist()

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                encoder_dims[0],
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.InstanceNorm2d(
                encoder_dims[0],
                eps=1e-5,
                affine=True,
            ),
        )
        self.branches = nn.ModuleList(
            ScaleBranch(
                encoder_dims[0],
                stage_dims,
                stage_depths,
                patch_size,
                state_dim=state_dim,
                scan_chunk_size=scan_chunk_size,
                drop_path_rates=branch_drop_rates,
            )
            for patch_size in patch_sizes
        )
        self.fusions = nn.ModuleList(
            MsGF(
                [stage_dims[stage_index]] * len(patch_sizes),
                stage_dims[stage_index],
            )
            for stage_index in range(3)
        )
        self.deep_projection = nn.Sequential(
            ConvNormAct(
                encoder_dims[3],
                encoder_dims[4],
                stride=2,
            ),
            ConvNormAct(encoder_dims[4], encoder_dims[4]),
        )

    def forward(self, image: Tensor) -> List[Tensor]:
        if image.ndim != 4:
            raise ValueError(
                "MS_ViDM expects a 2D image tensor [B, C, H, W]"
            )
        if image.shape[-2] % 32 or image.shape[-1] % 32:
            raise ValueError(
                "Input height and width must be divisible by 32, "
                f"received {tuple(image.shape[-2:])}."
            )

        f1 = self.stem(image)
        branch_outputs = [branch(f1) for branch in self.branches]

        input_height, input_width = image.shape[-2:]
        target_sizes = [
            (input_height // 4, input_width // 4),
            (input_height // 8, input_width // 8),
            (input_height // 16, input_width // 16),
        ]
        fused = [
            fusion(
                [outputs[index] for outputs in branch_outputs],
                target_sizes[index],
            )
            for index, fusion in enumerate(self.fusions)
        ]
        f2, f3, f4 = fused
        f5 = self.deep_projection(f4)
        return [f1, f2, f3, f4, f5]


class UpCatBlock(nn.Module):
    """Upsample, concatenate the encoder skip, and refine the result."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            ConvNormAct(in_channels + skip_channels, out_channels),
            ConvNormAct(out_channels, out_channels),
        )

    def forward(self, x: Tensor, skip: Optional[Tensor]) -> Tensor:
        if skip is None:
            x = F.interpolate(
                x,
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            )
            return self.refine(x)

        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.refine(torch.cat([x, skip], dim=1))


class UNetDecoder(nn.Module):
    """Five-step U-shaped decoder for encoder features F1--F5."""

    def __init__(
        self,
        encoder_dims: Sequence[int],
        decoder_dims: Sequence[int],
    ) -> None:
        super().__init__()
        if len(encoder_dims) != 5 or len(decoder_dims) != 5:
            raise ValueError("The encoder and decoder must each have five levels")

        blocks = []
        current_channels = encoder_dims[-1]
        for skip_channels, out_channels in zip(
            reversed(encoder_dims[:-1]),
            decoder_dims[:-1],
        ):
            blocks.append(
                UpCatBlock(
                    current_channels,
                    skip_channels,
                    out_channels,
                )
            )
            current_channels = out_channels
        blocks.append(UpCatBlock(current_channels, 0, decoder_dims[-1]))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != 5:
            raise ValueError("UNetDecoder expects features F1 through F5")
        x = features[-1]
        for block, skip in zip(
            self.blocks[:-1],
            reversed(features[:-1]),
        ):
            x = block(x, skip)
        return self.blocks[-1](x, None)


class MsD_UMamba(nn.Module):
    """Boundary-aware Multi-scale Differential U-Mamba segmentation model.

    Args:
        in_channels: Number of image channels.
        num_classes: Number of segmentation logits.
        encoder_depths: DAMamba block counts.  The first three values specify
            the three Mamba stages shown in the manuscript.
        encoder_dims: Channel counts for F1 through F5.
        decoder_dims: Output channels of the five decoder steps.
        img_size: Optional dataset metadata retained for factory compatibility.
            Forward inputs may use any size divisible by 32.
        spatial_dims: Input dimensionality. The current network supports 2D.
        patch_sizes: Patch sizes relative to the half-resolution stem map.
            ``(2, 4, 8)`` corresponds to effective image patches
            ``(4, 8, 16)``.
        state_dim: Latent SSM state dimension N.
        scan_chunk_size: Deprecated compatibility option. The official CUDA
            selective-scan kernel manages its own execution strategy.
        drop_path_rate: Maximum stochastic-depth probability.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        encoder_depths: Sequence[int] = (2, 2, 2, 2),
        encoder_dims: Sequence[int] = (16, 32, 48, 64, 128),
        decoder_dims: Sequence[int] = (128, 64, 48, 32, 16),
        *,
        img_size: Optional[int] = None,
        spatial_dims: int = 2,
        patch_sizes: Sequence[int] = (2, 4, 8),
        state_dim: int = 16,
        scan_chunk_size: int = 64,
        drop_path_rate: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        if in_channels < 1 or num_classes < 1:
            raise ValueError("in_channels and num_classes must be positive")
        if len(encoder_dims) != 5 or len(decoder_dims) != 5:
            raise ValueError(
                "encoder_dims and decoder_dims must each contain five values"
            )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.spatial_dims = spatial_dims
        self.encoder = MultiScaleEncoder(
            in_channels,
            encoder_dims,
            encoder_depths,
            patch_sizes=patch_sizes,
            state_dim=state_dim,
            scan_chunk_size=scan_chunk_size,
            drop_path_rate=drop_path_rate,
        )
        self.decoder = UNetDecoder(encoder_dims, decoder_dims)
        self.final = nn.Conv2d(
            decoder_dims[-1],
            num_classes,
            kernel_size=1,
        )

    def forward_features(self, image: Tensor) -> List[Tensor]:
        """Return F1--F5 for analysis and boundary-response visualization."""
        return self.encoder(image)

    def forward(self, image: Tensor) -> Tensor:
        features = self.forward_features(image)
        return self.final(self.decoder(features))


@register_model
def msd_umamba(pretrained: bool = False, **kwargs) -> MS_ViDM:
    """Construct MS_ViDM through timm's registry.

    Pretrained weights are not currently distributed. Standard metadata
    arguments injected by ``timm.create_model`` are accepted and discarded.
    """
    if pretrained:
        raise ValueError("Pretrained MS_ViDM weights are not available")
    for metadata_key in (
        "pretrained_cfg",
        "pretrained_cfg_overlay",
        "cache_dir",
    ):
        kwargs.pop(metadata_key, None)
    return MsD_UMamba(**kwargs)
