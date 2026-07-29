from pathlib import Path
from typing import Tuple

import monai.transforms as MT
import numpy as np
import torch
from einops import rearrange
from monai.utils import InterpolateMode


def get_center(image: np.ndarray) -> Tuple[float, float]:
    assert image.ndim == 4, "image must be 4D, got {}D".format(image.ndim)

    var = np.var(image, axis=-1)

    mean_coords_list = []
    for c in range(np.size(var, -1)):
        var_c = var[:, :, c]
        top_indices = np.argpartition(var_c.flatten(), -100)[-100:]
        coords = np.column_stack(np.unravel_index(top_indices, var_c.shape))
        mean_coords = np.mean(coords, 0)
        mean_coords_list.append(mean_coords)
    final_mean_coords = np.mean(mean_coords_list, axis=0)
    return tuple(final_mean_coords)


def get_center_norm_pct(fpath: Path):
    loader = MT.LoadImage(image_only=True, dtype=np.float32)
    image = loader(fpath).numpy()

    center = get_center(image)

    min, max = np.percentile(image, (1, 95))

    data_95 = image[image < max]
    mean, std = np.mean(data_95), np.std(data_95)

    return center, (mean, std), (min, max)


def load_cine_image(fpath: Path, center: Tuple[float, float], subtrahend: float, divisor: float, min: float, max: float):
    data = MT.LoadImage(dtype=torch.float)(fpath)
    h, w, d, t = data.shape
    ph, pw = data.pixdim[:2]
    ch, cw = center

    data = rearrange(data, 'h w d t -> (d t) h w')
    data = MT.clip(data, min, max)
    data = MT.NormalizeIntensity(subtrahend, divisor)(data)
    data = MT.Resize((h*ph, w*pw), mode=InterpolateMode.BILINEAR)(data)
    data = MT.SpatialCrop((ch*ph, cw*pw), (256, 256))(data)
    data = MT.ResizeWithPadOrCrop(256)(data)
    data = rearrange(data, '(d t) h w -> h w d t', d=d, t=t)

    return data.to(torch.float)


def load_cine_mask(fpath: Path, center: Tuple[float, float]):
    data = MT.LoadImage(dtype=torch.int)(fpath)
    h, w, d, t = data.shape
    ph, pw = data.pixdim[:2]
    ch, cw = center

    data = rearrange(data, 'h w d t -> (d t) h w')
    data = MT.Resize((h*ph, w*pw), mode=InterpolateMode.NEAREST_EXACT)(data)
    data = MT.SpatialCrop((ch*ph, cw*pw), (256, 256))(data)
    data = MT.ResizeWithPadOrCrop(256)(data)
    data = rearrange(data, '(d t) h w -> h w d t', d=d, t=t)

    return data.to(torch.int)


def load_frame_image(fpath: Path, center: Tuple[float, float], subtrahend: float, divisor: float, min: float, max: float):
    return load_frame_img(fpath, 2, center, subtrahend, divisor, min, max)


def load_frame_img(
    fpath: Path, spatial_dims: int,
    center: Tuple[float, float],
    subtrahend: float, divisor: float,
    min: float, max: float,
    img_size: int = 256,
) -> torch.Tensor:
    data = MT.LoadImage(dtype=torch.float)(fpath)
    data = MT.clip(data, min, max)
    data = MT.NormalizeIntensity(subtrahend, divisor)(data)

    if data.ndim != 3:
        raise ValueError

    h, w, d = data.shape
    ph, pw, pd = data.pixdim[:3]
    ch, cw = center
    hh, ww, dd = h*ph, w*pw, d*pd
    chh, cww, cdd = ch*ph, cw*pw, (dd-1)/2

    if spatial_dims == 2:
        data = rearrange(data, 'h w d -> d h w')
        data = MT.Resize((hh, ww), mode=InterpolateMode.BILINEAR)(data)
        data = MT.SpatialCrop((chh, cww), roi_size=img_size)(data)
        data = MT.ResizeWithPadOrCrop(256)(data)
        data = rearrange(data, 'd h w -> h w d')
    elif spatial_dims == 3:
        data = rearrange(data, 'h w d -> 1 h w d')
        data = MT.Resize((hh, ww, dd), mode=InterpolateMode.TRILINEAR)(data)
        data = MT.SpatialCrop((chh, cww, cdd), roi_size=img_size)(data)
        data = MT.ResizeWithPadOrCrop(256)(data)
        data = rearrange(data, '1 h w d -> h w d')
    else:
        raise ValueError

    return data.to(torch.float)


def load_frame_mask(fpath: Path, center: Tuple[float, float]):
    return load_frame_msk(fpath, 2, center)


def load_frame_msk(
    fpath: Path, spatial_dims: int,
    center: Tuple[float, float],
    img_size: int = 256,
) -> torch.Tensor:
    data = MT.LoadImage(dtype=torch.float)(fpath)

    if data.ndim != 3:
        raise ValueError

    h, w, d = data.shape
    ph, pw, pd = data.pixdim[:3]
    ch, cw = center
    hh, ww, dd = h*ph, w*pw, d*pd
    chh, cww, cdd = ch*ph, cw*pw, (dd-1)/2

    if spatial_dims == 2:
        data = rearrange(data, 'h w d -> d h w')
        data = MT.Resize((hh, ww), mode=InterpolateMode.NEAREST_EXACT)(data)
        data = MT.SpatialCrop((chh, cww), roi_size=img_size)(data)
        data = MT.ResizeWithPadOrCrop(256)(data)
        data = rearrange(data, 'd h w -> h w d')
    elif spatial_dims == 3:
        data = rearrange(data, 'h w d -> 1 h w d')
        data = MT.Resize((hh, ww, dd), mode=InterpolateMode.NEAREST_EXACT)(data)
        data = MT.SpatialCrop((chh, cww, cdd), roi_size=img_size)(data)
        data = MT.ResizeWithPadOrCrop(256)(data)
        data = rearrange(data, '1 h w d -> h w d')
    else:
        raise ValueError

    return data.to(torch.int)
