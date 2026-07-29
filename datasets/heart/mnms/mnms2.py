"""
Deep Learning Segmentation of the Right Ventricle in Cardiac MRI: The M&ms Challenge
https://doi.org/10.1109/JBHI.2023.3267857
https://www.ub.edu/mnms-2/
"""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Sequence, Tuple

import monai.transforms as MT
import numpy as np
import torch
from monai.utils import GridSampleMode

from ..._factory import register_dataset
from ..dataset import HeartDataset
from ..utils import (get_center_norm_pct, load_frame_image,
                     load_frame_mask)

__all__ = ['MnMs2', ]


class MnMs2(HeartDataset):
    __index: Dict = None

    def __init__(
        self,
        root: Path | str,
        train: bool = True,
        keys: Sequence[str] | str | None = None,
        img_size: int = 224,
        **kwargs
    ) -> None:
        super().__init__()

        self.root = Path(root, 'MnMs2').expanduser()
        self._create_cache()

        self.train = train
        if keys is None:
            if train:
                self.keys = ('training', 'validation',)
            else:
                self.keys = ('testing',)
        else:
            self.keys = (keys,) if isinstance(keys, str) else keys
        assert all(self.__index.get(key) is not None for key in self.keys)

        self.img_size = img_size
        self.eval_transform = MT.Compose([
            MT.MapLabelValued('mask', (1, 3), (3, 1), dtype=torch.int),
            MT.ResizeWithPadOrCropd(('image', 'mask'), img_size),
        ])
        self.train_transform = MT.Compose([
            MT.MapLabelValued('mask', (1, 3), (3, 1), dtype=torch.int),
            MT.RandZoomd(('image', 'mask'), 1., 0.9, 1.1, mode=(GridSampleMode.BILINEAR, GridSampleMode.NEAREST), keep_size=False),
            MT.RandRotated(
                ('image', 'mask'),
                range_x=math.radians(15),
                prob=0.5,
                mode=(GridSampleMode.BILINEAR, GridSampleMode.NEAREST),
            ),
            MT.RandFlipd(('image', 'mask'), prob=0.5),
            MT.RandSpatialCropd(('image', 'mask'), img_size),
        ]) if train else self.eval_transform

    def __len__(self) -> int:
        return sum(len(self.__index[key]) for key in self.keys)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key, idx = self._get_value_by_index(index)
        cache = self.cache_folder/'{}{:04d}.pt' .format(key, idx)
        sample = torch.load(cache, weights_only=False)
        sample = self.train_transform(sample)
        return sample['image'].to(torch.float32), sample['mask'].to(torch.int64)

    @property
    def cache_file(self) -> Path:
        return self.cache_folder/'index.json'

    @property
    def cache_folder(self) -> Path:
        return self.root/'cache'

    @property
    def raw_folder(self) -> Path:
        return self.root/'raw'

    def _create_cache(self, expire: int = 7) -> None:
        self.cache_folder.mkdir(exist_ok=True)
        if self.cache_file.exists():
            with self.cache_file.open('r') as fp:
                index = json.load(fp)
                cache_time = datetime.fromisoformat(index['time'])
                if datetime.now() - cache_time < timedelta(days=expire):
                    self.__index = index
                    return

        index = {
            'time': datetime.now().isoformat(),
            'training': [],
            'validation': [],
            'testing': [],
        }
        for file in self.raw_folder.rglob('*SA_CINE.nii.gz'):
            patient = file.parent.stem
            print('Creating cache for {} ...'.format(patient))
            key = 'training' if patient <= '160' else 'validation' if patient <= '200' else 'testing'
            center, norm, pct = get_center_norm_pct(file)

            for frame in ('ED', 'ES'):
                image_path = file.with_name('{}_SA_{}.nii.gz'.format(patient, frame))
                mask_path = file.with_name('{}_SA_{}_gt.nii.gz'.format(patient, frame))
                image = load_frame_image(image_path, center, *norm, *pct)
                mask = load_frame_mask(mask_path, center)

                for slice_idx in range(mask.size(-1)):
                    if torch.any(mask[..., slice_idx] > 0):
                        info = {
                            'patient': file.parent.relative_to(self.raw_folder).as_posix(),
                            'frame': frame,
                            'slice': slice_idx,
                        }
                        index[key].append(info)

                        sample = {
                            'image': image[..., slice_idx].unsqueeze(0),
                            'mask': mask[..., slice_idx].unsqueeze(0),
                        }
                        torch.save(sample, self.cache_folder/'{}{:04d}.pt'.format(key, len(index[key])-1))

        with self.cache_file.open('w') as fp:
            json.dump(index, fp)
        self.__index = index

    def _get_value_by_index(self, index: int) -> Tuple[str, int]:
        current_index = 0
        for key in self.keys:
            lst = self.__index[key]
            if index < current_index + len(lst):
                return key, index - current_index
            current_index += len(lst)
        else:
            raise IndexError(f'Index {index} out of range')

@register_dataset
def mnms2(root, **kwargs):
    dataset = MnMs2(root, **kwargs)
    cfg = dict(
        in_channels=1,
        img_size=dataset.img_size,
        num_classes=4,
    )
    return dataset, cfg
