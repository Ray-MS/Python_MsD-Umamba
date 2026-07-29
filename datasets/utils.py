import random
import time
import zipfile
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np
import urllib3
from torchvision.datasets.utils import check_integrity


class SplitMode(Enum):
    TRAIN = auto()
    VALID = auto()
    TEST = auto()


def download(host: str, folder: Path, filename: str, md5: str | None = None) -> None:
    folder.mkdir(exist_ok=True)
    url = f'{host}/{filename}'
    resp = urllib3.PoolManager().request('GET', url, preload_content=False)
    if resp.status == 200:
        content_disposition = resp.headers.get('Content-Disposition')
        if content_disposition:
            filename = content_disposition.split('filename=')[1].strip('"')
    if not check_integrity(folder / filename, md5):
        with open(folder / filename, 'wb') as f:
            for chunk in resp.stream(1024 * 1024):
                f.write(chunk)


def extract_zip(zip_path: Path, extract_dir: Path) -> bool:
    zip_stem = zip_path.stem

    with zipfile.ZipFile(zip_path, 'r') as zf:
        top_level_items = set()

        for name in zf.namelist():
            parts = Path(name).parts
            if parts:
                top_level_items.add(parts[0])

        extract_base = extract_dir if len(top_level_items) == 1 else extract_dir / zip_stem
        extract_base.mkdir(parents=True, exist_ok=True)

        for member in zf.infolist():
            member_path = Path(member.filename)

            if member.is_dir():
                continue

            target_path = extract_base / member_path
            if target_path.exists():
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src:
                target_path.write_bytes(src.read())


def load_nib(file: str | Path):
    import nibabel as nib

    file = Path(file)
    if not file.exists():
        raise FileNotFoundError
    data: nib.Nifti1Image = nib.load(file)
    return np.asarray(data.get_fdata(), data.get_data_dtype())


def shuffle(*args, seed: Optional[int] = None):
    seed = seed or int(time.time())
    for arg in args:
        random.seed(seed)
        random.shuffle(arg)
