from pathlib import Path

from torch.utils.data import Dataset


class HeartDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    @property
    def cache_folder(self) -> Path:
        return self.root/'cache'

    @property
    def raw_folder(self) -> Path:
        return self.root/'raw'
