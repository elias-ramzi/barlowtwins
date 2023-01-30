from os.path import join, expanduser, expandvars
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

NoneType = type(None)


def set_labels_to_range(labels: np.ndarray) -> np.ndarray:
    """
    set the labels so it follows a range per level of semantic
    """
    new_labels = []
    for lvl in range(labels.shape[1]):
        unique = sorted(set(labels[:, lvl]))
        conversion = {x: i for i, x in enumerate(unique)}
        new_lvl_labels = [conversion[x] for x in labels[:, lvl]]
        new_labels.append(new_lvl_labels)

    return np.stack(new_labels, axis=1)


class SOPDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        mode: str = 'train',
        transform: Optional[nn.Module] = None,
    ) -> NoneType:
        super().__init__()
        self.data_dir = expandvars(expanduser(data_dir))
        self.mode = mode
        self.transform = transform

        if mode == 'train':
            mode = ['train']
        elif mode == 'test':
            mode = ['test']
        elif mode == 'all':
            mode = ['train', 'test']
        else:
            raise ValueError(f"Mode unrecognized {mode}")

        self.paths = []
        labels = []
        super_labels = []
        for splt in mode:
            gt = pd.read_csv(join(self.data_dir, f'Ebay_{splt}.txt'), sep=' ')
            self.paths.extend(gt["path"].apply(lambda x: join(self.data_dir, x)).tolist())
            labels.extend((gt["class_id"] - 1).tolist())
            super_labels.extend((gt["super_class_id"] - 1).tolist())

        self.labels = np.stack([labels, super_labels], axis=1)
        self.labels = set_labels_to_range(self.labels)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, torch.from_numpy(self.labels[idx]), torch.tensor(idx)
