import json
from os.path import join, expanduser, expandvars

import numpy as np
import torch
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


class iNatDataset(Dataset):

    HIERARCHY_LEVEL = 7

    HIERARCHY = [
        "species",
        "genus",
        "family",
        "order",
        "class",
        "phylum",
        "kingdom",
    ]

    def __init__(
        self,
        data_dir: str,
        mode: str = 'train',
        hierarchy_mode: str = 'base',
        transform=None,
    ) -> NoneType:
        super().__init__()
        assert hierarchy_mode in ["full", "base"]

        self.data_dir = expandvars(expanduser(data_dir))
        self.mode = mode
        self.hierarchy_mode = hierarchy_mode
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
        for splt in mode:
            with open(join(self.data_dir, f'Inat_dataset_splits/Inaturalist_{splt}_set1.txt')) as f:
                paths = f.read().split("\n")
                paths.remove("")
            self.paths.extend([join(self.data_dir, pth) for pth in paths])

        with open(join(self.data_dir, 'train2018.json')) as f:
            db = json.load(f)['categories']
            self.db = {}
            for x in db:
                _ = x.pop("name")
                id_ = x.pop("id")
                x["species"] = id_
                self.db[id_] = x

        self.labels_name = [int(x.split("/")[-2]) for x in self.paths]

        self.hierarchy_name = {hier: set() for hier in self.HIERARCHY}
        for x in self.db.values():
            for hier in self.HIERARCHY:
                self.hierarchy_name[hier].add(x[hier])

        self.hierarchy_name_to_id = {hier: {} for hier in self.HIERARCHY}
        self.hierarchy_id_to_name = {hier: {} for hier in self.HIERARCHY}
        for hier in self.HIERARCHY:
            self.hierarchy_name_to_id[hier] = {cl: i for i, cl in enumerate(sorted(set(self.hierarchy_name[hier])))}
            self.hierarchy_id_to_name[hier] = {i: cl for i, cl in enumerate(sorted(set(self.hierarchy_name[hier])))}

        labels = []
        for x in self.labels_name:
            lb = []
            for hier in self.HIERARCHY:
                lb.append(self.hierarchy_name_to_id[hier][self.db[x][hier]])
            labels.append(lb)

        if self.hierarchy_mode == 'base':
            self.super_label_names = [x.split("/")[-3] for x in self.paths]
            super_labels_to_id = {scl: i for i, scl in enumerate(sorted(set(self.super_label_names)))}
            super_labels = [super_labels_to_id[x] for x in self.super_label_names]
            labels = [[x[0], super_labels[i]] for i, x in enumerate(labels)]

            self.super_id_to_names = {value: key for key, value in super_labels_to_id.items()}
            self.HIERARCHY_LEVEL = 2

        self.labels = np.stack(labels)
        self.labels = set_labels_to_range(self.labels)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, torch.from_numpy(self.labels[idx]), torch.tensor(idx)
