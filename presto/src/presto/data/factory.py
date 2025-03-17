#!/usr/bin/env python3

from dataclasses import dataclass, replace
import pickle
import os

import numpy as np
import torch as th

import h5py
import json

from presto.data.presto_shelf import PrestoDatasetShelf
from presto.network.layers import SinusoidalPositionalEncoding


@dataclass
class DataConfig:
    dataset_type: str = 'presto'  # or 'mnist'
    dataset_dir: str = '/input/presto-simple'
    relative: bool = False
    start_zero: bool = False
    normalize: bool = False
    add_task_cond: bool = False
    load_cloud: bool = False
    shelf: PrestoDatasetShelf.Config = PrestoDatasetShelf.Config()

    embed_init_goal: int = 0

    def __post_init__(self):
        self.shelf = replace(self.shelf,
                             normalize=self.normalize,
                             embed_init_goal=self.embed_init_goal)


def get_dataset(cfg: DataConfig,
                split: str = 'train',
                device: str = 'cuda:0'):
    if cfg.dataset_type == 'shelf':
        return PrestoDatasetShelf(replace(cfg.shelf, device=device),
                                 split)
    else:
        raise ValueError(F'Unknown dataset={cfg.dataset_type}')
