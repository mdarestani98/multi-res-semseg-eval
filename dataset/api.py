from typing import Tuple, Callable

from torch.utils.data import Dataset

from dataset import cityscapes, camvid
from utils.config import DotDict

DATASET = {'cityscapes': cityscapes.Handler, 'camvid': camvid.Handler}
DEFAULT = DATASET['cityscapes']


def get_dataset_from_config(cfg: DotDict, transforms: Tuple[Callable, ...] = None) -> Tuple[Dataset, ...]:
    handler = DATASET.get(cfg.name, DEFAULT)
    return handler.get_from_config(cfg, transforms)
