import os
import re
from typing import Tuple, Callable, Any, Union

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

from utils.config import DotDict
from utils.tools import DatasetHandler, normalize_array


class BaseDataset(Dataset):
    """Pseudo-abstract class for dataset creation"""
    def __init__(self, root: str, df: pd.DataFrame) -> None:
        super(BaseDataset, self).__init__()
        self.root = root
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Any:
        pass


class SegDataset(BaseDataset):
    """Semantic-Segmentation-ready dataset class"""
    def __init__(self, root: str, df: pd.DataFrame, transform: Callable = None) -> None:
        super().__init__(root, df)
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, str], ...]:
        row = self.df.iloc[[idx]].values.tolist()[0]
        image_path, annot_path = row[0], row[1]
        name = re.split(r'[ /\\]', image_path)[-1]
        image = cv2.imread(image_path)
        annot = cv2.imread(annot_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        edge = _generate_edges(label=annot)
        if self.transform is not None:
            image, annot, edge = self.transform([image, annot, edge])
        return image, annot, name, edge


class GeneralHandler(DatasetHandler):
    @staticmethod
    def get_from_config(cfg: DotDict, transforms: Tuple[Callable, ...] = None) -> Tuple[BaseDataset, ...]:
        """To get split datasets from config settings provided"""
        if transforms is None:
            transforms = [None] * 3
        elif len(transforms) == 1:
            transforms = transforms * 3
        if isinstance(cfg.df, str):
            df_path = os.path.join(cfg.root, cfg.df)
            if os.path.exists(df_path):
                df = pd.read_pickle(df_path)
            else:
                raise FileNotFoundError(f'pickle file for paths not found.')
            test_df, rem = train_test_split(df, train_size=normalize_array(cfg.train_val_test)[2], shuffle=True)
            val_df, train_df = train_test_split(rem, train_size=normalize_array(cfg.train_val_test[:2])[1], shuffle=True)
        elif isinstance(cfg.df, list):
            train_df = pd.read_pickle(os.path.join(cfg.root, cfg.df[0]))
            val_df = pd.read_pickle(os.path.join(cfg.root, cfg.df[1]))
            test_df = pd.read_pickle(os.path.join(cfg.root, cfg.df[2]))
        else:
            raise TypeError('dataset dataframe config should be string or list of strings.')
        return tuple([SegDataset(cfg.root, p, t) for (p, t) in zip([train_df, val_df, test_df], transforms)])


def _generate_edges(label: np.ndarray, edge_size: int = 4) -> np.ndarray:
    edge = cv2.Canny(label, 0.1, 0.2)
    kernel = np.ones((edge_size, edge_size), np.uint8)
    edge = (cv2.dilate(edge, kernel, iterations=1) > 50) * 1.0
    return edge
