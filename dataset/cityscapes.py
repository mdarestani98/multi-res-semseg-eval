import glob
import os.path
import pickle
import re
from collections import namedtuple
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import tqdm

from dataset.general import GeneralHandler

Label = namedtuple('Label', ['name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color'])

labels = [
    #      name                     id  trainId  category   catId  hasInstances ignoreInEval    color
    Label('unlabeled',              0,  255,    'void',         0, False,       True,           (0, 0, 0)),
    Label('ego vehicle',            1,  255,    'void',         0, False,       True,           (0, 0, 0)),
    Label('rectification border',   2,  255,    'void',         0, False,       True,           (0, 0, 0)),
    Label('out of roi',             3,  255,    'void',         0, False,       True,           (0, 0, 0)),
    Label('static',                 4,  255,    'void',         0, False,       True,           (0, 0, 0)),
    Label('dynamic',                5,  255,    'void',         0, False,       True,           (111, 74, 0)),
    Label('ground',                 6,  255,    'void',         0, False,       True,           (81, 0, 81)),
    Label('road',                   7,  0,      'flat',         1, False,       False,          (128, 64, 128)),
    Label('sidewalk',               8,  1,      'flat',         1, False,       False,          (244, 35, 232)),
    Label('parking',                9,  255,    'flat',         1, False,       True,           (250, 170, 160)),
    Label('rail track',             10, 255,    'flat',         1, False,       True,           (230, 150, 140)),
    Label('building',               11, 2,      'construction', 2, False,       False,          (70, 70, 70)),
    Label('wall',                   12, 3,      'construction', 2, False,       False,          (102, 102, 156)),
    Label('fence',                  13, 4,      'construction', 2, False,       False,          (190, 153, 153)),
    Label('guard rail',             14, 255,    'construction', 2, False,       True,           (180, 165, 180)),
    Label('bridge',                 15, 255,    'construction', 2, False,       True,           (150, 100, 100)),
    Label('tunnel',                 16, 255,    'construction', 2, False,       True,           (150, 120, 90)),
    Label('pole',                   17, 5,      'object',       3, False,       False,          (153, 153, 153)),
    Label('polegroup',              18, 255,    'object',       3, False,       True,           (153, 153, 153)),
    Label('traffic light',          19, 6,      'object',       3, False,       False,          (250, 170, 30)),
    Label('traffic sign',           20, 7,      'object',       3, False,       False,          (220, 220, 0)),
    Label('vegetation',             21, 8,      'nature',       4, False,       False,          (107, 142, 35)),
    Label('terrain',                22, 9,      'nature',       4, False,       False,          (152, 251, 152)),
    Label('sky',                    23, 10,     'sky',          5, False,       False,          (70, 130, 180)),
    Label('person',                 24, 11,     'human',        6, True,        False,          (220, 20, 60)),
    Label('rider',                  25, 12,     'human',        6, True,        False,          (255, 0, 0)),
    Label('car',                    26, 13,     'vehicle',      7, True,        False,          (0, 0, 142)),
    Label('truck',                  27, 14,     'vehicle',      7, True,        False,          (0, 0, 70)),
    Label('bus',                    28, 15,     'vehicle',      7, True,        False,          (0, 60, 100)),
    Label('caravan',                29, 255,    'vehicle',      7, True,        True,           (0, 0, 90)),
    Label('trailer',                30, 255,    'vehicle',      7, True,        True,           (0, 0, 110)),
    Label('train',                  31, 16,     'vehicle',      7, True,        False,          (0, 80, 100)),
    Label('motorcycle',             32, 17,     'vehicle',      7, True,        False,          (0, 0, 230)),
    Label('bicycle',                33, 18,     'vehicle',      7, True,        False,          (119, 11, 32)),
    Label('license plate',          -1, 255,    'vehicle',      7, False,       True,           (0, 0, 142)),
]

id2label = {label.id: label for label in labels}


Handler = GeneralHandler


def create_train_id(root: str) -> None:
    path_list = glob.glob(os.path.join(root, '**', '*_labelIds.png'), recursive=True)
    title = f'Converting LabelIds to TrainIds'
    progress = tqdm.tqdm(path_list, total=len(path_list), leave=True, ncols=150, miniters=1, smoothing=0.9, desc=title)
    for _, path in enumerate(progress):
        file = re.split(r'[\\/]', path)[-1]
        progress.set_description(f'{title} of {file}')
        label_ids = cv2.imread(path)
        train_ids = np.zeros_like(label_ids)
        for label_id in [label.id for label in labels]:
            idx = (label_ids == label_id)
            train_ids[idx] = id2label[label_id].trainId
        cv2.imwrite(f'{path[:-13]}_trainIds.png', train_ids)
    progress.close()


def create_dataframe(root: str, save: bool = True) -> Tuple[pd.DataFrame, ...]:
    splits = ['train', 'val', 'test']
    dfs = []
    for split in splits:
        df = pd.DataFrame(columns=['image', 'annotation'])
        image_list = glob.glob(os.path.join(root, 'leftImg8bit', split, '**', '*.png'), recursive=True)
        title = f'Gathering file addresses at {split}'
        progress = tqdm.tqdm(image_list, total=len(image_list), leave=True, ncols=150, miniters=1, smoothing=0.9,
                             desc=title)
        for _, path in enumerate(progress):
            file = re.split(r'[\\/]', path)[-1][:-16]
            city = file.split('_')[0]
            corr_annot_path = os.path.join(root, 'gtFine', split, city, f'{file}_gtFine_trainIds.png')
            if os.path.exists(corr_annot_path):
                row = pd.DataFrame({'image': [path], 'annotation': [corr_annot_path]})
                df = pd.concat([df, row])
        df = df.reset_index(drop=True)
        dfs.append(df)
        if save:
            df.to_pickle(os.path.join(root, f'{split}.pkl'), protocol=pickle.HIGHEST_PROTOCOL)
    return tuple(dfs)


def prepare_data(root: str) -> None:
    train_id_list = glob.glob(os.path.join(root, '**', '*_trainIds.png'), recursive=True)
    if len(train_id_list) == 0:
        create_train_id(root)
    splits = ['train', 'val', 'test']
    if not all([os.path.exists(os.path.join(root, f'{split}.pkl')) for split in splits]):
        create_dataframe(root, save=True)
