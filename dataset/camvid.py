import glob
import os.path
import pickle
import re

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple

from dataset.general import GeneralHandler
from utils import tools


Handler = GeneralHandler


def load_class_dict(root: str):
    class_df = pd.read_csv(os.path.join(root, 'class_dict.csv'))
    colors = [(r, g, b) for _, (r, g, b) in class_df[['r', 'g', 'b']].iterrows()]
    class_df['color'] = colors
    class_df = class_df.drop(columns=['class_11', 'r', 'g', 'b'])
    class_dict = {}
    for _, (color, train_id) in class_df[['color', 'trainId']].iterrows():
        class_dict[color] = train_id
    return class_dict


def convert_annotations(root: str):
    assert os.path.exists(os.path.join(root, 'class_dict.csv'))
    class_dict = load_class_dict(root)
    splits = ['train', 'val', 'test']
    for split in splits:
        annot_root = os.path.join(root, f'{split}_labels')
        file_list = os.listdir(annot_root)
        progress = tqdm(tools.IteratorTimer(file_list), ncols=120, total=len(file_list), smoothing=0.9, miniters=1,
                        leave=True, desc=f'Converting annotations to train IDs of {split} images')
        for _, file in enumerate(progress):
            if file.endswith('.png'):
                filename = file[:-6]
                mask = cv2.imread(os.path.join(annot_root, file))
                maskout = np.zeros((mask.shape[0], mask.shape[1]), dtype=int)
                for k in class_dict.keys():
                    idx = (mask == np.expand_dims(k[::-1], (0, 1)))
                    idx = (idx.sum(2) == 3)
                    maskout[idx] = class_dict[k]
                cv2.imwrite(os.path.join(annot_root, filename + '_gtFine_color.png'), mask)
                cv2.imwrite(os.path.join(annot_root, filename + '_gtFine_labelTrainIds.png'), maskout)
                os.remove(os.path.join(annot_root, file))


def create_dataframe(root: str, save: bool = True) -> Tuple[pd.DataFrame, ...]:
    splits = ['train', 'val', 'test']
    dfs = []
    for split in splits:
        df = pd.DataFrame(columns=['image', 'annotation'])
        image_list = glob.glob(os.path.join(root, split, '*.png'))
        title = f'Gathering file addresses at {split}'
        progress = tqdm(image_list, total=len(image_list), leave=True, ncols=150, miniters=1, smoothing=0.9,
                        desc=title)
        for _, path in enumerate(progress):
            file = re.split(r'[\\/]', path)[-1][:-4]
            corr_annot_path = os.path.join(root, split + '_labels', f'{file}_gtFine_labelTrainIds.png')
            if os.path.exists(corr_annot_path):
                row = pd.DataFrame({'image': [path], 'annotation': [corr_annot_path]})
                df = pd.concat([df, row])
        df = df.reset_index(drop=True)
        dfs.append(df)
        if save:
            df.to_pickle(os.path.join(root, f'{split}.pkl'), protocol=pickle.HIGHEST_PROTOCOL)
    return tuple(dfs)
