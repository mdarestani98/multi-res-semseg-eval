from __future__ import annotations

import copy
import itertools
import os
from collections.abc import Generator
from typing import List, Union, Any, Dict, Iterable

import numpy as np
import yaml

from dataset.augmentation import transforms


class DotDict(dict):
    """Nested dictionary to use as parent.child.child"""

    def __init__(self, nested_dict: dict = None):
        if nested_dict is None:
            nested_dict = {}
        for k, v in nested_dict.items():
            if type(v) is dict:
                nested_dict[k] = DotDict(v)
        super(DotDict, self).__init__(nested_dict)

    def has(self, item) -> bool:
        return item in self.keys()

    def replace(self, key_query: str, value: Any):
        temp_pointer = self
        keys = key_query.split('.')
        for k in keys[:-1]:
            temp_pointer = temp_pointer[k]
        temp_pointer[keys[-1]] = value

    def replace_subset(self, changes: DotDict):
        for k, v in changes.items():
            self.replace(k, v)

    def update(self, other: Dict[str, Any], **kwargs) -> None:
        for k in other.keys():
            self[k] = other[k]

    def __getattr__(self, item):
        if item not in self.keys():
            return None
        return super(DotDict, self).get(item)

    def __getitem__(self, key: Union[str, Iterable[str]]) -> Any:
        if isinstance(key, str):
            return super(DotDict, self).get(key)
        else:
            res = DotDict({})
            for k in key:
                res[k] = super(DotDict, self).get(k)
            return res

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        super(DotDict, self).__delitem__(item)

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        y = DotDict({})
        memo[id(self)] = y
        for key, value in self.items():
            y[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return y

    def __str__(self) -> str:
        text = '{\n'
        for k, v in self.items():
            if type(v) is np.ndarray:
                v_tab = f'numpy array of shape {v.shape}'
            else:
                v_tab = str(v).replace('\n', '\n\t')
            text += f'\t{k}: {v_tab},\n'
        if len(text) == 2:
            return text + '}'
        return text[:-2] + '\n}'


def load_cfg_from_yaml(file: str):
    """Load config from a .yaml file"""

    if not file.endswith('.yaml'):
        if os.path.exists(file + '.yaml'):
            file += '.yaml'
    with open(file, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg = DotDict(cfg)
    return cfg


def conditional_load_yaml(entry: Any) -> DotDict:
    if type(entry) is str:
        return load_cfg_from_yaml(entry)
    return entry


def list2dict(ls: Union[List[DotDict], DotDict], nickname: str = 'item') -> DotDict:
    if isinstance(ls, list):
        ls = list(map(lambda it: DotDict(it), ls))
    else:
        ls = [ls]
    for i, item in enumerate(ls):
        if not item.has('nickname'):
            item.nickname = f'{nickname}{i}'
    return DotDict({item.nickname: item for item in ls})


def init_curate(cfg: DotDict):
    assert cfg.has('data')
    assert cfg.has('train')
    if cfg.train.criteria is None:
        cfg.train.criteria = []
    cfg.train.criteria = list2dict(cfg.train.criteria, 'loss')
    assert cfg.train.has('trainable')
    cfg.train.trainable = list2dict(cfg.train.trainable, 'trainable')
    if cfg.train.frozen is None:
        cfg.train.frozen = []
    assert cfg.train.has('augmentation')
    assert cfg.train.has('metrics')


def final_curate(cfg: DotDict):
    cfg.data = conditional_load_yaml(cfg.data)
    cfg.train.augmentation.value_scale = cfg.data.value_scale
    cfg.train.augmentation.data_size = cfg.data.base_size
    for tr in cfg.train.trainable.values():
        tr.checkpoint.save = cfg.train.save_path
        create_paths(tr.checkpoint, cfg.data.name, cfg.train.exp_name)
        if tr.iterations is not None:
            tr.epochs = tr.iterations // (cfg.data.len * cfg.data.train_val_test[0] // cfg.train.batch_size)
        tr.scheduler.epochs = tr.epochs
        tr.model = conditional_load_yaml(tr.model)
        tr.model.device = cfg.train.device
        if tr.model.out_channel == 'data':
            tr.model.out_channel = cfg.data.no_classes
        if tr.output_keys is None:
            if tr.model.has('output_keys'):
                tr.output_keys = tr.model.output_keys
                tr.model.pop('output_keys')
            else:
                tr.output_keys = ['pred']
        assert 'pred' in tr.output_keys
        if tr.model.criteria is not None:
            tr.model.criteria = list2dict(tr.model.criteria, 'loss')
            cfg.train.criteria.update(tr.model.criteria)
            tr.model.pop('criteria')
    for k, criterion in cfg.train.criteria.items():
        cfg.train.criteria[k] = conditional_load_yaml(criterion)
        cfg.train.criteria[k].ignore_index = cfg.data.ignore_index
    for fr in cfg.train.frozen.values():
        fr.model = conditional_load_yaml(fr.model)
        fr.model.device = cfg.train.device
        if fr.output_keys is None:
            fr.output_keys = ['pred']
        assert 'pred' in fr.output_keys

    if not cfg.train.has('type'):
        if len(cfg.train.frozen) == 0:
            if len(cfg.train.trainable) == 1:
                cfg.train.type = 'simple'
            else:
                cfg.train.type = 'adversarial'
        else:
            if any(fr.nickname == 'teacher' for fr in cfg.train.frozen.values()):
                cfg.train.type = 'kd'

    if cfg.train.no_classes == 'data':
        cfg.train.no_classes = cfg.data.no_classes
    cfg.train.metrics.no_classes = cfg.train.no_classes

    if not cfg.train.has('display_metric'):
        cfg.train.display_metric = 'iou'

    cfg.train.writer = DotDict()
    cfg.train.writer.logdir = os.path.join(cfg.train.save_path, cfg.data.name, 'sw-logs')
    cfg.train.writer.exp_name = cfg.train.exp_name


def create_paths(cfg: DotDict, data_name: str, exp_name: str):
    path_list = [cfg.save, os.path.join(cfg.save, data_name)]
    cfg.paths = DotDict()
    cfg.paths.sw_log = os.path.join(path_list[-1], 'sw-logs')
    cfg.paths.model = os.path.join(path_list[-1], 'models')
    cfg.paths.results_dir = os.path.join(path_list[-1], 'results')
    cfg.paths.result = os.path.join(path_list[-1], 'results', exp_name)
    cfg.paths.temp = os.path.join(path_list[-1], 'temp')
    cfg.paths.fps = os.path.join(path_list[-1], 'latency')
    path_list += cfg.paths.values()
    for p in path_list:
        if not os.path.exists(p):
            os.mkdir(p)


def get_transforms_from_config(cfg: DotDict) -> DotDict:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if type(cfg.size) in [list, tuple]:
        cfg.size = tuple(cfg.size)
        if type(cfg.data_size) not in [list, tuple]:
            cfg.data_size = [cfg.data_size, cfg.data_size]
        base_scale = max([i / j for i, j in zip(cfg.size, cfg.data_size)])
    else:
        base_scale = cfg.size / cfg.data_size
        cfg.size = (cfg.size, cfg.size)
    if type(cfg.scale) not in [list, tuple]:
        cfg.scale = (base_scale * 1.1, base_scale * 1.1 * cfg.scale)

    train_transform = transforms.Compose([
        transforms.RandomGaussianBlur(),
        transforms.RandomHorizontalFlip(),
        transforms.RandScale(cfg.scale),
        transforms.Crop(cfg.size, crop_type='rand', padding=mean, value_scale=cfg.value_scale),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std, value_scale=cfg.value_scale)
    ])
    inference_transform = transforms.Compose([
        transforms.Resize(cfg.size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std, value_scale=cfg.value_scale)
    ])
    return DotDict({'train': train_transform,
                    'val': inference_transform,
                    'test': inference_transform})


def product_dict(**kwargs) -> Generator[DotDict, None, None]:
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield DotDict(dict(zip(keys, instance)))
