from __future__ import annotations

import copy
import os.path
import random
from collections.abc import Generator

from utils.config import DotDict, load_cfg_from_yaml, product_dict, init_curate, final_curate
from utils.experiment import get_experiment_from_config
from utils.tools import _parse_str

PATH_DICT = DotDict({'yaml': 'yaml-config',
                     'data': 'datasets',
                     'models': 'models',
                     'criteria': 'criteria'})


class SuperExperiment:
    def __init__(self, cfg: DotDict) -> None:
        cfg = check_config(cfg)
        self.base_config = load_cfg_from_yaml(cfg.base_config)
        init_curate(self.base_config)
        self.repeats = cfg.repeats
        del cfg.base_config, cfg.repeats
        self.raw_changes = cfg

    def _change_generator(self) -> Generator[DotDict, None, None]:
        return product_dict(**self.raw_changes)

    def _seed_generator(self):
        init = random.randint(0, 2147480000)
        for i in range(self.repeats):
            yield init + i

    def get_new_config(self, change: DotDict, exp_name: str) -> DotDict:
        change['train.exp_name'] = exp_name
        new_config = copy.deepcopy(self.base_config)
        new_config.replace_subset(change)
        return new_config

    def run_all(self):
        for change in self._change_generator():
            temp_change = copy.deepcopy(change)
            seed_gen = self._seed_generator()
            for i, seed in enumerate(seed_gen):
                change = copy.deepcopy(temp_change)
                exp_name = create_exp_name(change, i)
                change['manual_seed'] = seed
                curate_change(change)
                config = self.get_new_config(change, exp_name)
                run_single_experiment(config, exp_name)


def check_config(cfg: DotDict) -> DotDict:
    assert cfg.has('base_config')
    cfg.base_config = os.path.join(PATH_DICT.yaml, cfg.base_config)
    assert cfg.has('data')
    assert cfg.has('models')
    for k, name_list in cfg.models.items():
        cfg[f'models.{k}'] = name_list
    del cfg['models']
    if cfg.has('criteria'):
        for k, name_list in cfg.criteria.items():
            cfg[f'criteria.{k}'] = name_list
        del cfg['criteria']
    assert cfg.has('save_path')
    cfg.save_path = [cfg.save_path]
    if cfg.repeats is None:
        cfg.repeats = 1

    return cfg


def run_single_experiment(config: DotDict, exp_name: str):
    final_curate(config)
    experiment = get_experiment_from_config(config)
    new_bs = experiment.adjust_bs()
    info = decode_tag(exp_name)
    info.update({'batch_size': new_bs})
    run_no = info.pop('run')
    new_name = create_exp_name(info, run_no)
    for tr in config.train.trainable.values():
        if os.path.exists(os.path.join(tr.checkpoint.paths.model, f'{new_name}_{tr.nickname}.pth')):
            return
    if not new_name == exp_name:
        for tr in config.train.trainable.values():
            os.rmdir(tr.checkpoint.paths.result)
            os.mkdir(os.path.join(tr.checkpoint.paths.results_dir, new_name))
    tr = config.train.trainable.network
    temp = os.listdir(tr.checkpoint.paths.temp)
    if len(temp) > 0:
        temp = temp[-1]
        tr.load_checkpoint(temp, weights_only=False)
    experiment.change_exp_name(new_name)
    experiment.train_whole()
    experiment.save_results()


def curate_change(change: DotDict):
    for k, v in change.items():
        if type(v) is str and not k == 'save_path':
            if not v.endswith('.yaml'):
                v += '.yaml'
            change[k] = os.path.join(PATH_DICT.yaml, PATH_DICT[k.split('.')[0]], v)
    key = list(change.keys())
    val = list(change.values())
    for k, v in zip(key, val):
        del_flag = True
        if k in ['save_path', 'batch_size', 'manual_seed']:
            change[f'train.{k}'] = v
        elif k == 'lr':
            models = [key.split('.')[1] for key in change.keys() if 'models' in key]
            for model in models:
                change[f'train.trainable.{model}.optimizer.lr'] = v
        elif k == 'image_size':
            change['train.augmentation.size'] = v
        elif 'models' in k:
            change[f'train.trainable.{k.split(".")[1]}.model'] = v
        elif 'criteria' in k and not k.split('.')[0] == 'train':
            change[f'train.{k}'] = v
        else:
            del_flag = False
        if del_flag:
            del change[k]


def create_exp_name(change: DotDict, run_no: int, item_delim: str = '#', kw_delim: str = '@') -> str:
    if change.has('data'):
        name = f'{change.data}{item_delim}'
    else:
        name = ''
    if any(['models' in k for k in change.keys()]):
        name += item_delim.join([k.split('.')[1] + kw_delim + v for k, v in change.items() if 'models' in k]) + \
                item_delim
    if any(['criteria' in k for k in change.keys()]):
        name += item_delim.join([k.split('.')[1] + kw_delim + v for k, v in change.items()
                                 if 'criteria' in k and len(k.split('.')) < 4]) + item_delim
    for k, v in change.items():
        if any(key in k for key in ['models', 'criteria']) or k in ['save_path', 'data'] or len(k.split('.')) > 3:
            continue
        name += f'{k}{kw_delim}{v}{item_delim}'
    if run_no >= 0:
        name += f'run{kw_delim}{run_no}{item_delim}'
    return name[:-len(item_delim)]


def decode_tag(tag: str) -> DotDict:
    dataset_name = tag.split('#')[0]
    kw_tags = tag.split('#')[1:]
    decoded = {'data': dataset_name}
    for kw in kw_tags:
        k, v = kw.split('@')
        decoded[k] = _parse_str(v)
    return DotDict(decoded)
