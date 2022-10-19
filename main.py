import argparse
import random

import numpy as np
import torch
from torch.backends import cudnn

from utils.config import DotDict, load_cfg_from_yaml, init_curate, final_curate
from utils.experiment import get_experiment_from_config


def get_parser(desc: str = None, default_file: str = None, add_exp_name: bool = True) -> DotDict:
    if desc is None:
        desc = 'Ablation Study of U-Net, PyTorch'
    if default_file is None:
        default_file = 'yaml-config/redacted/new/unet.yaml'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--config', type=str, default=default_file)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_yaml(args.config)
    if add_exp_name:
        file_name = args.config.split('/')[-1].split('.')[0]
        cfg.train.exp_name = file_name
    return cfg


def set_manual_seed(seed: int = None) -> None:
    if seed is not None:                   # set every randomizer to a specific seed to be reproducible
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = False                             # CuDNN has heuristics as to which algorithm to pick
        cudnn.deterministic = True


def main():
    cfg = get_parser()
    init_curate(cfg)
    final_curate(cfg)
    set_manual_seed(cfg.train.manual_seed)
    experiment = get_experiment_from_config(cfg)
    experiment.adjust_bs()
    experiment.train_whole()
    experiment.save_results(speed_test=True)


if __name__ == '__main__':
    main()
