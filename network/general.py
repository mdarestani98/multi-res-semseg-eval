import torch.optim
from torch import nn
from typing import Iterator

from network import scheduler
from network.segmentation import fastscnn, unet, pidnet, hrnet, sfnet, ppliteseg, regseg, bisenet, ddrnet
from utils.config import DotDict


NETWORK = {'unet': unet.Handler, 'hrnet': hrnet.Handler, 'fastscnn': fastscnn.Handler, 'pidnet': pidnet.Handler,
           'sfnet': sfnet.Handler, 'ppliteseg': ppliteseg.Handler, 'regseg': regseg.Handler, 'bisenet': bisenet.Handler,
           'ddrnet': ddrnet.Handler}
DEFAULT_NETWORK = NETWORK['unet']

OPTIMIZER = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}
DEFAULT_OPTIMIZER = OPTIMIZER['sgd']

SCHEDULER = {'poly': scheduler.Poly}
DEFAULT_SCHEDULER = SCHEDULER['poly']


def get_model_from_config(cfg: DotDict) -> nn.Module:
    handler = NETWORK.get(cfg.name, DEFAULT_NETWORK)
    return handler.get_from_config(cfg)


def get_optimizer_from_config(cfg: DotDict, params: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
    optim = OPTIMIZER.get(cfg.name, DEFAULT_OPTIMIZER)
    cfg_excluded = {k: v for k, v in cfg.items() if 'name' not in k}            # exclude 'name' from keys
    return optim(params, **cfg_excluded)


def get_scheduler_from_config(cfg: DotDict, optimizer: torch.optim.Optimizer):
    sch = SCHEDULER.get(cfg.name, DEFAULT_SCHEDULER)
    cfg_excluded = {k: v for k, v in cfg.items() if not k == 'name'}
    return torch.optim.lr_scheduler.LambdaLR(optimizer, sch.get_func(**cfg_excluded))
