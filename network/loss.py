from collections.abc import Callable
from typing import Union, List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from utils.config import DotDict, product_dict
from utils.tools import remove_similar_key_values


class SimpleLoss(nn.Module):
    def __init__(self, loss: nn.Module, coef: float):
        super(SimpleLoss, self).__init__()
        self.loss = loss
        self.coef = coef
        
    def forward(self, pred: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        return self.loss(pred, target) * self.coef


class ConditionalLoss(nn.Module):
    def __init__(self, loss: nn.Module, coef: float, process: Callable[[Tuple[Tensor, ...]], Tensor]) -> None:
        super(ConditionalLoss, self).__init__()
        self.loss = loss
        self.coef = coef
        self.process = process

    def forward(self, *data) -> Tensor:
        target = self.process(*data)
        return self.loss(data[0], target) * self.coef


class Criterion(nn.Module):
    def __init__(self, name: str, loss: Union[Callable[[], Tensor], nn.Module], input_keys: List[str] = None) -> None:
        super(Criterion, self).__init__()
        self.name = name
        self.loss = loss
        if input_keys is None:
            input_keys = ['pred', 'target']
        self.input_keys = input_keys

    def forward(self, *data) -> Tensor:
        return self.loss(*data)


class Criteria:
    def __init__(self, criteria_dict: Dict[str, Union[Callable[[], Tensor], nn.Module]]) -> None:
        if criteria_dict is None:
            criteria_dict = {}
        self._criteria_dict = criteria_dict

    def add_criterion(self, criterion: Criterion) -> None:
        self._criteria_dict[criterion.name] = criterion

    def forward(self, data_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        res = {}
        if len(self._criteria_dict.keys()) > 1:
            for name, criterion in self._criteria_dict.items():
                if any(ikey not in data_dict.keys() for ikey in criterion.input_keys):
                    continue
                data = [data_dict[ikey] for ikey in criterion.input_keys]
                res[name] = criterion(*data)
            return res
        else:
            criterion = list(self._criteria_dict.values())[0]
            all_keys = {}
            for ikey in criterion.input_keys:
                all_keys[ikey] = [k for k in data_dict.keys() if ikey in k]
            all_list = list(product_dict(**all_keys))
            all_simplified = remove_similar_key_values(all_list, default='pred')
            names = [list(entry.values())[0] for entry in all_simplified]
            names = [name.replace('pred', 'raw_loss') for name in names]
            for name, entry in zip(names, all_list):
                data = [data_dict[k] for k in entry.values()]
                res[name] = criterion(*data)
            return res


class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, bd_pre, bd_gt):
        return weighted_bce(bd_pre, bd_gt)


def weighted_bce(pred, target):
    log_p = pred.permute(0, 2, 3, 1).contiguous().view(1, -1)
    target_t = target.view(1, -1).to(log_p.device, float)

    pos_index = (target_t >= 0.5)
    neg_index = (target_t < 0.5)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss


def boundary_mask_threshold(pred: Tensor, bd_pre: Tensor, target: Tensor, ignore_index: int = 255,
                            threshold: float = 0.8) -> Tensor:
    return torch.where(torch.sigmoid(bd_pre[:, 0, ...]) > threshold, target, torch.ones_like(target) * ignore_index)


class ExpAbsLoss(nn.Module):
    def __init__(self, reduction: str = 'mean', alpha: float = 1.0):
        super(ExpAbsLoss, self).__init__()
        self.reduction = reduction
        self.alpha = max(alpha, 1e-4)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = torch.sigmoid(pred)
        error = torch.abs(pred - target) * self.alpha
        exp_error = torch.exp(error)
        loss = 2 * (1 - exp_error * (1 - error)) / self.alpha ** 2
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class OhemCrossEntropyLoss(nn.Module):
    """
    Implements the ohem cross entropy loss function.
    Args:
        thresh (float, optional): The threshold of ohem. Default: 0.7.
        min_kept (int, optional): The min number to keep in loss computation. Default: 10000.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, thresh=0.7, min_kept=10000, ignore_index=255):
        super(OhemCrossEntropyLoss, self).__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.EPS = 1e-5

    def forward(self, logit, label):
        """
        Forward computation.
        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
        """
        if len(label.shape) != len(logit.shape):
            label = torch.unsqueeze(label, 1)

        # get the label after ohem
        n, c, h, w = logit.shape
        label = label.reshape((-1, )).to(label.device, dtype=torch.int64)
        valid_mask = (label != self.ignore_index).to(label.device, dtype=torch.int64)
        num_valid = valid_mask.sum()
        label = label * valid_mask

        prob = F.softmax(logit, dim=1)
        prob = prob.permute((1, 0, 2, 3)).reshape((c, -1))

        if self.min_kept < num_valid and num_valid > 0:
            # let the value which ignored greater than 1
            prob = prob + (1 - valid_mask)

            # get the prob of relevant label
            label_onehot = F.one_hot(label, c)
            label_onehot = label_onehot.transpose(1, 0)
            prob = prob * label_onehot
            prob = torch.sum(prob, dim=0)

            threshold = self.thresh
            if self.min_kept > 0:
                index = prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                threshold_index = int(threshold_index.item())
                if prob[threshold_index] > self.thresh:
                    threshold = prob[threshold_index]
                kept_mask = (prob < threshold).to(prob.device, dtype=torch.int64)
                label = label * kept_mask
                valid_mask = valid_mask * kept_mask

        # make the invalid region as ignore
        label = label + (1 - valid_mask) * self.ignore_index

        label = label.reshape((n, h, w))
        valid_mask = valid_mask.reshape((n, h, w)).to(label.device, dtype=torch.float32)
        loss = F.cross_entropy(logit, label, ignore_index=self.ignore_index)
        loss = loss * valid_mask
        avg_loss = torch.mean(loss) / (torch.mean(valid_mask) + self.EPS)

        label.stop_gradient = True
        valid_mask.stop_gradient = True
        return avg_loss


def get_criteria_from_config(cfg: DotDict) -> Criteria:
    criteria_dict = {}
    for name, cf in cfg.items():
        criteria_dict[name] = Criterion(name, get_criterion_from_config(cf), input_keys=cf.input_keys)
    return Criteria(criteria_dict)


def get_criterion_from_config(cfg: DotDict):
    if cfg.name == 'ce':
        return SimpleLoss(nn.CrossEntropyLoss(reduction=cfg.reduction, ignore_index=cfg.ignore_index), cfg.coef)
    elif cfg.name == 'boundary':
        return SimpleLoss(BoundaryLoss(), cfg.coef)
    elif cfg.name == 'boundary-ce':
        return ConditionalLoss(nn.CrossEntropyLoss(reduction=cfg.reduction, ignore_index=cfg.ignore_index), cfg.coef,
                               process=lambda *data: boundary_mask_threshold(data[0], data[1], data[2],
                                                                             cfg.ignore_index, cfg.threshold))
    elif cfg.name == 'l1':
        return SimpleLoss(nn.L1Loss(reduction=cfg.reduction), cfg.coef)
    elif cfg.name == 'l2':
        return SimpleLoss(nn.MSELoss(reduction=cfg.reduction), cfg.coef)
    elif cfg.name == 'exp-abs':
        return SimpleLoss(ExpAbsLoss(reduction=cfg.reduction, alpha=cfg.alpha), cfg.coef)
    elif cfg.name == 'ohem-ce':
        return SimpleLoss(OhemCrossEntropyLoss(cfg.threshold, cfg.min_kept, cfg.ignore_index), coef=cfg.coef)
    else:
        raise ValueError('loss function not found, got {}'.format(cfg.name))
