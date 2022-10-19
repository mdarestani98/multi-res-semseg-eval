import torch
import torch.nn.functional as F

from torch import Tensor


def ce_error_map(logit: Tensor, target: Tensor, **kwargs) -> Tensor:
    raw = F.cross_entropy(logit, target, reduction='none').detach()
    return one_class_target_wrapper(1.0 - F.threshold(1.0 - raw, 0.0, 0.0))


def abs_error_map(logit: Tensor, target: Tensor, no_classes: int = -1, **kwargs) -> Tensor:
    prob = F.softmax(logit, dim=1)
    raw = 0.5 * torch.sum(torch.abs(prob - F.one_hot(target, no_classes).permute((0, 3, 1, 2))), dim=1)
    return one_class_target_wrapper(raw)


def entropy_map(logit: Tensor, **kwargs) -> Tensor:
    prob = F.softmax(logit, dim=1)
    raw = -torch.sum(prob * torch.log2(prob), dim=1)
    return one_class_target_wrapper(raw)


def abs_or_entropy_map(logit: Tensor, target: Tensor, no_classes: int = -1, **kwargs) -> Tensor:
    raw = abs_error_map(logit, target, no_classes, **kwargs) + entropy_map(logit, **kwargs)
    return one_class_target_wrapper(1.0 - F.threshold(1.0 - raw, 0.0, 0.0))


def one_class_target_wrapper(score: Tensor):
    score = score.unsqueeze(1)
    return torch.cat([score, 1.0 - score], dim=1)


SCORE_FUNC = {'ce': ce_error_map, 'abs': abs_error_map, 'ent': entropy_map, 'abs-ent': abs_or_entropy_map}
DEFAULT_SCORE = SCORE_FUNC['ent']


def get_score_function(name: str):
    return SCORE_FUNC.get(name, DEFAULT_SCORE)
