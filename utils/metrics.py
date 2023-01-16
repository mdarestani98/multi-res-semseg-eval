from typing import Dict, Any, Union, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from utils.config import DotDict
from utils.tools import PrintableTime


class Metric(object):
    def __init__(self, name: str, func=None, output_keys: Union[List[str], str] = 'value',
                 input_keys: List[str] = None):
        if input_keys is None:
            input_keys = ['pred', 'target']
        self.name = name
        self.func = func
        self.data = {'value': 0.0, 'sum': None, 'average': None, 'count': 0}
        if isinstance(output_keys, str):
            output_keys = [output_keys]
        assert all([(key in list(self.data.keys()) + ['invisible']) for key in output_keys]), 'Invalid output key'
        self.visible = 'invisible' not in output_keys
        self.output_keys = [key for key in output_keys if not key == 'invisible']
        self.input_keys = input_keys

    def process(self, data: Tuple[Any]):
        if self.func is None:
            return data[0]
        return self.func(data)

    def update(self, data: Any, count: int = 1):
        data = self.process(data)
        self.data['value'] = data
        self.data['sum'] = data if self.data['sum'] is None else self.data['sum'] + data * count
        self.data['count'] += count
        self.data['average'] = self.data['sum'] / self.data['count']

    def get_value(self, key: str = None):
        if key is None:
            return self.data[self.output_keys[0]]
        return self.data[key]

    def get_dict(self):
        if self.visible:
            if len(self.output_keys) > 1:
                return {self.name + '.' + key: self.data[key] for key in self.output_keys}
            return {self.name: self.data[self.output_keys[0]]}
        return {}

    def finalize(self):
        for key in self.output_keys:
            if isinstance(self.data[key], np.ndarray):
                self.data[key] = float(np.mean(self.data[key]))

    def reset(self):
        self.data = {'value': 0.0, 'sum': None, 'average': None, 'count': 0}

    def __str__(self):
        output_dict = self.get_dict()
        output_list = []
        for (name, value) in output_dict.items():
            if isinstance(value, (int, PrintableTime)):
                item = name + ': {:}'.format(value)
            elif isinstance(value, float):
                if value > 0.001:
                    item = name + ': {:.4f}'.format(value)
                else:
                    item = name + ': {:.1e}'.format(value)
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    item = name + ': {:.4f}'.format(np.squeeze(value)[0])
                else:
                    item = name + ': {}'.format(value.tolist())
            else:
                item = ''
            output_list.append(item)
        output = ',\n'.join(output_list)
        return output

    def __float__(self):
        if type(self.get_value()) is np.ndarray:
            return self.get_value().mean()
        return self.get_value()


class DependentMetric(Metric):
    def __init__(self, name: str, func, dependencies: List[str]):
        super(DependentMetric, self).__init__(name, func, output_keys='value', input_keys=[])
        self.dependencies = dependencies


class MetricList(object):
    def __init__(self, metric_dict=None):
        if metric_dict is None:
            metric_dict = {}
        self._metric_dict = {}
        for (name, func) in metric_dict.items():
            self._metric_dict[name] = Metric(name, func)
        self._dependents = []

    def add_metric(self, metric: Metric):
        self._metric_dict[metric.name] = metric
        if isinstance(metric, DependentMetric):
            self._dependents.append(metric.name)

    def update(self, data_dict: Dict[str, Any], count: int = 1):
        # all dependents get updated once all others get updated
        # it is important to update all metrics at once / to provide all variables at once
        for (name, metric) in self._metric_dict.items():
            if not isinstance(metric, DependentMetric):
                # ignore metric if data not available - best if used only for 'test metrics'
                if any(ikey not in data_dict.keys() for ikey in metric.input_keys):
                    continue
                data = tuple([data_dict[ikey] for ikey in metric.input_keys])
                metric.update(data, count)
        for m in self._dependents:
            dep_data = [self._metric_dict[dep].get_value() for dep in self._metric_dict[m].dependencies]
            self._metric_dict[m].update(tuple(dep_data), count)

    def snapshot(self):
        result = {}
        for metric in self._metric_dict.values():
            result = dict(result, **metric.get_dict())
        return result

    def get_names(self, ignore_invisible: bool = True):
        names = list(self._metric_dict.keys())
        names = [name for name in names if self._metric_dict[name].visible or not ignore_invisible]
        return names

    def finalize(self):
        for metric in self._metric_dict.values():
            metric.finalize()

    def reset(self):
        for metric in self._metric_dict.values():
            metric.reset()

    def __getitem__(self, name: str):
        return self._metric_dict[name]

    def __str__(self):
        string_list = []
        for metric in self._metric_dict.values():
            string_list.append(str(metric))
        string_list.sort()
        output = ',\n'.join(string_list)
        return output


def generate_results(metrics: MetricList, nickname: str):
    metrics.finalize()
    ms = metrics.snapshot()
    results = {}
    for name in metrics.get_names():
        if name in ms.keys():
            results[nickname + '.' + name] = ms[name]
        elif 'time' in name:
            results[nickname + '.' + name] = ms[name + '.sum']  # time has two outputs
    return results


def get_metrics_from_config(cfg: DotDict, arch_type: str = None) -> DotDict:
    if arch_type is None:
        arch_type = 'segmenter'
    metrics_dict = {}
    for key in cfg.keys():
        if key not in ['general', 'no_classes', 'ignore_index']:
            cfg.general += [key + '.' + name for name in cfg[key]]
            metrics = _parse_metrics(cfg[key], model_type=arch_type)
            metrics_dict[key] = metrics
    general_metrics = _parse_metrics(cfg.general, model_type=arch_type, is_general=True)
    metrics_dict['general'] = general_metrics
    metrics_dict['no_classes'] = cfg.no_classes
    metrics_dict['ignore_index'] = cfg.ignore_index
    return DotDict(metrics_dict)


def _parse_metrics(metrics_names: List[str], model_type: str, is_general: bool = False) -> MetricList:
    metrics = MetricList()
    if not is_general:
        if 'classifier' in model_type:
            if any('f1' in name.lower() for name in metrics_names):
                metrics_names = metrics_names + ['precision', 'recall']
            if any(item in name.lower() for name in metrics_names
                   for item in ['accuracy', 'sensitivity', 'specificity', 'precision', 'recall']):
                metrics.add_metric(Metric('FP', func=_fp, output_keys=['invisible', 'sum']))
                metrics.add_metric(Metric('TP', func=_tp, output_keys=['invisible', 'sum']))
                metrics.add_metric(Metric('FN', func=_fn, output_keys=['invisible', 'sum']))
                metrics.add_metric(Metric('TN', func=_tn, output_keys=['invisible', 'sum']))
        elif model_type == 'segmenter':
            if any(item in name.lower() for name in metrics_names for item in ['iou', 'accuracy']):
                metrics.add_metric(Metric('area intersection', func=_intersection, output_keys=['invisible', 'sum'],
                                          input_keys=['pred', 'target', 'no_classes', 'ignore_index']))
                metrics.add_metric(Metric('area union', func=_union, output_keys=['invisible', 'sum'],
                                          input_keys=['pred', 'target', 'no_classes', 'ignore_index']))
                metrics.add_metric(Metric('area target', func=_target, output_keys=['invisible', 'sum'],
                                          input_keys=['target', 'no_classes', 'ignore_index']))
        if any('f1' in name.lower() for name in metrics_names):
            metrics_names = metrics_names + ['precision', 'recall']
    metrics_names = list(set(metrics_names))

    for name in metrics_names:
        name_lower = name.lower()
        if not is_general:
            if 'time' in name_lower:  # time: need both end time and per epoch
                metrics.add_metric(Metric(name, output_keys=['sum', 'average'], input_keys=[name_lower]))
            elif any(item in name_lower for item in ['mse', 'l2']):  # average MSE loss
                metrics.add_metric(Metric(name, func=_mse, output_keys='average'))
            elif 'L1' in name_lower:  # average L1 loss
                metrics.add_metric(Metric(name, func=_l1, output_keys='average'))
            elif 'loss' in name_lower:  # average loss w/o function
                metrics.add_metric(Metric(name, output_keys='average', input_keys=[name_lower]))
            elif 'accuracy' in name_lower:
                if 'classifier' in model_type:
                    metrics.add_metric(DependentMetric(
                        name,
                        func=lambda data: (data[0] + data[1]) / (data[0] + data[1] + data[2] + data[3] + 1e-10),
                        dependencies=['TP', 'TN', 'FP', 'FN'])
                    )
                elif model_type == 'segmenter':
                    metrics.add_metric(DependentMetric(name, func=lambda data: data[0] / (data[1] + 1e-10),
                                                       dependencies=['area intersection', 'area target']))
            elif 'sensitivity' in name_lower:
                metrics.add_metric(DependentMetric(name, func=lambda data: data[0] / (data[0] + data[1] + 1e-10),
                                                   dependencies=['TP', 'FN']))
            elif 'specificity' in name_lower:
                metrics.add_metric(DependentMetric(name, func=lambda data: data[0] / (data[0] + data[1] + 1e-10),
                                                   dependencies=['TN', 'FP']))
            elif 'precision' in name_lower:
                metrics.add_metric(DependentMetric(name, func=lambda data: data[0] / (data[0] + data[1] + 1e-10),
                                                   dependencies=['TP', 'FP']))
            elif 'recall' in name_lower:
                metrics.add_metric(DependentMetric(name, func=lambda data: data[0] / (data[0] + data[1] + 1e-10),
                                                   dependencies=['TP', 'FN']))
            elif 'f1' in name_lower:
                metrics.add_metric(DependentMetric(
                    name,
                    func=lambda data: 2 / (1 / (data[0] + 1e-10) + 1 / (data[1] + 1e-10)),
                    dependencies=['precision', 'recall'])
                )
            elif 'iou' in name_lower:
                metrics.add_metric(DependentMetric(name, func=lambda data: data[0] / (data[1] + 1e-10),
                                                   dependencies=['area intersection', 'area union']))
            else:
                raise ValueError('Invalid metric; got {}'.format(name))
        else:
            if 'epoch' in name_lower:  # epoch: only value needed
                ikey, okey = ['epoch'], ['value']
            elif 'learning rate' in name_lower:  # lr: same as epoch, only value
                ikey, okey = ['lr'], ['value']
            elif 'time' in name_lower:
                ikey, okey = [name_lower], ['sum', 'average']
            else:
                ikey, okey = [name_lower], ['value']
            metrics.add_metric(Metric(name, output_keys=okey, input_keys=ikey))

    return metrics


""" TODO: following functions does not consider ignore_index
    probably add another item to data tuple (being ignore_index) and mask error/score maps with that
"""


def _fp(data: Tuple[torch.Tensor, torch.Tensor]):
    pred, target = torch.clone(data[0]).reshape((-1)), torch.clone(data[1]).reshape((-1))
    return torch.sum(pred / target == float('inf')).detach().cpu().numpy()


def _tp(data: Tuple[torch.Tensor, torch.Tensor]):
    pred, target = torch.clone(data[0]).reshape((-1)), torch.clone(data[1]).reshape((-1))
    return torch.sum(pred / target == 1).detach().cpu().numpy()


def _fn(data: Tuple[torch.Tensor, torch.Tensor]):
    pred, target = torch.clone(data[0]).reshape((-1)), torch.clone(data[1]).reshape((-1))
    return torch.sum(pred / target == 0).detach().cpu().numpy()


def _tn(data: Tuple[torch.Tensor, torch.Tensor]):
    pred, target = torch.clone(data[0]).reshape((-1)), torch.clone(data[1]).reshape((-1))
    return torch.sum(torch.isnan(pred / target)).detach().cpu().numpy()


def _intersection(data: Tuple[torch.Tensor, torch.Tensor, int, int]):
    pred, target, k = torch.clone(data[0]), torch.clone(data[1]), data[2]
    pred, target = pred.view(-1), target.view(-1)
    # assert pred.shape == target.shape
    return torch.histc(pred[pred == target], bins=k, min=0, max=k - 1).detach().cpu().numpy()


def _union(data: Tuple[torch.Tensor, torch.Tensor, int, int]):
    pred, target, k, ignore_index = torch.clone(data[0]), torch.clone(data[1]), data[2], data[3]
    pred, target = pred.view(-1), target.view(-1)
    pred = pred[target != ignore_index]
    return _target(data[1:]) - _intersection(data) + torch.histc(pred, bins=k, min=0, max=k - 1).detach().cpu().numpy()


def _target(data: Tuple[torch.Tensor, int, int]):
    target, k, ignore_index = torch.clone(data[0]), data[1], data[2]
    target = target.view(-1)
    target = target[target != ignore_index]
    return torch.histc(target, bins=k, min=0, max=k - 1).detach().cpu().numpy()


def _mse(data: Tuple[torch.Tensor, torch.Tensor]):
    pred, target = data
    return F.mse_loss(pred, target, reduction='mean').item()


def _l1(data: Tuple[torch.Tensor, torch.Tensor]):
    pred, target = data
    return F.l1_loss(pred, target, reduction='mean').item()
