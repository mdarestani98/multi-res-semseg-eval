from __future__ import annotations

import copy
import os.path
import traceback
from collections.abc import Iterable
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.tensorflow_stub.errors import DataLossError
from tqdm import tqdm

from utils import tools
from utils.config import DotDict, product_dict
from utils.super_experiment import create_exp_name, decode_tag
from utils.tools import replace_all_punctuations, remove_similar_key_values

DEFAULT_SIZE_GUIDANCE = {
    "compressedHistograms": 1,
    "images": 1,
    "scalars": 0,  # 0 means load all
    "histograms": 1,
}
DEFAULT_METRICS = {
    'state': ['train', 'val'],
    'metric': ['accuracy', 'iou', 'loss']
}


class ExpResult(DotDict):
    def __init__(self, df: pd.DataFrame, tag: str):
        assert all(key in df.keys() for key in ['step', 'value', 'metric'])
        data = aggregate_table(df, 'metric', value_tags=['step', 'value'])
        super(ExpResult, self).__init__({'metrics': data})
        self.info = decode_tag(tag)


class MultiExpResult(DotDict):
    def __init__(self, exp_dict: DotDict):
        super(MultiExpResult, self).__init__(exp_dict)

    def get_slice(self, info: Dict[str, Any], metrics: List[str] = None) -> MultiExpResult:
        res = DotDict({})
        for k, v in self.items():
            info_pd = product_dict(**info)
            if any([inf.items() <= v.info.items() for inf in info_pd]):
                if metrics is not None:
                    res[k] = DotDict({'metrics': v.metrics[metrics], 'info': v.info})
                else:
                    res[k] = DotDict({'metrics': v.metrics, 'info': v.info})
        return MultiExpResult(res)

    def _unique_exp(self) -> List[str]:
        exp = []
        for k, v in self.items():
            exp.append('#'.join(k.split('#')[:-1]))
        exp = list(set(exp))
        return exp

    def calculate_averages(self):
        exp = self._unique_exp()
        res = {}
        for tag in exp:
            temp = {}
            n = 0
            for k, v in self.items():
                if tag in k:
                    if len(temp) == 0:
                        temp = copy.deepcopy(v.metrics)
                        tk = list(temp.keys())
                        for m in tk:
                            temp[m + '^2'] = temp[m] ** 2
                        n = 1
                    else:
                        for m in temp.keys():
                            if not m.endswith('^2'):
                                temp[m] += v.metrics[m]
                            else:
                                temp[m] += v.metrics[m[:-2]] ** 2
                        n += 1
            metrics = [k for k in temp.keys() if not k.endswith('^2')]
            final = {}
            for m in metrics:
                final[m + '_mean'] = temp[m] / n
                final[m + '_std'] = ((temp[m + '^2'] - temp[m] ** 2 / n) / (n - 1)) ** 0.5
            res[tag] = {'metrics': final, 'info': decode_tag(tag)}
        return MultiExpResult(DotDict(res))

    def to_mat(self, path: str):
        temp = [{'tag': replace_all_punctuations(k, '_'), 'data': v.metrics}
                for k, v in remove_similar_key_values_exp(self).items()]
        temp2 = [{'tag': t['tag'], 'data': []} for t in temp]
        for i in range(len(temp)):
            temp2[i]['data'] = [{'metric': replace_all_punctuations(k, '_'), 'values': v}
                                for k, v in temp[i]['data'].items()]
        io.savemat(path, {'exp': np.array(temp2)})


def aggregate_table(df: pd.DataFrame, keys_column: str, value_tags: Iterable[str]) -> DotDict:
    keys = df[keys_column].unique().tolist()
    res = {}
    for k in keys:
        res[k] = df[df[keys_column] == k].loc[:, value_tags].to_numpy(dtype=float)
    return DotDict(res)


def load_single_event(path: str) -> pd.DataFrame:
    df = pd.DataFrame(columns=['metric', 'value', 'step'])
    try:
        event_acc = EventAccumulator(path, size_guidance=DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {'metric': [tag] * len(step), 'value': values, 'step': step}
            r = pd.DataFrame(r)
            df = pd.concat([df, r])
    except DataLossError:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return df


def load_multiple_events(paths: Iterable[str]) -> pd.DataFrame:
    df = pd.DataFrame()
    for path in paths:
        sdf = load_single_event(path)
        if df.shape[0] == 0:
            df = sdf
        else:
            df = pd.concat([df, sdf], ignore_index=True)
    return df


def load_single_run(root: str, run_name: str) -> pd.DataFrame:
    paths = [f'{root}/{metric.state}/{metric.metric}/{run_name}' for metric in product_dict(**DEFAULT_METRICS)]
    return load_multiple_events(paths)


def load_multiple_runs(root: str) -> DotDict:
    test_path = f'{root}/{DEFAULT_METRICS["state"][0]}/{DEFAULT_METRICS["metric"][0]}'
    runs = list(filter(lambda s: os.path.isdir(os.path.join(test_path, s)), os.listdir(test_path)))
    all_runs = {}
    title = f'Loading all runs in directory {root}'
    progress = tqdm(tools.IteratorTimer(runs), ncols=150, total=len(runs), smoothing=0.9, miniters=1, leave=True,
                    desc=title)
    for _, run in enumerate(progress):
        all_runs[run] = ExpResult(load_single_run(root, run), run)
    return DotDict(all_runs)


def remove_similar_key_values_exp(exp: MultiExpResult) -> MultiExpResult:
    dicts = [v.info for v in exp.values()]
    simplified_info = remove_similar_key_values(dicts)
    return MultiExpResult(DotDict({create_exp_name(k, -1): copy.deepcopy(v)
                                   for k, v in zip(simplified_info, exp.values())}))


def plot_mean_std(exp: MultiExpResult):
    metrics = []
    for v in exp.values():
        for k in v.metrics.keys():
            if 'mean' in k:
                metrics.append(k[:-5])
    metrics = list(set(metrics))
    all_info = [v.info for v in exp.values()]
    all_info = remove_similar_key_values(all_info)
    legends = [create_exp_name(info, -1) for info in all_info]
    fig, axs = plt.subplots(1, len(metrics), sharex=True)
    for i, m in enumerate(metrics):
        for v in exp.values():
            temp_mean = v.metrics[m + '_mean'][:, 1]
            temp_std = v.metrics[m + '_std'][:, 1]
            temp_step = v.metrics[m + '_mean'][:, 0]
            axs[i].plot(temp_step, temp_mean, 'k-')
            axs[i].fill_between(temp_step, temp_mean - temp_std, temp_mean + temp_std, alpha=0.5)
        axs[i].legend(legends)
    plt.show()
