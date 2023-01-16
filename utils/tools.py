from __future__ import annotations

import ast
import copy
import os
import string
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Union, List, Any, Dict, Tuple, Protocol

import numpy as np
import torch
import torch.nn as nn
import tqdm
from tensorboardX import SummaryWriter

from utils.config import DotDict


def normalize_array(array: Union[List[Any], np.ndarray], norm: str = 'L1'):
    temp = np.array(array)
    if norm == 'L1':
        temp = temp / np.abs(temp).sum()
    elif norm == 'L2':
        temp = temp / ((temp ** 2).sum()) ** 0.5
    if isinstance(array, list):
        return temp.tolist()
    return temp


def dict2sw_format(d: Dict[str, Any]):
    unique_main_tags = {}
    for k in d.keys():
        if '.' in k:
            sub_tag = k.split('.')[0]
            main_tag = k[len(sub_tag) + 1:]
            if main_tag in unique_main_tags.keys():
                unique_main_tags[main_tag].append(k)
            else:
                unique_main_tags[main_tag] = [k]
        else:
            unique_main_tags[k] = d[k]
    for k in unique_main_tags.keys():
        temp = {}
        if type(unique_main_tags[k]) is list:
            for kk in unique_main_tags[k]:
                sub_tag = kk.split('.')[0]
                temp[sub_tag] = d[kk]
            unique_main_tags[k] = temp
    return unique_main_tags


def thorough_remove(path: str):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s'.format(file_path, e))


def calculate_latency(model: nn.Module, size: Tuple[int, int], device: str = 'cuda', iterations: int = 100) -> float:
    model = model.to(device)
    model.eval()
    x = torch.randn(1, 3, size[0], size[1]).to(device)
    with torch.no_grad():
        elapsed_time = 0
        while elapsed_time < 1:
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t_start = time.time()
            for _ in range(iterations):
                model(x)
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            elapsed_time = time.time() - t_start
            iterations *= 2
        FPS = iterations / elapsed_time
        iterations = int(FPS * 6)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(x)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations
    torch.cuda.empty_cache()
    return latency


def calculate_fps(model: nn.Module, size: Tuple[int, int], device: str = 'cuda', iterations: int = 100) -> float:
    return 1 / calculate_latency(model, size, device, iterations)


def find_size_fixed_fps(model: nn.Module, desired_fps: float, device: str = 'cuda', size_quant: int = 32,
                        max_int: int = 32, iterations: int = 100) -> Tuple[int, float]:
    nearest_size = None
    nearest_fps = None
    for s in range(size_quant, max_int * size_quant, size_quant):
        fps = calculate_fps(model, (s, s), device, iterations)
        print(s, fps)
        if nearest_fps is None and fps < desired_fps:
            nearest_size, nearest_fps = s, fps
            break
        if fps < desired_fps and abs(desired_fps - fps) < abs(desired_fps - nearest_fps):
            nearest_size, nearest_fps = s, fps
            break
        nearest_size, nearest_fps = s, fps
    return nearest_size, nearest_fps


def get_latency_curve(model: nn.Module, device: str = 'cuda', size_quant: int = 32, max_size: int = 1024,
                      iterations: int = 100) -> np.ndarray:
    res = []
    title = 'Calculate latency for size:'
    progress = tqdm.tqdm(range(size_quant, max_size + size_quant, size_quant), title, ncols=100, smoothing=0.9)
    for _, s in enumerate(progress):
        progress.set_description(title + f' ({s}, {s})')
        latency = calculate_latency(model, (s, s), device, iterations)
        res.append([s, latency])
    return np.array(res)


def profile(model: nn.Module, size: Tuple[int, int], device: str = 'cuda', repetition: int = 100):
    from torch.profiler import profile, ProfilerActivity, schedule

    def trace_handler(p: profile):
        print(p.key_averages().table(sort_by='self_cuda_time_total'))

    x = torch.randn((1, 3, size[0], size[1])).to(device)
    model = model.to(device)
    model.eval()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=5 * repetition, warmup=repetition, active=repetition, repeat=2),
                 on_trace_ready=trace_handler) as prof:
        for _ in range(20 * repetition):
            model(x)
            prof.step()
    torch.cuda.empty_cache()


def multiple_profile(model: nn.Module, sizes: List[int], device: str = 'cuda', repetition: int = 100):
    for s in sizes:
        print(f'{s}:')
        profile(model, (s, s), device, repetition)


class IteratorTimer:
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = self.iterable.__iter__()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterable)

    def __next__(self):
        start = time.time()
        n = next(self.iterator)
        self.last_duration = (time.time() - start)
        return n

    next = __next__


class PrintableTime(object):
    def __init__(self, value: float = 0):
        self.time_struct = {'second': 0.0, 'minute': 0, 'hour': 0}
        self.value = value
        self.make_struct(value)

    def make_struct(self, value):
        s = value % 60
        m = value // 60
        h = int(m / 60)
        m = int(m - 60 * h)
        self.time_struct = {'second': s, 'minute': m, 'hour': h}

    def __add__(self, other):
        if isinstance(other, (int, float)):
            value = self.value + other
        else:
            value = self.value + other.value
        return PrintableTime(value)

    def __truediv__(self, other: float):
        return PrintableTime(self.value / other)

    def __mul__(self, other):
        return PrintableTime(self.value * other)

    def __str__(self):
        if self.time_struct['hour'] > 0:
            out = '{:d}h {:d}m {:.1f}s'.format(self.time_struct['hour'],
                                               self.time_struct['minute'],
                                               self.time_struct['second'])
        elif self.time_struct['minute'] > 0:
            out = '{:d}m {:.1f}s'.format(self.time_struct['minute'], self.time_struct['second'])
        else:
            out = '{:.2f}s'.format(self.time_struct['second'])
        return out


class MySummaryWriter(object):
    """Two SummaryWriter instances to log the result of training in two separate formats"""
    def __init__(self, logdir: str, exp_name: str):
        if logdir is None:
            self.general_writer = None
            self.individual_writer = None
            return
        self.exp_name = exp_name
        self.general_writer = SummaryWriter(os.path.join(logdir, 'general'))
        self.individual_writer = SummaryWriter(os.path.join(logdir, 'experiments'))

    def add_results(self, results: dict, epoch: int):
        if self.general_writer is None:
            return
        for k, v in results.items():
            self.general_writer.add_scalars(k.replace('.', '/'), {self.exp_name: v}, global_step=epoch)
        for k, v in dict2sw_format(results).items():
            if type(v) is dict:
                self.individual_writer.add_scalars(self.exp_name + '/' + k, v, global_step=epoch)
            else:
                self.individual_writer.add_scalar(self.exp_name + '/' + k, v, global_step=epoch)

    def close(self):
        if self.general_writer is None:
            return
        self.general_writer.close()
        self.individual_writer.close()


class DatasetHandler(ABC):
    """Dummy class to handle dataset getter for different datasets"""

    @staticmethod
    @abstractmethod
    def get_from_config(cfg: DotDict, transforms: Tuple[Callable, ...]) -> Any:
        ...


class NetworkHandler(ABC):
    """Dummy class to handle network getter for different networks"""
    @staticmethod
    @abstractmethod
    def get_from_config(cfg: DotDict) -> Any:
        ...


class HasOutputKeys(Protocol):
    output_keys: List[str]


def dictionarify(method: Callable[[HasOutputKeys, ...], Any]):
    @wraps(method)
    def _impl(self: HasOutputKeys, *method_args, **method_kwargs) -> Dict[str, Any]:
        method_output = method(self, *method_args, **method_kwargs)
        if type(method_output) is not tuple:
            method_output = [method_output]
        method_output = list(method_output)
        names = self.output_keys
        wrapped = {names[i]: out for i, out in enumerate(method_output)}
        return wrapped
    return _impl


def replace_all_punctuations(text: str, replace_to: str):
    res = text
    for s in string.punctuation:
        res = res.replace(s, replace_to)
    return res


def remove_similar_key_values(dicts: List[DotDict[str, Any]], default: str = 'data') -> List[DotDict[str, Any]]:
    if len(dicts) > 1:
        similarities = [k for k in dicts[0].keys() if all([(k, dicts[0][k]) in d.items() for d in dicts])]
        res = []
        for d in dicts:
            d_copy = copy.deepcopy(d)
            for k in similarities:
                del d_copy[k]
            res.append(d_copy)
        return res
    d_copy = DotDict()
    d_copy[default] = dicts[0][default]
    return [d_copy]


def _parse_str(s: str):
    try:
        val = ast.literal_eval(s)
    except ValueError:
        val = s
    return val
