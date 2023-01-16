import os
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import scipy
import torch
from torch import Tensor
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.api import get_dataset_from_config
from network.analyze.difficulty import get_score_function
from network.general import get_model_from_config, get_optimizer_from_config, get_scheduler_from_config
from network.loss import get_criteria_from_config
from utils import tools
from utils.config import DotDict, get_transforms_from_config
from utils.metrics import get_metrics_from_config, generate_results, MetricList
from utils.tools import MySummaryWriter, PrintableTime, thorough_remove, get_latency_curve, dictionarify


class TorchItem(ABC):
    """Abstract class for torch module handlers; There are two types at the moment, Trainable and Frozen."""

    @abstractmethod
    def load_checkpoint(self, *args):
        ...

    @abstractmethod
    def forward(self, *args):
        ...

    @abstractmethod
    def eval(self):
        ...

    @abstractmethod
    def train(self):
        ...


class TrainableHandler(TorchItem):
    """Handler class for Trainable torch modules with respective optimizer and scheduler."""

    def __init__(self, cfg: DotDict) -> None:
        self.model = get_model_from_config(cfg.model)
        self.device = cfg.model.device
        self.model = self.model.to(self.device)
        self.optimizer = get_optimizer_from_config(cfg.optimizer, self.model.parameters())
        self.scheduler = get_scheduler_from_config(cfg.scheduler, self.optimizer)
        self.current_epoch = 0
        self.epochs = cfg.epochs
        self.evaluate = cfg.evaluate
        self.paths = cfg.checkpoint.paths
        if cfg.checkpoint.weights is not None:
            self.load_checkpoint(cfg.checkpoint.weights, weights_only=True)
        if cfg.checkpoint.resume is not None:
            self.load_checkpoint(cfg.checkpoint.resume, weights_only=False)
        self.output_keys = cfg.output_keys

    @dictionarify
    def forward(self, x: Any) -> Any:
        return self.model(x)

    def backward(self, loss: Tensor) -> None:
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

    def step(self) -> bool:
        if self.finished():
            return False
        self.current_epoch += 1
        self.scheduler.step()
        return True

    def finished(self):
        return self.current_epoch == self.epochs

    def eval(self) -> None:
        self.model.eval()

    def train(self) -> None:
        self.model.train()

    def get_current_lr(self) -> float:
        return self.scheduler.get_last_lr()[0]

    def load_checkpoint(self, path: str, weights_only: bool) -> None:
        if weights_only:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage.cuda())
            self.model.load_state_dict(checkpoint['state_dict'])
            self.current_epoch = checkpoint['epoch'] + 1
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

    def save_checkpoint(self, path: str, weights_only: bool = False) -> None:
        checkpoint = {'state_dict': self.model.state_dict()}
        if not weights_only:
            checkpoint['optimizer'] = self.optimizer.state_dict()
            checkpoint['scheduler'] = self.scheduler.state_dict()
            checkpoint['epoch'] = self.current_epoch
        torch.save(checkpoint, path)


class FrozenHandler(TorchItem):
    """Handler class for Frozen torch modules."""

    def __init__(self, cfg: DotDict):
        self.model = get_model_from_config(cfg.model)
        self.device = cfg.model.device
        self.model = self.model.to(self.device)
        self.load_checkpoint(cfg.checkpoint.weights)
        self.output_func = get_score_function(cfg.output_func)
        self.output_keys = cfg.output_keys

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage.cuda())
        self.model.load_state_dict(checkpoint['state_dict'])

    @dictionarify
    def forward(self, x: Any) -> Any:
        return self.model(x)

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()


class Experiment(ABC):
    """Main class for training experiments. Contains multiple trainable and frozen modules and can train and run
    inference with desired settings and metrics. """

    def __init__(self, cfg: DotDict) -> None:
        set_manual_seed(cfg.train.manual_seed)
        transform = get_transforms_from_config(cfg.train.augmentation)
        dataset = get_dataset_from_config(cfg.data, tuple(transform.values()))
        self.datasets = DotDict({'train': dataset[0], 'val': dataset[1], 'test': dataset[2]})
        self.batch_size = cfg.train.batch_size
        self.workers = cfg.train.workers
        self.loaders = DotDict({key: DataLoader(self.datasets[key], batch_size=self.batch_size if key == 'train' else 1,
                                                shuffle=key == 'train',
                                                num_workers=self.workers if key == 'train' else 1,
                                                pin_memory=key == 'train', drop_last=key == 'train')
                                for key in self.datasets.keys()})
        self.criteria = get_criteria_from_config(cfg.train.criteria)
        self.trainables = DotDict({key: TrainableHandler(tr) for key, tr in cfg.train.trainable.items()})
        self.frozens = DotDict({key: FrozenHandler(pm) for key, pm in cfg.train.frozen.items()})
        self.metrics = get_metrics_from_config(cfg.train.metrics)
        self.writer = MySummaryWriter(cfg.train.writer.logdir, cfg.train.exp_name)
        self.name = cfg.train.exp_name
        self.no_classes = cfg.train.no_classes
        self.display_metric = cfg.train.display_metric

    def train_whole(self, repr_trainable: TrainableHandler):
        while not repr_trainable.finished():
            results = self.train_inference_epoch(train=True)
            file_name = os.path.join(repr_trainable.paths.temp, f'train_epoch_{repr_trainable.current_epoch}.pth')
            repr_trainable.save_checkpoint(path=file_name, weights_only=False)
            if repr_trainable.current_epoch > 1:
                delete_name = os.path.join(repr_trainable.paths.temp,
                                           f'train_epoch_{repr_trainable.current_epoch - 2}.pth')
                os.remove(delete_name)
            if repr_trainable.evaluate:
                val_results = self.train_inference_epoch(train=False)
                results.update(val_results)
            results.update({'lr': repr_trainable.get_current_lr()})
            self.writer.add_results(results, repr_trainable.current_epoch)
            self.metrics.general.update(results)
            for key in self.metrics.keys():
                if key in ['train', 'val']:
                    self.metrics[key].reset()
            repr_trainable.step()
            self.writer.close()

    @abstractmethod
    def train_inference_epoch(self, train: bool, verbose: bool = True) -> Dict[str, Any]:
        pass

    def _train_inference_epoch(self, network: TrainableHandler, train: bool, verbose: bool = True) -> Dict[str, Any]:
        torch.cuda.empty_cache()
        network.eval()
        if train:
            network.train()

        title = ('Training' if train else 'Evaluating') + f' epoch {network.current_epoch}'
        loader = self.loaders.get('train' if train else 'val')
        metrics = self.metrics['train' if train else 'val']
        progress = tqdm(tools.IteratorTimer(loader), ncols=150, total=len(loader), smoothing=0.9, miniters=1,
                        leave=True, desc=title, disable=not verbose)

        end_time = time.time()
        for _, sample in enumerate(progress):
            n = sample[0].size(0)
            sample = {'image': sample[0].to(network.device), 'target': sample[1].to(network.device),
                      'edge': sample[3].to(network.device)}
            predictions, losses, final_target = self._forward_batch(network, sample)
            if train:
                network.backward(losses['loss_total'])
            data_dict = {'time': PrintableTime((time.time() - end_time) / n),
                         'pred': self._default_func(predictions['pred']), 'target': final_target,
                         'no_classes': self.metrics.no_classes}
            data_dict.update({k: v.item() for k, v in losses.items()})
            metrics.update(data_dict, count=n)
            progress.set_description(f'{title} {self.display_metric}: {float(metrics[self.display_metric]):.4f}')
        progress.close()
        results = generate_results(metrics, 'train' if train else 'val')
        return results

    @abstractmethod
    def _forward_batch(self, network: TrainableHandler, sample: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor],
                                                                                            Dict[str, Tensor], Tensor]:
        """single batch forward with dictionaries of predictions and losses, plus the final target as returns
        Must be implemented in every Experiment subclasses"""
        pass

    def save_results(self, test: bool = True, save_model: bool = True) -> None:
        if save_model:
            for name, tr in self.trainables.items():
                final_model_path = os.path.join(tr.paths.model, f'{self.name}_{name}.pth')
                tr.save_checkpoint(path=final_model_path, weights_only=True)
                thorough_remove(tr.paths.temp)
        if test:
            self.inference_and_save(save=self.trainables['network'].paths.result)

    def inference_and_save(self, save: str, func: Callable[[Tensor], Tensor] = None, postfix: str = None) -> None:
        if func is None:
            func = self._default_func
        if postfix is None:
            postfix = 'labelTrainIds'
        postfix = f'_pred_{postfix}.png'
        for tr in self.trainables.values():
            tr.eval()
        torch.cuda.empty_cache()
        title = 'Final Evaluation and Save'
        progress = tqdm(tools.IteratorTimer(self.loaders.test), ncols=150, total=len(self.loaders.test), smoothing=0.9,
                        miniters=1, leave=True, desc=title)
        for _, sample in enumerate(progress):
            file_name = sample[2][0][:-4] + postfix
            sample = {'image': sample[0].to(self.trainables['network'].device),
                      'target': sample[1].to(self.trainables['network'].device)}
            pred, _, _ = self._forward_batch(self.trainables['network'], sample)
            pred = func(logit=pred['pred'], target=sample['target'], no_classes=self.no_classes).squeeze()
            cv2.imwrite(os.path.join(save, file_name), pred.detach().cpu().numpy() * 255)
        progress.close()

    @staticmethod
    def _default_func(logit: Tensor, **kwargs) -> Tensor:
        return logit.max(1)[1].squeeze()

    def change_bs(self, new_bs, change_epochs: bool = True) -> None:
        if new_bs == self.batch_size:
            return
        old_bs = self.batch_size
        self.batch_size = new_bs
        self.loaders = DotDict({key: DataLoader(self.datasets[key], batch_size=self.batch_size if key == 'train' else 1,
                                                shuffle=key == 'train',
                                                num_workers=self.workers if key == 'train' else 1,
                                                pin_memory=key == 'train', drop_last=key == 'train')
                                for key in self.datasets.keys()})
        if change_epochs:
            for tr in self.trainables.values():
                tr.epochs = tr.epochs // old_bs * self.batch_size

    def two_epoch_test(self) -> bool:
        try:
            for i in range(2):
                results = self.train_inference_epoch(train=True, verbose=False)
                val_results = self.train_inference_epoch(train=False, verbose=False)
                results.update(val_results)
                self.metrics.general.update(results)
                for key in self.metrics.keys():
                    if key in ['train', 'val']:
                        self.metrics[key].reset()
            for m in self.metrics.values():
                if isinstance(m, MetricList):
                    m.reset()
            return True
        except RuntimeError:
            return False

    def adjust_bs(self, adjust_epochs: bool = True) -> int:
        print('Testing different batch sizes, it may take a while.')
        while self.batch_size > 0:
            res = self.two_epoch_test()
            res_text = 'Pass' if res else 'Fail'
            print(f'Memory test for batch size {self.batch_size}: {res_text}')
            if res:
                break
            else:
                self.change_bs(self.batch_size - 1, adjust_epochs)
        return self.batch_size

    def change_exp_name(self, new_name):
        self.name = new_name
        self.writer.exp_name = new_name


class SimpleExperiment(Experiment):
    def __init__(self, cfg: DotDict) -> None:
        super(SimpleExperiment, self).__init__(cfg)
        self.trainee = self.trainables['network']

    def train_whole(self, repr_trainable: TrainableHandler = None):
        super(SimpleExperiment, self).train_whole(self.trainee)

    def train_inference_epoch(self, train: bool, verbose: bool = True) -> Dict[str, Any]:
        return self._train_inference_epoch(self.trainee, train, verbose)

    def _forward_batch(self, network: TrainableHandler, sample: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor],
                                                                                            Dict[str, Tensor], Tensor]:
        predictions = network.forward(sample['image'])
        predictions.update(sample)
        losses = self.criteria.forward(predictions)
        effective_loss = sum(losses.values())
        losses['loss'] = list(losses.values())[0]
        losses['loss_total'] = effective_loss
        return predictions, losses, sample['target']

    def save_results(self, test: bool = True, save_model: bool = True, speed_test: bool = False) -> None:
        if speed_test:
            lat = get_latency_curve(self.trainee.model, size_quant=64)
            scipy.io.savemat(os.path.join(self.trainee.paths.fps, self.name + '.mat'), mdict={'lat': lat})
        super(SimpleExperiment, self).save_results(test, save_model)


class InferenceExperiment(Experiment):
    def __init__(self, cfg: DotDict) -> None:
        super(InferenceExperiment, self).__init__(cfg)
        self.network = self.frozens['network']

    def train_inference_epoch(self, train: bool, verbose: bool = True) -> Dict[str, Any]:
        torch.cuda.empty_cache()
        self.network.eval()

        title = 'Evaluating'
        loader = self.loaders.get('val')
        metrics = self.metrics['val']
        progress = tqdm(tools.IteratorTimer(loader), ncols=150, total=len(loader), smoothing=0.9, miniters=1,
                        leave=True, desc=title, disable=not verbose)

        end_time = time.time()
        for _, sample in enumerate(progress):
            n = sample[0].size(0)
            sample = {'image': sample[0].to(self.network.device), 'target': sample[1].to(self.network.device),
                      'edge': sample[3].to(self.network.device)}
            predictions, losses, final_target = self._forward_batch(self.network, sample)
            data_dict = {'time': PrintableTime((time.time() - end_time) / n),
                         'pred': self._default_func(predictions['pred']), 'target': final_target,
                         'no_classes': self.metrics.no_classes}
            data_dict.update({k: v.item() for k, v in losses.items()})
            metrics.update(data_dict, count=n)
            progress.set_description(f'{title} {self.display_metric}: {float(metrics[self.display_metric]):.4f}')
        progress.close()
        results = generate_results(metrics, 'train' if train else 'val')
        return results

    def _forward_batch(self, network: TrainableHandler, sample: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor],
                                                                                            Dict[str, Tensor], Tensor]:
        predictions = network.forward(sample['image'])
        predictions.update(sample)
        return predictions, {}, sample['target']


def set_manual_seed(seed: int = None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True


EXP = {'simple': SimpleExperiment, 'inference-only': InferenceExperiment}
DEFAULT_EXP = EXP['simple']


def get_experiment_from_config(cfg: DotDict):
    print(f'Loading experiment with {cfg.train.type} style.')
    return EXP.get(cfg.train.type)(cfg)
