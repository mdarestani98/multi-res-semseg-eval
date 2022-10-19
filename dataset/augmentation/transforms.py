import math
import random
from typing import List, Tuple, Union, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn

from dataset.augmentation import functions


class Transform(nn.Module):
    """Base class for transforms with different transforms applied on images and labels."""
    def __init__(self, image_func: Callable, target_func: Callable, image_indices: List[int] = None) -> None:
        super(Transform, self).__init__()
        self.image_func = image_func
        self.target_func = target_func
        if image_indices is None:
            image_indices = [0]
        self.image_indices = image_indices

    def __call__(
            self,
            sample: Tuple[Union[np.ndarray, torch.Tensor], ...], **kwargs
    ) -> Tuple[Union[np.ndarray, torch.Tensor], ...]:
        res = []
        for i, item in enumerate(sample):
            if i in self.image_indices:
                res.append(self.image_func(item, **kwargs))
            else:
                res.append(self.target_func(item, **kwargs))
        return tuple(res)


class Compose(object):
    """Composes transforms: transforms.Compose([transforms.RandScale((0.5, 2.0)), transforms.ToTensor()])."""
    def __init__(self, transforms_list: List[Transform], image_indices: List[int] = None) -> None:
        self.transforms = transforms_list
        if image_indices is not None:
            for t in self.transforms:
                t.image_indices = image_indices

    def __call__(
            self,
            sample: Tuple[Union[np.ndarray, torch.Tensor], ...]
    ) -> Tuple[Union[np.ndarray, torch.Tensor], ...]:
        result = []
        for item in sample:
            if isinstance(item, np.ndarray):
                result.append(np.copy(item))
            else:
                result.append(torch.clone(item))
        result = tuple(result)
        for t in self.transforms:
            result = t(result)
        return result


class ToTensor(Transform):
    """Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    def __init__(self, image_indices: List[int] = None):
        super(ToTensor, self).__init__(functions.to_tensor_image, functions.to_tensor_target, image_indices)


class Normalize(Transform):
    """Normalizes the tensor with mean and standard deviation along channels: channel = (channel - mean) / std"""
    def __init__(self, mean: List[float], std: List[float] = None, value_scale: Union[int, float] = 1,
                 image_indices: List[int] = None, device: str = 'cpu'):
        super(Normalize, self).__init__(functions.normalize_image, functions.identity, image_indices)
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = [item * value_scale for item in mean]
        self.mean = torch.Tensor(self.mean).view((-1, 1, 1)).to(device)
        self.std = [item * value_scale for item in std]
        if std is not None:
            self.std = torch.Tensor(self.std).view((-1, 1, 1)).to(device)

    def __call__(self, sample, **kwargs):
        return super(Normalize, self).__call__(sample, mean=self.mean, std=self.std)


class Resize(Transform):
    """Resizes the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w)."""
    def __init__(self, size: Tuple[int, int], image_indices: List[int] = None):
        super(Resize, self).__init__(functions.resize_image, functions.resize_target, image_indices)
        self.size = size

    def __call__(self, sample, **kwargs):
        return super(Resize, self).__call__(sample, size=self.size)


class RandScale(Transform):
    """Randomly resizes image & label with scale factor in [scale_min, scale_max]"""
    def __init__(self, scale: Tuple[float, float], aspect_ratio: Tuple[int, int] = None,
                 image_indices: List[int] = None):
        super(RandScale, self).__init__(functions.scale_image, functions.scale_target, image_indices)
        if 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise RuntimeError("transforms.RandScale() scale param error.")
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise RuntimeError("transforms.RandScale() aspect_ratio param error.")

    def __call__(self, sample, **kwargs):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        return super(RandScale, self).__call__(sample, scale_fx=scale_factor_x, scale_fy=scale_factor_y)


class Crop(Transform):
    """
    Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size: Union[Tuple[int, int], int], crop_type='center', padding: List[float] = None,
                 value_scale: int = 1, image_indices: List[int] = None):
        super(Crop, self).__init__(functions.crop_image, functions.crop_target, image_indices)
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise RuntimeError("crop size error.")
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise RuntimeError("crop type error: rand | center.")
        padding = [padding] * 3 if not len(padding) == 3 else padding
        self.padding = [p * value_scale for p in padding]

    def __call__(self, sample, **kwargs):
        h, w = sample[0].shape[:2]
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        h, w = h + pad_h, w + pad_w
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        return super(Crop, self).__call__(sample, pad_h=pad_h, pad_w=pad_w, crop_h=(h_off, h_off + self.crop_h),
                                          crop_w=(w_off, w_off + self.crop_w), padding=self.padding)


class RandRotate(Transform):
    """Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]."""
    def __init__(self, rotate: Tuple[float, float], padding: List[float], p: float = 0.5, value_scale: int = 1,
                 image_indices: List[int] = None):
        super(RandRotate, self).__init__(functions.rotate_image, functions.rotate_target, image_indices)
        if rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("transforms.RandRotate() scale param error."))
        padding = [padding] * 3 if not len(padding) == 3 else padding
        self.padding = [pad * value_scale for pad in padding]
        self.p = p

    def __call__(self, sample, **kwargs):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = sample[0].shape[-2:]
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            return super(RandRotate, self).__call__(sample, rot_mat=matrix, padding=self.padding)
        return sample


class RandomHorizontalFlip(Transform):
    """Flips image horizontally at a random probability."""
    def __init__(self, p: float = 0.5, image_indices: List[int] = None):
        super(RandomHorizontalFlip, self).__init__(cv2.flip, cv2.flip, image_indices)
        self.p = p

    def __call__(self, sample, **kwargs):
        if random.random() < self.p:
            return super(RandomHorizontalFlip, self).__call__(sample, flipCode=1)
        return sample


class RandomVerticalFlip(Transform):
    """Flips image vertically at a random probability."""
    def __init__(self, p: float = 0.5, image_indices: List[int] = None):
        super(RandomVerticalFlip, self).__init__(cv2.flip, cv2.flip, image_indices)
        self.p = p

    def __call__(self, sample, **kwargs):
        if random.random() < self.p:
            return super(RandomVerticalFlip, self).__call__(sample, flipCode=0)
        return sample


class RandomGaussianBlur(Transform):
    """Adds Gaussian noise at a random probability."""
    def __init__(self, radius: int = 5, image_indices: List[int] = None):
        super(RandomGaussianBlur, self).__init__(cv2.GaussianBlur, functions.identity, image_indices)
        self.radius = radius

    def __call__(self, sample, **kwargs):
        if random.random() < 0.5:
            return super(RandomGaussianBlur, self).__call__(sample, ksize=(self.radius, self.radius), sigmaX=0)
        return sample


class RGB2BGR(Transform):
    """Converts image from RGB order to BGR order, for model initialized from Caffe."""
    def __init__(self, image_indices: List[int] = None):
        super(RGB2BGR, self).__init__(cv2.cvtColor, functions.identity, image_indices)

    def __call__(self, sample, **kwargs):
        return super(RGB2BGR, self).__call__(sample, code=cv2.COLOR_RGB2BGR)


class BGR2RGB(Transform):
    """Converts image from BGR order to RGB order, for model initialized from Pytorch."""
    def __init__(self, image_indices: List[int] = None):
        super(BGR2RGB, self).__init__(cv2.cvtColor, functions.identity, image_indices)

    def __call__(self, sample, **kwargs):
        return super(BGR2RGB, self).__call__(sample, code=cv2.COLOR_BGR2RGB)
