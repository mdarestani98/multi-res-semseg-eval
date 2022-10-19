from typing import Tuple, List

import cv2
import numpy as np
import torch


def identity(obj, **kwargs):
    return obj


def to_tensor_image(image: np.ndarray):
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    image = torch.from_numpy(image.transpose((2, 0, 1)))
    if not isinstance(image, torch.FloatTensor):
        image = image.float()
    return image


def to_tensor_target(target: np.ndarray):
    target = torch.from_numpy(target)
    if not isinstance(target, torch.LongTensor):
        target = target.long()
    return target


def normalize_image(image: torch.Tensor, mean: List[float], std: List[float], **kwargs):
    if std is None:
        image.sub_(mean)
    else:
        image.sub_(mean).div_(std)
    return image


def resize_image(image, size):
    return cv2.resize(image, size[::-1], interpolation=cv2.INTER_LINEAR)


def resize_target(target, size):
    return cv2.resize(target, size[::-1], interpolation=cv2.INTER_NEAREST)


def scale_image(image, scale_fx, scale_fy):
    return cv2.resize(image, None, fx=scale_fx, fy=scale_fy, interpolation=cv2.INTER_LINEAR)


def scale_target(target, scale_fx, scale_fy):
    return cv2.resize(target, None, fx=scale_fx, fy=scale_fy, interpolation=cv2.INTER_NEAREST)


def crop_image(image, pad_h: int, pad_w: int, crop_h: Tuple[int, int], crop_w: Tuple[int, int], padding=None):
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        if padding is None:
            raise (RuntimeError("transforms.Crop() need padding while padding argument is None."))
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                   cv2.BORDER_CONSTANT, value=padding)
    return image[crop_h[0]:crop_h[1], crop_w[0]:crop_w[1]]


def crop_target(target, pad_h: int, pad_w: int, crop_h: Tuple[int, int], crop_w: Tuple[int, int], **kwargs):
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        target = cv2.copyMakeBorder(target, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                    cv2.BORDER_CONSTANT, value=0)
    return target[crop_h[0]:crop_h[1], crop_w[0]:crop_w[1]]


def rotate_image(image, rot_mat, padding):
    h, w = image.shape[:2]
    return cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=padding)


def rotate_target(target, rot_mat, **kwargs):
    h, w = target.shape
    return cv2.warpAffine(target, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)
