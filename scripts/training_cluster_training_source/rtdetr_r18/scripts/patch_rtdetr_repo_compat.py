#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


COMPAT_FILE = """from __future__ import annotations

import torchvision

torchvision.disable_beta_transforms_warning()

try:
    from torchvision import datapoints as _datapoints  # type: ignore

    BoundingBoxCls = _datapoints.BoundingBox
    BoundingBoxFormat = _datapoints.BoundingBoxFormat
    ImageCls = _datapoints.Image
    VideoCls = _datapoints.Video
    MaskCls = _datapoints.Mask

    def make_bounding_boxes(data, *, format, spatial_size):
        return BoundingBoxCls(data, format=format, spatial_size=spatial_size)

    def get_spatial_size(boxes):
        return boxes.spatial_size

except ImportError:
    from torchvision import tv_tensors as _tv_tensors  # type: ignore

    BoundingBoxCls = _tv_tensors.BoundingBoxes
    BoundingBoxFormat = _tv_tensors.BoundingBoxFormat
    ImageCls = _tv_tensors.Image
    VideoCls = _tv_tensors.Video
    MaskCls = _tv_tensors.Mask

    def make_bounding_boxes(data, *, format, spatial_size):
        return BoundingBoxCls(data, format=format, canvas_size=spatial_size)

    def get_spatial_size(boxes):
        return boxes.canvas_size
"""


PATCHED_COCO_DATASET = """\"\"\"
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
\"\"\"

import torch
import torch.utils.data

import torchvision

torchvision.disable_beta_transforms_warning()

from pycocotools import mask as coco_mask

from src.core import register
from src.data.torchvision_compat import BoundingBoxFormat, MaskCls, make_bounding_boxes

__all__ = ['CocoDetection']


@register
class CocoDetection(torchvision.datasets.CocoDetection):
    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']

    def __init__(self, img_folder, ann_file, transforms, return_masks, remap_mscoco_category=False):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # ['boxes', 'masks', 'labels']:
        if 'boxes' in target:
            target['boxes'] = make_bounding_boxes(
                target['boxes'],
                format=BoundingBoxFormat.XYXY,
                spatial_size=img.size[::-1],
            )

        if 'masks' in target:
            target['masks'] = MaskCls(target['masks'])

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\\n ann_file: {self.ann_file}\\n'
        s += f' return_masks: {self.return_masks}\\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\\n   {repr(self._transforms)}'

        return s


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, remap_mscoco_category=False):
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.remap_mscoco_category:
            classes = [mscoco_category2label[obj["category_id"]] for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]

        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])

        return image, target


mscoco_category2name = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
"""


PATCHED_TRANSFORMS = """\"\"\"by lyuwenyu
\"\"\"


import torch
import torch.nn as nn

import torchvision

torchvision.disable_beta_transforms_warning()

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from PIL import Image
from typing import Any, Dict, List, Optional

from src.core import register, GLOBAL_CONFIG
from src.data.torchvision_compat import (
    BoundingBoxCls,
    BoundingBoxFormat,
    ImageCls,
    MaskCls,
    VideoCls,
    get_spatial_size,
    make_bounding_boxes,
)


__all__ = ['Compose', ]


RandomPhotometricDistort = register(T.RandomPhotometricDistort)
RandomZoomOut = register(T.RandomZoomOut)
# RandomIoUCrop = register(T.RandomIoUCrop)
RandomHorizontalFlip = register(T.RandomHorizontalFlip)
Resize = register(T.Resize)
RandomCrop = register(T.RandomCrop)
Normalize = register(T.Normalize)

if hasattr(T, "ToImageTensor"):
    @register
    class ToImageTensor(T.ToImageTensor):
        pass
else:
    @register
    class ToImageTensor(T.ToImage):
        pass

if hasattr(T, "ConvertDtype"):
    @register
    class ConvertDtype(T.ConvertDtype):
        def __init__(self, dtype=torch.float32) -> None:
            super().__init__(dtype=dtype)
else:
    @register
    class ConvertDtype(T.ToDtype):
        def __init__(self, dtype=torch.float32, scale=True) -> None:
            super().__init__(dtype=dtype, scale=scale)

if hasattr(T, "SanitizeBoundingBox"):
    @register
    class SanitizeBoundingBox(T.SanitizeBoundingBox):
        pass
else:
    @register
    class SanitizeBoundingBox(T.SanitizeBoundingBoxes):
        pass


@register
class Compose(T.Compose):
    def __init__(self, ops) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    transfom = getattr(GLOBAL_CONFIG[name]['_pymodule'], name)(**op)
                    transforms.append(transfom)
                elif isinstance(op, nn.Module):
                    transforms.append(op)
                else:
                    raise ValueError('')
        else:
            transforms = [EmptyTransform(), ]

        super().__init__(transforms=transforms)


@register
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register
class PadToSize(T.Pad):
    _transformed_types = (
        Image.Image,
        ImageCls,
        VideoCls,
        MaskCls,
        BoundingBoxCls,
    )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sz = F.get_spatial_size(flat_inputs[0])
        h, w = self.spatial_size[0] - sz[0], self.spatial_size[1] - sz[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return self._get_params(flat_inputs)

    def __init__(self, spatial_size, fill=0, padding_mode='constant') -> None:
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)

        self.spatial_size = spatial_size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register
class ConvertBox(T.Transform):
    _transformed_types = (
        BoundingBoxCls,
    )

    def __init__(self, out_fmt='', normalize=False) -> None:
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize

        self.data_fmt = {
            'xyxy': BoundingBoxFormat.XYXY,
            'cxcywh': BoundingBoxFormat.CXCYWH
        }

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if self.out_fmt:
            spatial_size = get_spatial_size(inpt)
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.out_fmt)
            inpt = make_bounding_boxes(inpt, format=self.data_fmt[self.out_fmt], spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(get_spatial_size(inpt)[::-1]).tile(2)[None]

        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)
"""


PATCHED_AMP = """import torch
import torch.nn as nn

try:
    from torch.amp import GradScaler as TorchGradScaler
    _GRADSCALER_SUPPORTS_DEVICE_TYPE = True
except ImportError:
    from torch.cuda.amp import GradScaler as TorchGradScaler
    _GRADSCALER_SUPPORTS_DEVICE_TYPE = False


from src.core import register
import src.misc.dist as dist


__all__ = ['GradScaler']


@register
class GradScaler(TorchGradScaler):
    def __init__(
        self,
        enabled: bool = True,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ) -> None:
        kwargs = dict(
            enabled=enabled,
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
        )
        if _GRADSCALER_SUPPORTS_DEVICE_TYPE:
            super().__init__("cuda", **kwargs)
        else:
            super().__init__(**kwargs)
"""


PATCHED_DET_ENGINE = """\"\"\"
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
\"\"\"

import math
import os
import sys
import pathlib
from typing import Iterable

import torch
import torch.amp

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)

    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values()).detach().item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    panoptic_evaluator = None

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator
"""


PATCHED_RTDETR = """\"\"\"by lyuwenyu
\"\"\"

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

from src.core import register


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = int(np.random.choice(self.multi_scale))
            x = F.interpolate(x, size=[sz, sz])

        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
"""


PATCHED_REGNET = """import torch
import torch.nn as nn

try:
    from transformers import RegNetModel
except Exception as exc:
    RegNetModel = None
    _REGNET_IMPORT_ERROR = exc
else:
    _REGNET_IMPORT_ERROR = None


from src.core import register

__all__ = ['RegNet']


@register
class RegNet(nn.Module):
    def __init__(self, configuration, return_idx=[0, 1, 2, 3]):
        super(RegNet, self).__init__()
        if RegNetModel is None:
            raise ImportError(
                "RegNet backbone requires transformers with a compatible torch installation."
            ) from _REGNET_IMPORT_ERROR
        self.model = RegNetModel.from_pretrained("facebook/regnet-y-040")
        self.return_idx = return_idx

    def forward(self, x):
        outputs = self.model(x, output_hidden_states=True)
        x = outputs.hidden_states[2:5]

        return x
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch the RT-DETR repo for torchvision 0.16 compatibility.")
    parser.add_argument("--repo-root", required=True, help="Path to the RT-DETR rtdetr_pytorch root")
    return parser.parse_args()


def write_if_different(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current != content:
        path.write_text(content, encoding="utf-8")


def replace_if_different(path: Path, old: str, new: str) -> None:
    current = path.read_text(encoding="utf-8")
    updated = current.replace(old, new)
    if updated != current:
        path.write_text(updated, encoding="utf-8")

def patch_config_grad_scaler(path: Path) -> None:
    current = path.read_text(encoding="utf-8")
    updated = current

    compat_import = (
        "try:\n"
        "    from torch.amp import GradScaler as TorchGradScaler\n"
        "    _GRADSCALER_SUPPORTS_DEVICE_TYPE = True\n"
        "except ImportError:\n"
        "    from torch.cuda.amp import GradScaler as TorchGradScaler\n"
        "    _GRADSCALER_SUPPORTS_DEVICE_TYPE = False"
    )
    compat_init = (
        "            if _GRADSCALER_SUPPORTS_DEVICE_TYPE:\n"
        "                self._scaler = TorchGradScaler(\"cuda\")\n"
        "            else:\n"
        "                self._scaler = TorchGradScaler()"
    )

    if "_GRADSCALER_SUPPORTS_DEVICE_TYPE" not in updated:
        updated = updated.replace("from torch.cuda.amp.grad_scaler import GradScaler", compat_import)
        updated = updated.replace("from torch.amp import GradScaler as TorchGradScaler", compat_import)

    updated = updated.replace("        self._scaler :GradScaler = None ", "        self._scaler :TorchGradScaler = None ")
    updated = updated.replace("    def scaler(self) -> GradScaler: ", "    def scaler(self) -> TorchGradScaler: ")
    updated = updated.replace("    def scaler(self, obj: GradScaler):", "    def scaler(self, obj: TorchGradScaler):")

    if "if _GRADSCALER_SUPPORTS_DEVICE_TYPE:" not in updated:
        updated = updated.replace("            self._scaler = GradScaler()", compat_init)
        updated = updated.replace("            self._scaler = TorchGradScaler(\"cuda\")", compat_init)

    if updated != current:
        path.write_text(updated, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not (repo_root / "tools" / "train.py").exists():
        raise FileNotFoundError(f"RT-DETR repo root not found or invalid: {repo_root}")

    write_if_different(repo_root / "src" / "data" / "torchvision_compat.py", COMPAT_FILE)
    write_if_different(repo_root / "src" / "data" / "coco" / "coco_dataset.py", PATCHED_COCO_DATASET)
    write_if_different(repo_root / "src" / "data" / "transforms.py", PATCHED_TRANSFORMS)
    write_if_different(repo_root / "src" / "optim" / "amp.py", PATCHED_AMP)
    write_if_different(repo_root / "src" / "solver" / "det_engine.py", PATCHED_DET_ENGINE)
    replace_if_different(
        repo_root / "src" / "core" / "yaml_config.py",
        "            self._scaler = create('scaler')",
        "            self._scaler = create('GradScaler')",
    )
    patch_config_grad_scaler(repo_root / "src" / "core" / "config.py")
    write_if_different(repo_root / "src" / "nn" / "backbone" / "regnet.py", PATCHED_REGNET)
    write_if_different(repo_root / "src" / "zoo" / "rtdetr" / "rtdetr.py", PATCHED_RTDETR)
    print(str(repo_root))


if __name__ == "__main__":
    main()
