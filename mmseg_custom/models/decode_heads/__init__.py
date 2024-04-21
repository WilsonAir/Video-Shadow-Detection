# Copyright (c) OpenMMLab. All rights reserved.

from .perpixel_head import PerPixelHead, PerPixelPlusHead
from .perpixel_head_label_decouple import PerPixelDecoupleHead
# from .perpixel_head_label_decouple_for_flops import PerPixelDecoupleHead


__all__ = [
    'PerPixelHead',
    'PerPixelDecoupleHead',
]
