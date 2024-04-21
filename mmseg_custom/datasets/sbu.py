import os
import os.path as osp
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

from typing import List


@DATASETS.register_module()
class SBUDataset(CustomDataset):
    CLASSES = ('background', 'shadow')
    PALETTE = [[0, 255, 0], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(SBUDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)