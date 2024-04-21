import os
import os.path as osp
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

from typing import List


@DATASETS.register_module()
class SBUDatasetVideo(CustomDataset):
    CLASSES = ('background', 'shadow')
    PALETTE = [[0, 255, 0], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(SBUDatasetVideo, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)


    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)

        results = dict(img_info=img_info, ann_info=ann_info)

        self.pre_pipeline(results)

        return self.pipeline(results), self.pipeline(results)


    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)