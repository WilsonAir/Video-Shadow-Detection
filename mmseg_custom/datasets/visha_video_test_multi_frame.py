import os
import os.path as osp

import mmcv
from typing import List
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class VishaDatasetVideoTestMulti(CustomDataset):
    CLASSES = ('background', 'shadow')
    PALETTE = [[0, 255, 0], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(VishaDatasetVideoTestMulti, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

    def load_annotations(self, video_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            video_dir (str): Path to video directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        # img_infos_list = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            videos = os.listdir(video_dir)
            for video_name in videos:
                img_list = []
                img_dir = os.path.join(video_dir, video_name)
                for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                    img_info = dict(filename=os.path.join(video_name, img))
                    if ann_dir is not None:
                        seg_map = img.replace(img_suffix, seg_map_suffix)
                        img_info['ann'] = dict(seg_map=os.path.join(video_name, seg_map))
                    img_list.append(img_info)
                img_list = sorted(img_list, key=lambda x: x['filename'])
                for i in range(len(img_list)):
                    img_list[i]['idx'] = i
                    img_list[i]['inv_idx'] = len(img_list) - i
                    img_infos.append(img_list[i])
        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

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

        frame_inter = 5

        if img_info['inv_idx'] > frame_inter:
            next_img_info = self.img_infos[idx + frame_inter]
            next_ann_info = self.get_ann_info(idx + frame_inter)
        else:
            next_img_info = self.img_infos[idx - frame_inter]
            next_ann_info = self.get_ann_info(idx - frame_inter)

        results = dict(img_info=img_info, ann_info=ann_info)
        next_results = dict(img_info=next_img_info, ann_info=next_ann_info)
        self.pre_pipeline(results)
        self.pre_pipeline(next_results)

        ref = self.pipeline(results)
        query = self.pipeline(next_results)
        return ref, query

    def get_ref(self, img_infos, frame_inter):
        next_img_info = img_infos.copy()
        next_img_info['filename'] = img_infos['filename'][:-5]+'5.jpg'
        next_img_info['ann'] = {'seg_map':img_infos['ann']['seg_map'][:-5]+'5.png'}

        return next_img_info


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

        frame_inter = 5

        # if img_info['inv_idx'] > frame_inter:
        #     next_img_info = self.img_infos[idx + frame_inter]
        # else:
        #     next_img_info = self.img_infos[idx - frame_inter]
        next_img_info = self.get_ref(img_info, frame_inter)
        next_results = dict(img_info=next_img_info)
        self.pre_pipeline(next_results)
        self.pre_pipeline(results)

        next_results['img_prefix'] = next_results['img_prefix'].replace('val','test')
        next_results['seg_prefix'] = next_results['seg_prefix'].replace('val','test')
        return {'vids': [self.pipeline(results), self.pipeline(next_results)]}
