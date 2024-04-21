# [IJCAI2024] Structure-Aware Spatial-Temporal Interaction Network for Video Shadow Detection

Our code is developed on top of [MMSegmentation v0.20.2](https://github.com/open-mmlab/mmsegmentation/tree/v0.20.2).

[//]: # (For details see [Vision Transformer Adapter for Dense Predictions]&#40;https://arxiv.org/abs/2205.08534&#41;.)

If you use this code for a paper please cite:

```
@article{chen2022vitadapter,
  title={Structure-Aware Spatial-Temporal Interaction Network for Video Shadow Detection},
  author={Wei, Housheng and Xing, Guanyu and Liao, Jingwei and Zhang, Yanci and Liu, Yanli},
  booktitle = {Proceedings of the Thirty-Three International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  year={2024}
}
```

## Usage

Install [MMSegmentation v0.20.2](https://github.com/open-mmlab/mmsegmentation/tree/v0.20.2).

```
# recommended environment: torch1.9 + cuda11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.12
pip install mmdet==2.22.0 # for Mask2Former
pip install mmsegmentation==0.20.2
cd ops & sh make.sh # compile deformable attention
```

## Data Preparation

Preparing ViSha Dataset as the structure of ADE20K according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.

### Data structure

Visha_release

-- train\
annotations\
body_detail\
images\
-- test

-- val

## Pretraining Sources

| Name          | Year | Type       | Data         | Repo                                                                                                    | Paper                                                                                                                                                                           |
| ------------- | ---- | ---------- | ------------ | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BEiTv2        | 2022 | MIM        | ImageNet-22K | [repo](https://github.com/microsoft/unilm/tree/master/beit2)                                            | [paper](https://arxiv.org/abs/2208.06366)                                                                                                                                       |

**COCO-Stuff-164K**

| Method | Backbone      | Pretrain                                                                                                                   | Lr schd | Crop Size | mIoU (SS/MS)                                                                                                                                                                            | #Param | Config                                                                                          | Download                                                                                                                                                                                                                    |
|:------:|:-------------:|:--------------------------------------------------------------------------------------------------------------------------:|:-------:|:---------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------:|:-----------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Mask2Former | ViT-Adapter-L | [BEiTv2-L](https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21k.pth) | 80k     | 896       | 52.3 / -                                                                                                                                                                                | 571M   | [config](./configs/coco_stuff164k/mask2former_beitv2_adapter_large_896_80k_cocostuff164k_ss.py) | [ckpt](https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/mask2former_beitv2_adapter_large_896_80k_cocostuff164k.zip)  |

> Note that due to the capacity limitation of *GitHub Release*, some files are provided as `.zip` packages. Please **unzip** them before load into model.



## Training

To train SSTI-Net on ViSha val with a single rtx gpu run:

```shell
sh run_train_video.sh
```

## Evaluate

```shell
sh run_multi_frame_image_demo.sh
```


## Main Reference:

1、 [Vision Transformer Adapter for Dense Predictions](https://arxiv.org/abs/2205.08534).
