# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os
import os.path as osp
import mmcv
import numpy as np
from tqdm import tqdm
from PIL import Image
from medpy import metric

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
import cv2

from mmseg.apis.inference import LoadImage
from mmcv.parallel import collate, scatter
from mmseg.datasets.pipelines import Compose
import torch

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_list(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)

                subname = path.split('/')
                images.append(os.path.join(subname[-2],subname[-1]))
    return images

def cal_fmeasure(precision, recall):
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])

    return max_fmeasure


def computeBER_mth(gt_path, pred_path):
    print(gt_path, pred_path)

    video_list = os.listdir(gt_path)
    pred_list = os.listdir(pred_path)

    nim = len(pred_list)

    stats = np.zeros((nim, 4), dtype='float')
    stats_jaccard = np.zeros(nim, dtype='float')
    stats_mae = np.zeros(nim, dtype='float')
    stats_fscore = np.zeros((256, nim, 2), dtype='float')

    soda_path = '/media/wilson/Wilson/DE/Python/video_shadow/scotch-and-soda-main/output/results/scotch_and_soda_visha_image/pred'

    index = 0
    for v in tqdm(range(0, len(video_list)), desc="Calculating Metrics:"):
        pred_list = os.listdir(os.path.join(gt_path, video_list[v]))
        for i in tqdm(range(0, len(pred_list)), desc="Calculating Metrics:"):
            # if not k % internal == 0:
            #     continue
            im = pred_list[i]
            GTim = np.asarray(Image.open(os.path.join(gt_path, video_list[v], im)).convert('L'))
            posPoints = GTim > 0.5
            negPoints = GTim <= 0.5
            countPos = np.sum(posPoints.astype('uint8'))
            countNeg = np.sum(negPoints.astype('uint8'))
            sz = GTim.shape
            GTim = GTim > 0.5

            # soda
            # Predim = np.asarray(
            #     Image.open(os.path.join(pred_path, im[:-13], im[-12:-4]+'.jpg')).convert('L').resize((sz[1], sz[0]), Image.NEAREST))

            Predim = np.asarray(
                Image.open(os.path.join(pred_path, video_list[v] + '_' + im)).convert('L').resize((sz[1], sz[0]), Image.NEAREST))

            # BER
            tp = (Predim > 102) & posPoints
            tn = (Predim <= 102) & negPoints
            countTP = np.sum(tp)
            countTN = np.sum(tn)
            stats[index, :] = [countTP, countTN, countPos, countNeg]

            # IoU
            pred_iou = (Predim > 102)
            stats_jaccard[index] = metric.binary.jc(pred_iou, posPoints)

            # MAE
            pred_mae = (Predim > 12)
            mae_value = np.mean(np.abs(pred_mae.astype(float) - posPoints.astype(float)))
            stats_mae[index] = mae_value

            # Precision and Recall for FMeasure
            eps = 1e-4
            for jj in range(0, 256):
                real_tp = np.sum((Predim > jj) & posPoints)
                real_t = countPos
                real_p = np.sum((Predim > jj).astype('uint8'))

                precision_value = (real_tp + eps) / (real_p + eps)
                recall_value = (real_tp + eps) / (real_t + eps)
                stats_fscore[jj, index, :] = [precision_value, recall_value]

            index += 1
    # Print BER
    posAcc = np.sum(stats[:, 0]) / np.sum(stats[:, 2])
    negAcc = np.sum(stats[:, 1]) / np.sum(stats[:, 3])
    pA = 100 - 100 * posAcc
    nA = 100 - 100 * negAcc
    BER = 0.5 * (2 - posAcc - negAcc) * 100
    print('BER, S-BER, N-BER:')
    print(BER, pA, nA)

    # Print IoU
    jaccard_value = np.mean(stats_jaccard)
    print('IoU:', jaccard_value)

    # Print MAE
    mean_mae_value = np.mean(stats_mae)
    print('MAE:', mean_mae_value)

    # Print Fmeasure
    precision_threshold_list = np.mean(stats_fscore[:, :, 0], axis=1).tolist()
    recall_threshold_list = np.mean(stats_fscore[:, :, 1], axis=1).tolist()
    fmeasure = cal_fmeasure(precision_threshold_list, recall_threshold_list)
    print('Fmeasure:', fmeasure)

    return {"BER": BER, "S-BER": pA, "N-BER": nA, "IoU": jaccard_value, "MAE": mean_mae_value, "Fmeasure": fmeasure}

def inference_segmentor(model, img_r, img_q):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data_r = dict(img=img_r)
    data_q = dict(img=img_q)
    data_r = test_pipeline(data_r)
    data_q = test_pipeline(data_q)
    data_r = collate([data_r], samples_per_gpu=1)
    data_q = collate([data_q], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data_r = scatter(data_r, [device])[0]
        data_q = scatter(data_q, [device])[0]
    else:
        data_r['img_metas'] = [i.data[0] for i in data_r['img_metas']]
        data_q['img_metas'] = [i.data[0] for i in data_q['img_metas']]

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **{'vids':[data_r, data_q]})
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('img', help='Image file')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument('--interval', type=int, default=1, help='frame interval in inference')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta---', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)
    if os.path.isdir(args.img):
        video_list = os.listdir(args.img)
        internal = args.interval
        for video_name in video_list:
            img_list = os.listdir(os.path.join(args.img, video_name))
            img_list.sort()
            for idx, img_name in tqdm(enumerate(img_list)):
                # if not idx % internal == 0:
                #     continue
                if idx < internal:
                    ref_idx = 0
                else:
                    ref_idx = idx - internal

                out_path = osp.join(args.out, video_name+'_'+img_name[:-4]+'.png')
                if os.path.exists(out_path):
                    continue

                img_root_r = os.path.join(args.img, video_name, img_list[ref_idx])
                img_root_q = os.path.join(args.img, video_name, img_name)
                result = inference_segmentor(model, img_root_r, img_root_q)
                # ori = cv2.imread(img_root)
                result_shadow = result
                # cv2.imshow('ori', ori)
                # cv2.imshow('test', np.uint8(result[0]*255))
                cv2.imwrite(out_path, np.uint8(result[0]*255))
                # cv2.waitKey(1)
        computeBER_mth(str(args.img).replace('images', 'annotations'), args.out)
    else:
        # test a single image
        result = inference_segmentor(model, args.img)

        # show the results
        if hasattr(model, 'module'):
            model = model.module
        img = model.show_result(args.img, result,
                                palette=get_palette(args.palette),
                                show=False, opacity=args.opacity)
        mmcv.mkdir_or_exist(args.out)
        out_path = osp.join(args.out, osp.basename(args.img))
        cv2.imwrite(out_path, img)
        # cv2.imwrite(out_path, np.uint8(result[0]*255))
        print(f"Result is save at {out_path}")

if __name__ == '__main__':
    main()
