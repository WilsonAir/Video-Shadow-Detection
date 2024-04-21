# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
import cv2 as cv
import numpy as np

from mmcv.cnn import Conv2d, build_plugin_layer, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.ops import point_sample
from mmcv.runner import ModuleList, force_fp32
from mmseg.models.builder import HEADS, build_loss
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from ...core import build_sampler, multi_apply, reduce_mean
from ..builder import build_assigner
from ..utils import get_uncertain_point_coords_with_randomness
from .label_decouple import Encoder, Decoder


def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()


class LDF(nn.Module):
    def __init__(self,in_channels=1024):
        super(LDF, self).__init__()
        self.conv5b = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4b = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3b = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2b = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv5d = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4d = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3d = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2d = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.encoder = Encoder()
        self.decoderb = Decoder()
        self.decoderd = Decoder()
        self.linearb = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.lineard = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linear = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        self.initialize()

    def forward(self, input, shape=(512, 512)):
        out2, out3, out4, out5 = input
        out2b, out3b, out4b, out5b = self.conv2b(out2), self.conv3b(out3), self.conv4b(out4), self.conv5b(out5)
        out2d, out3d, out4d, out5d = self.conv2d(out2), self.conv3d(out3), self.conv4d(out4), self.conv5d(out5)

        outb1 = self.decoderb([out5b, out4b, out3b, out2b])
        outd1 = self.decoderd([out5d, out4d, out3d, out2d])
        out1 = torch.cat([outb1, outd1], dim=1)

        outb2, outd2 = self.encoder(out1)

        outb2 = self.decoderb([out5b, out4b, out3b, out2b], outb2)
        outd2 = self.decoderd([out5d, out4d, out3d, out2d], outd2)
        out2 = torch.cat([outb2, outd2], dim=1)

        # if shape is None:
        #     shape = x.size()[2:]
        out1 = F.interpolate(self.linear(out1), size=shape, mode='bilinear')
        outb1 = F.interpolate(self.linearb(outb1), size=shape, mode='bilinear')
        outd1 = F.interpolate(self.lineard(outd1), size=shape, mode='bilinear')

        out2 = F.interpolate(self.linear(out2), size=shape, mode='bilinear')
        outb2 = F.interpolate(self.linearb(outb2), size=shape, mode='bilinear')
        outd2 = F.interpolate(self.lineard(outd2), size=shape, mode='bilinear')
        return outb1, outd1, out1, outb2, outd2, out2

    def initialize(self):
        weight_init_ldf(self)

def weight_init_ldf(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init_ldf(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


@HEADS.register_module()
class PerPixelDecoupleHead(BaseDecodeHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`mmcv.ConfigDict` | dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder position encoding. Defaults to None.
        loss_cls (:obj:`mmcv.ConfigDict` | dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`mmcv.ConfigDict` | dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`mmcv.ConfigDict` | dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`mmcv.ConfigDict` | dict): Training config of
            Mask2Former head.
        test_cfg (:obj:`mmcv.ConfigDict` | dict): Testing config of
            Mask2Former head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 num_queries=100,
                 num_transformer_feat_level=3,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(PerPixelDecoupleHead, self).__init__(
            in_channels=in_channels,
            channels=feat_channels,
            num_classes=(num_things_classes + num_stuff_classes),
            init_cfg=init_cfg,
            input_transform='multiple_select',
            **kwargs)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers. \
            attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        # assert pixel_decoder.encoder.transformerlayers. \
        #            attn_cfgs.num_levels == num_transformer_feat_level
        # pixel_decoder_ = copy.deepcopy(pixel_decoder)
        # pixel_decoder_.update(
        #     in_channels=in_channels,
        #     feat_channels=feat_channels,
        #     out_channels=out_channels)
        #
        # self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]
        #
        # self.predictor = Conv2d(
        #     self.pixel_decoder.mask_feature.out_channels, self.num_classes, kernel_size=1, stride=1, padding=0
        # )

        ### for label decouple
        self.ldf_head = LDF(in_channels[0])
        ### for label decouple end

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

    def init_weights(self):
        # for m in self.decoder_input_projs:
        #     if isinstance(m, Conv2d):
        #         caffe2_xavier_init(m, bias=0)

        # self.pixel_decoder.init_weights()
        # weight_init.c2_msra_fill(self.predictor)
        a = 1
        # for p in self.transformer_decoder.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_normal_(p)

    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape [num_queries,
                cls_out_channels].
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape [num_queries, h, w].
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels of all images.
                    Each with shape [num_queries, ].
                - label_weights_list (list[Tensor]): Label weights of all
                    images.Each with shape [num_queries, ].
                - mask_targets_list (list[Tensor]): Mask targets of all images.
                    Each with shape [num_queries, h, w].
                - mask_weights_list (list[Tensor]): Mask weights of all images.
                    Each with shape [num_queries, ].
                - num_total_pos (int): Number of positive samples in all
                    images.
                - num_total_neg (int): Number of negative samples in all
                    images.
        """
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
                                      mask_preds_list, gt_labels_list,
                                      gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks,
                           img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (num_gts, ).
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (num_gts, h, w).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
        """
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        # assign and sample
        assign_result = self.assigner.assign(cls_score, mask_points_pred,
                                             gt_labels, gt_points_masks,
                                             img_metas)
        sampling_result = self.sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries,))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries,))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds)

    def loss_single(self, cls_scores, mask_preds, gt_labels_list,
                    gt_masks_list, img_metas):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (num_gts, ).
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (num_gts, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
                                           gt_labels_list, gt_masks_list,
                                           img_metas)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1, 1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        return loss_cls, loss_mask, loss_dice

    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_mask_preds, gt_labels_list,
             gt_masks_list, img_metas):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape [num_decoder, batch_size, num_queries,
                cls_out_channels].
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape [num_decoder, batch_size, num_queries, h, w].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (n, ). n is the sum of number of stuff type
                and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        gt_new = []
        for item in gt_masks_list:
            if item.shape[0] == 1:
                gt_new.append(torch.ones(size=item.shape, device=item.device, dtype=item.dtype) - item)
            else:
                gt_new.append(item[-1:])
        gt = torch.cat(gt_new, dim=0)

        #### for distance transformation
        #### label pre
        gt_numpy = 255*gt.cpu().numpy().astype(np.uint8)
        body_coarse = cv.blur(gt_numpy, ksize=(5, 5))
        body_fuse = []
        for i in range(body_coarse.shape[0]):
            body_i = cv.distanceTransform(body_coarse[i], distanceType=cv.DIST_L2, maskSize=5)
            body_i = body_i ** 0.5

            tmp = body_i[np.where(body_i > 0)]
            if len(tmp) != 0:
                body_i[np.where(body_i > 0)] = np.floor(tmp / np.max(tmp) * 255)
            body_fuse.append(np.expand_dims(body_i, 0))
        body = np.concatenate(body_fuse, axis=0)
        detail_o = gt_numpy - body

        body = torch.from_numpy(body).unsqueeze(1).cuda().float()
        body = body / 255.
        detail = torch.from_numpy(detail_o).unsqueeze(1).cuda().float()
        detail = detail / 255.
        gt = gt.unsqueeze(1).float()

        outb1, outd1, out1, outb2, outd2, out2 = all_mask_preds[0]

        lossb1 = F.mse_loss(outb1, body)
        lossd1 = F.mse_loss(outd1, detail)
        loss1 = F.binary_cross_entropy_with_logits(out1, gt) + iou_loss(out1, gt)

        lossb2 = F.mse_loss(outb2, body)
        lossd2 = F.mse_loss(outd2, detail)
        loss2 = F.binary_cross_entropy_with_logits(out2, gt) + iou_loss(out2, gt)
        loss = (lossb1 + lossd1 + loss1 + lossb2 + lossd2 + loss2) / 2
        #### end transformation


        loss_dict = dict()
        loss_dict['loss_mask'] = (loss1 + loss2) / 2
        loss_dict['loss_body'] = (lossb1 + lossb2) / 2
        loss_dict['loss_detail'] = (lossd1 + lossd2) / 2


        return loss_dict

    def forward(self, feats, img_metas):
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 h, w).
        """
        # batch_size = len(img_metas)
        # mask_features, multi_scale_memorys = self.pixel_decoder(feats)
        # out_mask = self.predictor(mask_features)
        out = self.ldf_head(feats)

        cls_pred_list = [torch.tensor((0, 1), dtype=torch.float, device='cuda')]
        mask_pred_list = []

        # cls_pred_list.append(cls_pred)
        mask_pred_list.append(out)

        # from ..plugins.TSNE import tsne,pca
        # # tsne(feats[3].cpu().numpy())
        # r = pca(feats[0].cpu().detach())

        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(10, 5))
        # plt.subplot2grid((1, 2), (0, 0))
        # plt.title('PCA')
        # plt.imshow(r[0][0].numpy())
        # plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=2)
        # plt.title('1')
        # plt.imshow(feats[0][0].cpu().numpy()[1])
        # # plt.imshow(feats[0][0].cpu().numpy().sum(0))
        # plt.show()

        return cls_pred_list, mask_pred_list

    def forward_train(self, x, img_metas, gt_semantic_seg, gt_labels, gt_masks):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Multi-level features from the upstream network,
                each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_semantic_seg (list[tensor]):Each element is the ground truth
                of semantic segmentation with the shape (N, H, W).
            train_cfg (dict): The training config, which not been used in
                maskformer.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).

        Returns:
            losses (dict[str, Tensor]): a dictionary of loss components
        """

        # forward
        all_cls_scores, all_mask_preds = self(x, img_metas)

        # loss
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks,
                           img_metas)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Test segment without test-time aumengtation.

        Only the output of last decoder layers was used.

        Args:
            inputs (list[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            test_cfg (dict): Testing config.

        Returns:
            seg_mask (Tensor): Predicted semantic segmentation logits.
        """
        all_cls_scores, all_mask_preds = self(inputs, img_metas)
        mask_pred = all_mask_preds[0][-1]
        body_pred = all_mask_preds[0][-3]
        detail_pred = all_mask_preds[0][-2]
        # ori_h, ori_w, _ = img_metas[0]['ori_shape']

        # semantic inference
        # cls_score = F.softmax(cls_score, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        use_bcel_loss = False
        if use_bcel_loss:
            body_pred = body_pred.sigmoid()
            detail_pred = detail_pred.sigmoid()
        # seg_mask = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)


        if mask_pred.shape[0] == 1:
            mask_pred = torch.cat((
                torch.ones(mask_pred.shape, device=mask_pred.device, dtype=mask_pred.dtype) - mask_pred,
                mask_pred,
                torch.zeros((mask_pred.shape[0], self.num_classes-2, mask_pred.shape[2], mask_pred.shape[3]), device=mask_pred.device, dtype=mask_pred.dtype)
            ),
                dim=1
            )
        # mask_pred = F.interpolate(
        #     mask_pred, size=(ori_h, ori_w), mode="bilinear", align_corners=False
        # )


        # body_pred = F.interpolate(
        #     body_pred, size=(ori_h, ori_w), mode="bilinear", align_corners=False
        # )
        # detail_pred = F.interpolate(
        #     detail_pred, size=(ori_h, ori_w), mode="bilinear", align_corners=False
        # )

        return mask_pred

        # return mask_pred, body_pred, detail_pred



class PerPixelPlusHead(BaseDecodeHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`mmcv.ConfigDict` | dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder position encoding. Defaults to None.
        loss_cls (:obj:`mmcv.ConfigDict` | dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`mmcv.ConfigDict` | dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`mmcv.ConfigDict` | dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`mmcv.ConfigDict` | dict): Training config of
            Mask2Former head.
        test_cfg (:obj:`mmcv.ConfigDict` | dict): Testing config of
            Mask2Former head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 num_queries=100,
                 num_transformer_feat_level=3,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(PerPixelPlusHead, self).__init__(
            in_channels=in_channels,
            channels=feat_channels,
            num_classes=(num_things_classes + num_stuff_classes),
            init_cfg=init_cfg,
            input_transform='multiple_select',
            **kwargs)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers. \
            attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.transformerlayers. \
                   attn_cfgs.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]

        self.predictor = Conv2d(
            self.pixel_decoder.mask_dim, self.num_classes, kernel_size=1, stride=1, padding=0
        )

        # self.transformer_decoder = build_transformer_layer_sequence(
        #     transformer_decoder)
        # self.decoder_embed_dims = self.transformer_decoder.embed_dims
        #
        # self.decoder_input_projs = ModuleList()
        # # from low resolution to high resolution
        # for _ in range(num_transformer_feat_level):
        #     if (self.decoder_embed_dims != feat_channels
        #             or enforce_decoder_input_project):
        #         self.decoder_input_projs.append(
        #             Conv2d(
        #                 feat_channels, self.decoder_embed_dims, kernel_size=1))
        #     else:
        #         self.decoder_input_projs.append(nn.Identity())
        # self.decoder_positional_encoding = build_positional_encoding(
        #     positional_encoding)
        # self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        # self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # # from low resolution to high resolution
        # self.level_embed = nn.Embedding(self.num_transformer_feat_level,
        #                                 feat_channels)
        #
        # self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        # self.mask_embed = nn.Sequential(
        #     nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
        #     nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
        #     nn.Linear(feat_channels, out_channels))
        # self.conv_seg = None # fix a bug here (conv_seg is not used)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()
        weight_init.c2_msra_fill(self.predictor)

        # for p in self.transformer_decoder.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_normal_(p)

    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape [num_queries,
                cls_out_channels].
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape [num_queries, h, w].
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels of all images.
                    Each with shape [num_queries, ].
                - label_weights_list (list[Tensor]): Label weights of all
                    images.Each with shape [num_queries, ].
                - mask_targets_list (list[Tensor]): Mask targets of all images.
                    Each with shape [num_queries, h, w].
                - mask_weights_list (list[Tensor]): Mask weights of all images.
                    Each with shape [num_queries, ].
                - num_total_pos (int): Number of positive samples in all
                    images.
                - num_total_neg (int): Number of negative samples in all
                    images.
        """
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
                                      mask_preds_list, gt_labels_list,
                                      gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks,
                           img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (num_gts, ).
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (num_gts, h, w).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
        """
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        # assign and sample
        assign_result = self.assigner.assign(cls_score, mask_points_pred,
                                             gt_labels, gt_points_masks,
                                             img_metas)
        sampling_result = self.sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries,))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries,))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds)

    def loss_single(self, cls_scores, mask_preds, gt_labels_list,
                    gt_masks_list, img_metas):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (num_gts, ).
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (num_gts, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
                                           gt_labels_list, gt_masks_list,
                                           img_metas)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1, 1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        return loss_cls, loss_mask, loss_dice

    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_mask_preds, gt_labels_list,
             gt_masks_list, img_metas):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape [num_decoder, batch_size, num_queries,
                cls_out_channels].
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape [num_decoder, batch_size, num_queries, h, w].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (n, ). n is the sum of number of stuff type
                and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # num_dec_layers = len(all_cls_scores)
        # all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        # all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        # img_metas_list = [img_metas for _ in range(num_dec_layers)]
        # losses_cls, losses_mask, losses_dice = multi_apply(
        #     self.loss_single, all_cls_scores, all_mask_preds,
        #     all_gt_labels_list, all_gt_masks_list, img_metas_list)

        predictions = all_mask_preds.float()  # https://github.com/pytorch/pytorch/issues/48163
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = F.cross_entropy(
            predictions, gt_masks_list, reduction="mean", ignore_index=self.ignore_value
        )

        # losses = {"loss_sem_seg": loss * self.loss_weight}
        loss_dict = dict()
        # loss from the last decoder layer
        # loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = loss[-1]
        # loss_dict['loss_dice'] = losses_dice[-1]
        # loss from other decoder layers
        # num_dec_layer = 0
        # for loss_cls_i, loss_mask_i, loss_dice_i in zip(
        #         losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
        #     loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
        #     loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
        #     loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
        #     num_dec_layer += 1
        return loss_dict

    def forward(self, feats, img_metas):
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 h, w).
        """
        batch_size = len(img_metas)
        mask_features, multi_scale_memorys = self.pixel_decoder(feats)
        out_mask = self.predictor(mask_features)
        cls_pred_list = []
        mask_pred_list = []

        # cls_pred_list.append(cls_pred)
        mask_pred_list.append(out_mask)


        return cls_pred_list, mask_pred_list

    def forward_train(self, x, img_metas, gt_semantic_seg, gt_labels, gt_masks):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Multi-level features from the upstream network,
                each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_semantic_seg (list[tensor]):Each element is the ground truth
                of semantic segmentation with the shape (N, H, W).
            train_cfg (dict): The training config, which not been used in
                maskformer.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).

        Returns:
            losses (dict[str, Tensor]): a dictionary of loss components
        """

        # forward
        all_cls_scores, all_mask_preds = self(x, img_metas)

        # loss
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks,
                           img_metas)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Test segment without test-time aumengtation.

        Only the output of last decoder layers was used.

        Args:
            inputs (list[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            test_cfg (dict): Testing config.

        Returns:
            seg_mask (Tensor): Predicted semantic segmentation logits.
        """
        all_cls_scores, all_mask_preds = self(inputs, img_metas)
        mask_pred = all_mask_preds[-1]
        ori_h, ori_w, _ = img_metas[0]['ori_shape']

        # semantic inference
        # cls_score = F.softmax(cls_score, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        # seg_mask = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)

        mask_pred = F.interpolate(
            mask_pred, size=(ori_h, ori_w), mode="bilinear", align_corners=False
        )
        return mask_pred
