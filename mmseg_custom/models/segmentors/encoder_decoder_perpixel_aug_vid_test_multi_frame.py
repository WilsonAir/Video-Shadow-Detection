# Copyright (c) OpenMMLab. All rights reserved.
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.core import add_prefix
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.ops import resize
from mmcv.runner import BaseModule, auto_fp16


@SEGMENTORS.register_module()
class EncoderDecoderPerPixelPlusAugVideoTestMulti(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(EncoderDecoderPerPixelPlusAugVideoTestMulti, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        decode_head.update(train_cfg=train_cfg)
        decode_head.update(test_cfg=test_cfg)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img_r, img_q):
        """Extract features from images."""
        x_r, x_q = self.backbone(img_r, img_q)
        if self.with_neck:
            x_r = self.neck(x_r)
            x_q = self.neck(x_q)
        return x_r, x_q

    def encode_decode(self, img_r, img_q, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x, x2 = self.extract_feat(img_r, img_q)
        out = self._decode_head_forward_test(x2, img_metas)
        out = resize(
            input=out,
            size=img_q.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(
            self, x, img_metas, gt_semantic_seg, gt_masks_q, gt_labels_q
    ):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(
            x, img_metas, gt_semantic_seg, gt_labels_q, gt_masks_q
        )

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    # def forward_dummy(self, img):
    #     """Dummy forward function."""
    #     seg_logit = self.encode_decode(img, None)
    #
    #     return seg_logit
    #

    def forward_dummy(self, img):
        """Dummy forward function."""
        # seg_logit = self.encode_decode(img, img)

        x, x2 = self.extract_feat(img, img)
        out = self.decode_head.forward_test(x2)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_logit = out

        return seg_logit

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**{'vids':data_batch})
        # losses = self(data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch[0]['img_metas']))

        return outputs

    @auto_fp16(apply_to=('img',))
    # def forward(self, img, img_metas, return_loss=True, **kwargs):
    def forward(self, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        # return_loss = True
        if return_loss:
            # return self.forward_train(img, img_metas, **kwargs)
            data_batch = kwargs['vids']
            return self.forward_train(data_batch, **kwargs)
        else:
            # img_metas, img = kwargs.pop('img_metas'), kwargs.pop('img')
            # return self.forward_test(img, img_metas, **kwargs)
            data_batch = kwargs['vids']
            return self.forward_test(data_batch, **kwargs)

    # def forward_test(self, imgs, img_metas, **kwargs):
    def forward_test(self, data_batch, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        ref_data, query_data = data_batch
        img_metas_r, imgs_r = ref_data.values()
        img_metas_q, imgs_q = query_data.values()

        for var, name in [(imgs_q, 'imgs'), (img_metas_q, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')

        num_augs = len(imgs_q)
        if num_augs != len(img_metas_q):
            raise ValueError(f'num of augmentations ({len(imgs_q)}) != '
                             f'num of image meta ({len(img_metas_q)})')
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas_q:
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_['img_shape'] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            return self.simple_test(imgs_r[0], imgs_q[0], img_metas_q[0])#, **kwargs)
        else:
            return self.aug_test(imgs_r, imgs_q, img_metas_q)#, rescale=kwargs['rescale'])


    # def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
    def forward_train(self, data_batch, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        ref_data, query_data = data_batch
        img_metas_r, img_r, gt_semantic_seg_r, gt_masks_r, gt_labels_r = ref_data.values()
        img_metas_q, img_q, gt_semantic_seg_q, gt_masks_q, gt_labels_q = query_data.values()

        x_r, x_q = self.extract_feat(img_r, img_q)

        losses = dict()

        loss_decode_q = self._decode_head_forward_train(x_q, img_metas_q,
                                                        gt_semantic_seg_q,
                                                        gt_masks_q,
                                                        gt_labels_q)
        loss_decode_r = self._decode_head_forward_train(x_r, img_metas_r,
                                                        gt_semantic_seg_r,
                                                        gt_masks_r,
                                                        gt_labels_r)
        loss_fuse = {}
        for k, v in loss_decode_q.items():
            loss_fuse[k] = loss_decode_r[k] + v
        losses.update(loss_fuse)
        # losses.update(loss_decode_q)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x_q, img_metas_q, gt_semantic_seg_q)
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(self, imgs_r, img, img_meta, rescale, unpad=True):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

        avg_time = 0
        count = 0
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img_r = imgs_r[:, :, y1:y2, x1:x2]
                crop_img = img[:, :, y1:y2, x1:x2]
                start = time.time()
                crop_seg_logit = self.encode_decode(crop_img_r, crop_img, img_meta)
                end = time.time()
                avg_time += end-start
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1

                count += 1
        avg_time = avg_time/count
        print(avg_time)
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat

        if unpad:
            unpad_h, unpad_w = img_meta[0]['img_shape'][:2]
            # logging.info(preds.shape, img_meta[0])
            preds = preds[:, :, :unpad_h, :unpad_w]
        if rescale:
            preds = resize(preds,
                           size=img_meta[0]['ori_shape'][:2],
                           mode='bilinear',
                           align_corners=self.align_corners,
                           warning=False)
        return preds

    def whole_inference(self, imgs_r, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(imgs_r, img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, imgs_r, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(imgs_r, img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(imgs_r, img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))

        return output

    def simple_test(self, img_r, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img_r, img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs_r, imgs_q, img_metas_q, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs_r[0], imgs_q[0], img_metas_q[0], rescale)
        for i in range(1, len(imgs_q)):
        # for i in range(2):
            cur_seg_logit = self.inference(imgs_r[i], imgs_q[i], img_metas_q[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs_q)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
