# Copyright (c) OpenMMLab. All rights reserved.
from .encoder_decoder_mask2former import EncoderDecoderMask2Former
from .encoder_decoder_mask2former_aug import EncoderDecoderMask2FormerAug
from .encoder_decoder_mask2former_aug_vid import EncoderDecoderMask2FormerAugVideo
from .encoder_decoder_perpixel_aug_vid import EncoderDecoderPerPixelPlusAugVideo
from .encoder_decoder_perpixel_aug_vid_test_multi_frame import EncoderDecoderPerPixelPlusAugVideoTestMulti
# from .encoder_decoder_perpixel_label_decouple_aug_vid_only_for_test import EncoderDecoderPerPixelPlusAugVideo
from .encoder_decoder_mask2former_aug_vid_test_multi_frame import EncoderDecoderMask2FormerAugVideoTestMulti

__all__ = [
    'EncoderDecoderMask2Former',
    'EncoderDecoderMask2FormerAug',
    'EncoderDecoderMask2FormerAugVideo',
    'EncoderDecoderMask2FormerAugVideoTestMulti',
    'EncoderDecoderPerPixelPlusAugVideo',
    'EncoderDecoderPerPixelPlusAugVideoTestMulti',
]
