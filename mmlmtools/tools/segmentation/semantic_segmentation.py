# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

import mmcv

from mmlmtools.parsers import DefaultParser
from mmlmtools.schema import ToolMeta
from mmlmtools.types import ImageIO
from mmlmtools.utils import get_new_file_path, load_or_build_object, require
from ..base import BaseTool


class SemanticSegmentation(BaseTool):
    DEFAULT_TOOLMETA = ToolMeta(
        name='Segment the Image',
        description=(
            'This is a useful tool when you only want to segment the '
            'picture or segment all objects in the picture. like: '
            'segment all objects. '),
        inputs=['image'],
        outputs=['image'],
    )

    @require('mmsegmentation')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 seg_model: str = (
                     'mask2former_r50_8xb2-90k_cityscapes-512x1024'),
                 device: str = 'cpu'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.seg_model = seg_model
        self.device = device

    def setup(self):
        from mmseg.apis import MMSegInferencer

        self._inferencer = load_or_build_object(
            MMSegInferencer,
            model=self.seg_model,
            device=self.device)

    def apply(self, image: ImageIO) -> ImageIO:
        results = self._inferencer(image, return_datasamples=True)
        output_path = get_new_file_path(
            image, func_name='semseg-something')
        img = mmcv.imread(image)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        self._inferencer.visualizer.add_datasample(
            'results',
            img,
            data_sample=results,
            draw_gt=False,
            draw_pred=True,
            show=False,
            out_file=output_path)
        return ImageIO(output_path)
