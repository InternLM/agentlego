# Copyright (c) OpenMMLab. All rights reserved.

from typing import Callable, Union

import mmcv

from mmlmtools.parsers import DefaultParser
from mmlmtools.schema import ToolMeta
from mmlmtools.types import ImageIO
from mmlmtools.utils import get_new_file_path, load_or_build_object, require
from ..base import BaseTool


class ObjectDetection(BaseTool):
    DEFAULT_TOOLMETA = ToolMeta(
        name='Detect All Objects',
        description=('A useful tool when you only want to detect the picture '
                     'or detect all objects in the picture. like: detect all '
                     'objects. '),
        inputs=['image'],
        outputs=['image'],
    )

    @require('mmdet>=3.1.0')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 model: str = 'rtmdet_l_8xb32-300e_coco',
                 device: str = 'cpu'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.model = model
        self.device = device

    def setup(self):
        from mmdet.apis import DetInferencer
        self._inferencer = load_or_build_object(
            DetInferencer, model=self.model, device=self.device)

    def apply(self, image: ImageIO) -> ImageIO:
        image = image.to_path()
        print(image)
        results = self._inferencer(
            image, no_save_vis=True, return_datasample=True)
        output_image = get_new_file_path(image, func_name='detect-something')
        img = mmcv.imread(image)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        self._inferencer.visualizer.add_datasample(
            'results',
            img,
            data_sample=results['predictions'][0],
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=output_image,
            pred_score_thr=0.5)
        return ImageIO(output_image)
