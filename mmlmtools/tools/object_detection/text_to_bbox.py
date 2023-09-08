# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

import mmcv

from mmlmtools.parsers import DefaultParser
from mmlmtools.types import ImageIO
from mmlmtools.utils import get_new_file_path, load_or_build_object, require
from mmlmtools.utils.schema import ToolMeta
from ..base_tool import BaseTool


class TextToBbox(BaseTool):
    DEFAULT_TOOLMETA = ToolMeta(
        name='Detect the Given Object',
        description=('A useful tool when you only want to show the location '
                     'of given objects, or detect or find out given objects '
                     'in the picture. '),
        inputs=['image', 'text'],
        outputs=['image'],
    )

    @require('mmdet>=3.1.0')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 model: str = 'glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365',
                 device: str = 'cpu'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.model = model
        self.device = device

    def setup(self):
        from mmdet.apis import DetInferencer
        self._inferencer = load_or_build_object(
            DetInferencer, model=self.model, device=self.device)

    def apply(self, image: ImageIO, text: str) -> ImageIO:
        results = self._inferencer(
            inputs=image, texts=text, no_save_vis=True, return_datasample=True)

        output_image = get_new_file_path(
            image, func_name='detect-something')['predictions'][0]

        img = mmcv.imread(image, channel_order='rgb')

        self._inferencer.visualizer.add_datasample(
            'results',
            img,
            data_sample=results['predictions'][0],
            show=False,
            out_file=output_image)

        return ImageIO(output_image)
