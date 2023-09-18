# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

from mmlmtools.parsers import DefaultParser
from mmlmtools.schema import ToolMeta
from mmlmtools.types import ImageIO
from mmlmtools.utils import load_or_build_object, require
from ..base import BaseTool


class TextToBbox(BaseTool):
    DEFAULT_TOOLMETA = ToolMeta(
        name='Detect the Given Object',
        description=('A useful tool when you only want to show the location '
                     'of given objects, or detect or find out given objects '
                     'in the picture. like: locate persons in the picture'),
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
        image = image.to_path()
        results = self._inferencer(image, texts=text)
        output_image = results['visualization']
        return ImageIO(output_image)
