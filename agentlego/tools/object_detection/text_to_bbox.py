# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

from agentlego.parsers import DefaultParser
from agentlego.schema import ToolMeta
from agentlego.types import ImageIO
from agentlego.utils import load_or_build_object, require
from ..base import BaseTool


class TextToBbox(BaseTool):
    """A tool to detection the given object.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        model (str): The model name used to detect texts.
            Which can be found in the ``MMDetection`` repository.
            Defaults to ``glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365``.
        device (str): The device to load the model. Defaults to 'cpu'.
    """
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
        results = self._inferencer(image, texts=text, return_vis=True)
        output_image = results['visualization'][0]
        return ImageIO(output_image)
