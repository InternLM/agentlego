# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

from mmlmtools.parsers import DefaultParser
from mmlmtools.schema import ToolMeta
from mmlmtools.types import ImageIO
from mmlmtools.utils import load_or_build_object, require
from ..base import BaseTool


class ImageCaption(BaseTool):
    """A tool to describe an image.

    Args:
        toolmeta (dict | ToolMeta): The meta info of the tool. Defaults to
            the :attr:`DEFAULT_TOOLMETA`.
        parser (Callable): The parser constructor, Defaults to
            :class:`DefaultParser`.
        model (str): The model name used to inference. Which can be found
            in the ``MMPreTrain`` repository.
            Defaults to ``blip-base_3rdparty_caption``.
        device (str): The device to load the model. Defaults to 'cpu'.
    """

    DEFAULT_TOOLMETA = ToolMeta(
        name='Image Description',
        description=('A useful tool that returns the '
                     'description of the input image.'),
        inputs=['image'],
        outputs=['text'],
    )

    @require('mmpretrain')
    def __init__(self,
                 toolmeta: Union[dict, ToolMeta] = DEFAULT_TOOLMETA,
                 parser: Callable = DefaultParser,
                 model: str = 'blip-base_3rdparty_caption',
                 device: str = 'cpu'):
        super().__init__(toolmeta=toolmeta, parser=parser)
        self.model = model
        self.device = device

    def setup(self):
        from mmpretrain.apis import ImageCaptionInferencer
        self._inferencer = load_or_build_object(
            ImageCaptionInferencer, model=self.model, device=self.device)

    def apply(self, image: ImageIO) -> str:
        image = image.to_array()[:, :, ::-1]
        return self._inferencer(image)[0]['pred_caption']
